#!/usr/bin/env python3
"""
import_openclaw.py — Import OpenClaw daily notes into ~/.ragger/memories.db

Source: ~/OpenClawBU2026-05-10/workspace/memory/*.md  (canonical — superset of OpenClawOrig)

The LLM (configured via [summarizer] in ~/.ragger/settings.ini) reads each note and returns:
  - decisions[]:  durable choices/facts  → decisions table
  - summary:      narrative of the day   → summaries (level=session)

Dedup:
  - summary:  skip if (timestamp, level='session') already exists
  - decision: skip if first 120 chars match an existing decision row

Usage:
    python3 scripts/import/import_openclaw.py [--dry-run] [--verbose] [--no-llm]
    python3 scripts/import/import_openclaw.py --settings /path/to/settings.ini

    --no-llm   Insert raw text as session summaries only (no decision extraction).
               Useful when the inference endpoint is offline.
    --settings Override the settings.ini path (default ~/.ragger/settings.ini).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (
    RaggerConfig, open_db, mtime_ts, is_after_birthday,
    llm_classify_daily_note,
    summary_exists, decision_exists_by_text,
    insert_summary, insert_decision,
    is_junk_content, get_or_create_model,
)

BU_MEMORY = Path.home() / "OpenClawBU2026-05-10" / "workspace" / "memory"


def ts_from_stem(stem: str):
    """Parse filename stem → DB timestamp string, or None."""
    # YYYY-MM-DD
    m = re.match(r'^(\d{4}-\d{2}-\d{2})$', stem)
    if m:
        return f"{m.group(1)} 00:00:00"
    # YYYY-MM-DD-HHMMSS
    m = re.match(r'^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})(\d{2})$', stem)
    if m:
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
    # YYYY-MM-DD-HHMM
    m = re.match(r'^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})$', stem)
    if m:
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:00"
    # YYYY-MM-DD-slug  (e.g. 2026-04-19-request-timed-out-...)
    m = re.match(r'^(\d{4}-\d{2}-\d{2})-', stem)
    if m:
        return f"{m.group(1)} 00:00:00"
    return None


def extract_session_content(text: str) -> tuple:
    """
    For OpenClaw structured session files that start with:
        # Session: 2026-04-22 19:11:27 UTC
        - Session Key: ...
        - Session ID: ...
        - Source: ...

        ## Conversation Summary
        <actual content>

    Returns (real_timestamp_or_None, cleaned_text).
    If no header found, returns (None, original text).

    Also strips metadata preamble blobs injected by the OpenClaw gateway:
        user: Conversation info (untrusted metadata):
        ```json
        { ... }
        ```
        user: Sender (untrusted metadata):
        ```json
        { ... }
        ```
    and pure system-noise lines (session-start banners, aborted-run notices).
    """
    ts = None
    # Extract timestamp from header line
    m = re.search(r'^#\s*Session:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', text, re.MULTILINE)
    if m:
        ts = f"{m.group(1)} {m.group(2)}"

    # Strip everything up to and including "## Conversation Summary"
    split = re.split(r'^##\s*Conversation Summary\s*$', text, maxsplit=1, flags=re.MULTILINE)
    if len(split) == 2:
        text = split[1].strip()

    # Strip metadata preamble blobs: lines starting with known noise prefixes
    # followed by optional ```json ... ``` fenced blocks.
    # We do this by splitting into paragraphs and dropping noise ones.
    _NOISE_PARA_PREFIXES = (
        "user: Conversation info (untrusted metadata)",
        "user: Sender (untrusted metadata)",
        "user: A new session was started via /new",
        "user: Note: The previous agent run was aborted",
        "assistant: ⚠️",
        "assistant: ⚙️",
        "assistant: ℹ️",
    )
    paragraphs = re.split(r'\n{2,}', text)
    clean_paras = []
    for para in paragraphs:
        stripped = para.strip()
        if any(stripped.startswith(prefix) for prefix in _NOISE_PARA_PREFIXES):
            continue
        # Also drop bare "Logs: ..." lines as standalone paragraphs
        if re.match(r'^Logs:\s*\w', stripped):
            continue
        if stripped:
            clean_paras.append(stripped)

    text = "\n\n".join(clean_paras)
    return ts, text


def main():
    ap = argparse.ArgumentParser(description="Import OpenClaw daily notes into Ragger")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM pass; import raw text as session summaries only")
    ap.add_argument("--settings", default=None,
                    help="Path to settings.ini (default: ~/.ragger/settings.ini)")
    args = ap.parse_args()

    cfg = RaggerConfig(Path(args.settings).expanduser()) if args.settings else RaggerConfig()

    if not BU_MEMORY.exists():
        print(f"ERROR: {BU_MEMORY} not found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    import_model_id = get_or_create_model(cur, "openclaw-import") if not args.dry_run else 0

    mode = "DRY RUN — " if args.dry_run else ""
    llm_note = " (--no-llm: raw text only)" if args.no_llm else f" (LLM: {cfg.llm_model!r} @ {cfg.llm_base_url})"
    print(f"{mode}OpenClaw daily notes import{llm_note}")
    print(f"Source: {BU_MEMORY}")
    print(f"DB:     {cfg.db_path}\n")

    files = sorted(BU_MEMORY.glob("*.md"))
    n_sum_ins = n_sum_skip = n_dec_ins = n_dec_skip = 0

    for f in files:
        ts = ts_from_stem(f.stem)
        if ts is None:
            print(f"  [SKIP] can't parse timestamp: {f.name}")
            n_sum_skip += 1
            continue
        if not is_after_birthday(ts):
            if args.verbose:
                print(f"  [SKIP] {f.name} — before birthday")
            n_sum_skip += 1
            continue

        text = f.read_text(encoding="utf-8").strip()
        if not text or len(text) < cfg.minimum_chunk_size:
            if args.verbose:
                print(f"  [SKIP] {f.name} — too short ({len(text)} chars)")
            n_sum_skip += 1
            continue

        # Extract real timestamp and strip session header if present
        embedded_ts, text = extract_session_content(text)
        if embedded_ts:
            ts = embedded_ts

        if not text or len(text) < cfg.minimum_chunk_size:
            if args.verbose:
                print(f"  [SKIP] {f.name} — too short after header strip")
            n_sum_skip += 1
            continue

        if is_junk_content(text):
            if args.verbose:
                print(f"  [JUNK] {f.name} — pure system noise, skipping")
            n_sum_skip += 1
            continue

        date   = ts[:10]
        t_tags = "imported,openclaw,daily-note"
        d_tags = f"imported,openclaw,{date}"

        if args.no_llm:
            if summary_exists(cur, ts, "session"):
                if args.verbose:
                    print(f"  [DUP]  {f.name} @ {ts}")
                n_sum_skip += 1
            else:
                if args.verbose:
                    print(f"  [INSERT] summary {f.name} @ {ts} ({len(text)} chars)")
                if not args.dry_run:
                    insert_summary(cur, text, "session", ts, t_tags, model_id=import_model_id)
                n_sum_ins += 1
        else:
            print(f"  [{date}] classifying {f.name} …", end=" ", flush=True)
            try:
                result = llm_classify_daily_note(text, date, cfg=cfg)
            except RuntimeError as e:
                print(f"\n  ERROR: {e}")
                print("  Tip: use --no-llm if the inference endpoint is offline")
                con.rollback()
                con.close()
                sys.exit(1)

            summary_text = result.get("summary", "").strip()
            decisions    = [d.strip() for d in result.get("decisions", []) if d.strip()]
            print(f"{len(decisions)} decisions, summary {len(summary_text)} chars")

            # Session summary
            if not summary_text:
                n_sum_skip += 1
            elif summary_exists(cur, ts, "session"):
                if args.verbose:
                    print(f"    [DUP]  summary @ {ts}")
                n_sum_skip += 1
            else:
                if not args.dry_run:
                    insert_summary(cur, summary_text, "session", ts, t_tags, model_id=import_model_id)
                n_sum_ins += 1

            # Decisions
            for dec in decisions:
                if decision_exists_by_text(cur, dec):
                    if args.verbose:
                        print(f"    [DUP]  dec: {dec[:60]}…")
                    n_dec_skip += 1
                else:
                    if args.verbose:
                        print(f"    [DEC]  {dec[:70]}…")
                    if not args.dry_run:
                        insert_decision(cur, dec, d_tags, ts)
                    n_dec_ins += 1

    if not args.dry_run:
        con.commit()
    con.close()

    print(f"\nSummaries: {n_sum_ins} inserted, {n_sum_skip} skipped")
    print(f"Decisions: {n_dec_ins} inserted, {n_dec_skip} skipped")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
