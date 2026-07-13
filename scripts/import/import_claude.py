#!/usr/bin/env python3
"""
import_claude.py — Import Claude Code project memory files into ~/.ragger/memories.db

Source: ~/.claude/projects/*/memory/*.md

The LLM (configured via [summarizer] in ~/.ragger/settings.ini) reads each file and
classifies it as either:
  - "decision" → individual items split into decisions table rows
  - "summary"  → whole file as a project-level summary row

Skips:
  - Files with mtime before birthday (2026-02-13)
  - Files shorter than minimum_chunk_size (from [import] section)
  - Index/TOC files (only contain links to other files, no real content)

Dedup:
  - summaries: by first 120 chars of text + level='project'
  - decisions: by first 120 chars of text

Usage:
    python3 scripts/import/import_claude.py [--dry-run] [--verbose] [--no-llm]
    python3 scripts/import/import_claude.py --settings /path/to/settings.ini
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (
    RaggerConfig, open_db, mtime_ts, is_after_birthday,
    llm_classify_project_memory,
    summary_exists_by_text, decision_exists_by_text,
    insert_summary, insert_decision, get_or_create_model,
)

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def is_index_file(text: str) -> bool:
    """True if the file is purely a link-index (no prose content)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    non_link = [l for l in lines if not l.startswith("#") and not re.match(r"^- \[", l)]
    return len(non_link) == 0


def slug_from_project_dir(name: str) -> str:
    """Turn the filesystem-encoded project dir name into a readable slug.

    Claude Code encodes a project's absolute path into a directory name by
    replacing every path separator (and '.') with '-' — e.g. a project at
    "$HOME/CLionProjects/Ragger" becomes "-Volumes-WDBlack2-CLionProjects-
    Ragger" on this machine, or "-home-alice-CLionProjects-Ragger" on a
    typical Linux box. We strip the home-directory prefix in that same
    encoded form (derived from Path.home() at runtime) rather than matching
    a hardcoded machine-specific segment, so this works on any machine/user.
    """
    encoded_home_parts = re.sub(r"[/.]", "-", str(Path.home())).strip("-").split("-")
    parts = name.lstrip("-").split("-")
    if parts[:len(encoded_home_parts)] == encoded_home_parts:
        parts = parts[len(encoded_home_parts):]
    slug = "-".join(parts) if parts else name
    return slug[:50]


def main():
    ap = argparse.ArgumentParser(description="Import Claude project memory into Ragger")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM pass; import all files as project summaries")
    ap.add_argument("--settings", default=None,
                    help="Path to settings.ini (default: ~/.ragger/settings.ini)")
    args = ap.parse_args()

    cfg = RaggerConfig(Path(args.settings).expanduser()) if args.settings else RaggerConfig()

    if not CLAUDE_PROJECTS.exists():
        print(f"ERROR: {CLAUDE_PROJECTS} not found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    import_model_id = get_or_create_model(cur, "claude-opus-4-5") if not args.dry_run else 0

    mode     = "DRY RUN — " if args.dry_run else ""
    llm_note = " (--no-llm)" if args.no_llm else f" (LLM: {cfg.llm_model!r} @ {cfg.llm_base_url})"
    print(f"{mode}Claude project memory import{llm_note}")
    print(f"Source: {CLAUDE_PROJECTS}")
    print(f"DB:     {cfg.db_path}\n")

    n_sum_ins = n_sum_skip = n_dec_ins = n_dec_skip = 0

    for proj_dir in sorted(CLAUDE_PROJECTS.iterdir()):
        mem_dir = proj_dir / "memory"
        if not mem_dir.exists():
            continue
        slug = slug_from_project_dir(proj_dir.name)

        for f in sorted(mem_dir.glob("*.md")):
            ts   = mtime_ts(f)
            text = f.read_text(encoding="utf-8").strip()

            if not is_after_birthday(ts):
                if args.verbose:
                    print(f"  [SKIP] {slug}/{f.name} — before birthday")
                n_sum_skip += 1
                continue
            if len(text) < cfg.minimum_chunk_size:
                if args.verbose:
                    print(f"  [SKIP] {slug}/{f.name} — too short ({len(text)} chars)")
                n_sum_skip += 1
                continue
            if is_index_file(text):
                if args.verbose:
                    print(f"  [SKIP] {slug}/{f.name} — index/TOC file")
                n_sum_skip += 1
                continue

            tags_sum = f"imported,claude,{slug}"
            tags_dec = f"imported,claude,{slug},{f.stem}"

            if args.no_llm:
                if summary_exists_by_text(cur, text, "project"):
                    if args.verbose:
                        print(f"  [DUP]  {slug}/{f.name}")
                    n_sum_skip += 1
                else:
                    if args.verbose:
                        print(f"  [INSERT] summary {slug}/{f.name} @ {ts}")
                    if not args.dry_run:
                        insert_summary(cur, text, "project", ts, tags_sum, model_id=import_model_id)
                    n_sum_ins += 1
            else:
                print(f"  [{slug}/{f.name}] classifying …", end=" ", flush=True)
                try:
                    result = llm_classify_project_memory(text, f.name, slug, cfg=cfg)
                except RuntimeError as e:
                    print(f"\n  ERROR: {e}")
                    con.rollback()
                    con.close()
                    sys.exit(1)

                kind  = result.get("type", "summary")
                items = [i.strip() for i in result.get("items", []) if i.strip()]
                print(f"type={kind}, {len(items)} items")

                if kind == "decision":
                    for item in items:
                        if decision_exists_by_text(cur, item):
                            if args.verbose:
                                print(f"    [DUP]  dec: {item[:60]}…")
                            n_dec_skip += 1
                        else:
                            if args.verbose:
                                print(f"    [DEC]  {item[:70]}…")
                            if not args.dry_run:
                                insert_decision(cur, item, tags_dec, ts)
                            n_dec_ins += 1
                else:
                    for item in items:
                        if summary_exists_by_text(cur, item, "project"):
                            if args.verbose:
                                print(f"    [DUP]  summary")
                            n_sum_skip += 1
                        else:
                            if not args.dry_run:
                                insert_summary(cur, item, "project", ts, tags_sum, model_id=import_model_id)
                            n_sum_ins += 1

    if not args.dry_run:
        con.commit()
    con.close()

    print(f"\nSummaries: {n_sum_ins} inserted, {n_sum_skip} skipped")
    print(f"Decisions: {n_dec_ins} inserted, {n_dec_skip} skipped")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
