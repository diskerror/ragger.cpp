#!/usr/bin/env python3
"""
import_gemini.py — Import Gemini memory into ~/.ragger/memories.db

Source: ~/.gemini/GEMINI.md

The LLM cleans up and individually extracts each bullet as a decision row.

Usage:
    python3 scripts/import/import_gemini.py [--dry-run] [--verbose] [--no-llm]
    python3 scripts/import/import_gemini.py --settings /path/to/settings.ini
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (
    RaggerConfig, open_db, mtime_ts, is_after_birthday,
    llm_classify_flat_bullets,
    decision_exists_by_text,
    insert_decision,
)

GEMINI_MD = Path.home() / ".gemini" / "GEMINI.md"


def naive_bullets(text: str) -> list:
    """Fallback: extract bullet lines without LLM."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = re.sub(r'^[-*]\s*', '', line).strip()
        if len(item) >= 20:
            items.append(item)
    return items


def main():
    ap = argparse.ArgumentParser(description="Import Gemini GEMINI.md into Ragger")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    cfg = RaggerConfig(Path(args.settings).expanduser()) if args.settings else RaggerConfig()

    if not GEMINI_MD.exists():
        print(f"ERROR: {GEMINI_MD} not found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    ts   = mtime_ts(GEMINI_MD)
    text = GEMINI_MD.read_text(encoding="utf-8").strip()
    tags = "imported,gemini"

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Gemini GEMINI.md import")
    print(f"Source: {GEMINI_MD}  ({ts})")
    print(f"DB:     {cfg.db_path}\n")

    if not is_after_birthday(ts):
        # File is older than birthday — still import, it was written during setup
        pass

    if args.no_llm:
        items = naive_bullets(text)
    else:
        print(f"Classifying with {cfg.llm_model!r} @ {cfg.llm_base_url} …")
        try:
            items = llm_classify_flat_bullets(text, "Gemini GEMINI.md", cfg=cfg)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            print("Tip: use --no-llm if the inference endpoint is offline")
            items = naive_bullets(text)
            print(f"Falling back to naive extraction: {len(items)} items")

    n_ins = n_skip = 0
    for item in items:
        if decision_exists_by_text(cur, item):
            if args.verbose:
                print(f"  [DUP]  {item[:70]}…")
            n_skip += 1
        else:
            if args.verbose:
                print(f"  [INSERT] {item[:70]}…")
            if not args.dry_run:
                insert_decision(cur, item, tags, ts)
            n_ins += 1

    if not args.dry_run:
        con.commit()
    con.close()

    print(f"\nDecisions: {n_ins} inserted, {n_skip} skipped")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
