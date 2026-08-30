#!/usr/bin/env python3
"""
import_gemini.py — Import Gemini memory into ~/.ragger/memories.db

Source: ~/.gemini/GEMINI.md
        Pass a path argument to override.

The LLM cleans up and individually extracts each bullet as a decision row.

Usage:
    python3 scripts/import/import_gemini.py [--dry-run] [--verbose] [--no-llm]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (
    RaggerConfig, add_base_arg, add_source_arg, resolve_source, config_from_args, open_db, mtime_ts,
    llm_classify_flat_bullets,
    decision_exists_by_text,
    insert_decision,
)

DEFAULT_SOURCE = Path.home() / ".gemini" / "GEMINI.md"


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
    add_source_arg(ap, DEFAULT_SOURCE, "file", "Gemini GEMINI.md")
    add_base_arg(ap)
    args = ap.parse_args()

    cfg = config_from_args(args)
    SOURCE = resolve_source(args.source, DEFAULT_SOURCE, "file", "Gemini GEMINI.md")

    if not SOURCE.exists():
        print(f"ERROR: {SOURCE} not found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    ts   = mtime_ts(SOURCE)
    text = SOURCE.read_text(encoding="utf-8").strip()
    tags = "imported,gemini"

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Gemini GEMINI.md import")
    print(f"Source: {SOURCE}  ({ts})")
    print(f"DB:     {cfg.db_path}\n")

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
