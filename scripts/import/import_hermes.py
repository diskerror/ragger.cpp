#!/usr/bin/env python3
"""
import_hermes.py — Import Hermes MEMORY.md into ~/.ragger/memories.db

Source: ~/.hermes/memories/MEMORY.md

Hermes MEMORY.md is § -delimited blocks. The LLM cleans each block into a
self-contained decision/fact string and inserts it into the decisions table.

Usage:
    python3 scripts/import/import_hermes.py [--dry-run] [--verbose] [--no-llm]
    python3 scripts/import/import_hermes.py --settings /path/to/settings.ini
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (
    RaggerConfig, open_db, mtime_ts, is_after_birthday,
    llm_classify_flat_bullets,
    decision_exists_by_text,
    insert_decision,
)

HERMES_MEMORY = Path.home() / ".hermes" / "memories" / "MEMORY.md"


def main():
    ap = argparse.ArgumentParser(description="Import Hermes MEMORY.md into Ragger")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    cfg = RaggerConfig(Path(args.settings).expanduser()) if args.settings else RaggerConfig()

    if not HERMES_MEMORY.exists():
        print(f"ERROR: {HERMES_MEMORY} not found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    ts   = mtime_ts(HERMES_MEMORY)
    text = HERMES_MEMORY.read_text(encoding="utf-8").strip()
    tags = "imported,hermes"

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Hermes MEMORY.md import")
    print(f"Source: {HERMES_MEMORY}  ({ts})")
    print(f"DB:     {cfg.db_path}\n")

    # Split on § — each block is one memory entry
    blocks = [b.strip() for b in text.split("§") if b.strip() and len(b.strip()) >= 20]
    print(f"Found {len(blocks)} §-delimited blocks")

    if args.no_llm:
        items = blocks
    else:
        print(f"Classifying with {cfg.llm_model!r} @ {cfg.llm_base_url} …")
        try:
            items = llm_classify_flat_bullets(
                "\n\n---\n\n".join(blocks),
                "Hermes MEMORY.md",
                cfg=cfg,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}")
            print("Falling back to raw blocks (--no-llm mode)")
            items = blocks

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
