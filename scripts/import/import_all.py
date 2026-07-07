#!/usr/bin/env python3
"""
import_all.py — Run all agent memory import scripts in sequence.

Passes --dry-run / --verbose / --no-llm / --settings through to each script.

Usage:
    python3 scripts/import/import_all.py [--dry-run] [--verbose] [--no-llm]
    python3 scripts/import/import_all.py --settings /path/to/settings.ini

Order:
  1. import_openclaw.py  — OpenClaw daily notes (biggest, most structured)
  2. import_claude.py    — Claude Code project memory files
  3. import_gemini.py    — Gemini GEMINI.md bullets
  4. import_hermes.py    — Hermes MEMORY.md §-blocks
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "import_openclaw.py",
    "import_claude.py",
    "import_gemini.py",
    "import_hermes.py",
]

SCRIPT_DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description="Run all Ragger memory import scripts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    flags = []
    if args.dry_run:  flags.append("--dry-run")
    if args.verbose:  flags.append("--verbose")
    if args.no_llm:   flags.append("--no-llm")
    if args.settings: flags += ["--settings", args.settings]

    overall_ok = True
    for script in SCRIPTS:
        path = SCRIPT_DIR / script
        print(f"\n{'='*60}", flush=True)
        print(f"  {script}", flush=True)
        print(f"{'='*60}", flush=True)
        result = subprocess.run(
            [sys.executable, str(path)] + flags,
            check=False,
        )
        if result.returncode != 0:
            print(f"\n[FAILED] {script} exited with code {result.returncode}", flush=True)
            overall_ok = False

    print(f"\n{'='*60}", flush=True)
    print("All scripts complete." if overall_ok else "One or more scripts failed.", flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
