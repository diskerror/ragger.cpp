#!/usr/bin/env python3
"""
import_claude_export.py — Import Claude.ai conversation export into ~/.ragger/memories.db

Sources:
  ~/Desktop/claude convo/data-*/conversations.json

Each conversation becomes a session. Human+assistant message pairs become turns
inserted with model_id=NULL and embedding=NULL — exactly like live captures.
The daemon's housekeeping then summarizes and embeds them normally.

Content extraction:
  - 'text' field used when non-empty (plain string)
  - 'content' list: concatenates all 'text' type blocks; skips tool_use/tool_result
  - Attachments: appends filename + text content when present

Dedup:
  - Sessions: skip if guid already in sessions table
  - Turns: skip if (session_id, timestamp) already exists

Filtering:
  - Conversations with 0 messages skipped
  - Turns where both user and assistant text are empty skipped
  - Conversations before birthday (2026-02-13) skipped unless --all

Usage:
    python3 scripts/import/import_claude_export.py [--dry-run] [--verbose] [--all]
    python3 scripts/import/import_claude_export.py --settings /path/to/settings.ini

    --all    Include conversations before 2026-02-13 (default: skip pre-birthday)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import RaggerConfig, open_db, BIRTHDAY, is_junk_content

EXPORT_BASE = Path.home() / "Desktop" / "claude convo"
EXPORT_DIRS = [
    "data-78037790-ba82-4dfe-b096-592b723f805f-1780344373-2468d4bb-batch-0000",
    "data-90174c9a-ca89-4ced-9ccf-926233989df8-1780344573-79dc889c-batch-0000",
]
# Also pick up any unzipped batch dir we haven't listed
MODEL_NAME = "claude.ai-export"


# ── helpers ───────────────────────────────────────────────────────────────────

def iso_to_ts(iso: str) -> str:
    """ISO 8601 → 'YYYY-MM-DD HH:MM:SS' for DB storage."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso[:19].replace("T", " ")


def extract_text(msg: dict) -> str:
    """
    Pull readable text from a message.
    Prefers 'text' field; falls back to 'content' list (text blocks only).
    Appends attachment text when present.
    """
    text = (msg.get("text") or "").strip()

    if not text:
        content = msg.get("content") or []
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts)
        elif isinstance(content, str):
            text = content.strip()

    # Append attachment content
    for att in msg.get("attachments") or []:
        fname = att.get("file_name") or att.get("name") or ""
        body  = (att.get("extracted_content") or att.get("content") or "").strip()
        if body:
            text += f"\n\n[Attachment: {fname}]\n{body}"

    return text.strip()


def pair_turns(messages: list) -> list:
    """
    Pair human+assistant messages into (user_text, assistant_text, user_ts, asst_ts) tuples.
    Handles non-strict alternation by collecting consecutive human messages
    before each assistant reply.
    """
    pairs = []
    pending_human = []
    pending_ts    = None

    for msg in messages:
        sender = msg.get("sender", "")
        text   = extract_text(msg)
        ts     = iso_to_ts(msg.get("created_at", ""))

        if sender == "human":
            if pending_human:
                # Two human messages in a row — merge
                pending_human.append(text)
            else:
                pending_human = [text]
                pending_ts    = ts
        elif sender == "assistant":
            user_text  = "\n\n".join(t for t in pending_human if t)
            asst_text  = text
            if user_text or asst_text:
                # Skip turns that are pure system noise (agent errors, restarts, etc.)
                if not is_junk_content(user_text) or not is_junk_content(asst_text):
                    pairs.append((user_text, asst_text, pending_ts or ts, ts))
            pending_human = []
            pending_ts    = None

    # Trailing human message with no assistant reply
    if pending_human:
        user_text = "\n\n".join(t for t in pending_human if t)
        if user_text:
            pairs.append((user_text, "", pending_ts, pending_ts))

    return pairs


def get_or_create_session(cur, guid: str) -> int:
    """Return session_id for guid, inserting if needed. Returns -1 if already exists."""
    cur.execute("SELECT session_id FROM sessions WHERE guid=?", (guid,))
    row = cur.fetchone()
    if row:
        return -1  # already imported
    cur.execute("INSERT INTO sessions (guid) VALUES (?)", (guid,))
    return cur.lastrowid


def get_or_create_model(cur, name: str) -> int:
    cur.execute("SELECT model_id FROM models WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO models (name) VALUES (?)", (name,))
    return cur.lastrowid


def insert_turn(cur, user_text: str, asst_text: str,
                session_id: int, timestamp: str, model_id: int) -> None:
    """Insert a raw turn with NULL embedding — daemon backfills it."""
    cur.execute(
        """INSERT INTO turns
               (user_text, assistant_text, embedding, model_id, session_id, timestamp)
           VALUES (?, ?, NULL, ?, ?, ?)""",
        (user_text, asst_text, model_id, session_id, timestamp),
    )


def insert_placeholder_summary(cur, user_text: str, asst_text: str,
                                session_id: int, timestamp: str) -> None:
    """
    Insert a NULL-model_id summary placeholder (the same shape store_turn writes).
    The summarizer sees model_id IS NULL and promotes it on the next pass.
    """
    raw = ""
    if user_text:
        raw += f"User: {user_text}\n\n"
    if asst_text:
        raw += f"Assistant: {asst_text}"
    cur.execute(
        """INSERT INTO summaries
               (model_id, text, embedding, level, status, tags, timestamp, session_id)
           VALUES (NULL, ?, NULL, 'turn', 'complete', 'imported,claude-export', ?, ?)""",
        (raw.strip(), timestamp, session_id),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Import Claude.ai conversation exports into Ragger")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--verbose",  "-v", action="store_true")
    ap.add_argument("--all",      action="store_true",
                    help="Include conversations before birthday (2026-02-13)")
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    cfg = RaggerConfig(Path(args.settings).expanduser()) if args.settings else RaggerConfig()

    if not EXPORT_BASE.exists():
        print(f"ERROR: {EXPORT_BASE} not found", file=sys.stderr)
        sys.exit(1)

    # Collect all export dirs (listed + any others present)
    dirs = []
    for name in EXPORT_DIRS:
        d = EXPORT_BASE / name
        if d.exists():
            dirs.append(d)
    for d in EXPORT_BASE.iterdir():
        if d.is_dir() and d not in dirs:
            if (d / "conversations.json").exists():
                dirs.append(d)

    if not dirs:
        print("ERROR: no export directories found", file=sys.stderr)
        sys.exit(1)

    con = open_db(cfg)
    cur = con.cursor()

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Claude.ai export import")
    print(f"DB: {cfg.db_path}\n")

    model_id = get_or_create_model(cur, MODEL_NAME) if not args.dry_run else 0

    total_convs = total_skipped_conv = 0
    total_turns = total_skipped_turn = 0

    for export_dir in sorted(dirs):
        conv_file = export_dir / "conversations.json"
        print(f"── {export_dir.name[:50]} ──")

        with open(conv_file, encoding="utf-8") as f:
            data = json.load(f)
        convs = data if isinstance(data, list) else data.get("conversations", [])

        # Also import memories.json if present
        mem_file = export_dir / "memories.json"
        if mem_file.exists() and not args.dry_run:
            with open(mem_file, encoding="utf-8") as f:
                mdata = json.load(f)
            mems = mdata if isinstance(mdata, list) else [mdata]
            for m in mems:
                blob = m.get("conversations_memory", "").strip()
                if blob and len(blob) >= cfg.minimum_chunk_size:
                    # Check not already stored
                    key = blob[:120]
                    cur.execute(
                        "SELECT 1 FROM decisions WHERE substr(text,1,120)=?", (key,))
                    if not cur.fetchone():
                        cur.execute(
                            "INSERT INTO decisions (text, embedding, status, tags, timestamp)"
                            " VALUES (?, NULL, 'decision', 'imported,claude-export,memory',"
                            " datetime('now'))",
                            (blob,))
                        if args.verbose:
                            print(f"  [MEM] imported memories.json ({len(blob)} chars)")

        for conv in sorted(convs, key=lambda c: c.get("created_at", "")):
            name     = conv.get("name") or "(unnamed)"
            guid     = conv.get("uuid", "")
            created  = conv.get("created_at", "")
            messages = conv.get("chat_messages") or []

            if not messages:
                if args.verbose:
                    print(f"  [SKIP] {name[:50]} — no messages")
                total_skipped_conv += 1
                continue

            conv_ts = iso_to_ts(created)
            if not args.all:
                try:
                    dt = datetime.strptime(conv_ts[:10], "%Y-%m-%d")
                    if dt < BIRTHDAY:
                        if args.verbose:
                            print(f"  [SKIP] {name[:50]} — before birthday ({conv_ts[:10]})")
                        total_skipped_conv += 1
                        continue
                except Exception:
                    pass

            pairs = pair_turns(messages)
            if not pairs:
                if args.verbose:
                    print(f"  [SKIP] {name[:50]} — no pairable turns")
                total_skipped_conv += 1
                continue

            # Session dedup
            cur.execute("SELECT session_id FROM sessions WHERE guid=?", (guid,))
            if cur.fetchone():
                if args.verbose:
                    print(f"  [DUP]  {name[:50]} — session already imported")
                total_skipped_conv += 1
                continue

            print(f"  [{conv_ts[:10]}] {name[:55]:<55} {len(pairs):3d} turns")

            if not args.dry_run:
                cur.execute("INSERT INTO sessions (guid) VALUES (?)", (guid,))
                session_id: int = cur.lastrowid or 0

                for user_text, asst_text, user_ts, asst_ts in pairs:
                    if not user_text and not asst_text:
                        total_skipped_turn += 1
                        continue
                    ts = user_ts or asst_ts or conv_ts
                    insert_turn(cur, user_text, asst_text, session_id, ts, model_id)
                    insert_placeholder_summary(cur, user_text, asst_text, session_id, ts)
                    total_turns += 1
            else:
                total_turns += len(pairs)

            total_convs += 1

    if not args.dry_run:
        con.commit()

    con.close()

    print(f"\nConversations: {total_convs} imported, {total_skipped_conv} skipped")
    print(f"Turns:         {total_turns} inserted, {total_skipped_turn} skipped")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print("\nRun `ragger rebuild-embeddings` then start the daemon.")
        print("Housekeeping will summarize the new turns automatically.")


if __name__ == "__main__":
    main()
