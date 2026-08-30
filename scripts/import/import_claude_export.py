#!/usr/bin/env python3
"""
import_claude_export.py — Import Claude.ai conversation export into ~/.ragger/memories.db

Sources:
  One or more unzipped Claude.ai export directories, named on the command line.
  Each must contain a conversations.json (a directory holding several such
  export directories works too — they are discovered one level down).

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

Usage:
    python3 scripts/import/import_claude_export.py DIR [DIR ...] [--dry-run]
                                                   [--verbose]

    DIR      An unzipped export directory (contains conversations.json), or a
             directory containing several of them.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import (RaggerConfig, add_base_arg, config_from_args, open_db,
                     to_epoch, is_junk_content)

MODEL_NAME = "claude.ai-export"


def resolve_export_dirs(paths) -> list:
    """
    Turn the directories named on the command line into a list of export dirs.

    A path is taken as an export directory when it holds a conversations.json;
    otherwise its immediate children are scanned for one, so both
    "the batch I just unzipped" and "the folder I unzip everything into" work.
    Order is preserved and duplicates dropped, so overlapping arguments are
    harmless.
    """
    found, seen = [], set()

    def add(d: Path):
        r = d.resolve()
        if r not in seen and (r / "conversations.json").exists():
            seen.add(r)
            found.append(r)

    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            print(f"WARNING: {p} not found — skipping", file=sys.stderr)
            continue
        if not p.is_dir():
            print(f"WARNING: {p} is not a directory — skipping", file=sys.stderr)
            continue
        add(p)
        for child in sorted(p.iterdir()):
            if child.is_dir():
                add(child)
    return found


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
               (user_text, assistant_text, embedding, model_id, session_id,
                created_at)
           VALUES (?, ?, NULL, ?, ?, ?)""",
        (user_text, asst_text, model_id, session_id, to_epoch(timestamp)),
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
        # level 'turn' predates schema 0.12, which moved per-turn (L2) summaries
        # into their own turn_summaries table and left summaries for
        # episode/session/project. Import these as 'episode' — the coarsest
        # level that still means "one exchange's worth of context".
        """INSERT INTO summaries
               (model_id, text, embedding, level, tags, created_at, updated_at,
                session_id)
           VALUES (NULL, ?, NULL, 'episode', 'imported,claude-export', ?, ?, ?)""",
        (raw.strip(), to_epoch(timestamp), to_epoch(timestamp), session_id),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Import Claude.ai conversation exports into Ragger")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("dirs", nargs="+", metavar="DIR",
                    help="Unzipped Claude.ai export directory (contains "
                         "conversations.json), or a directory holding several")
    ap.add_argument("--verbose",  "-v", action="store_true")
    add_base_arg(ap)
    args = ap.parse_args()

    cfg = config_from_args(args)

    dirs = resolve_export_dirs(args.dirs)
    if not dirs:
        print("ERROR: no export directory containing conversations.json found "
              "in: " + ", ".join(str(d) for d in args.dirs), file=sys.stderr)
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
                            "INSERT INTO decisions (text, embedding, status, tags, created_at)"
                            " VALUES (?, NULL, 'current', 'imported,claude-export,memory',"
                            " unixepoch())",
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
