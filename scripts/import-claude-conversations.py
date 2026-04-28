#!/usr/bin/env python3
"""
import-claude-conversations.py — compile Claude conversation history into
Ragger, preserving original timestamps.

Handles two source formats:

  code  — Claude Code JSONL files under ~/.claude/projects/<slug>/*.jsonl
          Each line is a single event; we pair user-turn + next-assistant-text
          into a single memory entry.

  web   — claude.ai "Export Data" archive (conversations.json). A JSON array
          of conversation objects, each containing chat_messages with
          sender="human"/"assistant" and created_at timestamps.

Modes:

  --output FILE     Write compiled turns to a human-readable text file
                    (one exchange per block, chronological).
  --import          Post each exchange to the running Ragger daemon via HTTP,
                    with metadata.timestamp = original turn time so the memory
                    dates reflect when the conversation actually happened.

Filters:

  --session ID      Only include this Claude Code session.
  --since YYYY-MM-DD     Only include turns at or after this date.
  --until YYYY-MM-DD     Only include turns before this date.

Examples:

  # Compile all Claude Code conversations for this project to a file
  ./import-claude-conversations.py --format code \\
      --path ~/.claude/projects/-Volumes-WDBlack2-CLionProjects-Ragger \\
      --output ~/claude-history.txt

  # Import a claude.ai export with original timestamps
  ./import-claude-conversations.py --format web \\
      --path ~/Downloads/claude-export/conversations.json \\
      --import

Requires: Python 3.9+, `requests` (only for --import mode).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    timestamp: str          # ISO-8601 UTC
    user_text: str
    assistant_text: str
    session_id: str
    source: str             # "claude-code" or "claude-web"


# ---------------------------------------------------------------------------
# Claude Code JSONL
# ---------------------------------------------------------------------------

def _extract_text_from_content(content) -> str:
    """Content may be a string or a list of content blocks; return joined text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            # tool_use / tool_result / thinking blocks: skip — they're not
            # conversational text and would pollute semantic search.
        return "\n\n".join(p for p in parts if p).strip()
    return ""


def parse_claude_code(path: Path) -> Iterator[Turn]:
    """
    Iterate turns from Claude Code JSONL files. `path` may be a single file
    or a directory of *.jsonl files. We pair each user event with the next
    assistant text event in the same session.
    """
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]

    for jf in files:
        pending_user: dict | None = None
        session_id = jf.stem

        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = ev.get("type")
                if etype not in ("user", "assistant"):
                    continue

                # Skip tool_result user events (they have no conversational text)
                msg = ev.get("message", {})
                content = msg.get("content")
                text = _extract_text_from_content(content)
                if not text:
                    continue

                ts = ev.get("timestamp", "")
                sid = ev.get("sessionId", session_id)

                if etype == "user":
                    # If a previous user turn had no assistant reply, flush as
                    # a user-only entry so no input is lost.
                    if pending_user:
                        yield Turn(
                            timestamp=pending_user["ts"],
                            user_text=pending_user["text"],
                            assistant_text="",
                            session_id=pending_user["sid"],
                            source="claude-code",
                        )
                    pending_user = {"ts": ts, "text": text, "sid": sid}

                elif etype == "assistant":
                    if pending_user:
                        yield Turn(
                            timestamp=pending_user["ts"],
                            user_text=pending_user["text"],
                            assistant_text=text,
                            session_id=pending_user["sid"],
                            source="claude-code",
                        )
                        pending_user = None
                    # assistant-without-user (rare) is dropped — usually a
                    # continuation of a tool_use chain we already skipped.

        # Flush any trailing user turn at EOF
        if pending_user:
            yield Turn(
                timestamp=pending_user["ts"],
                user_text=pending_user["text"],
                assistant_text="",
                session_id=pending_user["sid"],
                source="claude-code",
            )


# ---------------------------------------------------------------------------
# claude.ai web export
# ---------------------------------------------------------------------------

def parse_claude_web(path: Path) -> Iterator[Turn]:
    """
    Parse a claude.ai data export. The archive contains conversations.json,
    a JSON array of { uuid, name, created_at, chat_messages: [...] }.
    Each chat message has sender ("human"/"assistant"), text, created_at.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        # Some exports wrap the list in an object
        data = data.get("conversations", [])

    for conv in data:
        sid = conv.get("uuid", "")
        msgs = conv.get("chat_messages") or conv.get("messages") or []

        pending_user: dict | None = None
        for m in msgs:
            sender = m.get("sender") or m.get("role")
            text = (m.get("text") or m.get("content") or "").strip()
            ts = m.get("created_at") or m.get("timestamp") or ""
            if not text:
                continue

            if sender in ("human", "user"):
                if pending_user:
                    yield Turn(
                        timestamp=pending_user["ts"],
                        user_text=pending_user["text"],
                        assistant_text="",
                        session_id=sid,
                        source="claude-web",
                    )
                pending_user = {"ts": ts, "text": text}

            elif sender in ("assistant", "claude"):
                if pending_user:
                    yield Turn(
                        timestamp=pending_user["ts"],
                        user_text=pending_user["text"],
                        assistant_text=text,
                        session_id=sid,
                        source="claude-web",
                    )
                    pending_user = None

        if pending_user:
            yield Turn(
                timestamp=pending_user["ts"],
                user_text=pending_user["text"],
                assistant_text="",
                session_id=sid,
                source="claude-web",
            )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def filter_turns(turns: Iterator[Turn], args) -> Iterator[Turn]:
    since = _parse_date(args.since) if args.since else None
    until = _parse_date(args.until) if args.until else None

    for t in turns:
        if args.session and t.session_id != args.session:
            continue
        if since or until:
            try:
                ts = datetime.fromisoformat(t.timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            if since and ts < since:
                continue
            if until and ts >= until:
                continue
        yield t


# ---------------------------------------------------------------------------
# Output sinks
# ---------------------------------------------------------------------------

def write_text_file(turns: Iterator[Turn], output: Path) -> int:
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"=== {t.timestamp}  [{t.source}:{t.session_id[:8]}] ===\n\n")
            f.write(f"User: {t.user_text}\n\n")
            if t.assistant_text:
                f.write(f"Assistant: {t.assistant_text}\n\n")
            f.write("\n")
            count += 1
    return count


def post_to_ragger(turns: Iterator[Turn], server: str, token: str | None) -> int:
    try:
        import requests  # noqa
    except ImportError:
        sys.exit("--import requires the `requests` package: pip install requests")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    stored = 0
    for t in turns:
        text = f"User: {t.user_text}"
        if t.assistant_text:
            text += f"\n\nAssistant: {t.assistant_text}"

        payload = {
            "text": text,
            "metadata": {
                "collection": "memory",
                "category": "conversation",
                "source": t.source,
                "session_id": t.session_id,
                "timestamp": t.timestamp,
            },
        }
        r = requests.post(f"{server}/store", json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        stored += 1
        if stored % 25 == 0:
            print(f"  stored {stored} turns...", file=sys.stderr)
    return stored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_token() -> str | None:
    p = Path.home() / ".ragger" / "token"
    try:
        return p.read_text().strip() or None
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Import Claude conversations into Ragger")
    ap.add_argument("--format", choices=["code", "web"], required=True,
                    help="code = Claude Code JSONL, web = claude.ai export")
    ap.add_argument("--path", required=True,
                    help="File or directory to read")
    ap.add_argument("--output", help="Write to text file instead of importing")
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="POST turns to the running Ragger daemon")
    ap.add_argument("--server", default="http://localhost:8432",
                    help="Ragger server URL (default: %(default)s)")
    ap.add_argument("--token", default=None,
                    help="Bearer token (default: ~/.ragger/token)")
    ap.add_argument("--session", help="Filter to a single session ID")
    ap.add_argument("--since", help="Only turns on/after this date (YYYY-MM-DD)")
    ap.add_argument("--until", help="Only turns before this date (YYYY-MM-DD)")

    args = ap.parse_args()
    if not args.output and not args.do_import:
        ap.error("specify --output FILE or --import")

    path = Path(os.path.expanduser(args.path))
    if not path.exists():
        sys.exit(f"path not found: {path}")

    parser = parse_claude_code if args.format == "code" else parse_claude_web
    turns = filter_turns(parser(path), args)

    if args.output:
        n = write_text_file(turns, Path(os.path.expanduser(args.output)))
        print(f"Wrote {n} turns to {args.output}")
    else:
        token = args.token or _default_token()
        n = post_to_ragger(turns, args.server.rstrip("/"), token)
        print(f"Imported {n} turns into Ragger at {args.server}")


if __name__ == "__main__":
    main()
