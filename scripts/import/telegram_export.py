#!/usr/bin/env python3
"""
telegram_export.py — Pull a Telegram chat's history via the API (Telethon)
and write it out in the same JSON shape as Telegram Desktop's own
"Export chat history" → JSON feature, so it can be fed straight into:

    ragger import conversations --format=telegram --self="Your Name" <output.json>

Why this exists: the Mac App Store build of Telegram Desktop has chat
export stripped out (Apple sandboxing restriction), so this script
reproduces the same output format directly against the Telegram API.

Usage:
    python3 telegram_export.py --chat "Reid Woodbury" --output result.json
    python3 telegram_export.py --chat 8451224366 --output result.json --limit 5000
    python3 telegram_export.py --chat "@RaggerBotBot" --since 2026-07-28 --output today.json
    python3 telegram_export.py --chat "@RaggerBotBot" --since 2026-07-01 --until 2026-07-28

--since/--until filter on the LOCAL calendar date (inclusive on both ends);
handy for backfilling just a gap without re-pulling all history (the Ragger
importer dedups anyway, but a date range keeps the dump small).

First run will prompt for phone number + login code (and 2FA password if
set) interactively; session is cached in telegram_export.session next to
this script while the export is in progress. On a successful run this
session file is deleted at the end — it's an authenticated credential and
shouldn't linger on disk. A re-run will simply re-prompt for login.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import (
    MessageService,
    User,
    Chat as TgChat,
    Channel,
)

SESSION_PATH = Path(__file__).parent / "telegram_export.session"

# Filled in from my.telegram.org — treat like an app credential, not a secret key.
API_ID = 33282431
API_HASH = "40cffc9ec1016d1f4692764cafd02e63"


def entity_display_name(entity) -> str:
    """Best-effort human-readable name for a Telegram user/chat entity."""
    if isinstance(entity, User):
        parts = [p for p in (entity.first_name, entity.last_name) if p]
        if parts:
            return " ".join(parts)
        if entity.username:
            return entity.username
        return str(entity.id)
    if isinstance(entity, (TgChat, Channel)):
        return entity.title or str(entity.id)
    return str(getattr(entity, "id", "unknown"))


async def resolve_chat(client: TelegramClient, chat_arg: str):
    """Resolve --chat (numeric id, @username, or display-name substring)."""
    # Try numeric id first
    try:
        return await client.get_entity(int(chat_arg))
    except (ValueError, TypeError):
        pass
    except Exception:
        pass

    # Try as-is (username, phone, etc.)
    try:
        return await client.get_entity(chat_arg)
    except Exception:
        pass

    # Fall back to scanning dialogs for a name match
    async for dialog in client.iter_dialogs():
        if chat_arg.lower() in (dialog.name or "").lower():
            return dialog.entity

    raise SystemExit(f"Could not resolve chat: {chat_arg!r}")


async def export(chat_arg: str, output_path: Path, limit: int | None, self_name: str | None,
                 since_date=None, until_date=None):
    async with TelegramClient(str(SESSION_PATH), API_ID, API_HASH) as client:
        me = await client.get_me()
        my_name = self_name or entity_display_name(me)

        entity = await resolve_chat(client, chat_arg)
        chat_title = entity_display_name(entity)
        chat_id = entity.id

        span = ""
        if since_date or until_date:
            span = f" [{since_date or '...'} .. {until_date or '...'}]"
        print(f"Exporting chat with {chat_title!r} (id={chat_id}) as {my_name!r}{span} ...",
              file=sys.stderr)

        # Coarse server-side optimization: when --since is given, tell Telegram
        # to start near that point instead of walking the entire history. We
        # subtract a day and still do exact local-date filtering below, so an
        # off-by-one in offset_date's UTC-vs-local boundary can never drop a
        # message we actually want. offset_date + reverse=True returns messages
        # newer than the offset, ascending (oldest-first).
        offset_date = None
        if since_date is not None:
            offset_date = datetime.combine(since_date, time.min).astimezone() - timedelta(days=1)

        messages = []
        count = 0
        async for msg in client.iter_messages(entity, limit=limit, reverse=True,
                                               offset_date=offset_date):
            count += 1
            if count % 500 == 0:
                print(f"  ...{count} messages", file=sys.stderr)

            if isinstance(msg, MessageService) or not (msg.message or "").strip():
                # Service events (pins, joins, etc.) or empty/media-only
                # messages with no text — skip, matches Desktop export's
                # "type":"service" filtering behavior in our importer.
                continue

            # Exact date-range gate on the LOCAL date (msg.date is UTC).
            # Iteration is ascending (reverse=True), so once we pass --until
            # every remaining message is also out of range → break early.
            local_dt = msg.date.astimezone()
            local_date = local_dt.date()
            if since_date is not None and local_date < since_date:
                continue
            if until_date is not None and local_date > until_date:
                break

            sender = await msg.get_sender()
            from_name = entity_display_name(sender) if sender else "unknown"

            messages.append({
                "id": msg.id,
                "type": "message",
                # msg.date is UTC (tz-aware); convert to local time to match
                # Ragger's local-time created_at convention.
                "date": local_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": from_name,
                "text": msg.message,
            })

        doc = {
            "name": chat_title,
            "type": "personal_chat",
            # No chat id — personal identifier, keep it out of the dump/DB.
            "messages": messages,
        }

        output_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(messages)} messages to {output_path}", file=sys.stderr)
        print(f"Your display name for --self: {my_name!r}", file=sys.stderr)


def parse_date_arg(s: str, flag: str):
    """Parse a YYYY-MM-DD CLI date into a date object; exit cleanly on garbage."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"{flag}: expected YYYY-MM-DD, got {s!r}")


def main():
    ap = argparse.ArgumentParser(description="Export a Telegram chat to Desktop-export-shaped JSON")
    ap.add_argument("--chat", required=True,
                     help="Chat to export: numeric id, @username, or a substring of the display name")
    ap.add_argument("--output", default="result.json", help="Output JSON path")
    ap.add_argument("--limit", type=int, default=None, help="Max messages to fetch (default: all)")
    ap.add_argument("--since", default=None,
                     help="Only messages on/after this LOCAL date (YYYY-MM-DD), inclusive")
    ap.add_argument("--until", default=None,
                     help="Only messages on/before this LOCAL date (YYYY-MM-DD), inclusive")
    ap.add_argument("--self", dest="self_name", default=None,
                     help="Override your display name (defaults to your Telegram account name)")
    args = ap.parse_args()

    since_date = parse_date_arg(args.since, "--since")
    until_date = parse_date_arg(args.until, "--until")
    if since_date and until_date and since_date > until_date:
        raise SystemExit(f"--since ({since_date}) is after --until ({until_date})")

    asyncio.run(export(args.chat, Path(args.output), args.limit, args.self_name,
                       since_date, until_date))

    # Export succeeded (asyncio.run raises on failure, so we only get here on
    # success) — the session file is an authenticated credential, not build
    # output; delete it so it doesn't linger on disk between runs.
    if SESSION_PATH.exists():
        SESSION_PATH.unlink()
        print(f"Removed {SESSION_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
