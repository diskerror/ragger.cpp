"""
helpers.py — Shared utilities for Ragger memory import scripts.

Configuration comes from the same place the daemon gets it: the `settings`
table in memories.db. Ragger stopped reading or writing settings.ini on
2026-08-29 — defaults are compiled into the binary from default-settings.txt
and the DB is the only store — so an import script that parsed the INI was
reading a file that no longer exists, and before that a stale one.

Everything hangs off one base directory, ~/.ragger, exactly as the daemon does
(see ragger_base_dir() in src/util/fs.cpp). The DB is always <base>/memories.db;
there is no separate db_path setting to disagree with it.

Settings read from the DB (with the default-settings.txt fallbacks applied when
a row is absent, since the daemon's compiled-in defaults are not visible here):
  summarizer_api_url    LLM base URL
  summarizer_model      model name
  summarizer_api_key    bearer token (may be empty)
  summarizer_max_tokens cap on LLM response
  minimum_chunk_size    min chars to bother storing
"""

import argparse
import json
import re
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE_DIR = Path.home() / ".ragger"

# Mirrors default-settings.txt. Kept in sync by hand: the daemon's defaults are
# compiled into the binary, so there is no file for Python to read them from.
_DEFAULTS = {
    "summarizer_api_url":    "http://localhost:1234/v1",
    "summarizer_model":      "qwen3-4b-instruct-2507",
    "summarizer_api_key":    "",
    "summarizer_max_tokens": "1024",
    "minimum_chunk_size":    "300",
}


def add_source_arg(ap: argparse.ArgumentParser, default: Path, kind: str,
                   label: str) -> None:
    """
    Register the optional source path.

    Every importer reads from its tool's conventional location under $HOME
    (~/.claude, ~/.gemini, ...), so the common case takes no argument. The
    positional exists for when that location has moved — a relocated OpenClaw
    workspace, a copy restored from a backup, a second profile.

    `kind` is "file" or "dir"; it drives both the metavar and the validation.
    """
    ap.add_argument("source", nargs="?", default=None,
                    metavar=kind.upper(),
                    help=f"{label} (default: {default})")


def resolve_source(given, default: Path, kind: str, label: str) -> Path:
    """Validate the source path and fail with a usable message if it is wrong."""
    p = Path(given).expanduser() if given else default
    if not p.exists():
        raise SystemExit(
            f"ERROR: {label} not found at {p}\n"
            f"       Pass the path explicitly if it has moved."
        )
    if kind == "dir" and not p.is_dir():
        raise SystemExit(f"ERROR: {p} is not a directory ({label})")
    if kind == "file" and not p.is_file():
        raise SystemExit(f"ERROR: {p} is not a file ({label})")
    return p


def add_base_arg(ap: argparse.ArgumentParser) -> None:
    """
    Register the undocumented --ragger-base override.

    Mirrors the daemon's own testing-only flag (see main.cpp / ragger_base_dir)
    so a test run can point every script at a throwaway tree instead of the
    real ~/.ragger. Hidden from --help for the same reason it is hidden there:
    it is for testing, not for users.
    """
    ap.add_argument("--ragger-base", default=None, help=argparse.SUPPRESS)


def config_from_args(args: argparse.Namespace) -> "RaggerConfig":
    """Build a RaggerConfig honouring --ragger-base if it was given."""
    base = getattr(args, "ragger_base", None)
    return RaggerConfig(Path(base).expanduser() if base else None)


class RaggerConfig:
    """
    The settings the import scripts need, read from the `settings` table in
    <base>/memories.db. Absent rows fall back to _DEFAULTS.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
        self.db_path = self.base_dir / "memories.db"
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"memories.db not found at {self.db_path}. "
                f"Run `ragger start` once to create it, or pass --ragger-base."
            )
        self._settings = self._read_settings()

    def _read_settings(self) -> dict:
        # Read-only: an import script must never be the thing that creates or
        # migrates the settings table.
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT key, value FROM settings").fetchall()
        except sqlite3.OperationalError:
            rows = []          # no settings table yet — defaults will do
        finally:
            con.close()
        return {k: v for k, v in rows}

    def _get(self, key: str) -> str:
        v = self._settings.get(key)
        if v is None or v == "":
            return _DEFAULTS.get(key, "")
        return v

    # summarizer / inference
    @property
    def llm_base_url(self) -> str:
        return self._get("summarizer_api_url").rstrip("/")

    @property
    def llm_model(self) -> str:
        # No alias resolution: the [models] aliasing section was removed from
        # Ragger, and Config::resolve_model() is a pass-through.
        return self._get("summarizer_model").strip()

    # Retained so callers that distinguished the raw value keep working; with
    # aliasing gone the two are the same thing.
    llm_model_raw = llm_model

    @property
    def llm_api_key(self) -> str:
        return self._get("summarizer_api_key").strip()

    @property
    def llm_max_tokens(self) -> int:
        return int(self._get("summarizer_max_tokens"))

    # import
    @property
    def minimum_chunk_size(self) -> int:
        return int(self._get("minimum_chunk_size"))

    def __repr__(self):
        return (
            f"RaggerConfig(base={self.base_dir}, db={self.db_path}, "
            f"llm={self.llm_base_url}, model={self.llm_model!r})"
        )


# ── DB connection ─────────────────────────────────────────────────────────────

def open_db(cfg: Optional[RaggerConfig] = None) -> sqlite3.Connection:
    if cfg is None:
        cfg = RaggerConfig()
    path = cfg.db_path
    if not path.exists():
        raise FileNotFoundError(f"memories.db not found at {path}")
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    return con


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def mtime_ts(path: Path) -> str:
    """File modification time as a DB timestamp string (YYYY-MM-DD HH:MM:SS)."""
    t = datetime.fromtimestamp(path.stat().st_mtime)
    return t.strftime("%Y-%m-%d %H:%M:%S")

# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_call(system: str, user: str, cfg: Optional[RaggerConfig] = None,
             max_tokens: Optional[int] = None) -> str:
    """
    POST to the configured LLM endpoint (reads the summarizer_* settings).
    Returns the assistant message text, or raises RuntimeError on failure.
    """
    if cfg is None:
        cfg = RaggerConfig()

    model   = cfg.llm_model or "default"
    tokens  = max_tokens if max_tokens is not None else cfg.llm_max_tokens
    api_key = cfg.llm_api_key

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": tokens,
        "temperature": 0.1,
    }
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{cfg.llm_base_url}/chat/completions",
        data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"LLM endpoint unreachable ({cfg.llm_base_url}): {e}\n"
            f"Check the summarizer_api_url setting in {cfg.db_path} "
            f"(or set it in the Ragger dashboard)."
        )


def _clean_json(raw: str) -> str:
    """Strip markdown fences and thinking tags from LLM output."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def llm_classify_daily_note(text: str, date_str: str,
                             cfg: Optional[RaggerConfig] = None) -> dict:
    """
    Ask the LLM to split a daily note into decisions and a session summary.

    Returns:
        {
            "decisions": ["decision text 1", ...],   # may be empty
            "summary":   "condensed narrative of the day's work"
        }
    """
    system = (
        "You are a memory archivist. Parse a daily AI-assistant work log and split it into:\n"
        "1. DECISIONS — durable choices, preferences, or facts that were established "
        "(architectural decisions, tool choices, workflows, user preferences, "
        "lessons learned). Each should be self-contained, 1–4 sentences.\n"
        "2. SESSION_SUMMARY — a concise narrative (2–6 sentences) of what was worked "
        "on that day, suitable as a session memory.\n\n"
        "Output ONLY valid JSON, no explanation, no markdown fences:\n"
        '{"decisions": ["...", "..."], "summary": "..."}'
    )
    raw = llm_call(system, f"Date: {date_str}\n\n{text}", cfg=cfg, max_tokens=1500)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {"decisions": [], "summary": text[:800]}


def llm_classify_project_memory(text: str, filename: str, project: str,
                                 cfg: Optional[RaggerConfig] = None) -> dict:
    """
    Classify a Claude project memory file.

    Returns:
        {
            "type":  "decision" | "summary",
            "items": ["text1", ...]
        }
    """
    system = (
        "You are a memory archivist. Given a project memory file from Claude Code, "
        "classify and extract its content:\n"
        "- If it is a list of decisions, facts, or preferences: type='decision', "
        "split each bullet into a separate item in 'items'.\n"
        "- If it is a narrative, index, or mixed prose: type='summary', return the "
        "whole text as a single item.\n\n"
        'Output ONLY valid JSON: {"type": "decision", "items": ["..."]} or '
        '{"type": "summary", "items": ["..."]}'
    )
    raw = llm_call(system, f"Project: {project}\nFile: {filename}\n\n{text}",
                   cfg=cfg, max_tokens=1024)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {"type": "summary", "items": [text[:600]]}


def llm_classify_flat_bullets(text: str, source_label: str,
                               cfg: Optional[RaggerConfig] = None) -> list:
    """
    Turn a bullet-list memory file (Gemini GEMINI.md, Hermes MEMORY.md §-blocks)
    into a clean list of individual decision strings.

    Returns: ["item1", "item2", ...]
    """
    system = (
        "You are a memory archivist. Given a list of memory bullets or blocks from an "
        "AI assistant, return each as a clean, self-contained fact or decision string. "
        "Remove formatting syntax (bullets, brackets, markdown links). "
        "Keep content accurate — do not paraphrase.\n\n"
        'Output ONLY valid JSON array: ["item1", "item2", ...]'
    )
    raw = llm_call(system, f"Source: {source_label}\n\n{text}", cfg=cfg, max_tokens=1024)
    try:
        result = json.loads(_clean_json(raw))
        if isinstance(result, list):
            return [str(i).strip() for i in result if str(i).strip()]
        return []
    except json.JSONDecodeError:
        return []


# ── Content quality filters ───────────────────────────────────────────────────

# Lines that are pure system noise — no real conversation value.
_JUNK_LINE_PATTERNS = [
    re.compile(r'^assistant:\s*⚠️'),          # Agent failed before reply
    re.compile(r'^assistant:\s*⚙️'),          # Restarting OpenClaw / in-process restart
    re.compile(r'^user:\s*System:\s*\['),      # Gateway/system notifications
    re.compile(r'^assistant:\s*ℹ️'),          # Info system messages
    re.compile(r'^Logs:\s*\w'),               # "Logs: openclaw logs --follow" footer
    re.compile(r'^user:\s*Conversation info \(untrusted metadata\)'),  # metadata blob
    re.compile(r'^user:\s*Sender \(untrusted metadata\)'),             # metadata blob
    re.compile(r'^user:\s*A new session was started via /new'),        # session init noise
    re.compile(r'^user:\s*Note: The previous agent run was aborted'),  # aborted run notice
]


def is_junk_content(text: str) -> bool:
    """
    Return True if the text contains no real conversation content —
    i.e. every non-blank line is a system-noise line (agent errors,
    restart notices, gateway notifications).

    Used by import scripts to skip sessions that are pure system noise.
    Note: does NOT strip or alter text — caller decides what to do.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    return all(
        any(pat.match(line.strip()) for pat in _JUNK_LINE_PATTERNS)
        for line in lines
    )


# ── Dedup guards ──────────────────────────────────────────────────────────────

def summary_exists(cur: sqlite3.Cursor, timestamp: str, level: str) -> bool:
    cur.execute(
        "SELECT 1 FROM summaries WHERE created_at=? AND level=?",
        (to_epoch(timestamp), level),
    )
    return cur.fetchone() is not None


def summary_exists_by_text(cur: sqlite3.Cursor, text: str, level: str) -> bool:
    key = text.strip()[:120]
    cur.execute(
        "SELECT 1 FROM summaries WHERE substr(text,1,120)=? AND level=?",
        (key, level),
    )
    return cur.fetchone() is not None


def decision_exists_by_text(cur: sqlite3.Cursor, text: str) -> bool:
    key = text.strip()[:120]
    cur.execute("SELECT 1 FROM decisions WHERE substr(text,1,120)=?", (key,))
    return cur.fetchone() is not None


# ── Insert helpers ────────────────────────────────────────────────────────────

def to_epoch(ts) -> int:
    """
    Normalise a timestamp to the Unix epoch seconds the schema stores.

    Schema 0.12 keeps every time column as INTEGER unixepoch (turns.created_at,
    summaries.created_at/updated_at, decisions.created_at). These scripts were
    written against an older schema whose columns were named `timestamp` and
    held 'YYYY-MM-DD HH:MM:SS' text, so callers still pass strings.
    """
    if isinstance(ts, (int, float)):
        return int(ts)
    t = str(ts).strip()
    if not t:
        return int(datetime.now().timestamp())
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(t[:len(datetime.now().strftime(fmt))], fmt).timestamp())
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return int(datetime.now().timestamp())


def get_or_create_model(cur: sqlite3.Cursor, name: str) -> int:
    """Return model_id for name, inserting a new row if needed."""
    cur.execute("SELECT model_id FROM models WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO models (name) VALUES (?)", (name,))
    return cur.lastrowid or 0


def insert_summary(
    cur: sqlite3.Cursor,
    text: str,
    level: str,
    timestamp: str,
    tags: str,
    status: str = "complete",     # accepted and ignored; see below
    session_id: Optional[int] = None,
    model_id: Optional[int] = None,
) -> None:
    # Schema 0.12 dropped summaries.status and renamed timestamp -> created_at
    # (INTEGER unixepoch), adding a NOT NULL updated_at. `status` is kept in the
    # signature so existing callers keep working; it has nowhere to go.
    epoch = to_epoch(timestamp)
    cur.execute(
        """INSERT INTO summaries
               (model_id, text, embedding, level, tags, created_at, updated_at,
                session_id)
           VALUES (?, ?, NULL, ?, ?, ?, ?, ?)""",
        (model_id, text, level, tags, epoch, epoch, session_id),
    )


def insert_decision(
    cur: sqlite3.Cursor,
    text: str,
    tags: str,
    timestamp: str,
    status: str = "current",
) -> None:
    # status must be one of current | roadmap | superseded | deprecated — the
    # recall pipeline only surfaces 'current'. The old default here was
    # "decision", which is not a valid status, so every imported decision was
    # invisible to search. (Three rows in the live DB still carry the older
    # 'fact' / 'lesson' values from that era.)
    cur.execute(
        """INSERT INTO decisions (text, embedding, status, tags, created_at)
           VALUES (?, NULL, ?, ?, ?)""",
        (text, status, tags, to_epoch(timestamp)),
    )
