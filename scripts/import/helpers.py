"""
helpers.py — Shared utilities for Ragger memory import scripts.

Configuration is read from ~/.ragger/settings.ini (the same file the Ragger
daemon uses), so inference endpoint, model, DB path, etc. are all in one place.

Relevant INI sections used:
  [storage]    db_path          → memories.db location
  [summarizer] api_url          → LLM base URL (e.g. http://192.168.0.107:1234/v1)
               model            → model name / alias
               api_key          → bearer token (may be empty)
               max_tokens       → cap on LLM response (default 1024)
  [models]     <alias> = <id>   → model alias resolution
  [import]     minimum_chunk_size → min chars to bother storing (default 300)

Birthday cutoff: 2026-02-13 (confirmed "Day One" from OpenClaw 2026-02-13.md)
"""

import configparser
import json
import re
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

SETTINGS_PATH = Path.home() / ".ragger" / "settings.ini"
BIRTHDAY = datetime(2026, 2, 13)


def _load_settings(path: Path = SETTINGS_PATH) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";",),   # the INI uses ; for inline comments
        comment_prefixes=("#", ";"),
    )
    cfg.read(str(path))
    return cfg


def _expand(value: str) -> str:
    """Expand ~ in a settings value (configparser doesn't do this)."""
    return str(Path(value.strip()).expanduser()) if value.strip() else value


class RaggerConfig:
    """
    Thin wrapper around settings.ini that exposes the values the import
    scripts need.  Falls back to sensible defaults if a key is absent.
    """

    def __init__(self, settings_path: Optional[Path] = None):
        path = settings_path if settings_path is not None else SETTINGS_PATH
        if not path.exists():
            raise FileNotFoundError(f"settings.ini not found at {path}")
        self._cfg = _load_settings(path)
        self._path = path

    # storage
    @property
    def db_path(self) -> Path:
        raw = self._cfg.get("storage", "db_path", fallback="~/.ragger/memories.db")
        return Path(_expand(raw))

    # summarizer / inference
    @property
    def llm_base_url(self) -> str:
        return self._cfg.get("summarizer", "api_url",
                             fallback="http://192.168.0.107:1234/v1").rstrip("/")

    @property
    def llm_model_raw(self) -> str:
        """Model name as written in [summarizer] model = ..."""
        return self._cfg.get("summarizer", "model", fallback="").strip()

    @property
    def llm_model(self) -> str:
        """Resolve model aliases defined in [models]."""
        raw = self.llm_model_raw
        if not raw:
            return ""
        # check [models] section for an alias
        if self._cfg.has_section("models"):
            resolved = self._cfg.get("models", raw, fallback="").strip()
            if resolved:
                return resolved
        return raw

    @property
    def llm_api_key(self) -> str:
        return self._cfg.get("summarizer", "api_key", fallback="").strip()

    @property
    def llm_max_tokens(self) -> int:
        return int(self._cfg.get("summarizer", "max_tokens", fallback="1024"))

    # import
    @property
    def minimum_chunk_size(self) -> int:
        return int(self._cfg.get("import", "minimum_chunk_size", fallback="300"))

    def __repr__(self):
        return (
            f"RaggerConfig(db={self.db_path}, "
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


def is_after_birthday(ts_str: str) -> bool:
    """True if timestamp >= birthday (2026-02-13)."""
    try:
        dt = datetime.strptime(ts_str[:10], "%Y-%m-%d")
        return dt >= BIRTHDAY
    except Exception:
        return True   # unparseable → let it through


# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_call(system: str, user: str, cfg: Optional[RaggerConfig] = None,
             max_tokens: Optional[int] = None) -> str:
    """
    POST to the configured LLM endpoint (reads [summarizer] from settings.ini).
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
            f"Check [summarizer] api_url in {SETTINGS_PATH}"
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
        "SELECT 1 FROM summaries WHERE timestamp=? AND level=?",
        (timestamp, level),
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
    status: str = "complete",
    session_id: Optional[int] = None,
    model_id: Optional[int] = None,
) -> None:
    cur.execute(
        """INSERT INTO summaries
               (model_id, text, embedding, level, status, tags, timestamp, session_id)
           VALUES (?, ?, NULL, ?, ?, ?, ?, ?)""",
        (model_id, text, level, status, tags, timestamp, session_id),
    )


def insert_decision(
    cur: sqlite3.Cursor,
    text: str,
    tags: str,
    timestamp: str,
    status: str = "decision",
) -> None:
    cur.execute(
        """INSERT INTO decisions (text, embedding, status, tags, timestamp)
           VALUES (?, NULL, ?, ?, ?)""",
        (text, status, tags, timestamp),
    )
