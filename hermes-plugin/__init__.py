"""Ragger memory plugin — MemoryProvider interface for Hermes Agent.

Connects to the Ragger HTTP server (localhost:8432 by default) for local
semantic memory with hybrid vector + BM25 search. No pip dependencies —
uses stdlib urllib throughout.

Prerequisites:
    ragger start        # bring the daemon up (install-bin.sh sets this up)

Configuration is resolved live from the running daemon over its
pre-authenticated Unix socket (GET /config), falling back to the DB settings
table when the daemon is down, then to built-in defaults. No config file is
read — a gateway restart re-learns host/port/socket/auto_recall automatically.

To activate in Hermes, set in ~/.hermes/config.yaml:
    memory:
      provider: ragger
"""

from __future__ import annotations

import json
import logging
import os
import threading
import http.client
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_CIRCUIT_THRESHOLD = 5
_CIRCUIT_COOLDOWN = 120.0   # seconds before auto-reset


# ---------------------------------------------------------------------------
# Unix domain socket HTTP transport
# ---------------------------------------------------------------------------

class _UnixSocketHTTPConnection(http.client.HTTPConnection):
    """http.client.HTTPConnection that dials a Unix domain socket instead of TCP."""

    def __init__(self, socket_path: str, timeout: float = 5.0):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


# ---------------------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------------------

def _read_token() -> str:
    """Read the daemon bearer token from ~/.ragger/token (empty if absent).

    Needed only for TCP requests; the Unix socket is pre-authenticated.
    """
    try:
        return (Path.home() / ".ragger" / "token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _config_from_daemon(socket_path: str, token: str) -> Optional[dict]:
    """Fetch live settings from a running daemon over the Unix socket.

    The socket is deterministic (~/.ragger/ragger.sock) and pre-authenticated,
    so this needs no prior knowledge of host/port/token — it is the bootstrap
    channel an agent-framework gateway can hit on restart to learn Ragger's
    live config instead of reading a file. Returns None if the daemon is down.
    """
    if not Path(socket_path).exists():
        return None
    conn = _UnixSocketHTTPConnection(socket_path, timeout=2.0)
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        conn.request("GET", "/config", headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            return None
        entries = json.loads(resp.read()).get("config", [])
    except Exception as exc:
        logger.debug("Ragger GET /config (socket): %s", exc)
        return None
    finally:
        conn.close()
    return {e["key"]: e.get("value", "") for e in entries if "key" in e}


def _config_from_db(db_path: Path) -> dict:
    """Read the settings table directly when the daemon is down.

    Offline fallback: the DB is the source of truth the daemon itself overlays,
    so reading it gives the same effective values without a running server.
    Missing rows simply fall through to built-in defaults. Never raises.
    """
    out: dict = {}
    if not db_path.exists():
        return out
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            for key, value in con.execute("SELECT key, value FROM settings"):
                out[key] = value
        finally:
            con.close()
    except Exception as exc:
        logger.debug("Ragger settings DB read failed: %s", exc)
    return out


def _as_bool(value: str, default: bool) -> bool:
    v = str(value).split("#")[0].strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return default


def _load_ragger_config() -> dict:
    """Resolve the plugin's connection + behavior settings from the daemon.

    Source of truth is the running daemon, queried over the deterministic,
    pre-authenticated Unix socket (GET /config). This replaces the old
    settings.ini reader: an agent-framework gateway that restarts can re-learn
    Ragger's live config (host/port/socket/auto_recall) with no config file.

    Resolution order per value:
      1. Live daemon over the socket (GET /config)
      2. The DB settings table read directly (daemon down)
      3. Built-in defaults below

    The bearer token still comes from ~/.ragger/token (the socket path is
    pre-authed, so bootstrapping needs no token; token is for later TCP use).

    auto_recall gates the recall/injection side only (prefetch → the
    <memory-context> block). It never gates turn capture, which is always-on.
    """
    ragger_base = Path.home() / ".ragger"
    socket_path = str(ragger_base / "ragger.sock")
    token = _read_token()

    cfg = {
        "host": "127.0.0.1",
        "port": "8432",
        "auth_token": token,
        "auto_recall": True,
        "socket_enable": True,
        "socket_path": socket_path,
    }

    # Prefer the live daemon; fall back to the DB when it is down.
    settings = _config_from_daemon(socket_path, token)
    if settings is None:
        settings = _config_from_db(ragger_base / "memories.db")

    if settings:
        raw_port = str(settings.get("port", "")).split("#")[0].strip()
        if raw_port:
            cfg["port"] = raw_port
        raw_bind = str(settings.get("bind", "")).split("#")[0].strip()
        if raw_bind and raw_bind != "0.0.0.0":
            cfg["host"] = raw_bind
        cfg["auto_recall"] = _as_bool(settings.get("auto_recall", ""), True)
        cfg["socket_enable"] = _as_bool(settings.get("socket_enable", ""), True)

    return cfg


# ---------------------------------------------------------------------------
# Shared instructions file (single source of truth across all Ragger hosts)
# ---------------------------------------------------------------------------
#
# docs/agent-memory-instructions.md is the canonical policy doc — installers
# copy it to ~/.ragger/agent-memory-instructions.md and `ragger mcp` also
# serves it to MCP clients via the `initialize` response. Rather than
# duplicating its wording in this schema's description (which drifts), we
# load the file at import time and extract the decision-capture-policy
# section directly. Falls back to a short static string if the file is
# missing so the plugin still works standalone.

_INSTRUCTIONS_PATH = Path.home() / ".ragger" / "agent-memory-instructions.md"
_DECISION_POLICY_FALLBACK = (
    "For 'decision' entries: capture generously (including generalized "
    "reusable takeaways, not just narrow pivots), leading with a punchy "
    "conclusion line, then rationale/scope/source context when known. "
    "(Fallback text — full policy normally loaded from "
    "~/.ragger/agent-memory-instructions.md, which was not found.)"
)


def _load_decision_capture_policy() -> str:
    """Extract the decision-capture-policy section from the shared
    instructions file so this schema's description stays in sync with
    docs/agent-memory-instructions.md without copy-pasting text."""
    try:
        text = _INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    except OSError:
        return _DECISION_POLICY_FALLBACK
    marker = '### Capture policy for `category: "decision"`'
    start = text.find(marker)
    if start == -1:
        return _DECISION_POLICY_FALLBACK
    # Section runs until the next "**Do not store" line or next "## " heading.
    rest = text[start:]
    end = len(rest)
    for stop_marker in ("\n**Do not store", "\n## "):
        idx = rest.find(stop_marker, len(marker))
        if idx != -1:
            end = min(end, idx)
    return rest[:end].strip()


_DECISION_CAPTURE_POLICY = _load_decision_capture_policy()

# ---------------------------------------------------------------------------
# Tool schemas exposed to the Hermes agent
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "ragger_search",
    "description": (
        "Search semantic memory for relevant facts, past context, decisions, and preferences. "
        "Uses hybrid vector + BM25 search. Call at conversation start or whenever "
        "prior context may be relevant."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for — a question, topic, or keyword.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 5).",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "ragger_store",
    "description": (
        "Store a fact, decision, preference, or piece of context into semantic memory "
        "for future retrieval. Use for anything worth remembering across conversations: "
        "user preferences, project decisions, key facts, lessons learned.\n\n"
        + _DECISION_CAPTURE_POLICY
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to store.",
            },
            "category": {
                "type": "string",
                "description": (
                    "Free-form label, not a database-enforced enum (Ragger does a plain "
                    "string match server-side — any string works, but stick to the "
                    "convention for consistent retrieval). Convention: 'fact', "
                    "'decision', 'preference', or 'lesson'. Default: 'fact'."
                ),
            },
        },
        "required": ["text"],
    },
}


DECISION_SCHEMA = {
    "name": "memory_decision",
    "description": (
        "Record a standing decision: a ruling plus the situation it applies to. "
        "`trigger` is REQUIRED and is written the way a FUTURE question would be "
        "asked (e.g. 'Use when choosing a full-text index strategy') -- search "
        "matches on it, so phrase it in the asker's words, not the conclusion's. "
        "To replace an earlier decision, pass its id in `supersedes`; the old one "
        "is kept and marked superseded, never overwritten. Prefer this over "
        "ragger_store(category='decision') for anything with a rationale. "
        "The result may carry `possible_supersedes`: other active decisions this "
        "one may be replacing (hint only — nothing was changed; review them)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The ruling itself, imperative."},
            "trigger": {"type": "string",
                        "description": "When this applies, in the vocabulary of a future query."},
            "rationale": {"type": "string",
                          "description": "Why; alternatives considered and rejected."},
            "scope": {"type": "string", "description": "What this does NOT apply to."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["current", "provisional", "pending"],
                       "description": "Default current. provisional = agreed, not yet proven."},
            "supersedes": {"type": "integer", "description": "decision id this replaces."},
            "evidence": {"type": "array", "items": {"type": "integer"},
                         "description": "turn_summary ids (from search results) this rests on."},
            "project": {"type": "string",
                        "description": "Project name or path this ruling belongs to (optional; auto-inferred from cwd when omitted)."},
            "cwd": {"type": "string",
                    "description": "Working directory for project inference (the plugin fills os.getcwd() automatically when project is omitted)."},
        },
        "required": ["text", "trigger"],
    },
}


MEMORY_NOTE_SCHEMA = {
    "name": "memory_note",
    "description": (
        "Store a note (fact, preference, lesson) into memory, optionally filed "
        "under a project. `project` is a project name or path; when omitted the "
        "server infers the project from `cwd` (the plugin fills os.getcwd() "
        "automatically). The result may carry `possible_supersedes`: current "
        "decisions this note may contradict (hint only -- nothing was changed; "
        "review them and use memory_decision for a real ruling). Pass evidence "
        "(turn_summary ids from search) when you are recording a synthesized "
        "insight; it is stored as an insight linked to those sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note text."},
            "project": {"type": "string",
                        "description": "Project name or path (optional)."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Optional tags."},
            "cwd": {"type": "string",
                    "description": "Working directory for project inference (auto-filled when project is omitted)."},
            "evidence": {"type": "array", "items": {"type": "integer"},
                         "description": "turn_summary ids this note synthesizes. When present the note is stored as an insight (response carries \"table\":\"insights\") instead of a plain note."},
        },
        "required": ["text"],
    },
}


MEMORY_PROJECT_SCHEMA = {
    "name": "memory_project",
    "description": (
        "Projects: named bodies of work. action 'list' returns all projects "
        "(optional status filter); 'create' needs name (unique; path optional, "
        "used for cwd inference); 'get' returns the project briefing packet -- "
        "current rulings, latest project summary, recent episodes, open "
        "questions -- read it before working on a project; 'update' changes "
        "status/description/path (only supplied fields)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "create", "get", "update"],
                       "description": "list | create | get | update."},
            "name": {"type": "string",
                     "description": "Project name (required for create/get/update)."},
            "status": {"type": "string",
                       "description": "active|paused|done|abandoned (vocabulary may grow). For list: filter."},
            "description": {"type": "string", "description": "What this project is."},
            "path": {"type": "string",
                     "description": "Project root path (absolute or ~/...), used for cwd inference."},
        },
        "required": ["action"],
    },
}


MEMORY_TIMELINE_SCHEMA = {
    "name": "memory_timeline",
    "description": (
        "What happened recently, as a deterministic newest-first list (no "
        "search ranking). granularity 'turn' = per-exchange summaries "
        "(falling back to raw user text), 'episode' = episode summaries "
        "(default), 'session' = per session its session summary or newest "
        "episode. Filters: project (name/path; cwd inference applies), "
        "session GUID ('current' = this conversation), since/until "
        "('YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', or relative '30d'/'12h'). "
        "Page through with the returned next_cursor; null means the end."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project name or path (optional)."},
            "cwd": {"type": "string",
                    "description": "Working directory for project inference (auto-filled when project is omitted)."},
            "session": {"type": "string", "description": "Only this session GUID ('current' = this one)."},
            "since": {"type": "string", "description": "Lower time bound."},
            "until": {"type": "string", "description": "Upper time bound (bare date = end of day)."},
            "granularity": {"type": "string", "enum": ["turn", "episode", "session"],
                            "description": "turn | episode | session (default episode)."},
            "limit": {"type": "integer", "description": "Page size (default 20, max 100)."},
            "cursor": {"type": "string", "description": "Opaque next_cursor from the previous page."},
        },
    },
}


MEMORY_UPDATE_SCHEMA = {
    "name": "memory_update",
    "description": (
        "Curate one row's METADATA by typed handle: decision status (legal "
        "transitions only; 'superseded' is set only by storing a superseding "
        "decision), tags (replaces the whole set), project (name/path; empty "
        "string clears). Agent notes accept tags/project; summarizer rollups "
        "and turn / turn_summary / document rows are not editable. Text is "
        "NEVER edited in place -- to change what something says, store a new "
        "decision (memory_decision with supersedes) or note. Returns the "
        "updated row."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Typed handle '<layer>:<rowid>'."},
            "status": {"type": "string",
                       "description": "Decision status: current | provisional | pending | roadmap | retired."},
            "tags": {"type": "string", "description": "Replacement tag string (comma-separated)."},
            "project": {"type": "string", "description": "Project name/path; empty string clears it."},
        },
        "required": ["id"],
    },
}


MEMORY_REVIEW_SCHEMA = {
    "name": "memory_review",
    "description": (
        "The pending queue. action 'list' returns pending decisions and "
        "insights oldest first (handle + trigger + snippet); 'approve' moves each id "
        "pending->current; 'reject' moves each id pending->retired. ids accept "
        "'decision:<id>' / 'insight:<id>' handles or bare integers (decision); "
        "a non-pending id fails only that entry."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "approve", "reject"],
                       "description": "list | approve | reject."},
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "Decision handles for approve/reject."},
        },
        "required": ["action"],
    },
}


MEMORY_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Layered search across all memory corpora (turns, turn summaries, "
        "session/episode summaries, documents, decisions, insights). Returns typed handles "
        "+ snippets only -- call memory_get with chosen handles for full text. "
        "`layers` narrows by kind; omit for all. Pass session='current' to search "
        "only this conversation, or exclude_session='current' to skip it. "
        "status: 'active' (default) hides superseded/retired decisions; 'any' "
        "includes them. since/until: 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', or "
        "relative ('30d', '12h')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "layers": {"type": "array", "items": {"type": "string"},
                       "description": "Subset of: turn, turn_summary, summary, document, decision, insight."},
            "since": {"type": "string", "description": "Only results at/after this time."},
            "until": {"type": "string", "description": "Only results at/before this time."},
            "status": {"type": "string",
                       "description": "Decisions only: active (default) | any | one exact status."},
            "session": {"type": "string",
                        "description": "Only this session GUID ('current' = this session)."},
            "exclude_session": {"type": "string",
                                "description": "Drop this session GUID ('current' = this session)."},
            "limit": {"type": "integer", "description": "Max results (default 10, max 50)."},
            "max_tokens": {"type": "integer",
                           "description": "Token budget (~chars/4) over snippets; 0 = none."},
            "project": {"type": "string",
                        "description": "Only results tied to this project (name or path)."},
            "cwd": {"type": "string",
                    "description": "Working directory for project inference (auto-filled when project is omitted)."},
        },
        "required": ["query"],
    },
}


MEMORY_GET_SCHEMA = {
    "name": "memory_get",
    "description": (
        "Read full memory rows by typed handles (from memory_search), or page "
        "through one session's conversation history. Exactly one mode per call: "
        "ids, or session (+optional after_turn/before_turn/limit). expand: none | "
        "context (neighbouring turns/chunks) | sources (a decision's or "
        "insight's evidence, a summary's constituent turn summaries) | chain "
        "(a decision's or insight's full supersede lineage, oldest to newest). Pass session='current' for this conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "Typed handles '<layer>:<rowid>' (mode 1)."},
            "expand": {"type": "string", "description": "none | context | sources | chain."},
            "session": {"type": "string",
                        "description": "Session GUID for history paging (mode 2; 'current' = this session)."},
            "after_turn": {"type": "integer",
                           "description": "History: turns with turn_id > this (0 = from start)."},
            "before_turn": {"type": "integer",
                            "description": "History: turns with turn_id < this (0 = to end)."},
            "limit": {"type": "integer", "description": "History page size (default 20, max 100)."},
        },
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class RaggerMemoryProvider(MemoryProvider):
    """Ragger local semantic memory via HTTP API."""

    def __init__(self):
        self._base_url = "http://127.0.0.1:8432"
        self._auth_token = ""
        self._auto_recall = True
        self._socket_enable = True
        self._socket_path = str(Path.home() / ".ragger" / "ragger.sock")
        self._prefetch_result = ""
        self._model = ""
        self._session_id = ""   # Hermes session id; provenance for memory_decision
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        # Circuit breaker
        import time as _time
        self._time = _time
        self._failures = 0
        self._breaker_until = 0.0

    @property
    def name(self) -> str:
        return "ragger"

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _breaker_open(self) -> bool:
        if self._failures < _CIRCUIT_THRESHOLD:
            return False
        if self._time.monotonic() >= self._breaker_until:
            self._failures = 0
            return False
        return True

    def _record_ok(self) -> None:
        self._failures = 0

    def _agent_tool_result(self, got):
        """Shared result handling for the Phase 2 agent tools (memory_search /
        memory_get): _post_status returned None (daemon down) or a
        (status, body) pair. A 4xx body is the server's ERROR message."""
        if got is None:
            self._record_fail()
            return json.dumps({
                "error": "Ragger request failed. Is the daemon running? Try: ragger start"
            })
        self._record_ok()
        status, body = got
        if status >= 400:
            return json.dumps({"error": body.strip()})
        try:
            return json.dumps(json.loads(body))
        except ValueError:
            return json.dumps({"error": "Ragger returned a non-JSON reply: " + body[:200]})

    def _record_fail(self) -> None:
        self._failures += 1
        if self._failures >= _CIRCUIT_THRESHOLD:
            self._breaker_until = self._time.monotonic() + _CIRCUIT_COOLDOWN
            logger.warning(
                "Ragger circuit breaker tripped after %d failures — "
                "pausing for %ds. Is 'ragger start' running?",
                self._failures, int(_CIRCUIT_COOLDOWN),
            )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._auth_token:
            h["Authorization"] = f"Bearer {self._auth_token}"
        return h

    def _get(self, path: str, timeout: float = 3.0) -> Optional[dict]:
        if self._socket_enable:
            return self._request_unix("GET", path, None, timeout)
        url = self._base_url + path
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("Ragger GET %s: %s", path, exc)
            return None

    def _post(self, path: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
        if self._socket_enable:
            return self._request_unix("POST", path, payload, timeout)
        url = self._base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("Ragger POST %s: %s", path, exc)
            return None

    def _post_status(self, path: str, payload: dict, timeout: float = 5.0):
        """POST and return (http_status, body_text), or None on a TRANSPORT
        failure (daemon down, timeout). Unlike _post(), a 4xx comes back as a
        value: a validation error is the caller's mistake and must neither be
        reported as 'daemon down' nor trip the circuit breaker."""
        body = json.dumps(payload).encode()
        if self._socket_enable:
            conn = _UnixSocketHTTPConnection(self._socket_path, timeout=timeout)
            try:
                conn.request("POST", path, body=body, headers=self._headers())
                resp = conn.getresponse()
                return resp.status, resp.read().decode("utf-8", "replace")
            except Exception as exc:
                logger.debug("Ragger POST %s (socket): %s", path, exc)
                return None
            finally:
                conn.close()
        req = urllib.request.Request(self._base_url + path, data=body,
                                     headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")
        except Exception as exc:
            logger.debug("Ragger POST %s: %s", path, exc)
            return None

    def _request_unix(self, method: str, path: str, payload: Optional[dict], timeout: float) -> Optional[dict]:
        """Send a request over the Ragger Unix domain socket."""
        conn = _UnixSocketHTTPConnection(self._socket_path, timeout=timeout)
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            conn.request(method, path, body=body, headers=self._headers())
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                logger.debug("Ragger %s %s: HTTP %d %s", method, path, resp.status, data[:200])
                return None
            return json.loads(data)
        except Exception as exc:
            logger.debug("Ragger %s %s (socket): %s", method, path, exc)
            return None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ragger daemon is reachable."""
        cfg = _load_ragger_config()
        if cfg["socket_enable"]:
            conn = _UnixSocketHTTPConnection(cfg["socket_path"], timeout=2.0)
            try:
                headers = {}
                if cfg["auth_token"]:
                    headers["Authorization"] = f"Bearer {cfg['auth_token']}"
                conn.request("GET", "/health", headers=headers)
                resp = conn.getresponse()
                data = json.loads(resp.read())
                return data.get("status") == "ok"
            except Exception:
                return False
            finally:
                conn.close()
        url = f"http://{cfg['host']}:{cfg['port']}/health"
        req = urllib.request.Request(url)
        if cfg["auth_token"]:
            req.add_header("Authorization", f"Bearer {cfg['auth_token']}")
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        cfg = _load_ragger_config()
        self._base_url = f"http://{cfg['host']}:{cfg['port']}"
        self._auth_token = cfg["auth_token"]
        self._auto_recall = cfg["auto_recall"]
        self._socket_enable = cfg["socket_enable"]
        self._socket_path = cfg["socket_path"]

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Capture the live conversing model so sync_turn can record it.

        Hermes switches models mid-session, so we refresh this each turn
        rather than caching it at initialize(). The model name flows to
        Ragger's /turn endpoint, which stores it on the turn + its L2
        summary (the row that records *who the user was talking to*).
        """
        model = kwargs.get("model")
        if model:
            self._model = model

    def system_prompt_block(self) -> str:
        return (
            "# Ragger Memory\n"
            "Local semantic memory is active. Use ragger_search to recall past context, "
            "facts, decisions, and preferences. Use ragger_store to save anything worth "
            "remembering across conversations."
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Kick off a background search to warm the prefetch cache.

        No-op when auto_recall is disabled (the Ragger `auto_recall`
        setting = false) — turn capture (sync_turn /
        post_turn_finalized) is unaffected by this flag.
        """
        if not self._auto_recall or self._breaker_open():
            return

        def _run():
            result = self._post("/search", {"query": query, "limit": 5}, timeout=4.0)
            if result is not None:
                self._record_ok()
                hits = result.get("results", [])
                if hits:
                    lines = [h.get("text", "") for h in hits if h.get("text")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {line}" for line in lines)
            else:
                self._record_fail()

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="ragger-prefetch",
        )
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return (and clear) the result of the last queue_prefetch call."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=4.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Ragger Memory\n{result}"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Push the completed turn to Ragger's capture_turn (HTTP POST /turn).

        The daemon stores the raw turn — and later summarizes it — only when
        [server] capture_turns is enabled; otherwise this is a server-side
        no-op. session_id groups turns for session summaries. Agent-driven
        ragger_store remains the path for explicitly-curated memories.
        """
        if self._breaker_open() or not user_content:
            return
        self._post_turn(user_content, assistant_content, session_id=session_id)

    def post_turn_finalized(self, turn_data: dict, **kwargs) -> None:
        """Hook called after *every* turn finalization, including interrupted ones.
        
        This is the capture point that bypasses Hermes's skip-on-interrupt logic.
        We POST to Ragger any turn with both user and assistant content,
        even if interrupted. Empty turns are skipped (no substance to capture).
        
        turn_data format (from Hermes):
            {
                "user": str,
                "assistant": str,
                "model": str,
                "session_id": str,
                "interrupted": bool,
            }
        """
        # Skip if user content is missing — nothing to capture at all
        if not turn_data or not turn_data.get("user"):
            return

        user = turn_data["user"]
        assistant = turn_data.get("assistant", "")
        session_id = turn_data.get("session_id", "")

        # Post asynchronously so we don't block turn finalization
        def _post_async():
            if assistant:
                # Normal path: both sides present — store as a full turn
                self._post_turn(user, assistant, session_id=session_id)
            else:
                # User message only (stopped before any response was generated).
                # Store the user text alone via /store so it's searchable,
                # tagged so it's identifiable as an incomplete prompt.
                self._post(
                    "/store",
                    {
                        "text": f"[incomplete prompt — no response generated]\n{user}",
                        "metadata": {
                            "category": "fact",
                            "source": "hermes-interrupted",
                            "session_id": session_id,
                        },
                    },
                )

        thread = threading.Thread(target=_post_async, daemon=True, name="ragger-post-turn")
        thread.start()

    def _get_session_title(self, session_id: str) -> str:
        """Look up the current Hermes session title from ~/.hermes/state.db.

        Hermes sets sessions.title once auto-titling completes (usually a few
        turns in), so early turns in a session will get empty/NULL back —
        that's expected, not an error. Best-effort: any failure (locked DB,
        missing file, etc.) returns "" rather than raising, since this must
        never block turn capture.
        """
        if not session_id:
            return ""
        try:
            import sqlite3
            state_db = Path.home() / ".hermes" / "state.db"
            conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT title FROM sessions WHERE id = ? LIMIT 1", (session_id,)
                ).fetchone()
                return row[0] if row and row[0] else ""
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("Ragger: session title lookup failed for %s: %s", session_id, exc)
            return ""

    def _post_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Internal: POST a turn to Ragger's /turn endpoint.
        
        Called from both sync_turn (normal path) and post_turn_finalized (catch-all).
        """
        if self._breaker_open() or not user_content:
            return
        payload = {
            "user": user_content,
            "assistant": assistant_content,
            "model": self._model,
            "session_id": session_id,
        }
        title = self._get_session_title(session_id)
        if title:
            payload["session_name"] = title
            payload["name_source"] = "hermes_title"
        result = self._post("/turn", payload, timeout=4.0)
        if result is None:
            self._record_fail()
        else:
            self._record_ok()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA, DECISION_SCHEMA,
                MEMORY_SEARCH_SCHEMA, MEMORY_GET_SCHEMA,
                MEMORY_NOTE_SCHEMA, MEMORY_PROJECT_SCHEMA,
                MEMORY_TIMELINE_SCHEMA, MEMORY_UPDATE_SCHEMA,
                MEMORY_REVIEW_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._breaker_open():
            return json.dumps({
                "error": "Ragger temporarily unavailable (circuit breaker tripped). "
                         "Check that 'ragger start' is running. Will auto-retry."
            })

        if tool_name == "ragger_search":
            query = args.get("query", "").strip()
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            limit = max(1, int(args.get("limit", 5)))
            result = self._post("/search", {"query": query, "limit": limit})
            if result is None:
                self._record_fail()
                return json.dumps({
                    "error": "Ragger search failed. Is the daemon running? Try: ragger start"
                })
            self._record_ok()
            hits = result.get("results", [])
            if not hits:
                return json.dumps({"result": "No relevant memories found."})
            items = [
                {"text": h.get("text", ""), "score": round(h.get("score", 0), 3)}
                for h in hits
            ]
            return json.dumps({"results": items, "count": len(items)})

        elif tool_name == "ragger_store":
            text = args.get("text", "").strip()
            if not text:
                return json.dumps({"error": "Missing required parameter: text"})
            category = args.get("category", "fact").strip() or "fact"
            result = self._post(
                "/store",
                {
                    "text": text,
                    "metadata": {"category": category, "source": "hermes-agent"},
                },
            )
            if result is None:
                self._record_fail()
                return json.dumps({
                    "error": "Ragger store failed. Is the daemon running? Try: ragger start"
                })
            self._record_ok()
            return json.dumps({"result": "Stored.", "id": result.get("id", "")})

        elif tool_name == "memory_decision":
            text = str(args.get("text", "")).strip()
            trigger = str(args.get("trigger", "")).strip()
            if not text or not trigger:
                return json.dumps({"error": "memory_decision requires both text and trigger"})
            payload = {"text": text, "trigger": trigger, "session": self._session_id}
            for key in ("rationale", "scope", "status"):
                if args.get(key):
                    payload[key] = str(args[key])
            if args.get("tags"):
                payload["tags"] = [str(t) for t in args["tags"]]
            if args.get("supersedes") is not None:
                payload["supersedes"] = int(args["supersedes"])
            if args.get("evidence"):
                payload["evidence"] = [int(x) for x in args["evidence"]]
            # Project (Phase 3): explicit wins; else pass cwd so the server
            # infers (longest-prefix match). No match = unfiled, not an error.
            if args.get("project"):
                payload["project"] = str(args["project"])
            else:
                payload["cwd"] = str(args.get("cwd") or os.getcwd())
            got = self._post_status("/decision", payload)
            if got is None:
                self._record_fail()
                return json.dumps({
                    "error": "Ragger store failed. Is the daemon running? Try: ragger start"
                })
            self._record_ok()   # the daemon answered, even if it said 400
            status, body = got
            if status >= 400:
                return json.dumps({"error": body.strip()})
            try:
                return json.dumps(json.loads(body))
            except ValueError:
                return json.dumps({"error": "Ragger returned a non-JSON reply: " + body[:200]})

        elif tool_name == "memory_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            payload: dict = {"query": query}
            for key in ("since", "until", "status"):
                if args.get(key):
                    payload[key] = str(args[key])
            if args.get("layers"):
                payload["layers"] = [str(x) for x in args["layers"]]
            # 'current' resolves to this Hermes session before posting.
            if args.get("session"):
                s = str(args["session"])
                payload["session"] = self._session_id if s == "current" else s
            if args.get("exclude_session"):
                s = str(args["exclude_session"])
                payload["exclude_session"] = self._session_id if s == "current" else s
            if args.get("limit") is not None:
                payload["limit"] = int(args["limit"])
            if args.get("max_tokens") is not None:
                payload["max_tokens"] = int(args["max_tokens"])
            # Project filter (Phase 3): explicit wins; else cwd inference.
            if args.get("project"):
                payload["project"] = str(args["project"])
            else:
                payload["cwd"] = str(args.get("cwd") or os.getcwd())
            got = self._post_status("/memory/search", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_note":
            text = str(args.get("text", "")).strip()
            if not text:
                return json.dumps({"error": "Missing required parameter: text"})
            payload = {"text": text}
            if args.get("tags"):
                payload["tags"] = [str(t) for t in args["tags"]]
            if args.get("project"):
                payload["project"] = str(args["project"])
            else:
                payload["cwd"] = str(args.get("cwd") or os.getcwd())
            if args.get("evidence"):
                # Phase 5: evidence routes the write to store_insight server-side.
                payload["evidence"] = [int(e) for e in args["evidence"]]
            got = self._post_status("/memory/note", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_project":
            action = str(args.get("action", "")).strip()
            if action not in ("list", "create", "get", "update"):
                return json.dumps({"error": "memory_project action must be list|create|get|update"})
            payload = {"action": action}
            for key in ("name", "status", "description", "path"):
                if args.get(key) is not None:
                    payload[key] = str(args[key])
            got = self._post_status("/memory/project", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_timeline":
            payload: dict = {}
            for key in ("since", "until", "granularity", "cursor"):
                if args.get(key):
                    payload[key] = str(args[key])
            for key in ("limit",):
                if args.get(key) is not None:
                    payload[key] = int(args[key])
            if args.get("session"):
                s = str(args["session"])
                payload["session"] = self._session_id if s == "current" else s
            if args.get("project"):
                payload["project"] = str(args["project"])
            else:
                payload["cwd"] = str(args.get("cwd") or os.getcwd())
            got = self._post_status("/memory/timeline", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_update":
            rid = str(args.get("id", "")).strip()
            if not rid:
                return json.dumps({"error": "Missing required parameter: id"})
            payload = {"id": rid}
            for key in ("status", "tags", "project"):
                if args.get(key) is not None:
                    payload[key] = str(args[key])
            got = self._post_status("/memory/update", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_review":
            action = str(args.get("action", "")).strip()
            if action not in ("list", "approve", "reject"):
                return json.dumps({"error": "memory_review action must be list|approve|reject"})
            payload = {"action": action}
            if args.get("ids"):
                payload["ids"] = [str(x) for x in args["ids"]]
            got = self._post_status("/memory/review", payload)
            return self._agent_tool_result(got)

        elif tool_name == "memory_get":
            payload: dict = {}
            if args.get("ids"):
                payload["ids"] = [str(x) for x in args["ids"]]
            if args.get("expand"):
                payload["expand"] = str(args["expand"])
            if args.get("session"):
                s = str(args["session"])
                payload["session"] = self._session_id if s == "current" else s
            for key in ("after_turn", "before_turn", "limit"):
                if args.get(key) is not None:
                    payload[key] = int(args[key])
            if not payload.get("ids") and not payload.get("session"):
                return json.dumps({"error": "memory_get needs either ids or session"})
            got = self._post_status("/memory/get", payload)
            return self._agent_tool_result(got)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)

    def get_config_schema(self) -> list:
        return [
            {
                "key": "port",
                "description": "Ragger HTTP port (managed by Ragger, via the dashboard or `ragger config set port`)",
                "default": "8432",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        # Ragger owns its own configuration (the settings table in
        # ~/.ragger/memories.db, edited via the dashboard or `ragger config
        # set`), not Hermes config — nothing to persist here.
        pass


def register(ctx) -> None:
    """Register Ragger as a Hermes memory provider plugin."""
    provider = RaggerMemoryProvider()
    ctx.register_memory_provider(provider)

    @ctx.hook("post_tool_call")
    def _forward_memory_to_ragger(tool_name: str, args: dict, result: str, **kwargs) -> None:
        """Forward memory(action='add') calls to Ragger store."""
        if tool_name != "memory":
            return
        action = args.get("action", "")
        if action not in ("add", "replace"):
            return
        content = args.get("content", "").strip()
        if not content:
            return
        target = args.get("target", "memory")
        category = "preference" if target == "user" else "fact"
        provider._post(
            "/store",
            {"text": content, "metadata": {"category": category, "source": "memory-tool"}},
        )

    @ctx.hook("post_turn_finalized")
    def _capture_interrupted_turn(turn_data: Optional[dict] = None, **kwargs) -> None:
        """Capture turns that Hermes skips on interrupt (e.g. user /stop).

        Completed turns are already synced via sync_turn() (the normal
        MemoryProvider path). Hermes fires this hook for EVERY turn,
        including interrupted ones, but deliberately skips the sync_turn
        path when interrupted=True (run_agent issue #15218). To avoid
        double-posting completed turns to /turn, we act ONLY on the
        interrupted case here — the gap that was dropping turns.
        """
        if not turn_data or not turn_data.get("interrupted"):
            return
        provider.post_turn_finalized(turn_data)
