"""Ragger memory plugin — MemoryProvider interface for Hermes Agent.

Connects to the Ragger HTTP server (localhost:8432 by default) for local
semantic memory with hybrid vector + BM25 search. No pip dependencies —
uses stdlib urllib throughout.

Prerequisites:
    ragger start        # bring the daemon up (install.sh sets this up)

Configuration is read from ~/.ragger/settings.ini [server]:
    port       = 8432       # HTTP port  (default: 8432)
    bind       =            # bind address (default: 127.0.0.1)
    auth_token =            # bearer token (optional; leave unset for localhost)

To activate in Hermes, set in ~/.hermes/config.yaml:
    memory:
      provider: ragger
"""

from __future__ import annotations

import json
import logging
import threading
import http.client
import socket
import urllib.error
import urllib.request
from configparser import ConfigParser
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

def _load_ragger_config() -> dict:
    """Read host/port/auth_token/auto_recall from ~/.ragger/settings.ini.

    auto_recall (bool, default True) gates the *recall/injection* side only
    (queue_prefetch/prefetch → the <memory-context> block stuffed into every
    prompt). It never touches sync_turn/post_turn_finalized — turn capture
    stays always-on regardless of this setting, by design (Reid: "ragger-
    memory must always be capturing turns").

    Ragger now defaults to a Unix domain socket at <ragger_base>/ragger.sock
    (settings.ini [server] socket_enable=true, the default since the
    --ragger-base refactor). TCP (host/port, gated by [server] bind=) is now
    opt-in. Prefer the socket when socket_enable is true; only fall back to
    TCP when the server was explicitly configured for it (bind= set).
    """
    ragger_base = Path.home() / ".ragger"
    cfg = {
        "host": "127.0.0.1",
        "port": "8432",
        "auth_token": "",
        "auto_recall": True,
        "socket_enable": True,
        "socket_path": str(ragger_base / "ragger.sock"),
    }
    cfg_path = ragger_base / "settings.ini"
    if cfg_path.exists():
        parser = ConfigParser()
        try:
            parser.read(cfg_path, encoding="utf-8")
            if parser.has_section("server"):
                srv = parser["server"]
                raw_port = srv.get("port", cfg["port"]).split("#")[0].strip()
                if raw_port:
                    cfg["port"] = raw_port
                raw_bind = srv.get("bind", "").split("#")[0].strip()
                if raw_bind and raw_bind != "0.0.0.0":
                    cfg["host"] = raw_bind
                raw_token = srv.get("auth_token", "").split("#")[0].strip()
                cfg["auth_token"] = raw_token
                raw_recall = srv.get("auto_recall", "").split("#")[0].strip().lower()
                if raw_recall in ("false", "0", "no", "off"):
                    cfg["auto_recall"] = False
                raw_socket_enable = srv.get("socket_enable", "").split("#")[0].strip().lower()
                if raw_socket_enable in ("false", "0", "no", "off"):
                    cfg["socket_enable"] = False
                elif raw_socket_enable in ("true", "1", "yes", "on"):
                    cfg["socket_enable"] = True
        except Exception as exc:
            logger.debug("Could not parse ~/.ragger/settings.ini: %s", exc)
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

        No-op when auto_recall is disabled (~/.ragger/settings.ini
        [server] auto_recall = false) — turn capture (sync_turn /
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

    def _post_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Internal: POST a turn to Ragger's /turn endpoint.
        
        Called from both sync_turn (normal path) and post_turn_finalized (catch-all).
        """
        if self._breaker_open() or not user_content:
            return
        result = self._post("/turn", {
            "user": user_content,
            "assistant": assistant_content,
            "model": self._model,
            "session_id": session_id,
        }, timeout=4.0)
        if result is None:
            self._record_fail()
        else:
            self._record_ok()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA]

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

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)

    def get_config_schema(self) -> list:
        return [
            {
                "key": "port",
                "description": "Ragger HTTP port (set in ~/.ragger/settings.ini)",
                "default": "8432",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        # Ragger configuration lives in ~/.ragger/settings.ini, not Hermes config.
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
