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
# Config reader
# ---------------------------------------------------------------------------

def _load_ragger_config() -> dict:
    """Read host/port/auth_token from ~/.ragger/settings.ini."""
    cfg = {"host": "127.0.0.1", "port": "8432", "auth_token": ""}
    cfg_path = Path.home() / ".ragger" / "settings.ini"
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
        except Exception as exc:
            logger.debug("Could not parse ~/.ragger/settings.ini: %s", exc)
    return cfg


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
        "user preferences, project decisions, key facts, lessons learned."
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
                    "Memory category: 'fact', 'decision', 'preference', or 'lesson'. "
                    "Default: 'fact'."
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
        url = self._base_url + path
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("Ragger GET %s: %s", path, exc)
            return None

    def _post(self, path: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
        url = self._base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("Ragger POST %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ragger daemon is reachable."""
        cfg = _load_ragger_config()
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
        """Kick off a background search to warm the prefetch cache."""
        if self._breaker_open():
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
    ctx.register_memory_provider(RaggerMemoryProvider())
