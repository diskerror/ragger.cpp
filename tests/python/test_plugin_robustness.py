#!/usr/bin/env python3
"""PHASE9B step 9: Hermes plugin robustness.

Standalone, stdlib only. NOT registered in CTest: the plugin imports
`agent.memory_provider`, and the interpreter that has Hermes on its path is
machine-specific. Run directly:

    python3 tests/python/test_plugin_robustness.py

Checks (see PHASE9B-STEPS.md step 9):
  * an HTTP 4xx from the daemon is the caller's mistake — it is returned as
    an error and must NOT count toward the circuit breaker;
  * transport failures still trip it, and the cooldown still closes it;
  * a non-integer argument is a validation error, not an exception;
  * the breaker state is safe under concurrent use.
"""

import importlib.util
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- stub agent.memory_provider before importing the plugin ------------

_agent = types.ModuleType("agent")
_mp = types.ModuleType("agent.memory_provider")


class MemoryProvider:  # the plugin only needs the base class to exist
    pass


_mp.MemoryProvider = MemoryProvider
_agent.memory_provider = _mp
sys.modules.setdefault("agent", _agent)
sys.modules["agent.memory_provider"] = _mp

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "hermes-plugin" / "__init__.py"
_spec = importlib.util.spec_from_file_location("ragger_plugin", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)

# --- stub daemon -------------------------------------------------------

_state = {"status": 400, "body": "ERROR: bad", "last_path": "", "last_body": ""}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = ""
        if length:
            body = self.rfile.read(length).decode()
            _state["last_path"] = self.path
            _state["last_body"] = body
        self.send_response(_state["status"])
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(_state["body"].encode())

    def log_message(self, *args):  # silence
        pass


_server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
threading.Thread(target=_server.serve_forever, daemon=True).start()
BASE_URL = "http://127.0.0.1:%d" % _server.server_address[1]


def make_provider(base_url=BASE_URL):
    p = plugin.RaggerMemoryProvider()
    p._socket_enable = False
    p._base_url = base_url
    p._auth_token = ""
    return p


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# --- 1. HTTP 4xx must not trip the breaker -----------------------------

p = make_provider()
for i in range(10):
    out = json.loads(p.handle_tool_call("ragger_search", {"query": "x"}))
    check("error" in out, "400 from /search should return an error")
    check("bad" in out["error"], "the server's message should reach the caller: %r" % out)
check(not p._breaker_open(),
      "ten HTTP 400s are ten caller mistakes, not a daemon outage — the breaker "
      "must stay closed (before the fix the 5th one tripped it)")
check(p._failures == 0, "a 4xx records OK, got failures=%d" % p._failures)

for i in range(10):
    out = json.loads(p.handle_tool_call("ragger_store", {"text": "hello"}))
    check("error" in out, "400 from /store should return an error")
check(not p._breaker_open(), "ragger_store 4xx must not trip the breaker either")

# --- 2. transport failures still trip it, cooldown still closes it ------

p2 = make_provider("http://127.0.0.1:1")   # nothing listens: connection refused
for i in range(4):
    p2.handle_tool_call("ragger_search", {"query": "x"})
    check(not p2._breaker_open(), "breaker should still be closed at %d failures" % (i + 1))
p2.handle_tool_call("ragger_search", {"query": "x"})
check(p2._breaker_open(), "five transport failures must trip the breaker")
open_msg = json.loads(p2.handle_tool_call("ragger_search", {"query": "x"}))
check("circuit breaker" in open_msg.get("error", ""), "open breaker reports itself as such")
p2._breaker_until = 0.0    # cooldown expired
check(not p2._breaker_open(), "the breaker must close when its cooldown expires")

# --- 3. bad arguments are validation errors, never exceptions -----------

p3 = make_provider()
out = json.loads(p3.handle_tool_call("ragger_search", {"query": "x", "limit": "abc"}))
check(out.get("error") == "limit must be an integer",
      "a non-integer limit is a named validation error, got %r" % out)
for args in ({"query": "x", "limit": None},
             {"query": "x", "limit": [3]},
             {"query": "x", "tags": "not-a-list"}):
    try:
        out = json.loads(p3.handle_tool_call("ragger_search", args))
    except Exception as exc:                       # noqa: BLE001
        raise AssertionError("handle_tool_call raised for %r: %r" % (args, exc))
    check("error" in out, "bad argument should return an error JSON: %r" % out)

try:
    out = json.loads(p3.handle_tool_call("memory_decision",
                                         {"text": "t", "trigger": "g", "supersedes": "x"}))
except Exception as exc:                           # noqa: BLE001
    raise AssertionError("memory_decision raised for a bad supersedes: %r" % exc)
check("error" in out and "integer" in out["error"], "supersedes is validated: %r" % out)

# --- 4. concurrent breaker use ------------------------------------------

p4 = make_provider()


def hammer():
    for _ in range(1000):
        p4._record_fail()


threads = [threading.Thread(target=hammer) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check(p4._failures >= 5, "failures must be counted, got %d" % p4._failures)
check(p4._breaker_open(), "the breaker must be open after concurrent failures")

# --- 5. memory_note sends session provenance + trigger (LR S59) ---------

_state["status"] = 200
_state["body"] = '{"id": "1", "status": "stored"}'
p5 = make_provider()
p5._session_id = "sess-59"
p5.handle_tool_call("memory_note", {"text": "n1", "trigger": "when probing"})
sent = json.loads(_state["last_body"])
check(_state["last_path"] == "/memory/note", "routed to /memory/note, got %r" % _state["last_path"])
check(sent.get("session") == "sess-59",
      "memory_note must send session provenance like memory_decision: %r" % sent)
check(sent.get("trigger") == "when probing", "trigger must be forwarded: %r" % sent)
p5.handle_tool_call("memory_note", {"text": "n2", "session": "current", "evidence": [3]})
sent = json.loads(_state["last_body"])
check(sent.get("session") == "sess-59", "'current' resolves to the Hermes session: %r" % sent)
check(sent.get("evidence") == [3], "evidence forwarded: %r" % sent)
p5.handle_tool_call("memory_note", {"text": "n3", "session": "explicit-guid"})
sent = json.loads(_state["last_body"])
check(sent.get("session") == "explicit-guid", "an explicit GUID passes through: %r" % sent)

_server.shutdown()
print("OK")
