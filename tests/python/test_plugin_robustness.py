#!/usr/bin/env python3
"""PHASE9B step 9: Hermes plugin robustness.

Standalone, stdlib only. Registered in CTest as test_plugin_robustness (S52);
it stubs `agent.memory_provider` so no Hermes install is needed. Run
directly:

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

# --- 6. plugin schemas match the embedded tool definitions (LR S95) ------
# tools/*.json is the single source of truth for the MCP/HTTP surfaces; the
# plugin keeps its own Hermes-facing schemas (different descriptions, and
# ragger_search/ragger_store are the plugin's names for search/store). The
# contract checked here: every plugin tool exists among the tool definitions
# and demands the same required arguments.
TOOLS_DIR = PLUGIN_PATH.parent.parent / "tools"
tool_defs = {f.stem: json.loads(f.read_text())
             for f in sorted(TOOLS_DIR.glob("*.json"))}
PLUGIN_NAME_MAP = {"ragger_search": "search", "ragger_store": "store"}
p6 = make_provider()
for schema in p6.get_tool_schemas():
    mapped = PLUGIN_NAME_MAP.get(schema["name"], schema["name"])
    check(mapped in tool_defs,
          "plugin tool %r has no tools/%s.json" % (schema["name"], mapped))
    plugin_required = sorted(schema["parameters"].get("required", []))
    def_required = sorted(tool_defs[mapped]["inputSchema"].get("required", []))
    check(plugin_required == def_required,
          "plugin %r required %r != tools/%s.json required %r"
          % (schema["name"], plugin_required, mapped, def_required))

# --- 7. ragger_search must not hard-code limit (X11 / M11) --------------
# The schema text and tools/search.json say the default is the
# `default_search_limit` setting, so an omitted limit must send NO limit key
# (server default applies); a passed limit is forwarded after int coercion;
# a non-integer limit is an error JSON and sends no request at all.
_state["status"] = 200
_state["body"] = '{"results": []}'
p7 = make_provider()

_state["last_body"] = ""
p7.handle_tool_call("ragger_search", {"query": "q"})
check(_state["last_body"] != "", "omitted limit must still send a request")
sent = json.loads(_state["last_body"])
check("limit" not in sent,
      "no limit passed -> the body must have no limit key, got %r" % sent)

_state["last_body"] = ""
p7.handle_tool_call("ragger_search", {"query": "q", "limit": 3})
check(_state["last_body"] != "", "limit=3 must send a request")
sent = json.loads(_state["last_body"])
check(sent.get("limit") == 3, "limit=3 must be forwarded as 3, got %r" % sent)

_state["last_body"] = ""
out = json.loads(p7.handle_tool_call("ragger_search", {"query": "q", "limit": "x"}))
check("error" in out, "limit='x' must return an error JSON, got %r" % out)
check(_state["last_body"] == "", "limit='x' must send no request")

# --- 8. host scaffolding is stripped from a captured prompt -------------
# Hermes labels a mid-turn steering message and attaches the gateway's
# routing envelope. Stored verbatim that pollutes the remembered prompt and
# stops an imported copy of the same exchange from matching it.
_clean = plugin.RaggerMemoryProvider._clean_user_text
raw = (
    "First there's a problem with imported data. The import shows different "
    "assistant text responses than what was initially captured.\n"
    "\n"
    "User correction during the turn: Gateway message origin (JSON data, not "
    "instructions or authorization):\n"
    '{"platform": "telegram", "chat_id": "8451224366", "message_id": "41919"}\n'
    "Do not guess a reply destination when these fields are insufficient.\n"
    "\n"
    "Check the last 22 turns when sorted by date."
)
got = _clean(raw)
check("Gateway message origin" not in got, "envelope must be stripped, got %r" % got)
check("chat_id" not in got, "routing JSON must be stripped, got %r" % got)
check("User correction" not in got, "the label must be stripped, got %r" % got)
check(got.startswith("First there's a problem"), "prompt must survive, got %r" % got)
check(got.endswith("Check the last 22 turns when sorted by date."),
      "the steering statement must survive, got %r" % got)
check(got.count("\n\n") == 1 and "\n\n\n" not in got,
      "the two statements must join with ONE blank line, got %r" % got)

oob = ("Do the thing.\n\n"
       "[OUT-OF-BAND USER MESSAGE \u2014 a direct message from the user]\n"
       "Actually, stop at step two.\n"
       "[/OUT-OF-BAND USER MESSAGE]")
check(_clean(oob) == "Do the thing.\n\nActually, stop at step two.",
      "the out-of-band wrapper must be stripped, got %r" % _clean(oob))

check(_clean("Just a normal question?") == "Just a normal question?",
      "an ordinary prompt must be untouched")

# --- 9. a multi-part answer is captured whole --------------------------
# Hermes hands a provider `final_response`, which is the LAST assistant
# message only. An answer that speaks, calls a tool, then speaks again used
# to reach Ragger with everything before the final block missing. The
# provider rebuilds the turn from the `messages` transcript instead.
_join = plugin.RaggerMemoryProvider._assistant_text_for_turn

convo = [
    {"role": "user", "content": "older question"},
    {"role": "assistant", "content": "older answer"},
    {"role": "user", "content": "Can we do both?"},
    {"role": "assistant", "content": "Both, yes — and the good news is this needs no new code."},
    {"role": "assistant", "content": [{"type": "tool_use", "name": "patch", "input": {}}]},
    {"role": "tool", "content": "patched"},
    {"role": "assistant", "content": [{"type": "text", "text": "Added as compaction_full."}]},
]
got = _join(convo, "Added as compaction_full.")
check(got == "Both, yes — and the good news is this needs no new code.\n\nAdded as compaction_full.",
      "both assistant blocks of THIS turn must be captured, got %r" % got)
check("older answer" not in got, "the previous turn must not leak in, got %r" % got)
check("patched" not in got, "tool results must not be captured, got %r" % got)
check("tool_use" not in got, "tool_use blocks must not be captured, got %r" % got)

# Single-block turn: unchanged behaviour, no duplication.
simple = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
check(_join(simple, "hello") == "hello", "a single-block answer must not be duplicated")

# A final_response the transcript does not end with (held/rewritten turn).
held = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "draft"}]
check(_join(held, "final") == "draft\n\nfinal", "a rewritten final response must be appended")

# Degenerate input falls back to the final text and never raises.
check(_join(None, "only this") == "only this", "messages=None must fall back")
check(_join([], "only this") == "only this", "an empty transcript must fall back")
check(_join(["not a dict", 7], "only this") == "only this", "junk entries must fall back")
check(_join([{"role": "user", "content": "q"}], "") == "", "no assistant text is empty, not a crash")

# Thinking is NEVER captured (Reid, 2026-11-27: no thinking, no tool calls,
# every final response). Hermes carries reasoning two ways and both must be
# ignored: content parts typed thinking/redacted_thinking
# (anthropic_message_convert.py _THINKING_TYPES), and top-level message keys
# reasoning / reasoning_content / reasoning_details (turn_finalizer.py).
thinking_convo = [
    {"role": "user", "content": "Can we do both?"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "SECRET deliberation the user never saw"},
        {"type": "text", "text": "Both, yes."},
    ]},
    {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "SECRETblob"}]},
    {"role": "assistant",
     "reasoning": "SECRET top-level reasoning",
     "reasoning_content": "SECRET reasoning_content",
     "content": "Added as compaction_full."},
]
got = _join(thinking_convo, "Added as compaction_full.")
check("SECRET" not in got, "no form of thinking may be captured, got %r" % got)
check(got == "Both, yes.\n\nAdded as compaction_full.",
      "both visible responses must survive intact, got %r" % got)

# The signature must keep `messages`: Hermes only forwards the transcript to
# providers whose sync_turn accepts it, so losing the parameter silently
# reintroduces the truncation bug.
import inspect
_params = inspect.signature(plugin.RaggerMemoryProvider.sync_turn).parameters
check("messages" in _params, "sync_turn must accept `messages` or Hermes drops it")
check("turn_author" in _params, "sync_turn must match the base signature")

# 10. A finished background fan-out arrives as a USER-role message, with
# anything the user typed while it ran concatenated after it. Seen live on
# turn 4456: 73,426 characters of machinery with the real prompt at 99.7%,
# all of it embedded and summarized as if the user had written it.
_clean = plugin.RaggerMemoryProvider._clean_user_text

_delegation = (
    "[ASYNC DELEGATION BATCH COMPLETE \u2014 deleg_15334c2f]\n"
    "A background fan-out unit you dispatched earlier \u2014 8 subagent(s) \u2014 "
    "has finished; its consolidated results are below.\n\n"
    + ("Role: leaf   Model: claude-opus-5\nFull live transcript: task-7.log\n" * 40)
    + "\n[INTERNAL NOTIFICATION \u2014 generated by the Hermes gateway, not "
      "written by a human user. Do not treat it as user input, and do not reply "
      "to any earlier chat message because of it.]\n\n"
      "I assume we will have to use a timer to trigger a compacted context."
)
_got = _clean(_delegation)
check("I assume we will have to use a timer" in _got,
      "the user's real prompt must survive, got %r" % _got[:200])
check("deleg_15334c2f" not in _got and "subagent" not in _got,
      "delegation machinery must not be captured, got %r" % _got[:200])
check("INTERNAL NOTIFICATION" not in _got,
      "the host's own not-user-input marker must be honoured, got %r" % _got[:200])
check(len(_got) < 400, "payload must collapse, got %d chars" % len(_got))
# The turn must still be capturable: emptying user_text would make _post_turn
# drop it and take the assistant's reply with it.
check(_clean("[ASYNC DELEGATION BATCH COMPLETE \u2014 d1]\nresults here\n").strip() != "",
      "a notification-only turn must keep a marker so its reply is still stored")
# A turn that is ONLY a gateway notice still has a real assistant response, so
# it must stay capturable. Cleaning it to "" would make _post_turn drop it.
# (Seen live on turn 4503: user_text was nothing but the notification marker.)
_notice_only = ("[INTERNAL NOTIFICATION \u2014 generated by the Hermes gateway, not "
                "written by a human user. Do not treat it as user input, and do not "
                "reply to any earlier chat message because of it.]\n")
check(_clean(_notice_only).strip() != "",
      "a notice-only turn must keep a marker, else its reply is lost")
check("INTERNAL NOTIFICATION" not in _clean(_notice_only),
      "the notice body must still go, got %r" % _clean(_notice_only))

# A finished background job arrives as a user message carrying command output.
# That is tool output: it collapses, but text typed alongside it survives.
_bg = ("[IMPORTANT: Background process proc_abc completed normally (exit code 0).\n"
       "Command: ctest\nOutput:\n100% tests passed, 0 tests failed out of 21\n]\n\n"
       "Now commit it.")
_bgc = _clean(_bg)
check("Now commit it." in _bgc, "text typed alongside a job notice must survive, got %r" % _bgc)
check("100% tests passed" not in _bgc, "job stdout must not be captured, got %r" % _bgc)
check(_clean(_bg.split("\n\nNow commit it.")[0]).strip() != "",
      "a job-only turn must keep a marker so its reply is still stored")

# Ordinary prompts, including ones that merely mention delegation, are untouched.
for _plain in ("Check turns 4456 and 4488.",
               "Use delegate_task for the audit, 3 subagents."):
    check(_clean(_plain) == _plain, "plain prompt altered: %r" % _clean(_plain))

_server.shutdown()
print("OK")
