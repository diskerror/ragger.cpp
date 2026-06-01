# Ragger — Structure & Redundancy Report

*Prepared for the next programmer. Scope: the C++ codebase at `CLionProjects/Ragger`
(`src/*.cpp` ≈ 9,930 lines, `include/ragger/*.h` ≈ 2,080 lines). Goal: reduce
duplicated code and make the layout easier to reason about, without changing behavior.*

---

## TL;DR

The code is functional and the module boundaries are mostly sensible, but small
utilities are **reinvented per file** because there is **no shared utility layer** —
`src/` and `include/ragger/` contain no `util`, `common`, or `helpers` unit. That single
gap is the root cause of most of the redundancy you noticed (the timestamp example is
one instance of a broad pattern).

The seven highest-value changes, in rough order of payoff-to-risk:

| # | Change | Touches | Risk | Why |
|---|--------|---------|------|-----|
| 1 | Add `ragger/util/time.*` and route all timestamp formatting through it | 5 files | low | The exact problem you flagged; still incomplete |
| 2 | Add `read_file_to_string()` + collapse the two `load_workspace_files` into one | 3 files | low–med | Two copies have already **diverged** |
| 3 | Use the existing `expand_path()` everywhere instead of raw `getenv("HOME")` | 3 files | low | Helper exists; three call sites bypass it |
| 4 | RAII `Stmt` wrapper for SQLite | `sqlite_backend.cpp` | med | `prepare/bind/step/finalize` repeated 50+× |
| 5 | One HTTP client (libcurl) instead of three stacks | `client.cpp` | med | Raw-socket client duplicates curl client |
| 6 | A route wrapper for auth + JSON error handling in `server.cpp` | `server.cpp` | med | Same 6-line block repeated ~8× |
| 7 | Delete dead code (old summarizer, `migrate_*`, multi-user verbs) | 3 files | low | ~400+ lines that fresh builds never run |

---

## Method

I read the per-file summaries in `AI_reference/` for architecture, then verified each
finding against the source with `grep`/line reads. Every claim below cites
`file:line` so you can confirm before acting.

---

## Findings

### 1. There is no shared utility layer (root cause)

```
$ ls src include/ragger | grep -iE 'util|common|helper'
(none)
```

Every `.cpp` is a feature module (`chat`, `server`, `sqlite_backend`, …). There is
nowhere for a 6-line helper to live, so each module grows its own copy. **Fix this
first** — create `include/ragger/util/` (`time.h`, `fs.h`, `sqlite.h`) plus
`src/util/*.cpp`, and the findings below mostly become "move + call."

---

### 2. Timestamp formatting — reinvented 5×, three real formats *(your example, still open)*

The `store_document()` path now shares one timestamp per import (issue #48), but
*formatting a timestamp string* is still hand-rolled all over. Every copy does the same
`system_clock::now() → to_time_t → gmtime_r/localtime_r → strftime` dance:

| Location | Format produced |
|----------|-----------------|
| [auth.cpp:142](src/auth.cpp:142) `iso_timestamp_now()` | `%Y-%m-%dT%H:%M:%SZ` (ISO-8601 UTC) |
| [main.cpp:77](src/main.cpp:77) inline | `%Y-%m-%dT%H:%M:%S` + `"Z"` (same thing, built differently) |
| [server.cpp:188](src/server.cpp:188) inline | `%Y-%m-%dT%H:%M:%SZ` (cutoff string) |
| [inference.cpp:524](src/inference.cpp:524) & [:578](src/inference.cpp:578) | `%Y%m%d_%H%M%S` + ms (dump filenames — two near-identical copies) |
| [sqlite_backend.cpp:448](src/sqlite_backend.cpp:448) `local_timestamp()` | `%F %T` (local human) |
| [logger.cpp:123](src/logger.cpp:123) | `%Y-%m-%d %H:%M:%S.` + µs |

Note also that [storage_types.h:29](include/ragger/storage_types.h:29) documents the
backend as stamping rows with **`now_iso()`** — but no such function exists anywhere in
the tree. The name is aspirational; the report comment is describing the helper this
codebase *wants*.

**Recommendation.** `ragger/util/time.h`:
```cpp
namespace ragger::tm {
  std::string iso8601_utc();                 // 2026-06-01T12:34:56Z   (auth, main, server)
  std::string iso8601_utc(std::time_t);      // same, for an explicit instant (server cutoff)
  std::string compact_local(std::time_t);    // 20260601_123456        (inference dumps)
  std::string human_local(std::time_t = 0);  // 2026-06-01 12:34:56    (sqlite_backend)
}
```
Then `auth.cpp`'s `iso_timestamp_now` and the four inline copies become one call each.
Keep the SQL `strftime('%Y-%m-%dT%H:%M:%SZ','now')` defaults in the schema as they are
(those run inside SQLite, not C++ — see [sqlite_backend.cpp:386](src/sqlite_backend.cpp:386)).

---

### 3. File reading & workspace-file loading — duplicated and **already diverged**

**(a) Read-file-to-string** via `istreambuf_iterator<char>` is open-coded in
`chat.cpp`, `chat_sessions.cpp`, and `main.cpp`. One `read_file_to_string(path)` in
`util/fs.h` replaces all of them.

**(b) `load_workspace_files`** is implemented **twice** with the same intent — load
`SYSTEM.md` then `SOUL.md`/`USER.md`/`MEMORY.md`, trim trailing whitespace, join with
`\n\n---\n\n`:

- [chat.cpp:49](src/chat.cpp:49) — REPL version, adds priority-budget truncation.
- [chat_sessions.cpp:221](src/chat_sessions.cpp:221) — server version, no truncation.

They have already drifted: the chat.cpp header comment claims the priority order is
`SOUL.md > USER.md > AGENTS.md > TOOLS.md > MEMORY.md`
([chat.cpp:53](src/chat.cpp:53)), but the actual loop only iterates
`{"SOUL.md","USER.md","MEMORY.md"}` ([chat.cpp:98](src/chat.cpp:98)) — identical to the
server copy. That stale comment is exactly the kind of bug two-copies-of-one-function
breeds.

**Recommendation.** One `load_workspace(std::optional<int> char_budget)` (truncate when
the budget is set, otherwise load whole) in a small `workspace.cpp`, called by both
`Chat` and `ChatSessionManager`.

---

### 4. `expand_path()` exists but three modules bypass it

[config.cpp:38](src/config.cpp:38) defines a correct `expand_path()` (handles `~`,
falls back to `getpwuid`) and most of the code uses it. But raw `getenv("HOME")` is
re-rolled in:

- [api_formats.cpp:44](src/api_formats.cpp:44), [:52](src/api_formats.cpp:52), [:63](src/api_formats.cpp:63)
- [sqlite_backend.cpp:435](src/sqlite_backend.cpp:435)
- [mcp.cpp:186](src/mcp.cpp:186) (`getpwuid` again)

These miss the `~`-expansion and the `getpwuid` fallback, so they behave differently
when `HOME` is unset. **Recommendation.** Call `expand_path()` (or a thin
`home_dir()`/`ragger_dir()` in `util/fs.h`) from all five sites.

---

### 5. SQLite access — `prepare/bind/step/finalize` repeated 50+ times

In `sqlite_backend.cpp`:

```
sqlite3_prepare_v2 : 54    sqlite3_step      : 53
sqlite3_finalize   : 61    sqlite3_bind_text : 67
```

Every accessor hand-writes the same five-step ritual (see the cluster around
[sqlite_backend.cpp:1644-1700](src/sqlite_backend.cpp:1644)). Two costs:

1. **Volume** — these ~50 functions are 80% boilerplate.
2. **Leak/early-return risk** — `finalize` is manual, so each early `return` must
   remember to finalize first (e.g. [sqlite_backend.cpp:1683](src/sqlite_backend.cpp:1683)
   finalizes on both the row and no-row branch). One missed path leaks a statement.

**Recommendation.** A small RAII wrapper in `util/sqlite.h`:
```cpp
struct Stmt {
    Stmt(sqlite3* db, std::string_view sql);   // prepare or throw
    ~Stmt();                                    // finalize, always
    Stmt& bind(int i, std::string_view);        // chainable
    Stmt& bind(int i, int);
    bool step();                                // SQLITE_ROW?
    std::string text(int col); int integer(int col);
};
```
turning the example at lines 1683-1700 into ~5 lines with no manual finalize. This is
the single biggest line-count reduction available; do it incrementally, one method at a
time, since each conversion is independently testable.

---

### 6. Three separate HTTP stacks

The project speaks HTTP three different ways:

- **Inbound server:** `httplib` ([server.cpp](src/server.cpp)).
- **Outbound to LLMs:** `libcurl` ([inference.cpp:1](src/inference.cpp), incl. SSE).
- **Outbound to the local daemon:** **raw POSIX sockets**, hand-built HTTP/1.1
  request/response parsing with a manual 5s timeout
  ([client.cpp](src/client.cpp) `http_request`).

The third is the redundant one: `RaggerClient` re-implements (status-line parsing,
header building, `Authorization: Bearer`, timeouts) what the curl path in
`inference.cpp` already does robustly. Maintaining a bespoke socket HTTP client is also
a quiet source of edge-case bugs (chunked encoding, partial reads).

**Recommendation.** Back `RaggerClient` with libcurl (already a dependency, already
used). Inbound stays on httplib — that's a different role and fine as-is.

---

### 7. `server.cpp` route handlers — same auth + error block ~8×

Each endpoint repeats:
```cpp
if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
try { ... }
catch (json::exception& e) { res.status = 400; res.set_content("JSON error: " + ...); }
catch (std::exception& e)  { res.status = 500; res.set_content("ERROR: "      + ...); }
```
See [server.cpp:475](src/server.cpp:475), [:493](src/server.cpp:493),
[:532](src/server.cpp:532), [:578](src/server.cpp:578), [:608](src/server.cpp:608),
[:633](src/server.cpp:633), [:669](src/server.cpp:669), [:693](src/server.cpp:693), …

It has also drifted into inconsistency — the 500 message is variously `"ERROR: "`,
`"Error: "`, and the 400 sometimes `"JSON error: "`. Clients can't rely on the shape.

**Recommendation.** A wrapper that does auth + uniform try/catch once:
```cpp
auto guarded = [&](auto handler) {
    return [=](const httplib::Request& rq, httplib::Response& rs) {
        auto user = authenticate(rq);
        if (!user) return unauthorized(rs);
        try { handler(*user, rq, rs); }
        catch (const json::exception& e) { json_error(rs, 400, e.what()); }
        catch (const std::exception& e)  { json_error(rs, 500, e.what()); }
    };
};
svr.Post("/store", guarded(handle_store));
```
This both removes the repetition and makes every error response consistent.

---

### 8. Summarization logic lives in three places (one of them dead)

The fading-memory rework (#22/#41) added per-turn summarization but left the old path
in place:

- **Live, new:** `Chat::summarize_turn` ([chat.cpp:412](src/chat.cpp:412)) — per-turn
  L2 + running-L3 merge with the `NO_MERGE` topic-shift sentinel.
- **Dead, old:** `Chat::bg_summarize` ([chat.cpp:358](src/chat.cpp:358)) and
  `Chat::summarize_conversation` ([chat.cpp:502](src/chat.cpp:502)) — the superseded
  batch path (the AI_reference summary itself calls these "superseded").
- **Live, separate:** the server's idle-session summarizer
  ([server.cpp housekeeping](src/server.cpp), see AI_reference §3).
- **Live no-op:** `Chat::check_pause_summary` ([chat.cpp:482](src/chat.cpp:482)) is now
  an empty hook still called from the run loop ([chat.cpp:607](src/chat.cpp:607)).

The two live summarizers build a "summarize these turns" prompt and store the result —
the same shape in two files. **Recommendation.** Extract a `Summarizer` (prompt
build + memory-model call + `store_summary`) used by both `Chat` and the server
housekeeping; delete `bg_summarize`, `summarize_conversation`, and the `check_pause_summary`
no-op (and their declarations in [chat.h:87](include/ragger/chat.h:87),
[chat.h:96](include/ragger/chat.h:96), [chat.h:102](include/ragger/chat.h:102)).

---

### 9. Dead / vestigial code to remove

- **`migrate_*` (5 functions)** — [sqlite_backend.cpp:330-429](src/sqlite_backend.cpp:330).
  Per the schema summary they're "dead for fresh DBs (never called from
  `create_schema`)"; pre-v2 data is exported out-of-band. Either delete or move behind
  an explicit `ragger migrate-legacy` verb so they're not mistaken for live schema code.
- **Multi-user CLI verbs** — `useradd`/`usermod`/`userdel`/`add-user`/`add-all`/
  `remove-user`/`passwd`, ~lines [main.cpp:965-1330](src/main.cpp:965), all marked
  *"Multi-user mode removed … deprecated"* ([main.cpp:1067](src/main.cpp:1067),
  [:1087](src/main.cpp:1087)). ~265 lines of CLI surface for a removed feature.
- **`collections` no-op** — threaded through `mcp.cpp`, `client.cpp`, `storage_backend.h`,
  `sqlite_backend.h` as a parameter that the lean v2 schema ignores
  ([sqlite_backend.h:74](include/ragger/sqlite_backend.h:74)). Fine to keep for wire
  compat, but document it in *one* place and stop forwarding it through internal C++
  signatures.

---

### 10. `main.cpp` is doing four jobs (1,412 lines)

It mixes (a) CLI option parsing, (b) a ~25-arm `else if (command == …)` dispatch
([main.cpp:713-1370](src/main.cpp:713)), (c) daemon/service control wrapping
`launchctl`/`systemctl` ([main.cpp:214-384](src/main.cpp:214)), and (d) import/embed
orchestration ([main.cpp:54](src/main.cpp:54)).

**Recommendation (lower priority, do after the helpers land).** Split into:
`cli_dispatch.cpp` (the verb table), `daemon_control.cpp` (the service wrappers, already
self-contained), and keep `do_import`/`do_chat` where they make sense. A
`struct Command { name; handler; help; }` table also lets `help`/`version` be generated
instead of hand-maintained.

---

## Proposed target layout

```
include/ragger/
  util/
    time.h        // iso8601_utc(), compact_local(), human_local()
    fs.h          // read_file_to_string(), expand_path()/home_dir(), ragger_dir()
    sqlite.h      // Stmt RAII wrapper
  workspace.h     // load_workspace(optional<int> budget)
  http_client.h   // single libcurl-backed client (RaggerClient + inference share it)
  summarizer.h    // prompt-build + memory-model call + store_summary
src/
  util/{time,fs,sqlite}.cpp
  workspace.cpp  summarizer.cpp
  cli/{cli_dispatch,daemon_control}.cpp   // main.cpp slimmed to argv → dispatch
  ... (existing feature modules unchanged)
```

---

## Suggested sequence (each step independently shippable)

1. **`util/time.*`** → migrate the 5 timestamp sites. (Smallest, proves the pattern.)
2. **`util/fs.*`** → `read_file_to_string` + funnel the `getenv("HOME")` sites through `expand_path`.
3. **Delete dead code** (§9, §8 old summarizer) — pure removal, shrinks the surface before you refactor it.
4. **Unify `load_workspace_files`** (§3b) on top of the new `fs` helper.
5. **`util/sqlite.h` `Stmt`** → convert `sqlite_backend.cpp` methods a few at a time.
6. **Route wrapper** in `server.cpp` (§7).
7. **Single HTTP client** (§6) and **`Summarizer`** (§8), then **split `main.cpp`** (§10).

Steps 1-4 are low-risk and reclaim the most "obvious" duplication for the least effort —
a good place to stop if time is short.

---

## What *not* to "simplify" (intentional, leave alone)

- **PIMPL in `embedder.cpp` / `tokenizer_wrapper.cpp`** — deliberately hides ONNX /
  tokenizers headers from public headers. Keep.
- **Logger no-op sinks** ([logger.cpp:24](src/logger.cpp:24)) — intentional so logging
  is safe before any logger is constructed. Keep.
- **Data-driven `api_formats.cpp`** — the JSON-format abstraction is good design, not
  redundancy; it's what *prevents* per-provider duplication.
- **Retained config key names** (`bm25_weight`, `rebuild_bm25`, the `collections` wire
  field) — kept for backward compatibility on purpose; rename only with a migration.
- **Two HTTP *roles*** (inbound httplib vs outbound) — that's a real boundary; only the
  *outbound* duplication (§6) is worth collapsing.

---

*Generated from a read-only review; no source files were modified.*
