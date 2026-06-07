# Ragger Code Review

**Date:** 2026-06-05  
**Reviewer:** Claude Code (claude-sonnet-4-6)  
**Scope:** All headers, sources, tests, AI_reference summaries, and docs files.

---

## Table of Contents

1. [Inconsistencies](#1-inconsistencies)
2. [Redundant Code](#2-redundant-code)
3. [Class Structure](#3-class-structure)
4. [Naming Clarity](#4-naming-clarity)
5. [AI_reference Accuracy](#5-ai_reference-accuracy)
6. [Doc vs Behavior](#6-doc-vs-behavior)

---

## 1. Inconsistencies

### 1.1 `backfill_embeddings` Only Covers `summaries`, Not All Four Tables

**Location:** `src/sqlite_backend.cpp:1334–1353`  
**Severity:** High

`backfill_embeddings` was written early (for the summaries-only era) and now queries only:

```sql
SELECT summary_id AS id, text FROM summaries WHERE embedding IS NULL
```

But the codebase now has four embedded tables: `turns`, `summaries`, `decisions`, and `documents`. The header comment on `StorageBackend::backfill_embeddings()` says "Embed only rows whose embedding column is NULL" with no table restriction. `rebuild_embeddings` correctly walks all four tables (lines 1273–1332). `count_embeddable_rows()` correctly counts all four tables (lines 1220–1227).

At startup, `RaggerMemory` calls `backfill_embeddings()` to recover deferred-embedding rows. A document imported with `defer_embedding=true` will never have its embedding backfilled on the next server start — it silently stays NULL forever (unless `rebuild-embeddings` is run manually).

**Suggested fix:** Extend `Impl::backfill_embeddings` to iterate the same `TableSpec` array used by `rebuild_embeddings`, using the same `turn_embed_text` join for turns. The four-table loop already exists; backfill just needs the same structure with a `WHERE embedding IS NULL` filter.

---

### 1.2 `capture_turns` and `build_context` Defaults Disagree Between Code and Docs

**Location:** `include/ragger/config.h:109,115` vs `src/config.cpp:82–83` (DEFAULT_CONFIG string) vs `docs/configuration.md:46–47`  
**Severity:** High

The `Config` struct defaults both flags to `true`:
```cpp
bool capture_turns = true;
bool build_context = true;
```

The embedded `DEFAULT_CONFIG` string (which is written to `~/.ragger/settings.ini` on first run) defaults both to `false`:
```ini
capture_turns  = false
build_context  = false
```

`docs/configuration.md` table shows the default as `true` for both.

The effective default depends on whether the user has a `settings.ini`. A fresh install writes `false`; a programmatic user who never calls `init_config` gets `true`. This is confusing. The docs are wrong about the effective default — a first-run user gets the INI value of `false`.

**Suggested fix:** Align the struct defaults with the DEFAULT_CONFIG string (both `false`). Update docs to match. Alternatively, change the INI default to `true` and update the struct. Pick one source of truth.

---

### 1.3 Mixed `column_text()` / Raw `reinterpret_cast<const char*>(sqlite3_column_text(...))` Usage

**Location:** `src/sqlite_backend.cpp` — 11 raw casts interspersed with ~30 `Stmt::column_text()` calls  
**Severity:** Medium

`include/ragger/util/sqlite.h` provides `Stmt::column_text(int c)` that wraps the `reinterpret_cast` and handles NULL. The `Impl` methods use it correctly in some places (e.g., lines 732–735 for `turns_by_session`) but bypass it in others with raw casts (e.g., lines 1002, 1016, 1068, 1253, 1305, 1341, 1382, 1411).

The raw pattern (1) is more verbose, (2) requires the caller to remember NULL checks, and (3) the existing `Stmt::column_text` already handles NULL correctly (returns `""`). Some raw-cast sites do explicit NULL checks; others don't.

**Suggested fix:** Replace all `reinterpret_cast<const char*>(sqlite3_column_text(s.raw(), i))` patterns with `s.column_text(i)`. The only case where the raw accessor is justified is when `column_text_opt()` (returning `nullopt` for NULL) is semantically needed — use that variant explicitly.

---

### 1.4 `float cleanup_max_age_hours` Parameter Passed to `int cleanup_old_conversations(int)`

**Location:** `src/server.cpp:131`, `src/mcp.cpp:255`, `include/ragger/config.h:126`, `include/ragger/storage_backend.h:233`  
**Severity:** Medium

`Config::cleanup_max_age_hours` is `float` (supports e.g. `0.5` for 30-minute retention). The `StorageBackend::cleanup_old_conversations()` interface takes `int max_age_hours`. The server passes a `float` to an `int` parameter — this silently truncates fractional hours.

Either:
- Change the interface to `float max_age_hours` (or `double`), or
- Accept the truncation and document it (and change the config field to `int` to make the truncation obvious).

The `cleanup_old_conversations` implementation already uses `std::chrono::duration<double, std::ratio<3600>>`, so it can handle fractional values — the narrowing is in the interface only.

---

### 1.5 Search Default Limit and Min-Score Hardcoded in Server and MCP, Ignored from Config

**Location:** `src/server.cpp:409–410`, `src/mcp.cpp:193–194`  
**Severity:** Medium

The CLI `search` command correctly reads config-driven defaults:
```cpp
// src/main.cpp:348–355
response = client.search(query, cfg.default_search_limit, cfg.default_min_score, ...);
```

But the HTTP server and MCP both hardcode:
```cpp
int   limit     = body.value("limit", 5);
float min_score = body.value("min_score", 0.0f);
```

`cfg.default_search_limit` and `cfg.default_min_score` are ignored entirely in the server/MCP paths. If a user sets `default_min_score = 0.4` (the value in the DEFAULT_CONFIG string) in settings, that floor applies when searching from the CLI but not when an agent calls the `search` MCP tool or POSTs to `/search`.

**Suggested fix:** Change defaults in server and MCP to `config().default_search_limit` and `config().default_min_score`.

---

### 1.6 `InferenceClient::_endpoints` Is a Public Member with a Private-Looking Name

**Location:** `include/ragger/inference.h:57`  
**Severity:** Low

`_endpoints` has a leading underscore, which in C++ conventionally signals a private or implementation-detail member. Yet it is `public` and accessed directly in `server.cpp:241` (`inference_->_endpoints.size()`). The rest of the public members (`model`, `memory_model`, `max_tokens`) follow no such prefix.

**Suggested fix:** Either make `_endpoints` a proper private with an accessor method (`endpoints_count()` or similar), or rename it to `endpoints` (no leading underscore).

---

### 1.7 Logger `trace` Level Is Silently Broken

**Location:** `src/logger.cpp:57,65–69`  
**Severity:** Low

The `goto`-chain logger uses `goto debug` for `log_level = "debug"`, `goto info` for `"info"`, etc. But for `"trace"` the code reads:

```cpp
if (lower_str == "trace") {} /* goto trace; */
```

The `goto trace` is commented out. The label `// trace:` is also commented out. This means setting `log_level = trace` in settings silently falls through to all lambdas being no-ops (everything assigned in the fall-through default block above). No trace output is produced.

**Suggested fix:** Reinstate the `trace:` label and the `goto trace;`, or use a different dispatch mechanism (a `switch` or level-comparison integer) that doesn't rely on fall-through-via-goto.

---

## 2. Redundant Code

### 2.1 Tag-Normalizing Logic Duplicated in `store()` and `update_text()`

**Location:** `src/sqlite_backend.cpp:545–562` and `src/sqlite_backend.cpp:1077–1094`  
**Severity:** Medium

The block that:
1. Reads `metadata["tags"]` (handling both array and string forms)
2. Appends "keep" if `metadata.value("keep", false)` and "keep" not already present
3. Appends "bad" if `metadata.value("bad", false)` and "bad" not already present

…is copy-pasted verbatim in both `Impl::store()` and `Impl::update_text()`. The two blocks are identical (lines 545–562 vs. 1077–1094).

**Suggested fix:** Extract a free function (or private static method) in `Impl`:
```cpp
static std::string tags_from_metadata(const json& metadata) {
    std::string tags_str;
    if (metadata.contains("tags")) { /* ... */ }
    if (metadata.value("keep", false) && tags_str.find("keep") == std::string::npos) { /* ... */ }
    if (metadata.value("bad",  false) && tags_str.find("bad")  == std::string::npos) { /* ... */ }
    return tags_str;
}
```

---

### 2.2 Keep-Tag Protection Check Duplicated in Three Places

**Location:** `src/sqlite_backend.cpp:1378–1385` (`delete_memory`), `1404–1418` (`delete_batch` inner loop), and `1062–1073` (`update_text`)  
**Severity:** Medium

Each of the three functions opens a `SELECT tags FROM summaries WHERE summary_id = ?` statement and checks `tags.find("keep") != std::string::npos` independently. The pattern is four lines of boilerplate each time.

**Suggested fix:** Extract a private helper:
```cpp
bool has_keep_tag(int summary_id) {
    Stmt s(db, "SELECT tags FROM summaries WHERE summary_id = ?");
    s.bind(1, summary_id);
    if (!s.step()) return false;  // doesn't exist — treat as unprotected
    return s.column_text(0).find("keep") != std::string::npos;
}
```

This also eliminates the inconsistency where `update_text` distinguishes `!exists` from `protected_row` while `delete_memory` does not.

---

### 2.3 `delete_batch` Issues N Sequential SELECT+DELETE, No Transaction

**Location:** `src/sqlite_backend.cpp:1401–1443`  
**Severity:** Medium

`delete_batch` loops over every ID, opens a fresh prepared `Stmt` and runs a `SELECT tags` check — one round-trip per ID — then builds the delete SQL. On a batch of 50 IDs this is 50 separate selects. There is no transaction wrapping the whole operation, so the delete is also non-atomic.

More critically: the keep-tag filter is done in application code with individual SQLite round-trips. This could be done entirely in SQL:
```sql
DELETE FROM summaries
WHERE summary_id IN (?,?,...)
  AND tags NOT LIKE '%keep%';
```

**Suggested fix:** Wrap in `exec("BEGIN")` / `exec("COMMIT")`, and use a single SQL statement with a `tags NOT LIKE '%keep%'` predicate rather than N selects in a loop.

---

### 2.4 Timestamp Extraction in `turns_by_session` and `turns_by_session_desc` Duplicate the Row-Struct Pattern

**Location:** `src/sqlite_backend.cpp:729–739` and `905–916`  
**Severity:** Low

Both `turns_by_session` and `turns_by_session_desc` run near-identical SQL (one `ASC`, one `DESC`) and populate `TurnRecord` structs from the same five columns. The only differences are the ORDER BY direction and whether a LIMIT clause is appended.

**Suggested fix:** Extract a private helper taking an `order` parameter and an optional limit, reducing the two SQL strings to one parameterized template. Same applies to `turn_summaries_by_session_desc` and `session_summaries_desc`, which differ only by the `level` predicate value.

---

### 2.5 Duplicate Include in `main.cpp`

**Location:** `src/main.cpp:42,49`  
**Severity:** Low

```cpp
#include "ragger/sqlite_backend.h"   // line 42
// ...
#include "ragger/sqlite_backend.h"   // line 49
```

`sqlite_backend.h` is included twice. The `#pragma once` guard makes this harmless but it signals a stale edit.

---

### 2.6 JSON Response Body Construction Duplicated Across HTTP Endpoints

**Location:** `src/server.cpp` — route handlers for `/search` (lines 423–428), `/search_by_metadata` (554–557), and `/session/<guid>` (489–497)  
**Severity:** Low

Each builds a `json::array()` from a result vector using identical `push_back` calls. The search-result shape (`id`, `text`, `score`, `metadata`, `timestamp`) appears three times in slightly different forms (the metadata search omits `score`). A small local lambda or free function would reduce duplication and ensure the wire shape stays consistent if a field is ever added.

---

## 3. Class Structure

### 3.1 `SqliteBackend` Is Doing Too Much — User Management Belongs Elsewhere

**Location:** `include/ragger/sqlite_backend.h:133–159`, `src/sqlite_backend.cpp:1698–1779`  
**Severity:** Medium

`SqliteBackend` (and `StorageBackend`) implements two unrelated concerns:
1. **Memory storage**: turns, summaries, decisions, documents, embeddings — the core RAG pipeline.
2. **User/auth management**: `users` table CRUD, token management, password hashing storage, settings key-value store.

The DB-only constructor (`explicit SqliteBackend(const std::string& db_path)`) exists specifically to use the class for user-management without loading an embedder. This is a signal that the concerns should be separated.

The storage interface `StorageBackend` is intended as a swappable abstraction for the memory layer. Mixing user-management into it means any future alternative backend would also need to implement the full user/auth API even if it doesn't care about single-user token auth.

**Suggested fix:** Extract user management into a separate `UserStore` class (or a thin `UserBackend` interface) backed by a separate SQLite connection to the same file. The `users` and `settings` tables live in the same DB file and that's fine — the access layer doesn't have to be the same class.

---

### 3.2 `StorageBackend` Has Nested Structs That Should Be Top-Level

**Location:** `include/ragger/storage_backend.h:134–165`  
**Severity:** Low

`StorageBackend::DraftSummary` and `StorageBackend::SummaryRecord` are defined as nested structs inside the abstract base class. `TurnRecord` (also in this header, but defined outside the class at line 27) is correctly top-level.

Consumers of these structs (e.g., `SummarizerService`, `build_context` in `memory.cpp`) must qualify them as `StorageBackend::DraftSummary` and `StorageBackend::SummaryRecord`. Moving them to `storage_types.h` alongside `TurnRecord`, `SearchResult`, and `DocumentChunk` would unify all data-transfer types in one header and remove the class-qualification noise.

---

### 3.3 `RaggerMemory` Facade Is an Incomplete Wrapper — Callers Bypass It Via `backend()`

**Location:** `include/ragger/memory.h:115`, `src/summarizer_service.cpp:79,222,269`, `src/server.cpp:138,251,259,297,459`  
**Severity:** Medium

`RaggerMemory` is the intended facade. The `backend()` accessor leaks the `StorageBackend*` into callers who then call methods directly on it:
- `summarizer_service.cpp` calls `memory_.backend()->unsummarized_turns()`, `draft_summaries()`, `sessions_needing_close()`, `update_summary_text()`, `set_summary_tags()`, `turns_by_session_desc()`
- `server.cpp` calls `memory.backend()->get_user_by_token_hash()`, `create_user()`, `db_path()`, `turns_by_session_desc()`
- `memory.cpp` itself calls `memory.backend()->get_setting()`, `turns_by_session_desc()`, `session_summaries_desc()`, etc. in `build_context`

Many of these are methods that were never forwarded through `RaggerMemory`. The facade is incomplete. The bypass is workable but it means:
1. Tests of `SummarizerService` depend on `SqliteBackend` specifically, not on the interface.
2. Adding a second backend implementation would require tracing all the bypass call-sites.

**Suggested fix (pragmatic):** Add forwarding methods to `RaggerMemory` for the high-use backend bypasses: `unsummarized_turns()`, `draft_summaries()`, `sessions_needing_close()`, `set_summary_tags()`, `turns_by_session_desc()`, `turn_summaries_by_session_desc()`, `session_summaries_desc()`. The user-management calls are a separate concern (see §3.1) and should be routed through a `UserStore` rather than through `RaggerMemory`.

---

### 3.4 `Server::Impl` Is Over-Large — Route Handlers Are Defined Inside a Struct

**Location:** `src/server.cpp:56–582`  
**Severity:** Low

`Server::Impl` is a ~530-line nested struct that contains infrastructure (the two `httplib::Server` instances, member data), the auth logic, the housekeeping timer, the inference init, and the full `setup_routes()` method with every route handler inlined as lambdas.

This is workable but makes individual handlers harder to test or find. If routes grow further, extracting handlers as free functions (taking `const UserInfo&, const httplib::Request&, httplib::Response&` and a reference to the memory/inference state) would improve navigability without requiring architectural changes.

---

## 4. Naming Clarity

### 4.1 `user_db_path` Constructor Parameter Is Dead Code

**Location:** `include/ragger/memory.h:28`, `src/memory.cpp:20,32`  
**Severity:** High

The `RaggerMemory` constructor signature:
```cpp
explicit RaggerMemory(const std::string& db_path = "",
                      const std::string& model_dir = "",
                      const std::string& user_db_path = "",
                      bool skip_embedding_guard = false);
```

`user_db_path` is accepted as a parameter but **never used**. The constructor comment at line 31 says "Create primary backend - single user DB only now" and only `db_path` is passed to `SqliteBackend`. The test `test_search_merging_dual_db` passes a second path as the third argument (`TEMP_DB2`) and then asserts `mem.count() == 2` — but since `user_db_path` is ignored, the second store goes to `TEMP_DB1` (not `TEMP_DB2`). This test cannot actually be testing what its name implies.

**Suggested fix:** Remove the `user_db_path` parameter from the constructor (it was the old dual-DB path from before the single-user refactor). Update the test to remove the misleading third argument and the comment that says "TEMP_DB1 = common, TEMP_DB2 = user".

---

### 4.2 `store()` Parameter `common` Is Dead Code

**Location:** `include/ragger/memory.h:35`, `src/memory.cpp:88–91`  
**Severity:** High

```cpp
std::string store(const std::string& text, json metadata = {},
                  bool common = false,
                  bool defer_embedding = false);
```

The implementation:
```cpp
std::string RaggerMemory::store(..., bool common, bool defer_embedding) {
    // common flag is now ignored - single-user mode only
    return backend_->store(text, std::move(metadata), defer_embedding);
}
```

The `common` parameter is accepted, documented in the API, and silently ignored. The server's `POST /store` handler reads `body.value("common", false)` and passes it to `mem.store(text, metadata, common)` — it has no effect.

**Suggested fix:** Remove `common` from the `store()` signature (and from the HTTP handler and MCP tool schemas). Leave a short migration note in the commit if any external clients might be passing it.

---

### 4.3 `set_summary_tags` vs `update_summary_text` — Inconsistent Operation Name Pattern

**Location:** `include/ragger/storage_backend.h:116`, `include/ragger/memory.h` (absent)  
**Severity:** Low

The summary update methods are:
- `update_summary_text(summary_id, text, model_name)` — updates text + embedding
- `set_summary_status(summary_id, status)` — sets status column
- `set_summary_tags(summary_id, tags)` — sets tags column

Two use `set_`, one uses `update_`. For SQL single-column writes, `set_` is arguably more accurate, but `update_text` follows the `update_*` pattern used in the main memory API (`update_text`, `update_document_embedding`). The `update_` prefix is also consistent with how the operation is described in comments ("Replace a summary's text + embedding").

Additionally, `set_summary_tags` is present in `StorageBackend` and `SqliteBackend` but is **not forwarded through `RaggerMemory`**. This forces `SummarizerService` to reach through `backend()`. If `set_summary_status` is forwarded (it is, `memory.h:73`), `set_summary_tags` should be too.

---

### 4.4 `ERR_MCP_TEXT_REQUIRED` Used for Two Semantically Different Errors

**Location:** `src/mcp.cpp:184,214`  
**Severity:** Low

```cpp
if (tool_name == "store") {
    if (text.empty())
        return error_result(ragger::lang::ERR_MCP_TEXT_REQUIRED);  // line 184
...
} else if (tool_name == "capture_turn") {
    if (user.empty())
        return error_result(ragger::lang::ERR_MCP_TEXT_REQUIRED);  // line 214
```

The error string `ERR_MCP_TEXT_REQUIRED = "Error: text parameter required"` is correct for `store` (which has a `text` field) but misleading for `capture_turn` (which has a `user` field). An agent seeing "text parameter required" when `capture_turn` fails due to a missing `user` field will be confused.

**Suggested fix:** Add `ERR_MCP_USER_REQUIRED = "Error: user parameter required"` to `lang/en.h` and use it in the `capture_turn` branch.

---

### 4.5 `AllEmbeddings` Struct Is Defined in `storage_types.h` but Never Used

**Location:** `include/ragger/storage_types.h:49–55`  
**Severity:** Low

```cpp
struct AllEmbeddings {
    std::vector<int>                ids;
    std::vector<std::string>        texts;
    std::vector<std::vector<float>> embeddings;
    std::vector<json>               metadata;
    std::vector<std::string>        timestamps;
};
```

Searching the codebase finds no use of `AllEmbeddings` anywhere in source or tests. It appears to be a leftover from an earlier export-all-embeddings API idea.

**Suggested fix:** Remove the struct, or add a comment explaining what it's reserved for.

---

### 4.6 `MSG_SUMMARIZED_SESSION`, `ERR_SESSION_SUMMARY_FAIL`, `MSG_HOUSEKEEPING_EXPIRED`, `WARN_SUMMARIZER_DRAFT` Defined but Unused

**Location:** `include/ragger/lang/en.h:317–318,321,167`  
**Severity:** Low

Several lang strings are defined but never referenced in any source file:
- `MSG_SUMMARIZED_SESSION` — "Summarized session {} ({} turns)"
- `ERR_SESSION_SUMMARY_FAIL` — "Session summarization failed: {}"
- `MSG_HOUSEKEEPING_EXPIRED` — "Housekeeping: {} sessions expired, {} conversations cleaned"
- `WARN_SUMMARIZER_DRAFT` — "Summarizer draft retry failed for summary_id {}: {}"

`WARN_SUMMARIZER_DRAFT` is particularly notable: the draft-retry path in `SummarizerService::worker_loop` catches exceptions and logs with `WARN_SUMMARIZER_L2`, not `WARN_SUMMARIZER_DRAFT`. So the dedicated draft-retry warning string is never emitted.

---

## 5. AI_reference Accuracy

### 5.1 `src_sqlite_backend.cpp.summary.md` — `rebuild_bm25()` Does Not Exist

**Location:** `AI_reference/src_sqlite_backend.cpp.summary.md:65`  
**Severity:** High

The summary states:
> `rebuild_bm25()` issues FTS5 `'rebuild'` on all four FTS tables.

This function does not exist anywhere in `src/sqlite_backend.cpp`, `include/ragger/storage_backend.h`, `include/ragger/sqlite_backend.h`, or `src/memory.cpp`. The `StorageBackend` interface has no `rebuild_bm25` virtual method. Neither does `RaggerMemory`.

This is a reference to a function that was removed or never implemented. The `rebuild-bm25` verb is also mentioned in `AI_reference/src_main.cpp.summary.md` but is absent from `src/main.cpp`. FTS5 indexes are rebuilt implicitly via sync triggers on INSERT/UPDATE/DELETE — there is no manual rebuild entry point.

**Action required:** Remove all references to `rebuild_bm25()` from AI_reference summaries. If a manual `fts5('rebuild')` pass is needed (e.g., after direct SQLite manipulation), add it or document the gap.

---

### 5.2 `src_memory.cpp.summary.md` — Mentions `rebuild_bm25()` and Dual-DB Architecture That No Longer Exists

**Location:** `AI_reference/src_memory.cpp.summary.md:49,28–32`  
**Severity:** High

The summary mentions:
> - **Maintenance:** `rebuild_bm25()` and `rebuild_embeddings()` trigger index refreshes.

`rebuild_bm25()` does not exist (see §5.1).

The summary also describes (section "Turns"):
> `capture_turn(memory, user, assistant, model, session_id)` ... The HTTP handler then reads the row's timestamp back and hands `(turn_id, user, assistant, model, session_id, ts)` to `SummarizerService::enqueue_turn` for L2 summarization.

This is accurate for the HTTP path in `server.cpp`, but the summary implies this happens inside `capture_turn()` itself. The actual `capture_turn()` free function in `memory.cpp:198–210` only calls `store_turn` — the enqueue call happens in the server's `/turn` handler. The summary conflates the two.

**Action required:** Remove `rebuild_bm25()` reference. Clarify that `capture_turn()` only stores the turn; `SummarizerService::enqueue_turn` is called by the server handler.

---

### 5.3 `src_main.cpp.summary.md` — Mentions `rebuild-bm25` Command That Is Absent

**Location:** `AI_reference/src_main.cpp.summary.md:23`  
**Severity:** High

> **`rebuild-bm25` / `rebuild-embeddings`**: refresh FTS indices / regenerate all embeddings

`rebuild-bm25` is not handled in `src/main.cpp`. The only rebuild command present is `rebuild-embeddings` (line 603). The FTS indices are maintained automatically via triggers.

**Action required:** Remove `rebuild-bm25` from this summary.

---

### 5.4 `src_memory.cpp.summary.md` — `user_db_path` and Dual-DB Description Still Present

**Location:** `AI_reference/src_memory.cpp.summary.md:4` (the opening description)  
**Severity:** Medium

The summary opens by describing `RaggerMemory` as orchestrating two components. The code comment in `memory.cpp:31` says "single user DB only now." The `user_db_path` parameter is present but ignored. The AI_reference does not flag this dead parameter.

**Action required:** Note that `user_db_path` is accepted but ignored; the dual-DB architecture was removed.

---

### 5.5 `src_client.cpp.summary.md` — `register_user` Posts to `/register` Which Doesn't Exist in Server

**Location:** `AI_reference/src_client.cpp.summary.md`, `src/client.cpp:184`  
**Severity:** Medium

The summary accurately describes `register_user()`. However, `RaggerClient::register_user()` POSTs to `/register`, which is not registered in `Server::Impl::setup_routes()`. The server has no `/register` endpoint. This client method will always receive a 404.

This is not an AI_reference inaccuracy per se (the summary reflects the code correctly), but it's a consistency bug in the code: the client exposes an API that the server doesn't implement.

**Action required:** Flag this gap clearly in the summary. Longer-term: either add a `/register` endpoint to the server, or remove `register_user()` from the client.

---

### 5.6 `include_ragger_storage_backend.h.summary.md` — Does Not Cover `TurnRecord`, `DraftSummary`, `SummaryRecord` Placement Issue

**Location:** `AI_reference/include_ragger_storage_backend.h.summary.md`  
**Severity:** Low

The summary lists `DraftSummary` and `SummaryRecord` as part of the types section, but doesn't note that they are **nested inside the class** while `TurnRecord` is at namespace scope in the same header. This asymmetry isn't flagged.

**Action required:** Minor update noting the inconsistent placement of these types.

---

## 6. Doc vs Behavior

### 6.1 `docs/http-api.md` — Describes `auth_token` Config Key That Doesn't Exist

**Location:** `docs/http-api.md:35–47`  
**Severity:** High

The HTTP API docs show:
```ini
[server]
auth_token = your-secret-token-here
```

No `auth_token` key exists in `Config`, `load_config()`, or the `DEFAULT_CONFIG` string. The actual authentication mechanism is:
1. A random token stored in `~/.ragger/token` (generated by `ensure_token()`)
2. That token's SHA-256 hash is stored in the `users` table via `bootstrap_auth()`
3. Requests present the raw token in `Authorization: Bearer`
4. `_check_auth` hashes it and looks it up in `get_user_by_token_hash()`

There is no INI key to set a manual auth token. The docs also say "if no `auth_token` is set, the server allows unauthenticated access" — this is also wrong; the server auto-generates a token on first start via `ensure_token()`.

**Suggested fix:** Rewrite the Authentication section of `docs/http-api.md` to describe the actual token bootstrap flow: "On first start, a token is auto-generated and written to `~/.ragger/token`. Use `ragger add-self` to register it, then include it in `Authorization: Bearer <token>`."

---

### 6.2 `docs/configuration.md` — `capture_turns`/`build_context` Default Shown as `true`, Effective Default Is `false`

**Location:** `docs/configuration.md:46–47`  
**Severity:** High

The `[server]` table shows `capture_turns` default as `true` and `build_context` default as `true`. But the DEFAULT_CONFIG string (the file written to disk on first run) sets both to `false`. A fresh-install user reads the docs, sees the default is `true`, but their actual config file has `false`. (See also §1.2.)

---

### 6.3 `docs/configuration.md` — `[embed]` Defaults Disagree with Config Struct and DEFAULT_CONFIG

**Location:** `docs/configuration.md:129–131`, `include/ragger/config.h:58–60`, `src/config.cpp` (DEFAULT_CONFIG)  
**Severity:** Medium

The docs table shows:
- `timeout_ms` default: `10000`
- `max_workers` default: `8`

The `Config` struct defaults:
```cpp
int embed_timeout_ms  = 5000;
int embed_max_workers = 4;
```

The DEFAULT_CONFIG string:
```ini
timeout_ms = 5000
max_workers = 4
```

The docs are wrong on both values (by a factor of 2 each). The struct and DEFAULT_CONFIG agree with each other; the docs are stale.

---

### 6.4 `docs/search-and-rag.md` — `--collection` Import Flag Documented but No-Op

**Location:** `docs/search-and-rag.md:177–185`  
**Severity:** Medium

The search-and-rag docs show several import examples using `--collection docs` or `--collection reference`:
```bash
ragger import notes.md --collection docs
ragger import doc1.md doc2.md doc3.md --collection reference
```

The `--collection` flag is accepted by `main.cpp` (line 233) but the code at line 84 says:
```cpp
// (The lean v2 documents schema has no collection column, so `collection` isn't stored.)
```

The `DocumentChunk` struct has no `collection` field. The flag is silently ignored. A user who imports with `--collection` expecting to filter by it later will be puzzled.

**Suggested fix:** Add a deprecation notice to the docs. Either: (a) note that `--collection` is accepted but unused, advise using `--tags` instead, or (b) remove the examples that pass `--collection`.

---

### 6.5 `docs/search-and-rag.md` — Documents Are Not Included in `search()` But Docs Imply They Are

**Location:** `docs/search-and-rag.md:6–18`  
**Severity:** Medium

The search doc describes:
> 1. Documents are split into paragraph-sized chunks... Each chunk is:
>    1. Embedded into a 384-dimensional vector...
>    2. Indexed for BM25 keyword search via SQLite's built-in FTS5...
>    3. Stored alongside the original text and metadata

This description implies that importing a document and then calling `ragger search <query>` will find content from that document. It will not. The `search()` method in `SqliteBackend::Impl` only queries the `summaries` table. The `documents` table is searched only through the FTS5 index (which exists for `documents_fts`) but `search()` does not include a documents pass. The doc comment in `storage_backend.h:184` explicitly notes: "Documents (L5) are not yet merged into general search."

**Suggested fix:** Add a callout box to the search docs noting that L5 documents (imported files) are not currently included in `search`/`store` results — they're reserved for a future document-search recipe. Direct users to watch for the documents-search feature.

---

### 6.6 `docs/configuration.md` — `query_log` Documented Under `[logging]` but Docs Also Place It Under `[search]`

**Location:** `docs/search-and-rag.md:248`, `docs/configuration.md:136`  
**Severity:** Low

`docs/search-and-rag.md` shows:
```ini
[search]
query_log = true
```

But `query_log` is parsed under `[logging]` in `config.cpp:418`, and `docs/configuration.md` correctly places it in the `[logging]` table. The search-and-rag doc uses the wrong section name.

---

### 6.7 `docs/configuration.md` — Mentions `[server]` Config Search Order Including `/etc/ragger.ini`

**Location:** `include/ragger/config.h:7` (header comment), and implicitly `docs/configuration.md:1–3`  
**Severity:** Low

The header comment for `config.h` says:
```
Search order: --config= → /etc/ragger.ini → ~/.ragger/settings.ini
```

The `find_system_config()` implementation only checks `cli_path` (from `--config=`) and then falls back to bootstrapping `~/.ragger/settings.ini`. There is no `/etc/ragger.ini` check. The header comment is stale from a multi-user design that was removed.

`docs/configuration.md` correctly says "There is no system-level overlay; the file you see is the file the daemon reads." But the header comment contradicts it.

**Suggested fix:** Update the header comment in `config.h` to remove the `/etc/ragger.ini` reference.

---

*End of review. Total findings: 6 High-severity, 12 Medium-severity, 11 Low-severity.*
