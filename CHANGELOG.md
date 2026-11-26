# Changelog

## v0.17.0.2
- Fix: a database migrated by a pre-release 0.17 dev build keeps the old `episodes` shape (`supersedes`, `created_at` instead of `modified_on`), so the summarizer failed with `no such column: modified_on`. Opening the database now rebuilds the table into the released shape, preserving every row and the id sequence; it refuses rather than drop a non-null `supersedes` value, and aborts if the rebuild would add a foreign-key violation. Fresh and already-correct databases are untouched.
- Fix: `import-conversations --format=telegram` stored assistant replies that did not match what live capture had stored for the same exchange. Three causes: Markdown was lost (the export holds rendered text plus entity spans, now rebuilt into Markdown — code, fenced blocks with language, links, bold, italic, strikethrough, spoiler, blockquote); the gateway's status preamble sits inside the same chat message as the reply, since the message is edited while the turn runs (leading tool-progress lines, provider warnings and truncated continuations are now stripped, detected by glyph shape rather than a fixed emoji list); and the agent's reply to a user message dropped as a command was glued onto the previous turn.
- Fix: imported turns duplicated turns Ragger had already captured. The dedup window was symmetric around the time the user's message was sent, but a captured turn is stamped when the reply finished, minutes later; the window now reaches back 90 seconds and forward an hour. Text matching also allows for mid-turn steering, where the captured prompt is the imported message plus the steering text, with the host's routing envelope spliced between them: a captured turn matches on equality, on starting with the imported message, or on a shared 120-character prefix.
- `import-conversations` now files an imported turn under the real session that was current at that time; a synthetic `telegram-import-NNNN` session is only a fallback. A session's effective start is the earlier of `started_on` and the timestamp in its guid, so a session whose opening turns were never captured still claims the turns an import is filling in. A session that ended more than a day before the turn is not a match.
- `import-conversations --format=telegram` joins consecutive messages from the same sender with a blank line instead of a single newline, matching what capture stores.
- Hermes plugin: the host's scaffolding is removed from a captured prompt — the mid-turn steering label, the gateway routing envelope (`Gateway message origin` and its JSON), and `[OUT-OF-BAND USER MESSAGE]` markers. The user's statements are joined with a blank line.

## v0.17.0.1
- Fix: a database migrated by a pre-release 0.17 dev build is stamped 0.17 but has no `episodes.locked` column, so the summarizer failed with `no such column: e.locked` (critical log, summarizer stopped). Opening the database now adds the column when it is missing (`ALTER TABLE episodes ADD COLUMN locked INTEGER NOT NULL DEFAULT 0`); fresh and already-correct databases are unchanged.
- Fix: `import-conversations --since=20261125` silently matched nothing ("No turns matched."): dates were compared as text against `YYYY-MM-DD`. `--since`, `--until` and `--date` now accept `YYYY-MM-DD`, `YYYYMMDD` and `YYMMDD` (century 20) and reject malformed or impossible dates with an error. `--date=YYYYMMDD` used to be stored as the raw string in the timestamp column.

## v0.17.0
- `memory_search` gains opt-in **split mode** (`split: true`, with optional `min_vector_score` / `min_fts_score`): two separate runs (vector, FTS), each its top `limit` above a per-type minimum, returned as three disjoint lists `groups.both` / `groups.vector_only` / `groups.fts_only` with native scores only (`vector_score` = raw cosine, `fts_score` = raw keyword score, absent = `null`) — no blended `score`, no `matched_by`. The blended default is unchanged; `vector_available: false` when no embedder is ready. The blended number is min-max normalized per pool and cannot separate on-topic from off-topic hits (measured 2026-11-16); raw cosine can (0.38+ vs 0.20-).
- Auto-recall now defaults to **decisions only**: new setting `auto_recall_mode` (`decisions` | `flat`, dashboard Server section, live). Plugins inject active decisions matched to the prompt as `[knowledge:5] <trigger> → <ruling>` (status appended when not `current`); short prompts borrow the previous user messages. `flat` keeps the old behavior; `auto_recall = false` still turns recall off. The Hermes plugin reads the setting from the daemon; the OpenClaw plugin has `autoRecallMode` in its own config.
- New setting `auto_recall_min_similarity` (float, dashboard Server section, live, default `0.30`): the minimum raw-cosine vector similarity for a memory to be injected by auto-recall; `0` injects the top results regardless. The default sits inside the gap measured 2026-11-16 across 10 prompts — best off-topic hit 0.20, weakest on-topic top hit 0.38 — so it drops weaker secondary hits (0.27 seen) without cutting on-topic ones. It is a per-embedder calibration, not a tuned optimum: re-measure after changing the embedding model. Hal wires the Hermes and OpenClaw plugins to read it.
- Hermes and OpenClaw auto-recall (`decisions` mode) now query `memory_search` with `split: true` and `min_vector_score` = the similarity setting (OpenClaw: `autoRecallMinSimilarity`), injecting `both` then `vector_only`; the blended score is no longer used for the gate. Older daemons fall back to ungated results.
- A model reply cut off at the token limit (`finish_reason` `length`, Anthropic `stop_reason` `max_tokens`) is now a failed call, not a stored summary: the row is retried like any other failure. The check is per format (`response_finish_reason` / `finish_truncated` in `formats/*.json`); a format file without them gets the OpenAI shape, and an older installed `anthropic.json` is not checked until `formats/` is refreshed by the installer.
- New settings on the Summarizer page: `summarizer_title_prompt` (session titles, the Name sessions job; no placeholders, the title length is stated in the text, default 3 to 8 words) and `summarizer_trigger_prompt` (the Back-fill triggers job; no placeholders). Empty prompt uses the built-in default.
- `log_level`, `log_max_size_mb` and `log_max_age_days` are now marked restart-required (they always were: the logger is built once at startup; the dashboard said Live).
- Removed the `embed_timeout_ms`, `embed_retries` and `embed_max_workers` settings and the `[embed]` INI section, with `EmbedExecutor` and its test: nothing had created an EmbedExecutor since the import path stopped using it, so the settings had no effect. Old INI `[embed]` keys and saved setting rows are ignored.
- Fix: saving `summarizer_model`, `summarizer_api_url`, `summarizer_api_key`, `summarizer_max_tokens` or an `inference_*` setting on the dashboard (or `PUT /config`) now rebuilds the inference client at once. Before, the daemon kept calling the old model and endpoint until a restart or SIGHUP, although the settings are marked Live.
- New setting `summarizer_episode_prompt` (Episode Prompt): `{N}` is the member count, then two `{}` for target and cap. The Summarizer page lists the prompts in the order turn, episode, session title, knowledge trigger. All four built-in defaults moved to `lang/en.h` (`PROMPT_*`), out of `config.h` and `summarizer.cpp`.
- New built-in prompts. Turn summary: keeps who said what and says a proposal is not a decision unless the user accepted it in that exchange. Session title: 3 to 8 words, title only. Trigger: phrased as the future question, must not state the answer, one example included. Existing summaries are not rewritten; use the Regenerate back-fill on a sample to compare first.
- **FTS naming.** The keyword/text-index signal is now called FTS everywhere it is named: config keys `fts_enabled` / `fts_weight` (the old `bm25_enabled` / `bm25_weight` are still read), `matched_by` value `"fts"` (was `"lexical"`), `SearchResult::fts_score`. The scoring is TF-IDF, not BM25. The `stats.db` column `bm25_score` keeps its name (existing files).
- **Layer numbers retired.** The early L1–L6 labels are gone from the code, docs and messages: they said "turn summary" (L2), "episode" (L3), "documents" (L5) and "knowledge" (L6), and the project layer (L4) no longer exists. The back-fill kinds are now `turn-summary-null`, `turn-summary-empty` and `turn-summary-regenerate` (were `l2-null`, `l2-empty`, `l2-regenerate`); internal names followed (`backfill_turn_summary`, `stale_turn_summary_candidates`, `TurnSummaryRefresh`, ...). Older changelog entries keep their original wording.
- `GET /health` and the MCP `status` tool report `capture_turns` (bool), `summarizer_model` / `summarizer_engine` (strings; empty = not configured) and `summarizer_reachable` (bool: the summarizer endpoint answers `GET /v1/models`; 1 s timeout, cached 15 s). The Hermes and OpenClaw plugins log a warning once when capture is off or no summarizer is configured.
- OpenClaw plugin ported to the 0.17 agent API: `memory_search` (layered), `memory_get` (typed handles), `memory_timeline`, `memory_note`, `memory_decision`, `memory_update`, `document_search`, plus `memory_store` and `memory_status`. The old count tool (`memory_get` returning a number) is gone. Tool definitions are generated from `tools/*.json` by `scripts/gen-openclaw-tools.py` (`--check` in `scripts/test.sh`). **Untested against a live OpenClaw** (stub-server run only); README and `docs/openclaw.md` say so.
- `store` (HTTP, MCP, CLI) rejects `category: "decision"`: a decision needs a trigger, so every decision goes through `memory_decision` / `POST /decision`; `ragger decision add` requires `--trigger`. The Hermes plugin rejects it locally with a pointer to `memory_decision`, and the OpenClaw plugin gains a `memory_decision` tool (HTTP `/decision`, MCP `memory_decision`). `docs/agent-memory-instructions.md` no longer suggests `store` for decisions; re-run `install-bin.sh` and the plugin installers to pick up the new files (an edited instructions copy is kept, the shipped one is written beside it as `.new`).
- removed: `cleanup_max_age_hours` and the retention purge of raw turns. Honest history has no opt-out; turns are never auto-deleted. A `cleanup_max_age_hours` row left in the `settings` table by an older release is an unknown key and is silently dropped. The MCP-mode housekeeping thread (whose only job was the purge) is gone; the housekeeping pass keeps the summarizer catch-up and the embedding backfill
- `semqlite_config` moves into Ragger's database as the shared facts-about-the-file table SemanticSQLite reads. The embedding identity stamps (`embedding_model`, `vector_type`, `dimensions`) now live there under SemanticSQLite's names (`embedding_model`, `embedding_vector_type`, `embedding_dims`); Ragger also publishes `embedding_embedder` (`onnx`, or `http://host:port` for an external engine), `embedding_offset` (0: 0.17 stores bare vectors, the version is a column) and `embedding_api_key_env`. `semqlite memories.db` then runs `EMBED()` with no setup. Ragger's change-management state (`desired_*`, `embedding_version`, `reembedding*`) stays in `settings`
- The legacy `settings` stamp rows move over once, idempotently, whenever the user schema is created (not tied to `db_version`, so an already-stamped 0.17 file is fixed on next open). A stamp already present in `semqlite_config` wins. Nothing is published while embeddings are degraded
- With an external engine the recorded identity model is the endpoint's own model name (`embedding_external_model`), so `semqlite_config.embedding_model` is usable by the http embedder
- API keys are environment variables. `embedding_external_api_key` and `summarizer_api_key` settings are replaced by `embedding_api_key_env` and `summarizer_api_key_env`, which hold only the variable NAME (validated as an identifier, so a pasted key is rejected). The dashboard no longer sends a key, the daemon reads its own. A launchd daemon does not inherit the shell environment: set the variables in the plist. Import scripts read the same variable
- Views: only the date-conversion views remain (9). Dropped the two `knowledge_*_evidence_view` passthroughs and the five `<table>_terms_view` junction views; they are dropped by name from existing files on open. The view-coverage test now checks every remaining view converts a date
- Agent instructions rewritten for the full tool set (quick vs layered tools, which tool for decisions vs notes, `trigger` phrasing, supersede/evidence, curation, per-host tool names, when to write an insight). The Hermes plugin now puts the whole file in the system prompt (static fallback if missing). Every frontend installer (Hermes, OpenClaw, Claude Code, Claude Desktop, install-bin) installs it if absent and, when your copy differs from the shipped one, keeps yours and writes `.new` beside it
- Docs/tests: `semqlite_config` migration test (model-free), API-key environment test

**Schema**
- One `knowledge` table (kind: decision | insight | note) replaces `decisions`, `insights` and note summaries; typed handles are `knowledge:N` — `decision:N` / `insight:N` are rejected with a pointer to the new form
- `episodes` is a real table; membership is recorded on turn summaries (`episode_id`)
- `episodes` has no `supersedes`: episode text is derived and replaced in place, so there is no supersede chain. The column, `idx_episodes_supersedes` and the `episodes_view` column are gone; the unique key on `(session_id, span_start)` now covers every episode (it used to exempt superseded copies)
- `episodes.created_at` is now `modified_on`: the time the episode's text was last written — stamped on insert and on every in-place replace. The old rule `created_at = span_end` is gone (the span already says when the turns happened). The agent API keeps its `created` key for episodes; it now reports that write time. `scripts/rebuild-check.sql` checks `episodes_modified_before_span_end` instead of the old equality check
- `episodes` gains `locked` (INTEGER NOT NULL DEFAULT 0): 1 means the episode can no longer be rebuilt from its source (its turn summaries were culled). A locked episode is never re-summarized and never receives late turn summaries — the catch-up scan skips it, the regeneration handler returns without a write, and a late row covered only by locked episodes stays unattached (until the late-turn rule turns it into a new episode). Nothing sets the flag yet; the cull feature that does comes later. No API field, no dashboard control
- `turns` gains `started_on` (nullable INTEGER, last column): the time the user prompt arrived, sent by the capture hook as `started_at` on `POST /turn` / `capture_turn` (epoch seconds or a `YYYY-MM-DD HH:MM:SS` timestamp; rejected if malformed, more than a day in the future, or after the turn's own time). NULL = unknown, which keeps today's behaviour. Episode boundary timing now measures the gap from the END of one turn to the START of the next (a long reply is no longer idle time); on a same-exchange merge the earlier of stored and new value wins. The idle-close timer is unchanged (it still measures from the newest stored turn). The Hermes and OpenClaw plugins send `started_at` after this release; until then every turn stores NULL
- The `summaries` and `projects` tables are removed; sessions group only, and "project" is a tag (`project:<name>`)
- `sessions` gains `started_on` / `ended_on`, maintained from the session's turns on write
- Migration is 0.16→0.17 only; databases older than 0.16 are refused untouched — upgrade to v0.16.2.3 first

**Summarizer / episodes**
- Every new turn captured via `/turn` is enqueued to the summarizer immediately instead of waiting for the next catch-up tick
- No-daemon rule (accepted behavior, not a bug): an install that runs only `ragger mcp` — without a daemon — stores turns and knowledge but builds no summaries — turn summaries and episodes are created when a daemon with a configured summarizer endpoint next runs its catch-up; writes from the `ragger mcp` process do not invalidate a running daemon's caches, and the MCP process does not signal the daemon
- A turn that merges additional data after it was summarized carries a `stale` counter (merges not yet reflected in its summary text); the periodic catch-up refreshes the summary in place — text, embedding and index — and clears the counter only when no newer merge landed meanwhile
- Failing turns back off (5 min, doubling to 6 h) instead of retrying every tick
- Episodes close when a session has been idle past `episode_idle_minutes`, split on embedding-similarity boundaries (the existing rule), with membership recorded

**API**
- `store` with `metadata.category=decision` goes through one validated path on every transport (HTTP, MCP, CLI); decision-path responses carry `kind` instead of `table`, and `/memory/search` accepts `kinds`
- `chat()` requests carry a whole-request timeout, so a stalled endpoint can no longer freeze the summarizer worker

**Fixes**
- Embedder padding fix: short texts are no longer pooled over `[PAD]` positions — all stored embeddings must be rebuilt (see Upgrade notes)
- Shipped default settings actually apply on fresh installs
- Config is now an immutable snapshot; reloads publish a new snapshot instead of mutating strings under readers
- Turn summaries with NULL text are normalized and re-queued
- Write paths check statement results; failures throw instead of silently dropping writes or reporting wrong counts
- `ragger reindex` runs in one transaction — a mid-run failure leaves the old index intact
- CLI verbs (`import-docs`, `delete-doc`, `reindex`, `config set`, `recipe`) write audit lines to `activity.log`; `config set` signals the daemon to reload
- `import-docs` embeds in-process instead of spawning one model-loading subprocess per chunk

**Logic review 0.17 (capture quality, search, surfaces)**
- Episodes are stamped with their last member's time (`created_at` = span end); turn summaries keep their turn's time
- An interrupted turn merges into the stored turn instead of overwriting it
- Short exchanges are stored as real turn summaries (the source text copied, no model call at or under `summarizer_min_words`, default 20) — the empty `(skipped)` marker rows are gone; a poison-abandoned turn is the only empty row (model `bad`)
- A summary longer than its source is replaced by the source (a tie keeps the source); a copied summary records the original's model
- Episode summaries get their own prompt: the model is told the input is N consecutive summaries of one conversation span and must write standalone text without headers or wrapping quotes
- New back-fill verbs: `ragger backfill <kind>` (`turn-summary-null`, `turn-summary-empty`, `turn-summary-regenerate`, `triggers`, `stranded`, `session-names`) with `--dry-run`; they repair pre-rule rows, draft missing decision triggers, re-attach stranded summaries, and name untitled sessions
- `DELETE /memory/<id>` and `POST /delete_batch` left the HTTP surface (now 404); rows are retired by status (`memory_update`), not deleted
- `memory_note` accepts `trigger` (insights only — a plain note with a trigger is rejected) and `session` provenance
- `ragger decision set-status` goes through the same update path and validation as `memory_update`
- `memory_search` defaults to turn summaries, episodes and knowledge; documents have their own verb `document_search` (the `document` layer is rejected); raw `turn` text and session titles are searched only when named
- Sessions are searchable by title (lexical-only corpus)
- `memory_search` scores each requested corpus against its own candidate pool, so a weak corpus can no longer crowd another one's hits out of the results
- Both search verbs take `mode` (`hybrid` | `semantic` | `keyword`) and a `tags` filter (AND rule; arrays accepted everywhere, not just comma strings)
- Episode hits carry their session and span, and episodes satisfy the session filters
- History paging carries each turn's `turn_summary_id` and `episode_id`; `expand=context` returns neighbouring turns in session order; `expand=sources` walks knowledge → episodes → summaries → turn; `memory_timeline` gained `granularity=knowledge`
- Evidence points at episodes (`episode:<id>` or bare ids) on every transport; turn-summary handles are rejected with a pointer to the right form
- `memory_review` left the agent tool lists (MCP and plugin): there is no CLI `memory_review` verb; curation stays user-driven — the HTTP route `POST /memory/review` remains, and from the CLI the pending queue is reached with `ragger decision list --status pending` + `ragger decision set-status`
- Tool definitions live in `tools/*.json` — one source of truth compiled into the binary; new `GET /tools` serves the same gated array as MCP `tools/list`
- Every maintenance verb is dry-runnable (`--dry-run`), reports progress, is cancellable between batches, and resumable; `ragger reindex all` stays atomic
- The daemon gained a single-flight maintenance job runner (`POST /jobs/<name>`, `POST /jobs/cancel`, `GET /jobs`, progress as SSE `job` events) and the dashboard a Maintenance card; re-embedding stays a CLI operation the user runs

**Upgrade notes**
- Back up the database before upgrading (the migration leg makes its own pre-migration backup too — take one anyway)
- Run `ragger reindex all` — the tokenizer changes land in this cycle; stored term rows are stale
- Run `ragger re-embed all` — embeddings stored before the padding fix are degraded, and some rows' embed text changed
- Import your conversation history (e.g. Telegram) so the back-fill below runs against realistic data
- Repair episode times once through `semqlite`: `UPDATE episodes SET created_at = span_end WHERE created_at <> span_end;`, then `ragger reload-cache`
- Run `ragger backfill <kind> --dry-run` for each kind, then for real
- `ragger housekeeping` (or the daemon's normal pass) closes new episodes with the corrected time
- Run `scripts/rebuild-check.sql` and read the counts
- If your stored `summarizer_prompt` is wrapped in double quotes (older installs), fix it with `ragger config set`

## v0.16.2.6
- Follow c_lib's new library layout: `diskerror_double_metaphone` and `diskerror_stemmer_snowball` were merged into one `diskerror_search` target, so `ragger_core` links that instead. Pinned fallback c_lib bumped to `76eada1` (sources in `src/`, SQLite 3.54.1, merged search library)
- No behavior change; the DB schema stays 0.16

## v0.16.2.5
- Hermes plugin: hooks now actually register. `register()` used `@ctx.hook(...)`, which is not a Hermes API; it raised `AttributeError` on every load (logged as "Memory provider 'ragger' raised after registering (hook)") and left both hooks dead since they were written. They now use `ctx.register_hook(name, callback)`: `post_tool_call` forwards the built-in `memory` tool's add/replace calls to `/store`, and `post_turn_finalized` captures interrupted turns
- `install-hermes.sh` patches Hermes core to fire `post_turn_finalized` from `finalize_turn()` (upstream has no hook that carries message text on an interrupted turn: hermes-agent#108808). Idempotent; re-run after every `hermes update`
- `install-hermes.sh` restores the core patch that passes `model=` to `on_turn_start()`. Upstream never shipped it; the installer dropped it on a false belief that it had, and after a `hermes update` stashed the local edit (2026-09-15) every turn was stored with a NULL `model_id`. Handles both known call shapes
- Dropped the pinned `CMAKE_OSX_DEPLOYMENT_TARGET` of 26.0 (it caused "built for newer macOS" link warnings on newer hosts)

## v0.16.2.4
- Linux builds now prefer clang: when neither `CC`/`CXX` nor `-DCMAKE_CXX_COMPILER` is set, CMake picks `clang++`/`clang` (unversioned, then 21 down to 16 — Debian installs it as e.g. `clang++-19`). This matches c_lib, whose `WAVE.h` relies on a clang-accepted layout that GCC rejects, and matches macOS, where `gcc` is already Apple clang. No effect on macOS
- Fixed ambiguous `Stmt::bind(int, long long)` overload errors on Linux, where `int64_t` is `long` (macOS: `long long`); `util/sqlite.h` gains a `long long` overload enabled only where the two differ
- Verified: full build + all 18 tests pass on Debian 13 (clang 19), both with c_lib fetched from GitHub and with a sibling `../c_lib`
- Fetched dependencies are now pinned and cloned/downloaded at configure time instead of living in (or being silently missing from) the tree: `tokenizers-cpp` is fetched at the commit in `cmake/pins.cmake` (previously a git-ignored local copy, so a fresh clone could not build); c_lib's GitHub fallback is pinned the same way; ONNX Runtime archives are verified against pinned SHA-256 checksums
- `-DDEPS_LATEST=ON` tracks each upstream's newest commit for development and records what it used in `build/deps-resolved.cmake`; `scripts/pin-deps.sh` writes those commits into `cmake/pins.cmake` before tagging; `update_deps` target refreshes on demand
- README/getting-started corrected: tokenizers-cpp was documented as vendored in the repo

## v0.16.2.3
- README: clone `master` for the stable release (`dev` is the default branch)

## v0.16.2.2
- SQLite now comes from c_lib's vendored build (`diskerror_sqlite3`, SQLite 3.54.0) with `SQLITE_DEFAULT_FOREIGN_KEYS=1` — FK rules are enforced on every connection by default; `PRAGMA foreign_keys = OFF` still works for migrations. `-DRAGGER_SYSTEM_SQLITE=ON` links the system library instead
- Startup integrity check: every DB open verifies `PRAGMA foreign_keys` is actually on and refuses to run otherwise (`ERR_FOREIGN_KEYS_OFF`)
- c_lib is linked as prebuilt, optimized libraries from `../c_lib/build` (built by c_lib's `build_libs.sh`, run automatically and incrementally before each build; never cleaned by Ragger). A Ragger clean rebuild no longer recompiles c_lib or SQLite. Falls back to fetching c_lib from GitHub without a local checkout
- Version/commit/date are generated at build time (`cmake/build_info.cmake` → `build_info.cpp`): a new commit or tag recompiles one file instead of needing a reconfigure or full rebuild. Commit shows `-dirty` for uncommitted changes
- System `sqlite3` dev package no longer required to build

## v0.16.2
- SQLite backend reorganized: moved into `src/sqlite/`, `include/sqlite/`, namespaced `ragger::sqlite`; the 4,900-line `Impl` struct split into 8 cohesive files (schema, admin, cache, write, search, maintenance); classes renamed `SqliteBackend` → `sqlite::Backend`, `SqliteTextIndex` → `sqlite::TextIndex`
- Fixed `terms.term_id` AUTOINCREMENT leak; `reset_terms_table()` now drops+recreates `terms` and its junction tables instead of `DELETE FROM`, scoped to `reindex all`
- Tokenizer: stop digits, stray dashes, possessive `'s`, and `0x`/`x`-prefixed hex literals over 4 digits from being indexed as terms
- CLI backups (`re-embed`, `reindex`) now tar.gz the full DB file set (`.db`/`-wal`/`-shm`); pre-migration backup bundles `stats.db` too
- Fixed `reindex`'s live progress counter never activating; backup message now prints before the archiving step
- `rebuild-embeddings` renamed to `re-embed`; added `<table|all>` scope and per-table progress

## v0.16.1
- Refactored main.cpp.
- Altered stop-word lists. Single letters are stopped, Digits converted to words.
- Switched out Porter Stemmer for Snowball Stemmer.

## v0.16.0
- FTS5 retired entirely; replaced with a hand-rolled inverted index (flat `terms` table + five `<t>_terms` junction tables) for full control over tokenization, bigrams, and weighting
- TF-IDF scoring with live document frequency; four tunable weights (`fts_w_unigram`, `fts_w_bigram`, `fts_w_metaphone`, `fts_literal_enabled`); dead `phon_weight` setting removed
- Index engine extracted into `TextIndex`/`SqliteTextIndex`, out of `sqlite_backend.cpp`
- Tokenizer: Porter stemmer added to c_lib; two divergent stopword lists (unigram/bigram); n't contractions map to `not`; digits promoted to words; math symbols normalized to English
- Migration 0.15 → 0.16: text tables rebuilt (not ALTERed) to retire the `phon` column and add count columns; dangling `session_id` refs nulled; `ragger reindex [table|all]` replaces `rebuild_phon`
- Views: unconditional DROP-then-CREATE instead of `IF NOT EXISTS`; new `<t>_terms_view` per junction table; `scripts/refresh_views.sh` added
- Fixed `--host`/`--port` being ignored by `count`/`store`/`search`; fixed 289 assertions silently erased by `-DNDEBUG` across three test files; `CMAKE_OSX_DEPLOYMENT_TARGET` 14.0 → 26.0

## v0.15.1
- Embedding model filter: tightened e5 hint to `e5-`, widened external-model name filter with plausibility check
- StorageBackend gap closed for user/settings + schema introspection
- Plain-field INI keys driven through schema tables
- Server: new `respond_json()` helper collapses 27 duplicated sites
- New `util::HttpClient` RAII wrapper; refactored curl call sites
- New `IEmbedProvider` interface for Embedder and EmbedExecutor

## v0.15.0
- Documents table normalized into `document_sources`; in-process migration, schema update, 
  store/search rewired — the flat `documents` table couldn't express provenance (which file a doc came from) or multiple sources per document
- Settings.ini fully retired — DB is the only config source; onboarding removed. The INI was "half-live": read as base layer at startup with DB overlaid, so it was stale for every key the dashboard had written and still authoritative for keys it hadn't. `bootstrap_user_config()` would recreate a default settings.ini whenever missing, so deleting it just brought back a stale layer next launch
- Onboarding removed because it was the only writer of settings.ini — and wrote *only* there, never to the DB. Any key the dashboard had touched silently discarded its answer on re-run
- `auto_recall` promoted to first-class setting
- MCP: read-only `get_config` and `status` tools added so agents can introspect their own memory server
- db_version 0.15: embedding_version column, offset-based EmbeddingCodec, poison-abandon retry. Version tag moved out of the blob into a dedicated column so old blobs (offset=1) still decode correctly until rebuild-embeddings; new rows use offset=0
- `reset_abandoned_turn_summaries()` re-enqueues poison-abandoned turns once whatever caused failures is fixed — without it, one bad model would permanently strand those turns
- install.sh renamed to install-bin.sh (base binary+daemon installer)
- Import scripts fixed for schema 0.12 writes; read config from DB; take source paths
- Loose-ends cleanup: dead lang constants removed, Config::resolve_model() no-op deleted, whitespace-normalize declaration dropped, DB scan loops fail loud on mid-scan errors (silent partial scans were masking real failures)

## v0.14.4
- Onboarding and file-based config removed — settings table is the only store (see v0.15.0 for full rationale; this was the precursor commit that actually did it, with 0.15.0 following up on cleanup)
- Fixed silent partial re-embeds; degrade instead of dying on a bad model. A single malformed turn shouldn't take down the daemon

## v0.14.3
- Auto-heal legacy bare embedding_model to canonical provider/model form at config-overlay time — users who had `all-MiniLM-L6-v2` in their DB from before the convention was established now get `sentence-transformers/all-MiniLM-L6-v2` without manual intervention
- Embedding-model config: dynamic-enum validation + canonical provider/model form

## v0.14.2
- Embedding engine selection: internal (ONNX) / external (/v1/embeddings). Lets users with a fast GPU inference endpoint skip the ONNX path entirely
- Dashboard: field reorder, model dropdown, served-model mismatch check; README rewrite

## v0.14.1
- Filter null-text marker rows from turn_summaries read paths — these sentinel rows were internal bookkeeping leaking into search results

## v0.14.0
- Embedding storage dtype expanded (bf16 + int8 user-selectable); DB schema unchanged at 0.12
- Graceful degradation on embedding drift; recovery + audit logging added. Drift between configured model and stored vectors used to be a hard failure — now logged and recoverable
- c_lib moved to CMake FetchContent, replacing vendor symlink; include/ragger/ flattened to include/. Symlinked subprojects leak into `cmake --install` in irregular ways (tokenizers-cpp pulling msgpack-cxx); FetchContent keeps the install tree clean
- Logger and ProgramOptions relocated into c_lib — they had no Ragger-specific logic and belonged with the other shared utilities
- Vector quant benchmark: int4/binary simulation + two-stage rerank pass. Findings documented as evaluated-not-shipped: raw recall@10 drops to 0.88 (int4) / 0.63 (binary), too lossy for direct ranking; two-stage rerank recovers them but requires keeping f32 vectors around, defeating the storage win on Ragger's small brute-force corpus
- Home directory collapsed to `~/` in stored document import paths — full paths leaked into search results and broke portability when a user renamed their home dir
- Stale v2/v3/v4 schema reference SQL, episode-bloat cleanup script, offline migration removed

## v0.13.0
- Single-source decision-capture policy loaded at runtime from `docs/agent-memory-instructions.md`. Previously the wording was hand-copied into both plugin installers; changing the policy meant editing two files and re-running installs. Now edit one file, re-run installer (or copy), restart host — no code changes
- Web dashboard added; DB-backed config replaces settings.ini as source of truth. Two faces over one metadata source: CLI (`ragger config get/set`) and HTTP dashboard, both reading schema from `include/ragger/lang/en.h` and values from the DB settings table
- import-docs: strip leading `[ #]` and trailing spaces from stored document text — markdown artifacts were polluting search results

## v0.12.1
- Episode close uses sliding avg-similarity threshold to split sessions into multiple episodes. Replaces single-episode-per-close with per-turn boundary detection: cosine similarity of consecutive turn embeddings averaged with turn-summary embedding similarity, compared against a threshold that rises with idle time between turns. A long session about two unrelated topics now produces two episodes instead of one muddy summary
- Datetime added to turn_summary search metadata

## v0.12.0
- Summary status field dropped; `turn_summaries` table split out from the general summaries table — different lifecycle, different query patterns
- Unix-epoch timestamps throughout (replaces mixed string/integer representations)
- Session/project summaries: running-upsert replaced with boundary-detection. Redesign per Reid's clarified usage model: sessions and projects don't need a running summary rewritten in place — they need one immutable record per closed "run" of consecutive activity. A session run closes the instant a turn with a different session_id arrives; a project run closes on a time gap (default 7 days). Boundary detection is housekeeping-driven only, with a persisted watermark so failed/skipped closes are retried

## v0.11.1
- Poison turns abandoned after `max_turn_failures` (default 3). Summarizer now tracks consecutive inference failures per turn; after the threshold, stamps the placeholder with model 'bad' so it leaves the unsummarized queue permanently. Prevents one bad turn from blocking the entire pipeline — the root cause of a two-week backfill stall

## v0.11.0
- Schema v4 migration: column reorder to canonical `(id, keys, data, timestamps, embedding, phon)` layout, lookup timestamps added, version stamp in settings table. The old schema had inconsistent column ordering across tables; v4 makes it uniform for easier codegen and maintenance

## v0.10.2
- Built-in log rotation — daemon used to grow unbounded on activity.log
- Configurable catch-up batch size; newest-first summarization (oldest-first was starving recent turns)
- UTF-8 sanitization at tokenizer FFI boundary — malformed sequences crashed the tokenizer
- Portable deploy script + path-portability fix

## v0.10.1
- Telegram fuzzy dedup widened to 90s; inline tool-call trace text stripped from Claude import (tool traces were polluting summaries)
- Decisions: roadmap status, decision CLI verb, dedup/prefix fixes
- `import-conversations`: flat Markdown memory-log import + dedup fix

## v0.10.0
- Chat feature dropped entirely per project decision — Ragger is a memory server only; conversation lives in the agent. Removed ~90 lines of dead lang/en.h strings with zero live call sites
- Native TLS enforced via httplib::SSLServer (was just parsed and logged as unsupported). Every failure path falls back to working plain HTTP rather than refusing to start — a reachable daemon beats one that won't come up
- All paths consolidated under `--ragger-base`; settings.ini simplified
- timestamp→created_at rename; summaries columns reordered, updated_at NOT NULL
- Episode layer added (level='episode', boundary-triggered). Phase 1 was additive alongside the existing running-L3 path so behavior could be validated before removing per-turn churn in Phase 2
- Phase 2: session/project become boundary-triggered running rollups — removes the per-turn upsert overhead that made L3 generation O(turns) instead of O(closed runs)
- Conversation importer for Telegram/Claude — replaces a Python script with native C++ so it shares the same schema/codec paths as live capture
- Dual build dirs per RAGGER_STATS flag (`build/` off, `build-stats/` on); default OFF. Stats instrumentation adds measurable overhead; keeping two separate builds means release users pay nothing
- `install --stats` selects build-stats/ binary; drop build-stats.sh wrapper

## v0.9.12
- Phonetic "dolphining" sounds-like search (Double Metaphone) — catches typos and alternate spellings that exact-match misses
- RAGGER_STATS=ON default in build.sh; test both flag states

## v0.9.11
- Capture interrupted turns via `post_turn_finalized` hook — previously only completed turns were captured, losing context on user aborts
- Opt-in RAGGER_STATS retrieval instrumentation (StatsLogger unit-tested)

## v0.9.10
- L2 summarization deferred to housekeeping tick — running it inline at capture made store_turn slow; deferring keeps the write path fast while still producing summaries shortly after
- Housekeeping uses main memory backend for cleanup, not a separate SqliteBackend — two open handles on the same DB caused lock contention

## v0.9.9
- L6 decision write path + unified context-table embeddings

## v0.9.8
- Summarizer: strip_thinking, quality knobs, junk filters, import pipeline
- Search perf: normalize cache rows once at build, GEMV at query time — per-query normalization was a hot path
- Embedding blobs decoded by size, not the store_f16_ flag — the flag could drift from actual blob contents after manual DB edits; decoding by size is self-correcting
- `min_score` applied before top-k selection (was after, so low-scoring results displaced high-scoring ones in some cases)
- Daemon graceful SIGTERM/SIGINT — hard exit was losing in-flight summaries
- Raw placeholder L2 written at capture; NULL model_id = unsummarized sentinel. Makes "not yet summarized" queryable without a separate status column

## v0.9.7
- Layered running summaries: L3 from L2s, L4 from L3s — building session summaries directly from raw turns was expensive and lossy; the hierarchy compresses incrementally
- Summarizer size limits switched to percentages (target_pct / max_pct) — absolute char limits broke on long sessions
- `[inference.memory]` renamed to `[summarizer]`; prompt moved into settings.ini. The old name implied a separate inference endpoint for memory, which wasn't the case; making the prompt user-editable lets users tune summary style without recompiling
- Capture strips system-injected note from user text at capture time — notes like "[This response was interrupted...]" were polluting stored turns
- `-n/--num` added to search; default 3 results

## v0.9.6
- **Fading-memory schema (v2)** — separate `turns`, `summaries`, `documents`, `decisions` tables, each with FTS5 index; BM25 path removed. The single `memories` table couldn't distinguish between different content types, and BM25 didn't handle CJK or mixed-language text well
- Daemon-resident summarizer worker for L2/L3 generation — moving off the request thread means a slow inference call doesn't block other requests
- Recipe-based context assembly (`build_context`) with five built-in recipes. Different tasks need different context shapes: `natural_fading` for general conversation, `deep_recall` for exhaustive search, `tldr` for quick status, etc. Recipes are user-extensible via `~/.ragger/recipes/`
- New `ragger onboard` verb — guided first-run setup (later removed in v0.14.4 when DB became the only config store)
- Cross-platform build: Apple/Linux conditional, ONNX auto-fetch, install.sh fetches model
- Conversation importer ported to C++ (Telegram/Claude); L4 summaries import added
- f16 embedding storage halves blob size; drift guards on dtype/model/dimensions with empty-DB re-adopt path. Guards refuse to mix dtypes/models in the same table, which would silently corrupt search quality
- `rebuild-embeddings` covers all four embedded tables (previously just summaries)
- Summaries never longer than source; trivial turns skip inference — post-check rejects bloat from non-compliant models
- Reasoning-content fallback in api_formats for thinking models that return the visible answer in `choices[0].message.reasoning_content` instead of `.content`
- RAII `Stmt` wrapper for SQLite prepared statements — eliminates ~40 manual prepare/step/finalize sequences
- HTTP routes wrapped in single `guarded()` adapter for auth + error handling — was duplicated across every route handler
- Daemon/service control extracted into daemon_control.cpp — the main loop was 800+ lines; this isolated lifecycle management
- RaggerClient backed by libcurl instead of raw sockets — raw socket code didn't handle HTTP/1.1 keep-alive or redirects correctly
- **Removed**: chat REPL, web interface, LM proxy mode (replaced by `capture_turn`). The agent owns the conversation; Ragger owns the memory
