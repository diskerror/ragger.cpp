# Changelog

## v0.16.0 *In Progress*
The SQLite FTS5 module isn't working as desired or expected. Going to use my own full-text-search 
algorithm.
This is similar to the original BM25 used in the project but (hopefully) better.

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
