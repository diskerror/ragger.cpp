#pragma once
// Private implementation header for Backend::Impl.
// Declarations only -- method bodies live in schema.cpp, admin.cpp,
// cache.cpp, write.cpp, search.cpp, maintenance.cpp (see backend.cpp
// for the free helpers and public Backend:: wrapper methods).
//
// Purely-internal schema-migration helpers (pre-migration backup, reopen,
// PRAGMA reads, table rebuild, FK guard) are NOT declared here at all:
// they are file-local free functions taking Impl& in schema.cpp, so the
// module boundary is enforced by the compiler rather than convention.
#include "sqlite/backend.h"
#include "sqlite/status.h"
#include "sqlite/text_index.h"
#include "vector_codec.h"
#include "nlohmann_json.hpp"
#include <sqlite3.h>
#include <Eigen/Dense>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ragger::sqlite {

using json = nlohmann::json;

struct Backend::Impl {
    sqlite3*    db       = nullptr;
    // Custom v0.16 inverted-index engine (schema + reindex + TF-IDF scoring).
    // Holds a non-owning copy of `db`; constructed once `db` is open.
    std::optional<TextIndex> text_index_;
    Embedder*   embedder = nullptr;    // nullable — null for DB-only (user mgmt) mode
    bool        readonly_ = false;     // true for export-path readonly connections
    std::string db_path;
    // On-disk vector dtype (from config vector_type). All in-memory math is
    // f32; this only governs how embeddings are packed into the stored BLOB.
    vector_codec::VectorType vtype_ = vector_codec::VectorType::F16;
    // Current embedding version — read from settings at startup, written into
    // byte 0 of every new blob. Stale blobs (different version byte) are
    // re-embedded on cache load or by rebuild_embeddings().
    uint8_t embedding_version_ = 0;

    // Bind a float vector as the embedding blob (payload only, no version
    // byte — see bind_embedding_version) in the configured storage dtype.
    void bind_embedding(sqlite3_stmt* s, int idx,
                        const std::vector<float>& emb) const {
        // An empty vector means the embedder could not produce one — the
        // configured model is missing or unloadable, so Embedder is in its
        // disabled state and encode() returns {}. Store NULL rather than
        // encoding an empty vector. NULL is what the housekeeping backfill
        // looks for, so these rows are re-embedded automatically once the
        // configuration is fixed. This is the single choke point for every
        // write path, so no individual store() needs to know the embedder
        // is unavailable.
        if (emb.empty()) {
            sqlite3_bind_null(s, idx);
            return;
        }
        std::vector<uint8_t> blob = vector_codec::encode(vtype_, emb, /*offset=*/0);
        sqlite3_bind_blob(s, idx, blob.data(),
                          static_cast<int>(blob.size()),
                          SQLITE_TRANSIENT);
    }

    // Bind the embedding_version column that accompanies an embedding blob.
    // NULL iff the embedding itself is NULL (mirrors bind_embedding's own
    // empty-vector -> NULL rule) — the two columns are always NULL/non-NULL
    // together. Every INSERT/UPDATE that writes `embedding` must also write
    // `embedding_version` via this at the paired column index.
    void bind_embedding_version(sqlite3_stmt* s, int idx,
                                const std::vector<float>& emb) const {
        if (emb.empty()) {
            sqlite3_bind_null(s, idx);
            return;
        }
        sqlite3_bind_int(s, idx, static_cast<int>(embedding_version_));
    }

    // Embedding caches (review 2C): one struct, five instances — the five
    // tables (summaries, documents, decisions, insights, turn_summaries) each
    // hold the same six members. Vector scores come from the caches; keyword
    // scores come from the custom text index at query time, blended in
    // search(). Each cache is invalidated on writes to its table.
    struct EmbeddingCache {
        bool                        valid = false;
        std::vector<int>            ids;
        std::vector<std::string>    texts;
        Eigen::MatrixXf             embeddings;   // rows × dims, L2-normalized
        std::vector<json>           metadata;
        std::vector<std::string>    timestamps;
    };

    EmbeddingCache summary_cache_;   // summaries (L2/L3)
    EmbeddingCache doc_cache_;       // documents (L5)
    EmbeddingCache dec_cache_;       // decisions (L6)
    EmbeddingCache ins_cache_;       // insights (db 0.17)
    EmbeddingCache turn_cache_;      // turn_summaries (L2)
    EmbeddingCache episode_cache_;   // episodes (db 0.17, PHASE5 B.3)

    // Serializes all public-API access (H1): two httplib thread pools, the
    // housekeeping timer, and the SummarizerService worker share one backend.
    // The non-thread-safe Embedder and the three embedding caches are guarded
    // by this. Locked once at the Backend:: public boundary; Impl methods
    // never re-lock, so no recursive deadlock is possible.
    mutable std::mutex mu;

    // --- schema ---

    Impl(Embedder& emb, const std::string& path);

    /// DB-only constructor — no embedder.
    /// readonly=true: opens SQLITE_OPEN_READONLY, skips schema creation (for export).
    /// readonly=false: opens read-write and creates users/settings tables.
    explicit Impl(const std::string& path, bool readonly);

    ~Impl();

    // ---- helpers -------------------------------------------------------
    void exec(const char* sql);
    void exec(const std::string& sql);

    /// True if `table` has a column named `col`. Used to guard one-time
    /// ADD COLUMN migrations (SQLite has no ADD COLUMN IF NOT EXISTS).
    /// The table name is inlined (PRAGMA table_info doesn't take a bound
    /// parameter reliably); callers pass internal constants, never user input.
    bool column_exists(const std::string& table, const std::string& col);

    bool table_exists(const std::string& table);

    void create_schema();


    // ---- schema version --------------------------------------------------
    // The DB's schema version lives in settings['db_version'] as a string
    // (e.g. "0.12"). Absent means pre-versioning (legacy v3/v4 DB from
    // before this key existed) -- returns "" in that case. Startup hard-gates
    // on this via kExpectedDbVersion below; there is no in-binary migration.
    static constexpr std::string_view kExpectedDbVersion = "0.17";

    std::string db_version();

    void set_db_version(const std::string& v);

    // ---- in-binary version migrations ------------------------------------
    // Historically, DB schema upgrades were external shell scripts
    // (scripts/migrate_to_db*.sh) and the binary merely hard-gated on the
    // version. To make GitHub-cloned upgrades painless, the known upgrade
    // legs now run in-process on open, chained until the DB reaches
    // kExpectedDbVersion. Each leg mirrors migrate_documents_normalize()'s
    // shape: one BEGIN/try/COMMIT/ROLLBACK transaction, Logger progress,
    // and post-migration verification that aborts (rolls back) on any
    // inconsistency rather than silently shipping a corrupt index.
    //
    // The legs below are called ONLY from within schema.cpp (constructors,
    // create_schema(), run_pending_migrations()); their supporting helpers
    // (backup/reopen/pragma/table-rebuild) are file-local there.

    /// Chain the in-binary migration legs until the DB reaches
    /// kExpectedDbVersion (or a leg we don't know about is hit -> leave it for
    /// the hard gate to reject). `from_version` is the on-disk version captured
    /// before create_schema() mutated anything. Each leg advances db_version;
    /// the loop re-reads it so 0.12 flows 0.12 -> 0.15 -> (0.16 once its leg
    /// lands) under the single backup taken above.
    void run_pending_migrations(const std::string& from_version);

    /// 0.12 -> 0.15 migration (in-binary port of scripts/migrate_to_db0.15.sh).
    ///
    /// By the time this runs, create_schema() has ALREADY performed the
    /// structural half of the 0.15 upgrade on this existing DB:
    ///   - added embedding_version INTEGER to all five embedded tables
    ///     (ALTER ADD COLUMN guards, above), and
    ///   - normalized documents into document_sources via
    ///     migrate_documents_normalize(), which also CLEARS documents'
    ///     embedding/embedding_version/phon (they get re-embedded with the
    ///     title appended).
    ///
    /// The remaining data step is the embedding version-byte SPLIT for the
    /// four tables migrate_documents_normalize did NOT wipe: turns,
    /// turn_summaries, summaries, decisions. Pre-0.15 blobs stored a 1-byte
    /// version tag at byte 0 (EmbeddingCodec offset=1); 0.15 blobs are payload
    /// only (offset=0) with the tag living in the embedding_version column. So
    /// for every row with a non-NULL embedding: embedding_version = blob[0],
    /// embedding = blob[1:]. NULL embeddings stay NULL in both columns.
    ///
    /// Whole thing runs in one transaction. On any verification failure the
    /// transaction is rolled back (original rows untouched) and the throw
    /// propagates -- the pre-migration backup made above is the safety net.
    void migrate_0_12_to_0_15();


    /// One-time in-place migration to the normalized documents schema (0.15).
    /// Extracts per-document metadata (path/title/year/imported_at) from the
    /// old flat `documents` rows into `document_sources`, converts imported_at
    /// text -> unix epoch (stored text is a LOCAL wall-clock timestamp), points
    /// each chunk at its source via document_source_id, redistributes tags
    /// (source = intersection of a (path,title) group's chunk tags; chunk keeps
    /// its own set minus that intersection), clears embedding/embedding_version/
    /// phon on every doc row (so housekeeping re-embeds with title appended),
    /// then drops the four extracted columns. Whole thing runs in one
    /// transaction. Caller gates on column_exists("documents","path").
    void migrate_documents_normalize();

    /// 0.15 -> 0.16 migration (FTS5 teardown + custom index backfill).
    ///
    /// This is the big cutover:
    /// 1. Drop all FTS5 triggers (both text and phonetic)
    /// 2. Drop both FTS5 virtual tables
    /// 3. Drop the `phon` column from all 5 text tables
    /// 4. Call reindex_table() for each of the 5 tables to populate the custom index
    /// 5. Update db_version to 0.16
    ///
    /// The schema (terms, *_terms, count columns) is created by create_schema()
    /// BEFORE this runs, and reindex_table() (Step 6) populates them.
    void migrate_0_15_to_0_16();

    /// 0.16 -> 0.17 migration: the whole agent-API schema in one step.
    /// Rebuilds `decisions` (adds trigger, rationale, scope, supersedes,
    /// session_id) and `summaries` in the canonical column order, rebuilds
    /// `sessions` (drops project_id, adds started_on/ended_on span columns
    /// computed from the session's turns), creates insights and
    /// insight_evidence, drops the projects table and its indexes, and moves
    /// stray category words that leaked into decisions.status ('fact',
    /// 'lesson') into tags, setting status 'current'. decision_evidence and
    /// the orphan-protection trigger are created after migration in
    /// create_schema().
    void migrate_0_16_to_0_17();

    /// Custom terms index (v0.16) — replaces both FTS5 layers with a single
    /// flat term table + per-content-type junction tables. See
    /// ownCloud/Ragger/custom-search-schema.md for the design rationale and
    /// docs/plans/migration-plan-fts-index.md for the step-by-step build plan.
    ///
    /// The `terms` table holds literal (stemmed) AND metaphone tokens in ONE
    /// flat list with no type/flag column — they self-filter at query time
    /// because a stemmed word and its DMP code are different strings that both
    /// resolve to the same term_id. No doc_frequency column: df is computed
    /// live via COUNT(*) on the per-table *_terms join (Decision C).
    void create_terms_schema();

    // --- admin ---

    /// Users + settings tables — declarative, no in-place migration
    /// (single-user app; pre-v2 data is exported out-of-band). `users` mirrors
    /// the reference DDL plus `password_hash` for credentialed access.
    /// Shared by both constructors.
    void create_user_schema();

    // ---- User / settings CRUD (gap closure) --------------------------------

    std::optional<UserInfo> get_user_by_username(const std::string& username);

    std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash);

    std::optional<std::string> get_user_password(const std::string& username);

    void set_user_password(const std::string& username, const std::string& pw_hash);

    int create_user(const std::string& username, const std::string& token_hash);

    bool delete_user(const std::string& username);

    void update_user_token(const std::string& username, const std::string& new_hash);

    std::optional<std::string> get_setting(const std::string& key);

    void set_setting(const std::string& key, const std::string& value);

    // ---- Schema introspection (gap closure) --------------------------------

    std::vector<SchemaObject> list_schema_objects();

    std::vector<std::string> table_column_names(const std::string& table);

    int iterate_table_rows(const std::string& table,
                           const std::function<void(const ExportRow&)>& cb);

    /// Convenience views: epoch ints rendered as local datetime, embedding
    /// BLOBs collapsed to a has_embedding flag, and term_ids resolved to the
    /// readable term. These exist purely for human inspection (sqlite3 CLI,
    /// GUI browsers) -- no application code reads them.
    ///
    /// DROP-then-CREATE, deliberately NOT "CREATE VIEW IF NOT EXISTS". A view
    /// holds no data, so recreating is free -- and IF NOT EXISTS meant an
    /// existing DB kept a stale view definition FOREVER after this DDL changed
    /// (exactly how the v0.16 views were left still selecting the retired
    /// `phon` column). Unconditional recreation keeps every DB's views in
    /// lockstep with this code on every open.
    void create_views();

    // --- cache ---

    // Decode an embedding BLOB into a float vector of `dims`. Uses decode_any
    // which handles the current version-tagged format AND legacy formats (RV1
    // headers, raw f16/f32). On any failure (NULL, deferred-but-unbackfilled
    // row, corruption, or a dimension mismatch) a zero vector is returned;
    // the first such row per cache load is logged once (table + id).
    std::vector<float> decode_embedding_blob(const void* blob, int blob_bytes,
                                             int dims, const char* table,
                                             int row_id, bool& warned);

    // ---- epoch timestamp for INTEGER columns --------------------------
    // resolve_epoch(caller_ts_string): callers (importers) sometimes supply
    // a specific historical timestamp as a "%F %T" string rather than "now".
    // Parse it via the shared parse_db_timestamp() and convert to epoch
    // seconds; empty input -> current time. A non-empty string that fails
    // to parse is a caller error (malformed import timestamp) -- surfaced
    // as a thrown exception rather than silently binding garbage, matching
    // how other malformed-input cases in this file are handled.
    static int64_t resolve_epoch(const std::string& caller_ts);

    // ---- cache --------------------------------------------------------
    void invalidate_cache();
    void invalidate_doc_cache();
    void invalidate_dec_cache();
    void invalidate_ins_cache();
    void invalidate_turn_cache();
    void invalidate_episode_cache();
    /// Drop all six warm caches (summaries, documents, decisions, insights,
    /// turns). Use after any write that can change rows across several tables
    /// (embedding backfill/rebuild) so no table's stale rows survive.
    void invalidate_all_caches();

    // Shared body of all five ensure_*_cache() loaders (PHASE3 Step 5):
    // run `sql` (columns: id, text, embedding blob, ..., timestamp at
    // index 5), decode blobs under `label`, and let fill_meta build each
    // row's metadata JSON. Column layout stays per-table; only the loop
    // shape is shared. SQL text must match the pre-refactor loaders.
    void load_cache(EmbeddingCache& c, const char* label,
                    const std::string& sql,
                    const std::function<void(Stmt&, nlohmann::json&)>& fill_meta);

    // Loads the summaries table into the vector cache. Keyword scores come
    // from FTS5 (summaries_fts) at query time — see keyword_scores().
    void ensure_cache();

    // Loads the documents (L5) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (documents_fts) at query time. Metadata carries source="document" plus
    // the title so search() consumers can distinguish documents from summaries.
    void ensure_doc_cache();

    // Loads the decisions (L6) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (decisions_fts) at query time. Metadata carries source="decision" plus
    // the status so search() consumers can distinguish decisions from the
    // other corpora.
    void ensure_dec_cache();

    // Loads the insights table into the parallel vector cache. Mirrors
    // ensure_dec_cache() (same status allow-list); metadata carries
    // source="insight" plus the status/trigger.
    void ensure_ins_cache();

    // Loads the turn_summaries (L2) table into the parallel vector cache.
    // Mirrors ensure_dec_cache(): vector scores come from here, keyword
    // scores from FTS5 (turn_summaries_fts) at query time. WHERE text IS NOT
    // NULL excludes unfinalized placeholder rows (summarizer hasn't run yet)
    // — these must never surface in search results or feed the embedding
    // similarity matrix. Metadata carries source="turn_summary" plus
    // turn_id/session_id (turn_summaries has no tags/status/level columns).
    void ensure_turn_cache();
    /// Episode corpus (db 0.17, PHASE5 B.3): loaded from `episodes` through
    /// the shared load_cache(); marker rows (text='') never load.
    void ensure_episode_cache();

    // --- agent API Phase 2 reads (src/sqlite/agent_read.cpp) ---

    // Layered hybrid search over all caches with post-filters; see
    // StorageBackend::agent_search for the contract.
    std::vector<AgentHit> agent_search(const AgentSearchQuery& q);

    // Full-row reads / hierarchy expansion / history paging; see
    // StorageBackend::agent_get for the contract.
    json agent_get(const AgentGetQuery& q);

    // --- agent API Phase 4 timeline/curation/review (src/sqlite/agent_read.cpp) ---

    // Deterministic newest-first range scan, key-paginated; see
    // StorageBackend::agent_timeline for the contract.
    json agent_timeline(const TimelineQuery& q);

    // Metadata-only curation with the Phase 4 transition table; see
    // StorageBackend::agent_update for the contract.
    json agent_update(const UpdateRequest& req);

    // Pending-decision queue; reuses agent_update's transition code.
    json agent_review(const std::string& action,
                      const std::vector<std::string>& ids);

    // --- write ---

    // ---- path normalization -------------------------------------------
    static std::string normalize_path(const std::string& text);

    // Strip a leading "Decision #NNN", "Design Decision #NNN", or
    // "Decision Log #NNN" label (optionally with a "(context)" aside
    // and/or the word "Logged") from decision text, e.g.
    // "Decision #122 (Vienna, June 2026): Ragger adds ..." ->
    // "Ragger adds ...". The row's own decision_id + created_at already
    // carry this information — a hand/agent-authored numeric label baked
    // into the text is pure duplication that dilutes embedding/FTS search
    // (and drifts out of sync with the real decision_id over time).
    // Markdown emphasis markers ("**"/"*") wrapping the label are also
    // consumed. No-op if the text doesn't start with this pattern.
    static std::string strip_decision_number_prefix(const std::string& text);

    // ---- local timestamp ("%F %T" == "YYYY-MM-DD HH:MM:SS") ----------
    // Thin aliases over the shared formatter (ragger/util/time.h) so every
    // DB timestamp uses one local-time format.
    static std::string local_timestamp(std::time_t tt);
    static std::string local_timestamp();

    // ---- public API ---------------------------------------------------

    // v2: the generic store API writes a summary (L2/L3/L4). A memory-only
    // install records agent memories here. `level`/`status` come from
    // metadata when supplied; defaults suit a settled session-level note.
    // collection/category and the free-form metadata blob are gone in v2 —
    // metadata fields are either promoted to columns or dropped. FTS5 sync
    // triggers index the row, so there is no explicit BM25 step.
    std::string store(const std::string& raw_text, json metadata, bool defer_embedding);

    // ---- store_document: write Level 5 RAG chunk ----------------------

    // Resolve a (path, title) document to its document_sources row, inserting
    // one on first sighting. Returns {document_source_id, imported_at_epoch}.
    // imported_at_text is the caller's timestamp (TEXT 'YYYY-MM-DD HH:MM:SS'
    // or 'YYYYMMDD', interpreted LOCAL); empty -> now. On an existing source
    // the stored epoch is returned (the first import's time wins), so every
    // chunk of a document shares one imported_at.
    std::pair<long long, long long> get_or_create_document_source(
            const std::string& path, const std::string& title,
            int year, const std::string& tags,
            const std::string& imported_at_text);

    int store_document(const DocumentChunk& chunk, bool defer_embedding);

    // ---- turns (L1) raw exchange capture ------------------------------
    // Resolve a model name to its models.model_id, creating the row if it
    // doesn't exist. Empty name → 0 (callers bind NULL).
    int get_or_create_model(const std::string& name);

    /// Resolve a session GUID to its compact integer id, inserting on first
    /// sighting. Empty guid → 0 (NULL session_id), mirroring get_or_create_model.
    /// If session_name is non-empty, sets sessions.name (and name_source) on
    /// first sighting or updates it if the title has changed since last capture.
    int get_or_create_session(const std::string& guid,
                              const std::string& session_name = "",
                              const std::string& name_source = "");

    // Embedding for a turn is over the joined exchange (user + assistant),
    // using the same U+001F unit-separator as chat's summary turns.
    static std::string turn_embed_text(const std::string& u, const std::string& a);

    // Store a raw L1 turn. An empty assistant_text writes a *partial* row
    // (assistant + embedding NULL) for the prompt-arrival/finalize flow;
    // otherwise the exchange is embedded unless defer_embedding. FTS5 sync
    // triggers index user_text/assistant_text. Returns turn_id.
    // `source_timestamp` (non-empty, db format) overrides created_at for
    // historical imports; the regeneration-dedup window is skipped in that
    // case (it reasons about "now", which is meaningless for old turns).
    // Widen a session's derived span (PHASE4): started_on = MIN, ended_on =
    // MAX over stored turn epochs. NULL bounds start at the first value.
    // Bounds are never recomputed smaller when turns are later culled —
    // cheap and conservative: a stale-wide span is harmless, a wrong-narrow
    // one would hide the session from range queries.
    void touch_session_span(int session_id, int64_t epoch);

    int store_turn(const std::string& user_text, const std::string& assistant_text,
                   const std::string& model_name, bool defer_embedding,
                   const std::string& session_guid,
                   const std::string& source_timestamp = "",
                   const std::string& session_name = "",
                   const std::string& name_source = "");

    // Shared implementation for turn queries over a session GUID.
    // asc=true → oldest-first (ORDER BY turn_id ASC, no limit).
    // asc=false → newest-first (ORDER BY timestamp DESC, turn_id DESC) with optional limit.
    std::vector<TurnRecord> turns_by_session_impl(
            const std::string& session_guid, bool asc, int limit = 0);

    /// All turns belonging to a session GUID, oldest first.
    std::vector<TurnRecord> turns_by_session(const std::string& session_guid);

    // Finalize a partial turn: set assistant_text, (re)embed the exchange,
    // and record the model. Returns false if the turn doesn't exist.
    bool finalize_turn(int turn_id, const std::string& assistant_text,
                       const std::string& model_name);

    // ---- summaries (L2/L3) pipeline primitives (issue #22) ------------
    // Insert a summary row. level: 'turn' (L2) | 'session' (L3) | 'project'.
    // Embeds text, records model.
    // source_timestamp (non-empty) overrides the row's timestamp: L2 turn
    // summaries inherit the source turn's timestamp so the (session_id,
    // timestamp) pair links a turn to its summary (no FK column needed) and
    // stays stable across embedding-model changes.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& source_timestamp,
                      const std::string& tags);

    // ---- store_decision: write Level 6 curated decision/lesson ----------
    // Mirrors store_summary but the decisions table has no level/model/session
    // columns — just text/embedding/status/tags/timestamp. defer_embedding
    // leaves embedding NULL for a later backfill pass.
    int store_decision(const std::string& text, const std::string& status,
                       const std::string& tags,
                       const std::string& source_timestamp,
                       bool defer_embedding);

    // ---- store_structured_decision: agent API Phase 1 (db 0.17) ---------
    // Validates, embeds trigger+text, then in ONE transaction inserts the
    // row, flips the superseded row (if any) to 'superseded', and links the
    // evidence turn_summaries. Throws std::invalid_argument on bad input.
    StructuredDecisionResult store_structured_decision(const DecisionInput& in);

    // ---- store_insight: agent API Phase 5 (db 0.17) ----------------------
    // decisions and insights share ONE write path (store_structured_row);
    // only the table names, the error-message noun, and which field is
    // required differ:
    //   decisions: trigger REQUIRED, evidence optional.
    //   insights:  trigger optional, evidence REQUIRED — an insight with no
    //              sources is just a note (Phase 5 step 3).
    // Reuses DecisionInput / StructuredDecisionResult verbatim (P5 doc:
    // "Reuse the struct; it already has every field").
    struct StructuredRowSpec {
        const char* table;           // "decisions" | "insights"
        const char* pk;              // "decision_id" | "insight_id"
        const char* evidence_table;  // "decision_evidence" | "insight_evidence"
        const char* noun;            // word used in validation errors
        bool trigger_required;
        bool evidence_required;
        bool decisions_cache;        // invalidate_dec_cache() vs _ins_cache()
    };
    StructuredDecisionResult store_structured_row(const DecisionInput& in,
                                                  const StructuredRowSpec& spec);
    StructuredDecisionResult store_insight(const DecisionInput& in);

    // Document source by unique path (null json when absent).
    json find_document_source(const std::string& path);
    json list_document_sources();
    json delete_document_source(int document_source_id);

    // Cosine near-duplicate scan over active rows of one structured table
    // (possible_supersedes hint; Phase 2 step 5). Shared with the Phase 3
    // note path and, since Phase 5, the insights table.
    std::vector<std::pair<int,float>> supersede_hints(
        const std::vector<float>& emb, const std::vector<int>& exclude_ids,
        const char* table = "decisions", const char* pk = "decision_id");

    // Embed `text` (when the embedder is usable) and run supersede_hints
    // over it. Used by the Phase 3 note surface, where the stored row is a
    // summary but the hint compares against current decisions.
    std::vector<std::pair<int,float>> possible_supersedes_for_text(
        const std::string& text);

    // ---- catch-up / recipe helpers ------------------------------------
    // Turns lacking an L2 summary, linked by (session_id, timestamp).
    // LEFT JOIN keeps turns whose summary row doesn't exist; ordered
    // newest-first so the summarizer works on the most recently active
    // turns first. This matters most for a manual resummarize (nulling
    // model_id on a batch of `summaries` rows via direct SQL): the turns
    // the user actually cares about right now are usually the recent
    // ones, and with catch_up_batch_size capping each tick, newest-first
    // means those show up with real summaries soonest instead of waiting
    // behind a long tail of old backlog.
    std::vector<TurnRecord> unsummarized_turns(int limit);

    // ---- turn_id-based turn_summaries helpers (v0.12.0) ----------------
    bool turn_summary_exists(int turn_id);

    // Housekeeping retry for poison-abandoned turns: mark_turn_summarized()
    // permanently stamps a turn_summaries row with model "bad" and text=''
    // when a turn hits max_turn_failures (summarizer_service.cpp). That row
    // deliberately blocks re-enqueue via unsummarized_turns()'s LEFT JOIN
    // exclusion -- otherwise a persistently-poison turn would spin forever.
    // But whatever caused the failure (inference outage, config regression)
    // may since be fixed, and there's no other path back for these turns.
    // Deleting the abandoned row here lets the turn fall back into
    // unsummarized_turns()'s normal candidate set on the very next
    // enqueue_catch_up() pass, giving it a fresh set of max_turn_failures
    // attempts. Trivial-turn skips are NOT touched -- those are tagged with
    // the real summarizer model name, never "bad", so this only targets
    // genuine poison-abandon rows.
    int reset_abandoned_turn_summaries(int limit);

    bool finalize_turn_summary(int turn_id, const std::string& text,
                                const std::string& summary_model_name);

    // Create a turn_summaries row marking a turn as "done" with no summary
    // text (trivial-turn skip, or poison-turn abandonment) — mirrors
    // finalize_turn_summary's turns lookup, just with text left as ''
    // (empty string, not NULL -- an empty string unambiguously means
    // "intentionally blank"; NULL in this column would look like a data
    // integrity error rather than a deliberate skip/abandon marker).
    // Reid's design decision: no placeholder mechanism, so a turn_summaries
    // row now always represents completed work (real summary, trivial-skip,
    // or poison-abandon), never an in-progress sentinel.
    bool mark_turn_summarized(int turn_id, const std::string& model_name);


    // ---- episode layer (EPISODE_PLAN Phase 1) --------------------------
    // Most recent episode's span-end (stored in updated_at) for a session.
    // Empty when the session has no episode rows yet.
    std::string last_episode_end(const std::string& session_guid);

    // Non-draft L2 turn summaries for a session with timestamp > since_ts
    // (empty since_ts = all), oldest-first. Composes the closing episode.
    // since_ts arrives as a "%F %T" string (from last_episode_end(), which
    // renders the epoch column back to that format) -- parse it back to
    // epoch seconds to compare against the INTEGER created_at column.
    std::vector<SummaryRecord> l2_summaries_since(
            const std::string& session_guid, const std::string& since_ts);

    // Candidate turns for similarity-based episode detection: joins
    // turn_summaries against turns to fetch both embeddings (turn + summary)
    // plus the turn-summary text. Rows where EITHER embedding is NULL are
    // skipped (boundary detection requires both signals).
    std::vector<EpisodeCandidateTurn> episode_candidate_turns(
            const std::string& session_guid, const std::string& since_ts);

    // Close an episode (PHASE5 B.1): insert one `episodes` row and set
    // episode_id on every member turn_summary in ONE transaction — any bad
    // member (other session, already assigned, marker, missing) rolls the
    // whole close back. Unique-key collision on (session, span_start)
    // returns 0. See storage_backend.h for the full contract.
    int store_episode_close(const EpisodeClose& close);

    // Sessions whose open episode is ready to close: they have >=1 non-empty
    // L2 turn summary past the last episode's span_end, and their newest
    // turn is older than idle_minutes. Returns session GUIDs.
    std::vector<std::string> episodes_needing_close(int idle_minutes);

    // L2 (turn) summaries for a session, newest-first (reads turn_summaries
    // directly; PHASE5 B.5).
    std::vector<SummaryRecord> turn_summaries_by_session_desc(
            const std::string& session_guid, int limit);

    // This session's episodes, newest-first (span_start DESC).
    std::vector<SummaryRecord> episodes_by_session_desc(
            const std::string& session_guid, int limit);
    std::vector<TurnRecord> turns_by_session_desc(
            const std::string& session_guid, int limit);


    // Exact-match existence check for import idempotency: same text +
    // same created_at timestamp already present in summaries. Used by
    // `ragger import conversations`/`import summaries` so re-running an
    // import (or feeding overlapping exports) doesn't duplicate rows.
    // Normalizes `text` the same way store_summary() does before storing
    // (absolute home-dir paths -> "~/") — otherwise a chunk containing
    // e.g. "/Users/reid/..." never matches what's actually in the table
    // (already normalized to "~/...") and gets re-inserted on every run.
    bool summary_exists_exact(const std::string& text, const std::string& created_at);

    // Mirrors summary_exists_exact for the decisions table (same
    // normalize-before-compare fix applies here too — plus the
    // decision-number-prefix strip, since store_decision() strips that
    // before storing too).
    bool decision_exists_exact(const std::string& text, const std::string& created_at);

    // Fuzzy dedup against live-captured turns: exact user_text, timestamp
    // within ±window_seconds. Catches import overlap with turns Ragger
    // captured live (identical content, a few seconds of clock skew —
    // message send time vs. capture time, plus any tz-conversion rounding).
    bool turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                           int window_seconds);

    // Collapse whitespace, drop zero-width/variation-selector/control
    // characters, and case-fold so the same exchange pasted through two
    // different clients (Telegram markdown escaping vs. Claude.ai's raw
    // text) compares equal. Comparison-only — never used for storage.
    static std::string normalize_for_compare(const std::string& s);

    // Cross-source overlap lookup: scan turns for a user_text that
    // normalizes-equal (see normalize_for_compare), ignoring timestamp
    // entirely. Used to reconcile the same exchange captured from two
    // different importers whose timestamps have no reason to agree.
    std::optional<TurnRecord> find_turn_by_text(const std::string& user_text);

    // Upgrade an existing turn's timestamp/session_guid in place. Empty
    // args leave that field untouched. session_guid resolves/creates a
    // sessions row same as store_turn.
    bool update_turn_meta(int turn_id, const std::string& timestamp,
                          const std::string& session_guid);

    // Recipe ingredients (issue #23): recent summaries of a given level, and
    // current decisions — fetched by recency (not semantic search) for the
    // default tiered payload. Returned newest-first.
    std::vector<std::string> recent_summaries(const std::string& level, int limit);

    std::vector<std::string> current_decisions(int limit);

    // Set a decision's status (e.g. "roadmap" -> "current" once planned
    // work is done, or -> "superseded"/"deprecated" once stale).
    bool set_decision_status(int decision_id, const std::string& status);

    // Decisions with an arbitrary status, most recent first — e.g.
    // status="roadmap" to list planned/future work explicitly (roadmap
    // entries are deliberately excluded from current_decisions()'s
    // recall-pipeline query so unfinished plans don't clutter every
    // session's context).
    std::vector<std::string> decisions_by_status(const std::string& status, int limit);

    // Replace a summary's text + embedding, update its model. False if absent.
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name);

    // Replace a summary's tags column. Used by the summarizer to clear
    // "draft" once a row has been rewritten with a real summary.
    bool set_summary_tags(int summary_id, const std::string& tags);

    bool update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding);

    // --- search ---

    // Hybrid search over summaries: vector cosine (cached embeddings) blended
    // with the custom TF-IDF text index (literal + metaphone, v0.16). Both are
    // min-max normalized and combined with vector_weight/bm25_weight (bm25_weight
    // is now the single text-index blend weight; the old separate phon_weight
    // signal is folded into score_query() itself via fts_w_metaphone).
    //
    // NOTE: the lean v2 summaries table has no collection column, so the
    // `collections` filter is currently a no-op (kept for API/source compat).
    // Documents (L5) and decisions (L6) ARE merged into search: parallel passes
    // (ensure_doc_cache / ensure_dec_cache + a text_index_->score_query() pass)
    // are scored identically and merged with the summaries results into the
    // single ranked top-k returned here. Each SearchResult's metadata["source"]
    // is "summary", "document", or "decision" so callers can tell the corpora
    // apart.
    SearchResponse search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> /*collections*/);

    // Text-only search: FTS5 keyword + phonetic scoring across all four
    // corpora. No embedding caches, no embedder call. Used when embeddings
    // are degraded (drift mismatch at startup).
    SearchResponse search_text_only(const std::string& query, int limit);

    // --- maintenance ---

    int count() const;

    std::vector<std::pair<std::string, int64_t>> table_row_counts() const;

    // Total rows across the five embedded tables (turns, turn_summaries,
    // summaries, decisions, documents) — i.e. how many rows
    // `rebuild_embeddings()` will re-encode. (count() alone is just
    // summaries, which understates the rebuild scope.)
    int count_embeddable_rows(const std::string& table = "all") const;

    // True if any embedded table holds a non-NULL embedding. EXISTS short-
    // circuits on the first hit. Deferred (NULL-embedding) rows don't count —
    // they get the current model on backfill, so they aren't incompatible.
    bool has_embeddings() const;

    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). Returns every summary, score 0.
    std::vector<SearchResult> load_all(const std::string& /*collection*/);

    // Re-encode embeddings across all five embedded tables (turns, summaries,
    // decisions, documents, turn_summaries). Two modes, selected by `only_missing`:
    //   * false → re-encode every row (full rebuild; used by CLI
    //             `ragger re-embed` after a model/dtype change)
    //   * true  → only rows whose embedding is NULL or whose embedding_version
    //             column doesn't match the current version (cheap backfill
    //             at server startup) — a plain indexed integer comparison
    //             against the embedding_version column (db_version 0.15+),
    //             not a per-row blob decode.
    // When `progress` is set, prints a running "n/total" line to stdout
    // (interactive CLI). Returns the number of rows (re-)embedded.
    // The embed/index text formulas (documents' title suffix, decisions'
    // trigger prefix) live in sqlite/text_index.h -- the single source
    // shared with TextIndex::text_select_of() and the store paths.
    int embed_tables(Embedder& emb_ref, bool only_missing, bool progress,
                      const std::string& table_filter = "all");

    // Full re-encode of every embedded row (interactive, with progress),
    // or one table when `table` != "all".
    int rebuild_embeddings(Embedder& emb_ref, bool progress, const std::string& table);

    // Cheap backfill: embed rows left NULL or with stale version byte,
    // scoped to one table when `table` != "all".
    int backfill_embeddings(Embedder& emb_ref, const std::string& table);

    // A disabled embedder returns {} from encode(), which bind_embedding()
    // turns into NULL. That is right for a single store, but catastrophic for
    // a bulk pass: a full rebuild would walk every table replacing good
    // vectors with NULL. Refuse instead.
    bool embedder_usable(const Embedder& emb_ref) const;

    /// The embedder, or throw. Every write path that must encode calls
    /// this instead of dereferencing `embedder` directly: the DB-only
    /// constructor leaves it null, and a silent crash there is worse than
    /// an exception the caller can report (B §1.17).
    Embedder& require_embedder();

    uint8_t get_embedding_version() const;

    uint8_t increment_embedding_version();

    // (Re)build the custom index for a single text table. Delegates to the
    // TextIndex engine (schema + tokenize + TF-IDF live there now).
    int reindex_table(const std::string& table_name, bool progress);

    // Truncate the shared `terms` table + reset its AUTOINCREMENT counter.
    // Delegates to TextIndex; see its header comment for the "only
    // safe immediately before reindexing ALL five tables" caveat.
    void reset_terms_table();

    // Set a document's embedding (used by the import path after embedding
    // chunks via the subprocess executor). Returns true on a row update.
    // PHASE3 Step 6: shared write-back + per-table cache invalidation.
    void invalidate_cache_for(const TextTable& t);
    bool update_embedding(const TextTable& t, int id, const std::vector<float>& emb);

    bool update_document_embedding(int document_id, const std::vector<float>& emb);

    // Per-row embedding write-back for the other three context tables.
    // Mirrors update_document_embedding; each invalidates the cache that
    // backs its table's vector search.
    bool update_decision_embedding(int decision_id, const std::vector<float>& emb);

    bool update_summary_embedding(int summary_id, const std::vector<float>& emb);

    bool update_turn_embedding(int turn_id, const std::vector<float>& emb);

    std::vector<std::string> collections() const;

    // Returns true if the summaries row exists and has a "keep" tag.
    bool has_keep_tag(int summary_id);

    bool delete_memory(int memory_id);

    int delete_batch(const std::vector<int>& memory_ids);

    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit,
                                                 const std::string& after = "",
                                                 const std::string& before = "");

    void close();
};

} // namespace ragger::sqlite
