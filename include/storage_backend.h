/**
 * StorageBackend — Abstract interface for memory storage backends
 *
 * Defines the contract for storage implementations (e.g., SQLite).
 * Allows swapping implementations without changing client code.
 *
 * GAP CLOSED (fix/storage-gap):
 * UserStore (user_store.cpp) and export.cpp both previously bypassed this
 * interface with raw sqlite3_* calls. Both are now fully abstracted:
 *   - UserStore delegates all CRUD through sqlite::Backend (DB-only mode).
 *   - export.cpp uses list_schema_objects(), table_column_names(), and
 *     iterate_table_rows() from this interface. The db_path convenience
 *     overloads open a sqlite::Backend with readonly=true, which uses
 *     SQLITE_OPEN_READONLY and skips schema creation — a legitimate
 *     SQLite-level detail at the right abstraction level (interface specifies
 *     behavior; implementation optimizes with a readonly connection).
 * StatsLogger is intentionally exempt — separate SQLite instrumentation DB.
 */
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "storage_types.h"
#include "vector_codec.h"
#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

class Embedder;

/// One raw L1 turn as read back from the `turns` table. `model_name` is
/// empty when the turn recorded no model; `session_guid` is empty when the
/// turn was captured outside a session (turns.session_id IS NULL).
struct TurnRecord {
    int         turn_id;
    std::string user_text;
    std::string assistant_text;
    std::string model_name;
    std::string timestamp;
    std::string session_guid;
};

/// One row of the v0.12.0 `turn_summaries` table (turn_id-FK based, epoch
/// timestamps). Replaces the old (session_guid, source_timestamp)-keyed
/// linkage. `*_null` flags distinguish "0/absent" from a real 0 value for
/// nullable FK/int columns (turn_id is 0/absent if NULL — pruned turn).
struct TurnSummaryRecord {
    int turn_summary_id;
    std::string text;
    int turn_id;           // 0/absent if NULL (pruned)
    bool turn_id_null;
    int session_id;
    bool session_id_null;
    int turn_model_id;
    bool turn_model_id_null;
    int summary_model_id;
    bool summary_model_id_null;
    int64_t turn_datetime;   // epoch seconds
    int64_t summarized_on;   // epoch seconds
};

/**
 * Abstract base class for storage backends.
 *
 * This interface defines all methods that client code expects from a backend.
 * Concrete implementations (like sqlite::Backend) inherit from this and
 * implement the actual storage logic.
 *
 * One interface, deliberately (review 5B): splitting into narrower interfaces
 * (UserBackend / SchemaIntrospectable / EmbeddingManageable + multiple
 * inheritance) is deferred until a second implementation or a test-only
 * consumer actually needs a narrower type. Today every consumer talks to
 * RaggerMemory or the concrete sqlite::Backend, so a split would be pure
 * ceremony across ~80 methods. The section banners below are the navigability
 * aid; add new methods under the matching section, not at the end.
 */
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    /// Path to the database file.
    virtual std::string db_path() const = 0;

    // --- Core writes & document-source management -----------------------

    /// Store text with metadata. Returns memory ID.
    /// `defer_embedding`: write the row with embedding=NULL, skipping the
    /// embedder call. Caller is responsible for triggering a backfill.
    virtual std::string store(const std::string& text, json metadata = {},
                              bool defer_embedding = false) = 0;

    /// Store a Level 5 RAG document chunk into the `documents` table.
    /// FTS5 sync triggers index the row's text. Returns the new row id.
    /// `chunk.imported_at` is honoured when non-empty so every chunk of one
    /// import shares a single timestamp (issue #48).
    virtual int store_document(const DocumentChunk& chunk,
                               bool defer_embedding = false) = 0;

    /// Store a raw L1 conversation turn into the `turns` table. An empty
    /// `assistant_text` writes a partial row (embedding NULL) for the
    /// prompt-arrival/finalize flow. `model_name` resolves/creates a models
    /// row (turns.model_id). `session_guid` resolves/creates a sessions row
    /// (turns.session_id) — empty leaves it NULL. Returns the new turn_id.
    /// `source_timestamp` (db format) overrides created_at for historical
    /// imports (also skips the regeneration-dedup window).
    virtual int store_turn(const std::string& user_text,
                           const std::string& assistant_text,
                           const std::string& model_name = "",
                           bool defer_embedding = false,
                           const std::string& session_guid = "",
                           const std::string& source_timestamp = "",
                           const std::string& session_name = "",
                           const std::string& name_source = "") = 0;

    /// All raw turns belonging to a session GUID, oldest first. Empty if the
    /// session is unknown. Grouping primitive for session summaries/recipes.
    virtual std::vector<TurnRecord> turns_by_session(
        const std::string& session_guid) = 0;

    /// Finalize a partial turn: set assistant_text, embed the exchange, and
    /// record the model. Returns false if the turn_id doesn't exist.
    virtual bool finalize_turn(int turn_id,
                               const std::string& assistant_text,
                               const std::string& model_name = "") = 0;

    /// Set a document's embedding (import path: embed chunks out-of-process,
    /// then write the vectors back). Returns true if a row was updated.
    virtual bool update_document_embedding(int document_id,
                                           const std::vector<float>& emb) = 0;

    /// Store a curated L6 decision/lesson (decisions table). `status` defaults
    /// to "current" — the only status current_decisions() (the recall-
    /// pipeline layer) actually surfaces. Other recognized values:
    /// "roadmap" (planned/future work — deliberately excluded from the
    /// recall layer so every session isn't cluttered with unfinished plans;
    /// query it via decisions_by_status instead), "superseded", "deprecated".
    /// `tags` stored verbatim; `source_timestamp` overrides the row
    /// timestamp when non-empty. Returns the new decision_id.
    virtual int store_decision(const std::string& text,
                               const std::string& status = "current",
                               const std::string& tags = "",
                               const std::string& source_timestamp = "",
                               bool defer_embedding = false) = 0;

    /// Store a structured decision (agent API Phase 1, db 0.17). Throws
    /// std::invalid_argument on validation failure (empty text/trigger,
    /// bad status, unknown or inactive supersedes target, unknown evidence id).
    virtual KnowledgeResult store_knowledge(const KnowledgeInput& in) = 0;

    /// Look up a document source by its (unique) path. Returns null when no
    /// document has that path, else {"id","title","imported_at"}. Used by the
    /// importer to refuse importing the same file twice.
    virtual json find_document_source(const std::string& path) = 0;

    /// Drop every in-memory search cache (summaries, documents, decisions,
    /// insights, turns). They are rebuilt lazily from the DB on the next
    /// lookup. Called when something outside this process may have changed
    /// the DB file (CLI import/delete, hand edits).
    virtual void invalidate_all_caches() = 0;

    /// All imported documents, ordered by document_source_id (gaps from
    /// deletions are shown as-is). Each item: {id, title, path, year, tags,
    /// imported_at, chunks}. title/path/year may be null.
    virtual json list_document_sources() = 0;

    /// Delete one imported document: all its chunks (and their term-index
    /// rows) and its document_sources row, in one transaction. Returns
    /// {id, title, path, chunks}. std::invalid_argument if no such id.
    virtual json delete_document_source(int document_source_id) = 0;

    /// Store an insight (agent API Phase 5, db 0.17). Same shape as a
    /// structured decision; trigger is optional and evidence is REQUIRED
    /// (an insight with no sources is just a note). Throws

    // --- Agent API (typed handles; see PHASE2/3/4/5-STEPS.md) -----------

    /// Layered agent search (agent API Phase 2). Returns handles + snippets
    /// only (full text via agent_get). Throws std::invalid_argument on an
    /// unknown layer name or a malformed since/until bound.
    virtual std::vector<AgentHit> agent_search(const AgentSearchQuery& q) = 0;

    /// Agent row reads (agent API Phase 2): full rows by typed handle with
    /// optional hierarchy expansion (mode 1), or paged conversation history
    /// for one session (mode 2). Exactly one mode per call; both or neither
    /// throws std::invalid_argument.
    virtual json agent_get(const AgentGetQuery& q) = 0;

    /// Near-duplicate decision hints for free text: embeds the text and runs
    /// the same cosine scan as possible_supersedes (compares against current
    /// decisions). Empty when the embedder is degraded. Used by the Phase 3
    /// note surface so a note can flag a ruling it may contradict.
    virtual std::vector<std::pair<int,float>> possible_supersedes_for_text(
        const std::string& text) = 0;

    /// Timeline (agent API Phase 4): deterministic range scan, newest first,
    /// key-paginated by "<created_epoch>:<rowid>" cursors. No embeddings.
    /// Returns {"items":[{id,created,session,text}...], "next_cursor":...|null}.
    /// Bad granularity or malformed cursor -> std::invalid_argument.
    virtual json agent_timeline(const TimelineQuery& q) = 0;

    /// Curation (agent API Phase 4): metadata-only update of one typed
    /// handle. Text present -> always rejected (a text change is a new row
    /// with supersedes). turn/turn_summary/document rows are not agent-
    /// editable. summaries: only agent-written notes (model_id NULL); tags
    /// only. decisions: status must be a legal transition (see
    /// the Phase 4 table; 'superseded' is reached only via supersedes).
    /// Returns the updated row as agent_get would. std::invalid_argument on
    /// any rule violation.
    virtual json agent_update(const UpdateRequest& req) = 0;

    /// Review queue (agent API Phase 4): action "list" returns pending
    /// decisions oldest first; "approve"/"reject" move each id
    /// pending->current / pending->retired through the agent_update
    /// transition code. ids are decision handles (bare integers also
    /// accepted). A non-pending id fails only that id.
    /// Returns {"done":[ids],"failed":[{"id":...,"error":...}]}.
    virtual json agent_review(const std::string& action,
                              const std::vector<std::string>& ids) = 0;

    // --- Embedding write-back (deferred-embedding path) ------------------

    /// Write back a single row's embedding for the remaining context tables
    /// (documents already covered above). Returns true if a row was updated.
    virtual bool update_decision_embedding(int decision_id,
                                           const std::vector<float>& emb) = 0;
    virtual bool update_turn_embedding(int turn_id,
                                       const std::vector<float>& emb) = 0;

    // PHASE8: the old summary-store and summary embedding write-back are gone
    // with the `summaries` table. Importers write knowledge rows of kind='note' via
    // store_knowledge; re-embedding goes through update_embedding(knowledge).

    /// Exact-match existence check among knowledge notes (kind='note'):
    /// does a row already exist with this text and created_at timestamp?
    /// Used by bulk importers (conversation/summary import) to make
    /// re-running an import idempotent — safe to feed the same export file
    /// (or an overlapping one) twice without duplicating rows.
    virtual bool note_exists_exact(const std::string& text,
                                   const std::string& created_at) = 0;

    /// Exact-match existence check on decision rows (kind='decision'):
    /// same text + same created_at already present? Mirrors
    /// note_exists_exact — used by importers (e.g. memories.json →
    /// decisions) to stay idempotent across re-runs.
    virtual bool decision_exists_exact(const std::string& text,
                                       const std::string& created_at) = 0;

    /// Fuzzy existence check on the `turns` table: does a live-captured
    /// turn already exist with this exact user_text and a created_at
    /// within ±window_seconds of `ts`? Used by conversation importers to
    /// skip exchanges Ragger already captured live — the text content is
    /// identical but timestamps skew by a few seconds (message send time
    /// vs. capture time) so an exact timestamp match would miss them.
    virtual bool turn_exists_fuzzy(const std::string& user_text,
                                   const std::string& ts,
                                   int window_seconds) = 0;

    /// Cross-source overlap lookup: does a turn already exist whose
    /// user_text normalizes-equal to `user_text` (whitespace collapsed,
    /// control/zero-width/variation-selector chars stripped, case-folded)?
    /// Unlike turn_exists_fuzzy (same-source dedup, timestamp-windowed),
    /// this ignores timestamp entirely — it's for reconciling the *same*
    /// exchange captured from two different sources (e.g. pasted into
    /// Telegram vs. the original Claude.ai session) whose timestamps have
    /// no reason to agree. Returns the existing row so the caller can
    /// decide whether to upgrade its metadata via update_turn_meta().
    virtual std::optional<TurnRecord> find_turn_by_text(
        const std::string& user_text) = 0;

    /// Upgrade an existing turn's timestamp and/or session_guid in place
    /// (e.g. a Telegram-import row gets corrected to the authoritative
    /// timestamp/session from a later Claude-export import, or vice versa
    /// if run in the other order). Empty `timestamp`/`session_guid` leaves
    /// that field untouched. Returns false if the turn_id doesn't exist.
    virtual bool update_turn_meta(int turn_id,
                                  const std::string& timestamp = "",
                                  const std::string& session_guid = "") = 0;

    // PHASE8: the old summary text-update / tag-setter / recent-list are gone
    // with the `summaries` table (text is never rewritten in place;
    // recipes read episodes since PHASE5).
    /// Current (active) decisions, newest first.
    virtual std::vector<std::string> current_decisions(int limit) = 0;

    /// Set a decision's status (e.g. promote "roadmap" -> "current" once
    /// planned work is done, or mark a stale one "superseded"/"deprecated").
    /// Returns false if no row matched.
    virtual bool set_decision_status(int decision_id, const std::string& status) = 0;

    /// Decisions with the given status, most recent first. Use this for
    /// anything other than the recall pipeline's "current" default —
    /// e.g. status="roadmap" to list planned/future work explicitly.
    virtual std::vector<std::string> decisions_by_status(const std::string& status,
                                                         int limit) = 0;

    /// Turns that don't yet have a matching L2 ('turn') summary —
    /// LEFT JOIN summaries ON (session_id, timestamp) WHERE summary_id IS NULL.
    /// Returned newest-first, capped at `limit` (0 = unbounded). Powers the
    /// summarizer's startup catch-up and housekeeping retry pass.
    virtual std::vector<TurnRecord> unsummarized_turns(int limit = 0) = 0;

    // --- v0.12.0 turn_id-based turn-summary methods (schema migration) ---
    // NOTE: unsummarized_turns(int) is intentionally NOT a new overload here.
    // Per the migration plan, unsummarized_turns() keeps its EXISTING
    // signature/return type (std::vector<TurnRecord>) unchanged — only its
    // SQL body changes (JOIN against turn_summaries instead of summaries,
    // implemented in sqlite_backend.cpp). No interface change needed for it.
    virtual bool turn_summary_exists(int turn_id) = 0;
    virtual bool finalize_turn_summary(int turn_id, const std::string& text,
                                        const std::string& summary_model_name) = 0;
    virtual bool mark_turn_summarized(int turn_id, const std::string& model_name) = 0;

    /// Delete poison-abandoned turn_summaries rows (summary_model_id ->
    /// model "bad", text NULL) so the underlying turn falls back into
    /// unsummarized_turns()'s candidate set on the next catch-up pass.
    /// Housekeeping retry for turns that hit max_turn_failures before
    /// whatever caused the failures was fixed. Returns rows deleted.
    /// Does NOT touch legitimate trivial-turn skips (tagged with the real
    /// summarizer model name, never "bad").
    virtual int reset_abandoned_turn_summaries(int limit = 0) = 0;

    // --- episode layer (db 0.17) ---
    /// The span end of the most recent episode for this session (rendered
    /// "%F %T" local time), or empty if the session has no episodes yet.
    /// Marks where the next open episode begins.
    virtual std::string last_episode_end(const std::string& session_guid) = 0;

    /// L2 turn summaries for a session with timestamp strictly greater than
    /// `since_ts` (pass the previous episode's end; empty = all), oldest-first.
    /// These compose the closing episode. Returned as SummaryRecords so the
    /// caller can read each row's timestamp to derive the episode span.
    virtual std::vector<SummaryRecord> l2_summaries_since(
        const std::string& session_guid, const std::string& since_ts) = 0;

    /// Candidate turns for similarity-based episode detection: joins
    /// turn_summaries against turns to fetch both embeddings plus the
    /// turn-summary text, for all turns after the last episode's end.
    /// Oldest-first, skipping rows where either embedding is NULL.
    virtual std::vector<EpisodeCandidateTurn> episode_candidate_turns(
        const std::string& session_guid, const std::string& since_ts) = 0;

    /// Close an episode: insert one immutable `episodes` row and record its
    /// membership by setting `episode_id` on each member turn_summary in the
    /// same transaction. Contract (the order is the point): validate ->
    /// embed outside the transaction -> INSERT -> per-member UPDATE requiring
    /// total changes == members.size() (any member from another session,
    /// already in an episode, a marker row, or missing fails the whole close
    /// and rolls back) -> index -> commit -> invalidate the episode cache.
    /// A unique-key collision on (session, span_start) — someone already
    /// closed this span — returns 0 (nothing written; caller must not retry).
    /// Returns the new episode_id.
    virtual int store_episode_close(const EpisodeClose& close) = 0;

    /// Sessions with at least one non-empty L2 turn summary past their last
    /// episode's end AND whose most recent turn is older than `idle_minutes`
    /// (idle) — i.e. ready to have their open episode closed. Returns session
    /// GUIDs. Anonymous (session_id NULL) turns are skipped.
    virtual std::vector<std::string> episodes_needing_close(
        int idle_minutes) = 0;

    /// All turns belonging to a session GUID, newest-first up to `limit`
    /// (0 = unbounded). Mirrors `turns_by_session` but in reverse order and
    /// bounded — used by recipes that walk back from the latest prompt.
    virtual std::vector<TurnRecord> turns_by_session_desc(
        const std::string& session_guid, int limit = 0) = 0;

    /// Turn-level (L2) summaries for a session, newest-first up to `limit`
    /// (PHASE5 B.5: reads `turn_summaries` directly — the old
    /// summaries.level='turn' rows have been gone since 0.12). Empty-text
    /// marker rows are excluded.
    virtual std::vector<SummaryRecord> turn_summaries_by_session_desc(
        const std::string& session_guid, int limit = 0) = 0;

    /// This session's episodes, newest-first (span_start DESC) up to `limit`
    /// (0 = unbounded). `timestamp` carries the rendered span_end. Recipe
    /// episode layer (PHASE5 D.3).
    virtual std::vector<SummaryRecord> episodes_by_session_desc(
        const std::string& session_guid, int limit = 0) = 0;

    // PHASE8: update_text is gone — text is never rewritten in place;
    // corrections go through store_knowledge with supersedes, or
    // agent_update for status/tag changes.

    /// Hybrid search across the four corpora: vector cosine blended with the
    /// custom terms-index keyword relevance. (The `collections` filter is
    /// currently a no-op; kept for API compat.)
    virtual SearchResponse search(const std::string& query,
                                  int limit = 5,
                                  float min_score = 0.0f,
                                  std::vector<std::string> collections = {}) = 0;

    /// Text-only search using FTS5 keyword + phonetic scoring only (no
    /// embeddings). Used when embeddings are degraded (drift mismatch).
    virtual SearchResponse search_text_only(const std::string& query,
                                            int limit = 5) = 0;

    /// Number of stored memories.
    virtual int count() const = 0;

    /// Row counts for the user-facing tables (turns, summaries,
    /// turn_summaries, sessions, documents, decisions, statements, models),
    /// for the dashboard status pane. Ordered, name -> count. Missing tables
    /// are simply omitted rather than erroring.
    virtual std::vector<std::pair<std::string, int64_t>> table_row_counts() const = 0;

    /// True if any table holds a non-NULL embedding — i.e. the DB has vectors
    /// that an embedding-config change would invalidate. Used by the startup
    /// drift guard to decide between hard-error and re-adopt on an empty DB.
    virtual bool has_embeddings() const = 0;

    /// Total rows across the four embedded tables (turns, summaries, decisions,
    /// documents), or just one table when `table` != "all" — the true scope of
    /// a `rebuild_embeddings()`/`backfill_embeddings()` pass.
    virtual int count_embeddable_rows(const std::string& table = "all") const = 0;

    /// Load all memories. Returns vector of SearchResult (score=0).
    virtual std::vector<SearchResult> load_all(const std::string& collection = "") = 0;

    /// Rebuild embeddings for all stored documents, or for one table when
    /// `table` != "all" (turns, turn_summaries, summaries, decisions,
    /// documents). Returns row count (re-)encoded.
    virtual int rebuild_embeddings(Embedder& embedder, bool progress = true,
                                    const std::string& table = "all") = 0;

    /// Rebind the backend's embedder. RaggerMemory replaces its Embedder
    /// object (recovery, model update); the backend keeps a raw pointer, so
    /// every reassignment MUST be followed by this or the next store/search
    /// dereferences freed memory.
    virtual void set_embedder(Embedder& embedder) = 0;

    /// Set the dtype used to encode embedding blobs. Must be updated in
    /// lockstep with config().embedding_vector_type, BEFORE a rebuild —
    /// otherwise rebuild writes blobs in the old dtype under the new name.
    virtual void set_vector_type(vector_codec::VectorType t) = 0;

    /// Embed only rows whose embedding is NULL or has a stale version byte,
    /// scoped to one table when `table` != "all". Cheap; intended to run on
    /// startup and after deferred-embedding writes. Returns rows updated.
    virtual int backfill_embeddings(Embedder& embedder,
                                     const std::string& table = "all") = 0;

    /// Get the current embedding version from the settings table.
    virtual uint8_t embedding_version() const = 0;

    /// Increment the embedding version (mod 256) in the settings table and
    /// update the in-memory cached value. Returns the new version.
    virtual uint8_t increment_embedding_version() = 0;

    /// (Re)build the custom FTS index for a single text table.
    /// Tokenizes every record's text, extracts/inserts terms, populates the
    /// <table>_terms junction table, and updates unigram_count/bigram_count.
    /// Idempotent: call multiple times with the same result.
    /// table: one of the registry's text tables (see sqlite/tables.h)
    /// progress: if true, log progress (e.g., every 100 records)
    /// Returns: number of records reindexed.
    virtual int reindex_table(const std::string& table, bool progress = false) = 0;

    /// Truncate the shared cross-table `terms` list and reset its
    /// AUTOINCREMENT counter (starts term_ids fresh from 1). ON DELETE
    /// CASCADE empties every <table>_terms junction row too. Only safe
    /// immediately before reindexing ALL five text tables in the same run --
    /// reindexing a subset afterward would leave the other tables' junction
    /// rows referencing term_ids that no longer exist.
    virtual void reset_terms_table() = 0;

    /// Get distinct collection names.
    virtual std::vector<std::string> collections() const = 0;

    // --- Memory operations ---

    /// Delete a memory by ID. Returns true if deleted.
    virtual bool delete_memory(int memory_id) = 0;

    /// Delete multiple memories by ID. Returns count deleted.
    virtual int delete_batch(const std::vector<int>& memory_ids) = 0;

    /// Search by metadata field matching with optional temporal filtering.
    virtual std::vector<SearchResult> search_by_metadata(const json& metadata_filter,
                                                         int limit = 0,
                                                         const std::string& after = "",
                                                         const std::string& before = "") = 0;

    /// Close the backend and release resources.
    virtual void close() = 0;

    /// Delete old conversation entries older than specified hours. Returns count deleted.
    virtual int cleanup_old_conversations(float max_age_hours) = 0;

    // --- Users / settings table CRUD (gap closure: replaces UserStore's own sqlite3* connection) ---

    /// Look up a user by username. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_username(const std::string& username) = 0;

    /// Get hashed password for a user. Returns nullopt if not set.
    virtual std::optional<std::string> get_user_password(const std::string& username) = 0;

    /// Replace the user's token hash.
    virtual void update_user_token(const std::string& username, const std::string& new_hash) = 0;

    /// Create a new user. Returns the new user_id, or -1 on failure (e.g. duplicate username).
    virtual int create_user(const std::string& username, const std::string& token_hash) = 0;

    /// Delete a user by username. Returns true if a row was deleted.
    virtual bool delete_user(const std::string& username) = 0;

    /// Set or clear the password hash for a user.
    virtual void set_user_password(const std::string& username, const std::string& password_hash) = 0;

    /// Look up a user by token hash. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash) = 0;

    /// Get a settings value by key. Returns nullopt if not present.
    virtual std::optional<std::string> get_setting(const std::string& key) = 0;

    /// Set or replace a settings value.
    virtual void set_setting(const std::string& key, const std::string& value) = 0;

    /// Read a row of `semqlite_config` (the facts-about-the-file table shared
    /// with SemanticSQLite, keyed by SemanticSQLite's own names). nullopt if
    /// the key (or the table) is absent.
    virtual std::optional<std::string> get_semqlite_config(const std::string& key) = 0;

    /// Set or replace a `semqlite_config` row.
    virtual void set_semqlite_config(const std::string& key, const std::string& value) = 0;

    // --- Schema introspection (gap closure: replaces export.cpp's raw sqlite_master / PRAGMA calls) ---

    /// All schema objects (tables, indexes, triggers) from sqlite_master, ordered
    /// tables-first. FTS5 shadow tables are NOT filtered — callers filter via
    /// is_internal_table() if needed.
    virtual std::vector<SchemaObject> list_schema_objects() = 0;

    /// Column names for a table, in schema order (PRAGMA table_info).
    virtual std::vector<std::string> table_column_names(const std::string& table) = 0;

    /// Iterate every row of `table`, calling `cb` once per row. Each ExportCell
    /// carries the runtime type tag (Null/Integer/Float/Text/Blob) and value.
    /// Used by export_sql to dump arbitrary tables without knowing their schema.
    virtual void iterate_table_rows(const std::string& table,
                                    const std::function<void(const ExportRow&)>& cb) = 0;
};

} // namespace ragger
