/**
 * StorageBackend — Abstract interface for memory storage backends
 *
 * Defines the contract for storage implementations (e.g., SQLite).
 * Allows swapping implementations without changing client code.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ragger/storage_types.h"
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

/// One maximal run of turns closed by boundary-detection (session_id
/// change for session runs, time-gap for project runs). first_ts/last_ts
/// are epoch seconds (turns.created_at of the run's first/last turn).
/// session_guid is empty for project-boundary runs (session-unscoped).
struct ClosedRun {
    std::string session_guid;
    int64_t first_ts;
    int64_t last_ts;
    int first_turn_id;
    int last_turn_id;
};

/**
 * Abstract base class for storage backends.
 *
 * This interface defines all methods that client code expects from a backend.
 * Concrete implementations (like SqliteBackend) inherit from this and
 * implement the actual storage logic.
 */
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    /// Path to the database file.
    virtual std::string db_path() const = 0;

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

    /// Write back a single row's embedding for the remaining context tables
    /// (documents already covered above). Returns true if a row was updated.
    virtual bool update_decision_embedding(int decision_id,
                                           const std::vector<float>& emb) = 0;
    virtual bool update_summary_embedding(int summary_id,
                                          const std::vector<float>& emb) = 0;
    virtual bool update_turn_embedding(int turn_id,
                                       const std::vector<float>& emb) = 0;

    // --- summaries (L2/L3) pipeline (issue #22) ---
    /// Insert a summary (level 'turn'|'session'|'project'); embeds text,
    /// records model. Returns summary_id.
    /// `source_timestamp` (when non-empty) overrides the row's timestamp —
    /// L2 turn summaries inherit the source turn's timestamp so chronology
    /// is the linkage between turn and its summary (no FK column needed) and
    /// remains stable across embedding-model changes. `tags` is stored verbatim
    /// on the row (used to mark drafts: "draft" → housekeeping re-summarizes
    /// it when inference becomes available again).
    virtual int store_summary(const std::string& text, const std::string& level,
                              const std::string& model_name = "",
                              const std::string& session_guid = "",
                              const std::string& source_timestamp = "",
                              const std::string& tags = "") = 0;

    /// Exact-match existence check on the `summaries` table: does a row
    /// already exist with this text and created_at timestamp? Used by
    /// bulk importers (conversation/summary import) to make re-running an
    /// import idempotent — safe to feed the same export file (or an
    /// overlapping one) twice without duplicating rows.
    virtual bool summary_exists_exact(const std::string& text,
                                      const std::string& created_at) = 0;

    /// Exact-match existence check on the `decisions` table: same text +
    /// same created_at already present? Mirrors summary_exists_exact —
    /// used by importers (e.g. memories.json → decisions) to stay
    /// idempotent across re-runs.
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

    /// Replace a summary's text + embedding (and model). False if absent.
    virtual bool update_summary_text(int summary_id, const std::string& text,
                                     const std::string& model_name = "") = 0;
    /// Replace a summary's `tags` column (used to clear "draft" once the
    /// row has been rewritten with a real summary).
    virtual bool set_summary_tags(int summary_id, const std::string& tags) = 0;
    /// Recent summaries of a given level ('turn'|'session'|'project'), newest
    /// first — recipe ingredients for tiered payload assembly (issue #23).
    virtual std::vector<std::string> recent_summaries(const std::string& level,
                                                      int limit) = 0;
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

    /// Summary rows tagged 'draft' (heuristic fallback writes that the
    /// summarizer should rewrite when inference is back). Returned
    /// oldest-first, capped at `limit` (0 = unbounded).
    virtual std::vector<DraftSummary> draft_summaries(int limit = 0) = 0;

    /// Sessions whose most recent turn is older than `pause_minutes` AND
    /// that have a `level='session' status='current'` running summary,
    /// returned as session GUIDs. The summarizer's pause-timer uses this to
    /// pick sessions whose L3 running summary should be finalized.
    virtual std::vector<std::string> sessions_needing_close(
        int pause_minutes) = 0;

    // --- episode layer (EPISODE_PLAN Phase 1) ---
    /// The end timestamp (last turn) of the most recent `level='episode'` row
    /// for this session, or empty if the session has no episodes yet. Stored
    /// in the episode row's `updated_at` (timestamp=first turn, updated_at=last
    /// turn). Marks where the next open episode begins.
    virtual std::string last_episode_end(const std::string& session_guid) = 0;

    /// L2 ('turn') summaries for a session with timestamp strictly greater than
    /// `since_ts` (pass the previous episode's end; empty = all), oldest-first,
    /// excluding drafts. These compose the closing episode. Returned as
    /// SummaryRecords so the caller can read each row's timestamp to derive the
    /// episode span (first/last turn).
    virtual std::vector<SummaryRecord> l2_summaries_since(
        const std::string& session_guid, const std::string& since_ts) = 0;

    /// Insert one immutable `level='episode'` row for a session. `first_ts`
    /// becomes the row `timestamp`; `last_ts` is stored in `updated_at` (the
    /// span end). Mirrors store_summary (embed, phon, cache invalidation).
    /// Returns the new summary_id.
    virtual int store_episode(const std::string& text,
                              const std::string& model_name,
                              const std::string& session_guid,
                              const std::string& first_ts,
                              const std::string& last_ts) = 0;

    /// Sessions with at least one non-draft L2 turn summary past their last
    /// episode's end AND whose most recent turn is older than `idle_minutes`
    /// (idle) — i.e. ready to have their open episode closed. Returns session
    /// GUIDs. Anonymous (session_id NULL) turns are skipped.
    virtual std::vector<std::string> episodes_needing_close(
        int idle_minutes) = 0;

    /// All `level='episode'` summary texts for a session, oldest-first. The
    /// ordered episode corpus a session rollup is rebuilt from (Phase 2).
    virtual std::vector<std::string> episode_texts(
        const std::string& session_guid) = 0;

    /// Stamp a summary row's `updated_at` = now (last-regenerate time for a
    /// running rollup row). Returns false if the row is absent.
    virtual bool set_summary_updated_at(int summary_id) = 0;

    // --- boundary-detection scans (new session/project rollup design) ---

    /// Maximal runs of consecutive (by turn_id) turns sharing the same
    /// non-NULL session_id that have already been superseded by a later,
    /// different-session run (edge detection) and not yet reported past
    /// the persisted watermark. The most recent run is never included —
    /// it's still open. Ordered oldest-first.
    virtual std::vector<ClosedRun> sessions_needing_close_boundary() = 0;

    /// Maximal runs of turns (any session, including anonymous) separated
    /// by a time gap of >= gap_days days between consecutive turns (by
    /// turn_id), already superseded by a later run, not yet reported past
    /// the persisted watermark. The most recent run is never included.
    /// Returned ClosedRun.session_guid is always empty (project runs are
    /// session-unscoped). Ordered oldest-first.
    virtual std::vector<ClosedRun> projects_needing_close_boundary(
        int gap_days) = 0;

    /// Advance the persisted session-boundary watermark to turn_id (task D
    /// calls this only after a successful level='session' rollup insert).
    virtual void advance_session_boundary_watermark(int turn_id) = 0;

    /// Advance the persisted project-boundary watermark to turn_id (task D
    /// calls this only after a successful level='project' rollup insert).
    virtual void advance_project_boundary_watermark(int turn_id) = 0;

    /// Bounded text-gathering for a closed session run: episodes fully
    /// spanned by [first_ts, last_ts] plus trailing turn_summaries after
    /// the newest such episode (or from first_ts if none) up to last_ts,
    /// oldest-first. Unlike episode_texts()/l2_summaries_since(), this
    /// never reaches into a LATER run of the same session_guid.
    virtual std::vector<std::string> bounded_session_rollup_texts(
        const std::string& session_guid, int64_t first_ts,
        int64_t last_ts) = 0;

    /// Bounded text-gathering for a closed project run: every level='session'
    /// row (produced by store_session_summary) whose full span
    /// [created_at, COALESCE(updated_at,created_at)] falls entirely within
    /// [first_ts, last_ts], session-unscoped (project runs span sessions),
    /// oldest-first.
    virtual std::vector<std::string> bounded_project_rollup_texts(
        int64_t first_ts, int64_t last_ts) = 0;

    /// Insert one immutable level='session' row spanning a closed run.
    /// Mirrors store_episode's shape exactly (created_at=first_ts,
    /// updated_at=last_ts). Returns the new summary_id.
    virtual int store_session_summary(const std::string& text,
                                      const std::string& model_name,
                                      const std::string& session_guid,
                                      int64_t first_ts, int64_t last_ts) = 0;

    /// Insert one immutable level='project' row spanning a closed run.
    /// Session-unscoped (session_id NULL). Returns the new summary_id.
    virtual int store_project_summary(const std::string& text,
                                      const std::string& model_name,
                                      int64_t first_ts, int64_t last_ts) = 0;

    /// All turns belonging to a session GUID, newest-first up to `limit`
    /// (0 = unbounded). Mirrors `turns_by_session` but in reverse order and
    /// bounded — used by recipes that walk back from the latest prompt.
    virtual std::vector<TurnRecord> turns_by_session_desc(
        const std::string& session_guid, int limit = 0) = 0;

    /// Turn-level (L2) summaries for a session, newest-first up to `limit`.
    /// Returned with the summary row's timestamp (= the source turn's
    /// timestamp) so recipes can align them against raw turns.
    virtual std::vector<SummaryRecord> turn_summaries_by_session_desc(
        const std::string& session_guid, int limit = 0) = 0;

    /// The session's (level='session') summary rows, newest-first up to
    /// `limit`. Recipe's session-summary layer pulls from here.
    virtual std::vector<SummaryRecord> session_summaries_desc(
        const std::string& session_guid, int limit = 0) = 0;

    /// Replace text + metadata of an existing row. Re-embeds (or NULLs the
    /// embedding when `defer_embedding`); FTS5 sync triggers reindex the row.
    /// Preserves id and original timestamp. Returns false if the row
    /// doesn't exist or is protected (keep tag).
    virtual bool update_text(int memory_id,
                             const std::string& text,
                             json metadata = {},
                             bool defer_embedding = false) = 0;

    /// Hybrid search over summaries: vector cosine blended with FTS5 keyword
    /// relevance. (Lean v2 summaries has no collection column, so the
    /// `collections` filter is currently a no-op; kept for API compat.)
    virtual SearchResponse search(const std::string& query,
                                  int limit = 5,
                                  float min_score = 0.0f,
                                  std::vector<std::string> collections = {}) = 0;

    /// Number of stored memories.
    virtual int count() const = 0;

    /// True if any table holds a non-NULL embedding — i.e. the DB has vectors
    /// that an embedding-config change would invalidate. Used by the startup
    /// drift guard to decide between hard-error and re-adopt on an empty DB.
    virtual bool has_embeddings() const = 0;

    /// Total rows across the four embedded tables (turns, summaries, decisions,
    /// documents) — the true scope of a `rebuild_embeddings()` pass.
    virtual int count_embeddable_rows() const = 0;

    /// Load all memories. Returns vector of SearchResult (score=0).
    virtual std::vector<SearchResult> load_all(const std::string& collection = "") = 0;

    /// Rebuild embeddings for all stored documents. Returns doc count.
    virtual int rebuild_embeddings(Embedder& embedder) = 0;

    /// Embed only rows whose embedding column is NULL. Cheap; intended to
    /// run on startup and after deferred-embedding writes. Returns the
    /// number of rows updated.
    virtual int backfill_embeddings(Embedder& embedder) = 0;

    /// (Re)compute the phon (Double Metaphone "sounds-like") column for every
    /// context-table row. only_missing=true does only phon-NULL rows (cheap
    /// post-migration backfill); false recomputes all. Returns rows rewritten.
    virtual int rebuild_phon(bool only_missing, bool progress) = 0;

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
};

} // namespace ragger
