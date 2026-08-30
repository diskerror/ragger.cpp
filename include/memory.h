/**
 * RaggerMemory — high-level facade
 *
 * Mirrors ragger_memory/memory.py from the Python version.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "storage_backend.h"
#include "user_store.h"

namespace ragger {

using json = nlohmann::json;

class RaggerMemory {
public:
    /// Construct against a specific DB path. `skip_embedding_guard` bypasses
    /// the startup model/dtype/dimensions drift check (used by
    /// `rebuild-embeddings`, which intentionally re-encodes at the new
    /// config and rewrites the settings afterward). The model directory is
    /// always resolved from config (Config::resolved_model_dir(), which
    /// itself honors the --model-dir override) — it is not a constructor
    /// parameter.
    /// Throws std::invalid_argument if db_path is empty.
    explicit RaggerMemory(const std::string& db_path,
                          bool skip_embedding_guard = false);
    ~RaggerMemory();

    /// Store a memory. Stores to the configured DB.
    /// `defer_embedding`: write embedding=NULL; caller (or startup) backfills.
    std::string    store(const std::string& text, json metadata = {},
                         bool defer_embedding = false);

    /// Store a Level 5 RAG document chunk. Pass the same `chunk.imported_at`
    /// for every chunk of one import so they share a single timestamp
    /// (issue #48). Returns the new document id.
    int store_document(const DocumentChunk& chunk, bool defer_embedding = false);

    /// Capture a raw L1 conversation turn into the `turns` table. Empty
    /// assistant_text → partial row (finalize later). `session_guid` groups
    /// the turn under a session (empty leaves it NULL). Returns turn_id.
    int store_turn(const std::string& user_text,
                   const std::string& assistant_text = "",
                   const std::string& model_name = "",
                   bool defer_embedding = false,
                   const std::string& session_guid = "",
                   const std::string& source_timestamp = "",
                   const std::string& session_name = "",
                   const std::string& name_source = "");
    /// Finalize a partial turn (set assistant_text + embed + model).
    bool finalize_turn(int turn_id,
                       const std::string& assistant_text,
                       const std::string& model_name = "");

    /// Write back a document's embedding (import path).
    bool update_document_embedding(int document_id, const std::vector<float>& emb);

    /// Store a curated L6 decision/lesson. Embeds text unless deferred.
    /// `status` defaults to "current" — the only status current_decisions()
    /// (the recall-pipeline layer) actually surfaces. Other recognized
    /// values: "roadmap" (planned/future work — deliberately excluded from
    /// the recall layer so every session isn't cluttered with unfinished
    /// plans; query it explicitly instead), "superseded", "deprecated".
    int store_decision(const std::string& text,
                       const std::string& status = "current",
                       const std::string& tags = "",
                       const std::string& source_timestamp = "",
                       bool defer_embedding = false);

    /// Per-row embedding write-back for the remaining context tables.
    bool update_decision_embedding(int decision_id, const std::vector<float>& emb);
    bool update_summary_embedding(int summary_id, const std::vector<float>& emb);
    bool update_turn_embedding(int turn_id, const std::vector<float>& emb);

    /// Purge conversation turns older than max_age_hours. Returns rows deleted.
    int cleanup_old_conversations(float max_age_hours);

    // --- summaries (L2/L3) pipeline (issue #22) ---
    /// `source_timestamp` (non-empty) overrides the row's timestamp — L2
    /// inherits the source turn's timestamp so the (session_id, timestamp)
    /// pair links a turn to its summary (no FK column). `tags`="draft" marks
    /// a heuristic-fallback summary that should be re-summarized later.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& model_name = "",
                      const std::string& session_guid = "",
                      const std::string& source_timestamp = "",
                      const std::string& tags = "");
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name = "");
    std::vector<std::string> recent_summaries(const std::string& level, int limit);
    std::vector<std::string> current_decisions(int limit);

    /// Set a decision's status (e.g. promote "roadmap" -> "current" once
    /// planned work is done, or mark a stale one "superseded"/"deprecated").
    /// Returns false if no row matched.
    bool set_decision_status(int decision_id, const std::string& status);

    /// Decisions with the given status, most recent first. Use this for
    /// anything other than the recall pipeline's "current" default —
    /// e.g. status="roadmap" to list planned/future work explicitly.
    std::vector<std::string> decisions_by_status(const std::string& status, int limit);

    /// Exact-match idempotency check for bulk importers: does a summaries
    /// row already exist with this text and created_at timestamp?
    bool summary_exists_exact(const std::string& text, const std::string& created_at);

    /// Exact-match idempotency check for the decisions table. Mirrors
    /// summary_exists_exact.
    bool decision_exists_exact(const std::string& text, const std::string& created_at);

    /// Fuzzy overlap check for conversation importers: does a live-captured
    /// turn already exist with this exact user_text within ±window_seconds?
    bool turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                           int window_seconds);

    /// Cross-source overlap lookup (ignores timestamp) — see
    /// StorageBackend::find_turn_by_text.
    std::optional<TurnRecord> find_turn_by_text(const std::string& user_text);

    /// Upgrade an existing turn's timestamp/session_guid in place — see
    /// StorageBackend::update_turn_meta.
    bool update_turn_meta(int turn_id,
                          const std::string& timestamp = "",
                          const std::string& session_guid = "");

    /// Replace text + metadata of an existing memory (re-embeds unless
    /// `defer_embedding`, FTS5 triggers reindex, preserves id and original
    /// timestamp). Returns false if the row is missing or has the keep tag.
    bool update_text(int memory_id, const std::string& text, json metadata = {},
                     bool defer_embedding = false);

    /// Search. Returns results from the single configured DB.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});
    /// Total count in the single DB.
    int  count() const;

    /// Total rows across all five embedded tables (turns, turn_summaries,
    /// summaries, decisions, documents) — the rebuild/backfill scope, and a
    /// more accurate "how much is in here" figure than count() (summaries only).
    int  count_embeddable_rows() const;

    /// Load all memories (for export). Optionally filter by collection.
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild embeddings for all documents. Returns doc count.
    /// `progress` writes a \r-updated counter to stdout — right for the CLI,
    /// wrong for the daemon (it lands in the activity log as one enormous
    /// line); daemon callers pass false and get periodic Logger lines instead.
    int rebuild_embeddings(bool progress = true);

    // ---- Embedding identity: current vs desired + re-embed control -------
    struct EmbeddingStatus {
        std::string current_model, current_vtype, current_engine;
        int         current_dims = 0;
        std::string desired_model, desired_vtype, desired_engine;
        int         desired_dims = 0;
        bool        needs_update = false;   // current != desired
        bool        reembedding  = false;   // an update is in progress
        bool        repair_pending = false; // a previous update was interrupted
    };

    /// Snapshot of current vs desired embedding identity, read from the
    /// settings table (current) and config (desired).
    EmbeddingStatus embedding_status();

    /// True when no usable embedding model is loaded: new rows are stored with
    /// a NULL embedding and search is keyword-only. Distinct from
    /// embeddings_degraded(), which also covers a drift mismatch where the
    /// model itself loads fine. Surfaced to agents via MCP because a log line
    /// is not something the user will ever see.
    bool embeddings_unavailable() const;

    /// True when the embedding drift guard failed at startup — vectors are
    /// unusable but FTS5 text + phonetic search still works. Log the error
    /// and keep serving rather than refusing to start.
    bool embeddings_degraded() const { return embeddings_degraded_; }

    /// Re-check the embedding drift guard. If the stored settings now match
    /// the config (e.g. after `ragger rebuild-embeddings`), clear the
    /// degraded flag, backfill any NULL embeddings, and re-enable semantic
    /// search. Returns true if embeddings were recovered.
    bool try_recover_embeddings();

    /// True while a re-embed is running (search short-circuits on this).
    bool is_reembedding();

    /// Perform the staged re-embed: set the reembedding flag, re-encode every
    /// row with an embedder for the DESIRED model, promote current := desired
    /// in the settings table, swap the live embedder, then clear the flag.
    /// Returns rows re-encoded. Throws on failure (flag is always cleared).
    int update_embeddings();

    /// True when a re-embed was interrupted and the remaining rows still
    /// carry the previous model's vectors. Search degrades to text-only while
    /// this is set, and housekeeping finishes the job.
    bool repair_pending();

    /// Finish an interrupted re-embed. Resumes rather than restarts: only rows
    /// whose version byte is stale get re-encoded, using the identity that was
    /// promoted before the interrupted run began. No-op unless repair_pending()
    /// and no re-embed is currently running. Returns rows re-encoded.
    int resume_interrupted_reembed();

    /// Embed only rows whose embedding column is NULL. Returns count.
    int backfill_embeddings();

    /// (Re)compute the phon "sounds-like" column. only_missing=true backfills
    /// only NULL-phon rows; false recomputes all. Returns rows rewritten.
    int rebuild_phon(bool only_missing = false, bool progress = false);

    /// Get distinct collection names.
    std::vector<std::string> collections() const;

    /// Delete a memory by ID.
    bool delete_memory(int memory_id);

    /// Delete multiple memories by ID.
    int delete_batch(const std::vector<int>& memory_ids);

    /// Search by metadata field matching.
    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit = 0,
                                                 const std::string& after = "",
                                                 const std::string& before = "");

    /// Session turns newest-first (recipe walk-back direction).
    std::vector<TurnRecord> turns_by_session_desc(const std::string& session_guid,
                                                   int limit = 0);

    /// Settings access (user_store).
    std::optional<std::string> get_setting(const std::string& key);
    void set_setting(const std::string& key, const std::string& value);

    /// Access primary backend (internal — prefer RaggerMemory methods where possible).
    StorageBackend* backend() { return backend_.get(); }

    void close();

private:
    std::unique_ptr<Embedder>      embedder_;
    std::unique_ptr<StorageBackend> backend_;
    std::unique_ptr<UserStore>      user_store_;   // settings and user management
    bool                            embeddings_degraded_ = false;
#ifdef RAGGER_STATS
    std::unique_ptr<class StatsLogger> stats_;      // opt-in retrieval instrumentation
#endif
};

/// Outcome of capture_turn(). `captured` is false when turn capture is
/// disabled (config [server] capture_turns) or the turn is empty; `turn_id`
/// is the new turns row id when captured, else -1.
struct CaptureResult {
    bool captured;
    int  turn_id;
};

/// Shared entry point behind the `capture_turn` MCP tool and HTTP POST /turn:
/// ingest one agent-pushed raw turn (user+assistant) under a session GUID for
/// background summarization. No-op (captured=false) unless config [server]
/// capture_turns is true. Both transports disassemble their envelope and call
/// this; the gate and storage policy live here, once.
CaptureResult capture_turn(RaggerMemory& memory,
                           const std::string& user,
                           const std::string& assistant,
                           const std::string& model,
                           const std::string& session_id,
                           const std::string& session_name = "",
                           const std::string& name_source = "");

/// One assembled chunk in a recipe-built context, ordered oldest-first
/// (chronological) when emitted so an agent can inject them as-is. `kind`
/// echoes the recipe layer name (raw_turn / turn_summary / session_summary
/// / project_summary / decision). `timestamp` is the source-turn/source-row
/// timestamp where available (empty for cross-session items).
struct ContextChunk {
    std::string kind;
    std::string text;
    std::string timestamp;
};

/// Result of build_context(). `enabled` is false when the read side is off
/// (config [server] build_context, which requires capture_turns). When
/// enabled, `chunks` carries the recipe-assembled payload (oldest first);
/// `recipe_name` reports which recipe was applied (default or caller-named).
struct SessionContext {
    bool                       enabled;
    std::string                recipe_name;
    std::vector<ContextChunk>  chunks;
};

/// Shared entry point behind the `build_context` MCP tool and HTTP
/// GET /session/<id>: assemble a session's turns + summaries + decisions
/// into a recipe-shaped payload. Walks back from the latest turn in the
/// active session; each layer consumes a slice (raw_turn / turn_summary
/// pop turns chronologically; session/project/decisions pull recent
/// rows). max_tokens is enforced as a ceiling.
///
/// No-op (enabled=false) unless BOTH capture_turns and build_context are
/// on. `recipe_name` empty falls back to `[server] default_recipe`. If
/// that recipe doesn't exist, the first built-in is used.
SessionContext build_context(RaggerMemory& memory,
                             const std::string& session_id,
                             const std::string& recipe_name = "");

} // namespace ragger
