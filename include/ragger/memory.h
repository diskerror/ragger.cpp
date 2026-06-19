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
    /// Construct with optional override for DB path and model dir.
    /// `skip_embedding_guard` bypasses the startup model/dtype/dimensions drift
    /// check (used by `rebuild-embeddings`, which intentionally re-encodes at
    /// the new config and rewrites the settings afterward).
    explicit RaggerMemory(const std::string& db_path = "",
                          const std::string& model_dir = "",
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
                   const std::string& session_guid = "");
    /// Finalize a partial turn (set assistant_text + embed + model).
    bool finalize_turn(int turn_id,
                       const std::string& assistant_text,
                       const std::string& model_name = "");

    /// Write back a document's embedding (import path).
    bool update_document_embedding(int document_id, const std::vector<float>& emb);

    /// Store a curated L6 decision/lesson. Embeds text unless deferred.
    int store_decision(const std::string& text,
                       const std::string& status = "active",
                       const std::string& tags = "",
                       const std::string& source_timestamp = "",
                       bool defer_embedding = false);

    /// Per-row embedding write-back for the remaining context tables.
    bool update_decision_embedding(int decision_id, const std::vector<float>& emb);
    bool update_summary_embedding(int summary_id, const std::vector<float>& emb);
    bool update_turn_embedding(int turn_id, const std::vector<float>& emb);

    // --- summaries (L2/L3) pipeline (issue #22) ---
    /// `source_timestamp` (non-empty) overrides the row's timestamp — L2
    /// inherits the source turn's timestamp so the (session_id, timestamp)
    /// pair links a turn to its summary (no FK column). `tags`="draft" marks
    /// a heuristic-fallback summary that should be re-summarized later.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& status, const std::string& model_name = "",
                      const std::string& session_guid = "",
                      const std::string& source_timestamp = "",
                      const std::string& tags = "");
    std::optional<std::pair<int, std::string>>
        current_session_summary(const std::string& session_guid = "");
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name = "");
    bool set_summary_status(int summary_id, const std::string& status);
    std::vector<std::string> recent_summaries(const std::string& level, int limit);
    std::vector<std::string> current_decisions(int limit);

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

    /// Load all memories (for export). Optionally filter by collection.
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild embeddings for all documents. Returns doc count.
    int rebuild_embeddings();

    /// Embed only rows whose embedding column is NULL. Returns count.
    int backfill_embeddings();

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
                           const std::string& session_id);

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
