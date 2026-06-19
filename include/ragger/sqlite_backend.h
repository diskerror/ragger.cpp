/**
 * SQLite backend for memory storage
 *
 * Mirrors ragger_memory/sqlite_backend.py from the Python version.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ragger/storage_types.h"
#include "ragger/storage_backend.h"
#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

class Embedder;

class SqliteBackend : public StorageBackend {
public:
    SqliteBackend(Embedder& embedder, const std::string& db_path = "");

    /// DB-only constructor — no embedder required.
    /// Only schema/migration operations work; store/search will throw.
    explicit SqliteBackend(const std::string& db_path);
    ~SqliteBackend() override;

    /// Path to the database file.
    std::string db_path() const override;

    /// Store text with metadata. Returns memory ID.
    std::string store(const std::string& text, json metadata = {},
                      bool defer_embedding = false) override;

    /// Store a Level 5 RAG document chunk.
    int store_document(const DocumentChunk& chunk,
                       bool defer_embedding = false) override;

    /// Store / finalize a raw L1 conversation turn (turns table).
    int store_turn(const std::string& user_text,
                   const std::string& assistant_text,
                   const std::string& model_name = "",
                   bool defer_embedding = false,
                   const std::string& session_guid = "") override;
    std::vector<TurnRecord> turns_by_session(
        const std::string& session_guid) override;
    bool finalize_turn(int turn_id,
                       const std::string& assistant_text,
                       const std::string& model_name = "") override;

    /// Write back a document's embedding (import path).
    bool update_document_embedding(int document_id,
                                   const std::vector<float>& emb) override;

    /// Store a curated L6 decision/lesson. Returns decision_id.
    int store_decision(const std::string& text,
                       const std::string& status = "active",
                       const std::string& tags = "",
                       const std::string& source_timestamp = "",
                       bool defer_embedding = false) override;

    /// Write back a single row's embedding for any embedded context table.
    bool update_decision_embedding(int decision_id,
                                   const std::vector<float>& emb) override;
    bool update_summary_embedding(int summary_id,
                                  const std::vector<float>& emb) override;
    bool update_turn_embedding(int turn_id,
                               const std::vector<float>& emb) override;

    // --- summaries (L2/L3) pipeline (issue #22) ---
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& status,
                      const std::string& model_name = "",
                      const std::string& session_guid = "",
                      const std::string& source_timestamp = "",
                      const std::string& tags = "") override;
    std::optional<std::pair<int, std::string>>
        current_session_summary(const std::string& session_guid = "") override;
    std::optional<std::pair<int, std::string>>
        current_project_summary() override;
    std::vector<std::string>
        l2_summary_texts(const std::string& session_guid) override;
    std::vector<std::string>
        complete_l3_summary_texts() override;
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name = "") override;
    bool finalize_turn_summary(const std::string& session_guid,
                               const std::string& source_timestamp,
                               const std::string& text,
                               const std::string& model_name = "") override;
    bool mark_turn_summarized(const std::string& session_guid,
                              const std::string& source_timestamp,
                              const std::string& model_name) override;
    bool set_summary_status(int summary_id, const std::string& status) override;
    bool set_summary_tags(int summary_id, const std::string& tags) override;
    std::vector<std::string> recent_summaries(const std::string& level, int limit) override;
    std::vector<std::string> current_decisions(int limit) override;
    std::vector<TurnRecord> unsummarized_turns(int limit = 0) override;
    bool turn_summary_exists(const std::string& session_guid,
                             const std::string& source_timestamp) override;
    std::vector<DraftSummary> draft_summaries(int limit = 0) override;
    std::vector<std::string> sessions_needing_close(int pause_minutes) override;
    std::vector<TurnRecord> turns_by_session_desc(
        const std::string& session_guid, int limit = 0) override;
    std::vector<SummaryRecord> turn_summaries_by_session_desc(
        const std::string& session_guid, int limit = 0) override;
    std::vector<SummaryRecord> session_summaries_desc(
        const std::string& session_guid, int limit = 0) override;

    /// Replace text + metadata of an existing row.
    bool update_text(int memory_id,
                     const std::string& text,
                     json metadata = {},
                     bool defer_embedding = false) override;

    /// Hybrid search over summaries: vector cosine blended with FTS5 keyword
    /// relevance. (`collections` is currently a no-op — lean v2 summaries has
    /// no collection column; kept for API compat.)
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {}) override;

    /// Number of stored memories.
    int count() const override;

    /// True if any table holds a non-NULL embedding.
    bool has_embeddings() const override;

    /// Total rows across the four embedded tables (rebuild scope).
    int count_embeddable_rows() const override;

    /// Load all memories. Returns vector of SearchResult (score=0).
    std::vector<SearchResult> load_all(const std::string& collection = "") override;

    /// Rebuild embeddings for all stored documents. Returns doc count.
    int rebuild_embeddings(Embedder& embedder) override;
    int backfill_embeddings(Embedder& embedder) override;

    /// Get distinct collection names.
    std::vector<std::string> collections() const override;

    /// Delete a memory by ID. Returns true if deleted.
    bool delete_memory(int memory_id) override;

    /// Delete multiple memories by ID. Returns count deleted.
    int delete_batch(const std::vector<int>& memory_ids) override;

    /// Search by metadata field matching with optional temporal filtering. Returns vector of results.
    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit = 0,
                                                 const std::string& after = "",
                                                 const std::string& before = "") override;

    void close() override;

    /// Delete old conversation entries older than specified hours. Returns count deleted.
    int cleanup_old_conversations(float max_age_hours) override;


private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
