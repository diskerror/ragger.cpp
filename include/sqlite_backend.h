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

#include "storage_types.h"
#include "storage_backend.h"
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
                   const std::string& session_guid = "",
                   const std::string& source_timestamp = "",
                   const std::string& session_name = "",
                   const std::string& name_source = "") override;
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
                       const std::string& status = "current",
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
                      const std::string& model_name = "",
                      const std::string& session_guid = "",
                      const std::string& source_timestamp = "",
                      const std::string& tags = "") override;
    bool summary_exists_exact(const std::string& text,
                              const std::string& created_at) override;
    bool decision_exists_exact(const std::string& text,
                               const std::string& created_at) override;
    bool turn_exists_fuzzy(const std::string& user_text,
                           const std::string& ts,
                           int window_seconds) override;
    std::optional<TurnRecord> find_turn_by_text(
        const std::string& user_text) override;
    bool update_turn_meta(int turn_id,
                          const std::string& timestamp = "",
                          const std::string& session_guid = "") override;
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name = "") override;
    bool set_summary_tags(int summary_id, const std::string& tags) override;
    std::vector<std::string> recent_summaries(const std::string& level, int limit) override;
    std::vector<std::string> current_decisions(int limit) override;
    bool set_decision_status(int decision_id, const std::string& status) override;
    std::vector<std::string> decisions_by_status(const std::string& status, int limit) override;
    std::vector<TurnRecord> unsummarized_turns(int limit = 0) override;

    // --- v0.12.0 turn_id-based turn-summary methods (schema migration) ---
    bool turn_summary_exists(int turn_id) override;
    bool finalize_turn_summary(int turn_id, const std::string& text,
                               const std::string& summary_model_name) override;
    bool mark_turn_summarized(int turn_id, const std::string& model_name) override;
    int  reset_abandoned_turn_summaries(int limit = 0) override;

    std::vector<DraftSummary> draft_summaries(int limit = 0) override;
    std::vector<std::string> sessions_needing_close(int pause_minutes) override;
    std::string last_episode_end(const std::string& session_guid) override;
    std::vector<SummaryRecord> l2_summaries_since(
        const std::string& session_guid, const std::string& since_ts) override;
    std::vector<EpisodeCandidateTurn> episode_candidate_turns(
        const std::string& session_guid, const std::string& since_ts) override;
    int store_episode(const std::string& text,
                      const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& first_ts,
                      const std::string& last_ts) override;
    std::vector<std::string> episodes_needing_close(int idle_minutes) override;
    std::vector<std::string> episode_texts(const std::string& session_guid) override;
    bool set_summary_updated_at(int summary_id) override;
    std::vector<ClosedRun> sessions_needing_close_boundary() override;
    std::vector<ClosedRun> projects_needing_close_boundary(int gap_days) override;
    void advance_session_boundary_watermark(int turn_id) override;
    void advance_project_boundary_watermark(int turn_id) override;
    std::vector<std::string> bounded_session_rollup_texts(
        const std::string& session_guid, int64_t first_ts, int64_t last_ts) override;
    std::vector<std::string> bounded_project_rollup_texts(
        int64_t first_ts, int64_t last_ts) override;
    int store_session_summary(const std::string& text, const std::string& model_name,
                              const std::string& session_guid,
                              int64_t first_ts, int64_t last_ts) override;
    int store_project_summary(const std::string& text, const std::string& model_name,
                              int64_t first_ts, int64_t last_ts) override;
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

    /// Text-only fallback: FTS5 keyword + phonetic scoring, no embeddings.
    SearchResponse search_text_only(const std::string& query,
                                    int limit = 5) override;

    /// Number of stored memories.
    int count() const override;

    std::vector<std::pair<std::string, int64_t>> table_row_counts() const override;

    /// True if any table holds a non-NULL embedding.
    bool has_embeddings() const override;

    /// Total rows across the four embedded tables (rebuild scope).
    int count_embeddable_rows() const override;

    /// Load all memories. Returns vector of SearchResult (score=0).
    std::vector<SearchResult> load_all(const std::string& collection = "") override;

    /// Rebuild embeddings for all stored documents. Returns doc count.
    int rebuild_embeddings(Embedder& embedder, bool progress = true) override;
    int backfill_embeddings(Embedder& embedder) override;
    uint8_t embedding_version() const override;
    uint8_t increment_embedding_version() override;
    int rebuild_phon(bool only_missing, bool progress) override;

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
