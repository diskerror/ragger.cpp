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
class BM25Index;

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
    std::string store(const std::string& text, json metadata = {}) override;

    /// Search with hybrid vector + BM25. collections={} means all.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {}) override;

    /// Number of stored memories.
    int count() const override;

    /// Load all memories (for export). Returns vector of SearchResult (score=0).
    std::vector<SearchResult> load_all(const std::string& collection = "") override;

    /// Rebuild BM25 index from all stored documents. Returns doc count.
    int rebuild_bm25() override;

    /// Rebuild embeddings for all stored documents. Returns doc count.
    int rebuild_embeddings(Embedder& embedder) override;

    /// Get distinct collection names.
    std::vector<std::string> collections() const override;

    // --- Chat sessions (persistent conversation state) ---
    void save_chat_session(const std::string& session_id, const std::string& username,
                          const std::string& messages_json, const std::string& web_token = "") override;
    std::optional<std::string> get_chat_session(const std::string& session_id) override;
    void delete_chat_session(const std::string& session_id) override;

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
    int cleanup_old_conversations(int max_age_hours) override;

    // --- Bulk export / import ---
    std::vector<MemoryRecord> export_memories(const MemoryFilter& filter) override;
    int import_memories(const std::vector<MemoryRecord>& records, int user_id = -1) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
