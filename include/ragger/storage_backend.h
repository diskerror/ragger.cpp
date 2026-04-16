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
#include <vector>

#include "ragger/storage_types.h"
#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

class Embedder;

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
    virtual std::string store(const std::string& text, json metadata = {}) = 0;

    /// Search with hybrid vector + BM25. collections={} means all.
    virtual SearchResponse search(const std::string& query,
                                  int limit = 5,
                                  float min_score = 0.0f,
                                  std::vector<std::string> collections = {}) = 0;

    /// Number of stored memories.
    virtual int count() const = 0;

    /// Load all memories (for export). Returns vector of SearchResult (score=0).
    virtual std::vector<SearchResult> load_all(const std::string& collection = "") = 0;

    /// Rebuild BM25 index from all stored documents. Returns doc count.
    virtual int rebuild_bm25() = 0;

    /// Rebuild embeddings for all stored documents. Returns doc count.
    virtual int rebuild_embeddings(Embedder& embedder) = 0;

    /// Get distinct collection names.
    virtual std::vector<std::string> collections() const = 0;

    // --- Chat sessions (persistent conversation state) ---

    /// Save a chat session.
    virtual void save_chat_session(const std::string& session_id,
                                   const std::string& username,
                                   const std::string& messages_json,
                                   const std::string& web_token = "") = 0;

    /// Get a chat session. Returns nullopt if not found.
    virtual std::optional<std::string> get_chat_session(const std::string& session_id) = 0;

    /// Delete a chat session.
    virtual void delete_chat_session(const std::string& session_id) = 0;

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
    virtual int cleanup_old_conversations(int max_age_hours) = 0;

    // --- Bulk export / import (for cross-database migration) ---

    /// Export memories matching the given filter, including raw embedding blobs.
    virtual std::vector<MemoryRecord> export_memories(const MemoryFilter& filter) = 0;

    /// Import memory records (with raw embeddings). Returns count imported.
    /// If user_id >= 0, the user_id column is set on each inserted row.
    virtual int import_memories(const std::vector<MemoryRecord>& records, int user_id = -1) = 0;
};

} // namespace ragger
