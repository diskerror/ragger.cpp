/**
 * RaggerMemory — high-level facade
 *
 * Mirrors ragger_memory/memory.py from the Python version.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sqlite_backend.h"

namespace ragger {

class RaggerMemory {
public:
    /// Construct with optional override for DB path and model dir.
    /// If user_db_path is set, db_path = common DB, user_db_path = user's private DB.
    explicit RaggerMemory(const std::string& db_path = "",
                          const std::string& model_dir = "",
                          const std::string& user_db_path = "");
    ~RaggerMemory();

    /// Store a memory. In multi-DB mode, stores to user DB unless common=true.
    std::string    store(const std::string& text, json metadata = {},
                         bool common = false);

    /// Search. In multi-DB mode, merges results from both DBs by score.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});
    /// Total count across all DBs.
    int  count() const;

    /// True if operating with separate common + user databases.
    bool is_multi_db() const { return user_backend_ != nullptr; }

    /// Load all memories (for export). Optionally filter by collection.
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild BM25 index. Returns doc count.
    int rebuild_bm25();

    /// Get distinct collection names.
    std::vector<std::string> collections() const;

    /// Access primary backend (for user management, etc.)
    SqliteBackend* backend() { return backend_.get(); }

    void close();

private:
    std::unique_ptr<Embedder>      embedder_;
    std::unique_ptr<SqliteBackend> backend_;      // common DB (or only DB in single-user)
    std::unique_ptr<SqliteBackend> user_backend_;  // user's private DB (nullptr = single-user)
};

} // namespace ragger
