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
    explicit RaggerMemory(const std::string& db_path = "",
                          const std::string& model_dir = "");
    ~RaggerMemory();

    std::string    store(const std::string& text, json metadata = {});
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});
    int  count() const;

    /// Load all memories (for export). Optionally filter by collection.
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild BM25 index. Returns doc count.
    int rebuild_bm25();

    /// Get distinct collection names.
    std::vector<std::string> collections() const;

    void close();

private:
    std::unique_ptr<Embedder>      embedder_;
    std::unique_ptr<SqliteBackend> backend_;
};

} // namespace ragger
