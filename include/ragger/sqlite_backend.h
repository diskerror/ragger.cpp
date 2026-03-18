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

#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

struct SearchResult {
    int         id;
    std::string text;
    float       score;
    json        metadata;
    std::string timestamp;
};

struct SearchResponse {
    std::vector<SearchResult> results;
    json                      timing;
};

struct AllEmbeddings {
    std::vector<int>                ids;
    std::vector<std::string>        texts;
    std::vector<std::vector<float>> embeddings;
    std::vector<json>               metadata;
    std::vector<std::string>        timestamps;
};

class Embedder;
class BM25Index;

class SqliteBackend {
public:
    SqliteBackend(Embedder& embedder, const std::string& db_path = "");
    ~SqliteBackend();

    /// Store text with metadata. Returns memory ID.
    std::string store(const std::string& text, json metadata = {});

    /// Search with hybrid vector + BM25. collections={} means all.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});

    /// Number of stored memories.
    int count() const;

    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
