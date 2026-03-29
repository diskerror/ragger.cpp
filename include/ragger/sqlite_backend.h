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

    /// DB-only constructor — no embedder required.
    /// Only user management operations work; store/search will throw.
    explicit SqliteBackend(const std::string& db_path);
    ~SqliteBackend();

    /// Path to the database file.
    std::string db_path() const;

    /// Store text with metadata. Returns memory ID.
    std::string store(const std::string& text, json metadata = {});

    /// Search with hybrid vector + BM25. collections={} means all.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});

    /// Number of stored memories.
    int count() const;

    /// Load all memories (for export). Returns vector of SearchResult (score=0).
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild BM25 index from all stored documents. Returns doc count.
    int rebuild_bm25();

    /// Rebuild embeddings for all stored documents. Returns doc count.
    int rebuild_embeddings(Embedder& embedder);

    /// Get distinct collection names.
    std::vector<std::string> collections() const;

    /// User management
    struct UserInfo {
        int         id;
        std::string username;
        bool        is_admin;
        std::string token_hash;
        std::string preferred_model;  // empty = system default
    };

    /// Create a user. Returns user id.
    int create_user(const std::string& username, const std::string& token_hash,
                    bool is_admin = false);

    /// Look up user by token hash. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash);

    /// Look up user by username. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_username(const std::string& username);

    /// Update a user's token hash.
    void update_user_token(const std::string& username, const std::string& token_hash);

    /// Get when a user's token was last rotated. Returns nullopt if not set.
    std::optional<std::string> get_user_token_rotated_at(const std::string& username);

    /// Update when a user's token was last rotated (ISO timestamp).
    void update_user_token_rotated_at(const std::string& username, const std::string& timestamp);

    /// Get a user's preferred model. Returns nullopt if not set.
    std::optional<std::string> get_user_preferred_model(const std::string& username);

    /// Update a user's preferred model.
    void update_user_preferred_model(const std::string& username, const std::string& model);

    /// Set a user's password hash. Empty string clears it.
    void set_user_password(const std::string& username, const std::string& password_hash);

    /// Get a user's password hash. Returns nullopt if not set.
    std::optional<std::string> get_user_password(const std::string& username);

    /// Remove a user from the users table.
    void delete_user(const std::string& username);

    /// Delete a memory by ID. Returns true if deleted.
    bool delete_memory(int memory_id);

    /// Delete multiple memories by ID. Returns count deleted.
    int delete_batch(const std::vector<int>& memory_ids);

    /// Search by metadata field matching. Returns vector of results.
    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit = 0);

    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
