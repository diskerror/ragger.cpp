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

    // --- User management (single-user mode) ---
    /// Get user info by username. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_username(const std::string& username);

    /// Get hashed password for a user.
    std::optional<std::string> get_user_password(const std::string& username);

    /// Update user's token hash.
    void update_user_token(const std::string& username, const std::string& new_hash);

    /// Create a web session with given token.
    void create_web_session(const std::string& token, const std::string& username,
                           int user_id, int ttl_seconds);

    /// Get user's preferred model (empty string if none set).
    std::optional<std::string> get_user_preferred_model(const std::string& username);

    /// Set/clear user's preferred model.
    void update_user_preferred_model(const std::string& username, const std::string& model);

    /// Get timestamp when token was last rotated.
    std::optional<std::string> get_user_token_rotated_at(const std::string& username);

    /// Update the token rotated timestamp.
    void update_user_token_rotated_at(const std::string& username, const std::string& timestamp);

    /// Create a new user. Returns user_id or -1 on error.
    int create_user(const std::string& username, const std::string& token_hash);

    /// Delete a user by username. Returns true if deleted.
    bool delete_user(const std::string& username);

    /// Set/clear password hash for a user.
    void set_user_password(const std::string& username, const std::string& password_hash);

    /// Get user info by token hash. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash);

    /// Get web session by token. Returns nullopt if not found or expired.
    std::optional<UserInfo> get_web_session(const std::string& token);

    /// Get a settings value by key. Returns nullopt if key doesn't exist.
    std::optional<std::string> get_setting(const std::string& key) override;

    /// Set or update a settings value.
    void set_setting(const std::string& key, const std::string& value) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
