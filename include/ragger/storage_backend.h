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

    /// Load all memories. Returns vector of SearchResult (score=0).
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

    // --- User management (single-user mode) ---
    /// Get user info by username. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_username(const std::string& username) = 0;

    /// Get hashed password for a user.
    virtual std::optional<std::string> get_user_password(const std::string& username) = 0;

    /// Update user's token hash.
    virtual void update_user_token(const std::string& username, const std::string& new_hash) = 0;

    /// Create a web session with given token.
    virtual void create_web_session(const std::string& token, const std::string& username,
                                    int user_id, int ttl_seconds) = 0;

    /// Get user's preferred model (empty string if none set).
    virtual std::optional<std::string> get_user_preferred_model(const std::string& username) = 0;

    /// Set/clear user's preferred model.
    virtual void update_user_preferred_model(const std::string& username, const std::string& model) = 0;

    /// Get timestamp when token was last rotated.
    virtual std::optional<std::string> get_user_token_rotated_at(const std::string& username) = 0;

    /// Update the token rotated timestamp.
    virtual void update_user_token_rotated_at(const std::string& username, const std::string& timestamp) = 0;

    /// Create a new user. Returns user_id or -1 on error.
    virtual int create_user(const std::string& username, const std::string& token_hash) = 0;

    /// Delete a user by username. Returns true if deleted.
    virtual bool delete_user(const std::string& username) = 0;

    /// Set/clear password hash for a user.
    virtual void set_user_password(const std::string& username, const std::string& password_hash) = 0;

    /// Get user info by token hash. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash) = 0;

    /// Get web session by token. Returns nullopt if not found or expired.
    virtual std::optional<UserInfo> get_web_session(const std::string& token) = 0;

    /// Get a settings value by key. Returns nullopt if key doesn't exist.
    virtual std::optional<std::string> get_setting(const std::string& key) = 0;

    /// Set or update a settings value.
    virtual void set_setting(const std::string& key, const std::string& value) = 0;
};

} // namespace ragger
