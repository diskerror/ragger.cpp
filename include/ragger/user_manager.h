/**
 * UserManager — Abstract interface for user management
 *
 * Separates user/auth concerns from memory storage.
 * A different storage backend (e.g., Postgres) might use a completely
 * different user management strategy.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ragger/storage_types.h"

namespace ragger {

class UserManager {
public:
    virtual ~UserManager() = default;

    // --- User CRUD ---

    /// Create a user. Returns user id.
    virtual int create_user(const std::string& username, const std::string& token_hash) = 0;

    /// Look up user by token hash. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash) = 0;

    /// Look up user by username. Returns nullopt if not found.
    virtual std::optional<UserInfo> get_user_by_username(const std::string& username) = 0;

    /// Update a user's token hash.
    virtual void update_user_token(const std::string& username, const std::string& token_hash) = 0;

    /// Get when a user's token was last rotated. Returns nullopt if not set.
    virtual std::optional<std::string> get_user_token_rotated_at(const std::string& username) = 0;

    /// Update when a user's token was last rotated (ISO timestamp).
    virtual void update_user_token_rotated_at(const std::string& username, const std::string& timestamp) = 0;

    /// Get a user's preferred model. Returns nullopt if not set.
    virtual std::optional<std::string> get_user_preferred_model(const std::string& username) = 0;

    /// Update a user's preferred model.
    virtual void update_user_preferred_model(const std::string& username, const std::string& model) = 0;

    /// Set a user's password hash. Empty string clears it.
    virtual void set_user_password(const std::string& username, const std::string& password_hash) = 0;

    /// Get a user's password hash. Returns nullopt if not set.
    virtual std::optional<std::string> get_user_password(const std::string& username) = 0;

    /// Count of registered users.
    virtual int get_user_count() = 0;

    /// Remove a user.
    virtual void delete_user(const std::string& username) = 0;

    // --- Web sessions (password login, DB-backed) ---

    /// Create a web session. Returns void.
    virtual void create_web_session(const std::string& token,
                                    const std::string& username,
                                    int user_id,
                                    int ttl_seconds = 86400) = 0;

    /// Get a web session. Returns nullopt if not found/expired.
    virtual std::optional<UserInfo> get_web_session(const std::string& token) = 0;

    /// Delete a web session.
    virtual void delete_web_session(const std::string& token) = 0;

    /// Cleanup expired web sessions. Returns count deleted.
    virtual int cleanup_web_sessions() = 0;
};

} // namespace ragger
