/**
 * UserStore — user and settings management over the Ragger SQLite database.
 *
 * Separated from StorageBackend so that the memory-storage interface does not
 * need to carry auth concerns, and so that future storage backends don't have
 * to implement user management.  Callers that only need auth or settings (CLI
 * user commands, the HTTP server auth check) construct a UserStore directly;
 * they do not need a full RaggerMemory.
 */
#pragma once

#include "ragger/storage_types.h"

#include <memory>
#include <optional>
#include <string>

namespace ragger {

class UserStore {
public:
    /// Open the users/settings tables in the given database file.
    /// The file must already exist (created by SqliteBackend on first use).
    explicit UserStore(const std::string& db_path);
    ~UserStore();

    UserStore(const UserStore&)            = delete;
    UserStore& operator=(const UserStore&) = delete;

    /// Get user info by username. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_username(const std::string& username);

    /// Get hashed password for a user. Returns nullopt if not set.
    std::optional<std::string> get_user_password(const std::string& username);

    /// Replace the user's token hash.
    void update_user_token(const std::string& username, const std::string& new_hash);

    /// Create a new user. Returns the new user_id, or -1 on failure (e.g. duplicate username).
    int create_user(const std::string& username, const std::string& token_hash);

    /// Delete a user by username. Returns true if a row was deleted.
    bool delete_user(const std::string& username);

    /// Set or clear the password hash for a user.
    void set_user_password(const std::string& username, const std::string& password_hash);

    /// Get user info by token hash. Returns nullopt if not found.
    std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash);

    /// Get a settings value by key. Returns nullopt if not present.
    std::optional<std::string> get_setting(const std::string& key);

    /// Set or replace a settings value.
    void set_setting(const std::string& key, const std::string& value);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
