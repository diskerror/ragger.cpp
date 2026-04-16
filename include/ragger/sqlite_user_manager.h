/**
 * SqliteUserManager — SQLite implementation of UserManager
 *
 * Opens its own connection to the database file. Safe to use alongside
 * SqliteBackend on the same DB (WAL mode handles concurrent access).
 */
#pragma once

#include <memory>
#include <string>

#include "ragger/user_manager.h"

namespace ragger {

class SqliteUserManager : public UserManager {
public:
    /// Open (or create) the users/sessions tables in the given database.
    explicit SqliteUserManager(const std::string& db_path);
    ~SqliteUserManager() override;

    // --- User CRUD ---
    int create_user(const std::string& username, const std::string& token_hash) override;
    std::optional<UserInfo> get_user_by_token_hash(const std::string& token_hash) override;
    std::optional<UserInfo> get_user_by_username(const std::string& username) override;
    void update_user_token(const std::string& username, const std::string& token_hash) override;
    std::optional<std::string> get_user_token_rotated_at(const std::string& username) override;
    void update_user_token_rotated_at(const std::string& username, const std::string& timestamp) override;
    std::optional<std::string> get_user_preferred_model(const std::string& username) override;
    void update_user_preferred_model(const std::string& username, const std::string& model) override;
    void set_user_password(const std::string& username, const std::string& password_hash) override;
    std::optional<std::string> get_user_password(const std::string& username) override;
    int get_user_count() override;
    void delete_user(const std::string& username) override;

    // --- Web sessions ---
    void create_web_session(const std::string& token, const std::string& username,
                            int user_id, int ttl_seconds = 86400) override;
    std::optional<UserInfo> get_web_session(const std::string& token) override;
    void delete_web_session(const std::string& token) override;
    int cleanup_web_sessions() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
