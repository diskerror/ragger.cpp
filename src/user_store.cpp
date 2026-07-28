/**
 * UserStore — user and settings management over the Ragger SQLite database.
 */
#include "ragger/user_store.h"
#include "ragger/util/sqlite.h"
#include "ragger/util/time.h"
#include "ragger/config.h"

#include <sqlite3.h>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace ragger {

namespace fs = std::filesystem;

struct UserStore::Impl {
    sqlite3* db = nullptr;

    explicit Impl(const std::string& path) {
        std::string resolved = expand_path(path);
        fs::create_directories(fs::path(resolved).parent_path());
        if (sqlite3_open(resolved.c_str(), &db) != SQLITE_OK) {
            throw std::runtime_error(std::string("UserStore: cannot open DB: ") +
                                     sqlite3_errmsg(db));
        }
        sqlite3_busy_timeout(db, 5000);
        create_schema();
    }

    void create_schema() {
        auto exec = [&](const char* sql) {
            sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
        };
        exec(R"(CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            token_hash    TEXT NOT NULL,
            password_hash TEXT,
            created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
            updated_at    INTEGER NOT NULL DEFAULT (unixepoch())
        ))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS users_modified
            AFTER UPDATE ON users BEGIN
                UPDATE users SET updated_at = unixepoch()
                WHERE id = NEW.id;
            END)");
        exec(R"(CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ))");
    }

    ~Impl() {
        if (db) { sqlite3_close(db); db = nullptr; }
    }
};

UserStore::UserStore(const std::string& db_path)
    : pImpl(std::make_unique<Impl>(db_path)) {}

UserStore::~UserStore() = default;

std::optional<UserInfo> UserStore::get_user_by_username(const std::string& username) {
    Stmt s(pImpl->db, "SELECT id, username, token_hash FROM users WHERE username = ?");
    s.bind(1, username);
    if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
    return std::nullopt;
}

std::optional<std::string> UserStore::get_user_password(const std::string& username) {
    Stmt s(pImpl->db, "SELECT password_hash FROM users WHERE username = ?");
    s.bind(1, username);
    if (s.step()) {
        auto result = s.column_text(0);
        return result.empty() ? std::nullopt : std::make_optional(result);
    }
    return std::nullopt;
}

void UserStore::update_user_token(const std::string& username, const std::string& new_hash) {
    Stmt s(pImpl->db, "UPDATE users SET token_hash = ? WHERE username = ?");
    s.bind(1, new_hash).bind(2, username).step();
}

int UserStore::create_user(const std::string& username, const std::string& token_hash) {
    int64_t ts = db_epoch();
    Stmt s(pImpl->db,
           "INSERT INTO users (username, token_hash, created_at, updated_at) VALUES (?,?,?,?)");
    s.bind(1, username).bind(2, token_hash).bind(3, ts).bind(4, ts).step();
    return sqlite3_changes(pImpl->db) > 0
        ? static_cast<int>(sqlite3_last_insert_rowid(pImpl->db))
        : -1;
}

bool UserStore::delete_user(const std::string& username) {
    Stmt s(pImpl->db, "DELETE FROM users WHERE username = ?");
    s.bind(1, username).step();
    return sqlite3_changes(pImpl->db) > 0;
}

void UserStore::set_user_password(const std::string& username,
                                  const std::string& password_hash) {
    Stmt s(pImpl->db, "UPDATE users SET password_hash = ? WHERE username = ?");
    s.bind(1, password_hash).bind(2, username).step();
}

std::optional<UserInfo> UserStore::get_user_by_token_hash(const std::string& token_hash) {
    Stmt s(pImpl->db, "SELECT id, username, token_hash FROM users WHERE token_hash = ?");
    s.bind(1, token_hash);
    if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
    return std::nullopt;
}

std::optional<std::string> UserStore::get_setting(const std::string& key) {
    Stmt s(pImpl->db, "SELECT value FROM settings WHERE key = ?");
    s.bind(1, key);
    if (s.step()) return s.column_text_opt(0);
    return std::nullopt;
}

void UserStore::set_setting(const std::string& key, const std::string& value) {
    Stmt s(pImpl->db, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)");
    s.bind(1, key).bind(2, value).step();
}

} // namespace ragger
