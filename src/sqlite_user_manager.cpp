/**
 * SqliteUserManager — SQLite implementation of user management
 *
 * Separated from SqliteBackend so that storage and user management
 * are independent concerns. Opens its own DB connection.
 */
#include "ragger/sqlite_user_manager.h"
#include "ragger/config.h"
#include "ragger/lang.h"

#include <sqlite3.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ragger {

// -----------------------------------------------------------------------
// Impl
// -----------------------------------------------------------------------
struct SqliteUserManager::Impl {
    sqlite3*    db = nullptr;
    std::string db_path;

    explicit Impl(const std::string& path) : db_path(path) {
        namespace fs = std::filesystem;
        fs::create_directories(fs::path(db_path).parent_path());

        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Cannot open user DB: " + db_path);
        }
        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA busy_timeout=5000");

        create_schema();
    }

    ~Impl() {
        if (db) { sqlite3_close(db); db = nullptr; }
    }

    // ---- helpers -------------------------------------------------------
    void exec(const char* sql) {
        char* errmsg = nullptr;
        int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            std::string err = errmsg ? errmsg : "unknown error";
            sqlite3_free(errmsg);
            throw std::runtime_error(std::string(lang::ERR_SQL) + err);
        }
    }

    static std::string now_iso() {
        auto now = std::chrono::system_clock::now();
        auto tt  = std::chrono::system_clock::to_time_t(now);
        std::tm gm{};
        gmtime_r(&tt, &gm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &gm);
        return std::string(buf) + "Z";
    }

    // ---- schema --------------------------------------------------------
    void create_schema() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL,
                created    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                modified   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        )");
        migrate_add_column("users", "token_rotated_at", "TEXT");
        migrate_add_column("users", "preferred_model", "TEXT");
        migrate_add_column("users", "password_hash", "TEXT");
        exec(R"(
            CREATE TABLE IF NOT EXISTS web_sessions (
                token    TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                user_id  INTEGER NOT NULL,
                created  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                expires  TEXT NOT NULL
            )
        )");
    }

    void migrate_add_column(const char* table, const char* column, const char* type) {
        sqlite3_stmt* stmt = nullptr;
        std::string pragma = std::string("PRAGMA table_info(") + table + ")";
        sqlite3_prepare_v2(db, pragma.c_str(), -1, &stmt, nullptr);
        bool exists = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            auto col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col && std::string(col) == column) { exists = true; break; }
        }
        sqlite3_finalize(stmt);
        if (!exists) {
            std::string sql = std::string("ALTER TABLE ") + table +
                              " ADD COLUMN " + column + " " + type;
            exec(sql.c_str());
        }
    }
};

// -----------------------------------------------------------------------
// Public forwarding
// -----------------------------------------------------------------------
SqliteUserManager::SqliteUserManager(const std::string& db_path)
    : pImpl(std::make_unique<Impl>(db_path)) {}

SqliteUserManager::~SqliteUserManager() = default;

// --- User CRUD ---

int SqliteUserManager::create_user(const std::string& username,
                                    const std::string& token_hash) {
    auto now = Impl::now_iso();
    std::string sql = "INSERT INTO users (username, token_hash, created, modified) "
                      "VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, now.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, now.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return static_cast<int>(sqlite3_last_insert_rowid(pImpl->db));
}

std::optional<UserInfo> SqliteUserManager::get_user_by_token_hash(
        const std::string& token_hash) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT id, username, preferred_model FROM users WHERE token_hash = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo u;
        u.id = sqlite3_column_int(stmt, 0);
        u.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.token_hash = token_hash;
        const char* pref_model = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        u.preferred_model = (pref_model && pref_model[0] != '\0') ? std::string(pref_model) : "";
        sqlite3_finalize(stmt);
        return u;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::optional<UserInfo> SqliteUserManager::get_user_by_username(
        const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT id, username, token_hash, preferred_model FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo u;
        u.id = sqlite3_column_int(stmt, 0);
        u.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.token_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        const char* pref_model = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        u.preferred_model = (pref_model && pref_model[0] != '\0') ? std::string(pref_model) : "";
        sqlite3_finalize(stmt);
        return u;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

void SqliteUserManager::update_user_token(const std::string& username,
                                           const std::string& token_hash) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "UPDATE users SET token_hash = ? WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteUserManager::get_user_token_rotated_at(
        const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT token_rotated_at FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<std::string> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* val = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (val) result = std::string(val);
    }
    sqlite3_finalize(stmt);
    return result;
}

void SqliteUserManager::update_user_token_rotated_at(const std::string& username,
                                                      const std::string& timestamp) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "UPDATE users SET token_rotated_at = ? WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteUserManager::get_user_preferred_model(
        const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT preferred_model FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<std::string> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* val = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (val && val[0] != '\0') result = std::string(val);
    }
    sqlite3_finalize(stmt);
    return result;
}

void SqliteUserManager::update_user_preferred_model(const std::string& username,
                                                     const std::string& model) {
    sqlite3_stmt* stmt = nullptr;
    if (model.empty()) {
        int rc = sqlite3_prepare_v2(pImpl->db,
            "UPDATE users SET preferred_model = NULL WHERE username = ?",
            -1, &stmt, nullptr);
        if (rc != SQLITE_OK) throw std::runtime_error("Failed to prepare statement");
        sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        int rc = sqlite3_prepare_v2(pImpl->db,
            "UPDATE users SET preferred_model = ? WHERE username = ?",
            -1, &stmt, nullptr);
        if (rc != SQLITE_OK) throw std::runtime_error("Failed to prepare statement");
        sqlite3_bind_text(stmt, 1, model.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    }
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to execute statement");
    }
    sqlite3_finalize(stmt);
}

void SqliteUserManager::set_user_password(const std::string& username,
                                           const std::string& password_hash) {
    sqlite3_stmt* stmt = nullptr;
    if (password_hash.empty()) {
        int rc = sqlite3_prepare_v2(pImpl->db,
            "UPDATE users SET password_hash = NULL WHERE username = ?",
            -1, &stmt, nullptr);
        if (rc != SQLITE_OK) throw std::runtime_error("Failed to prepare statement");
        sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        int rc = sqlite3_prepare_v2(pImpl->db,
            "UPDATE users SET password_hash = ? WHERE username = ?",
            -1, &stmt, nullptr);
        if (rc != SQLITE_OK) throw std::runtime_error("Failed to prepare statement");
        sqlite3_bind_text(stmt, 1, password_hash.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    }
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to execute statement");
    }
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteUserManager::get_user_password(
        const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT password_hash FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<std::string> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* val = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (val && val[0] != '\0') result = std::string(val);
    }
    sqlite3_finalize(stmt);
    return result;
}

int SqliteUserManager::get_user_count() {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db, "SELECT count(*) FROM users", -1, &stmt, nullptr);
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW)
        count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return count;
}

void SqliteUserManager::delete_user(const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

// --- Web sessions ---

void SqliteUserManager::create_web_session(const std::string& token,
                                            const std::string& username,
                                            int user_id, int ttl_seconds) {
    auto now = std::chrono::system_clock::now();
    auto expires = now + std::chrono::seconds(ttl_seconds);
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto exp_t = std::chrono::system_clock::to_time_t(expires);
    char now_buf[32], exp_buf[32];
    std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_t));
    std::strftime(exp_buf, sizeof(exp_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&exp_t));

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "INSERT OR REPLACE INTO web_sessions (token, username, user_id, created, expires) "
        "VALUES (?, ?, ?, ?, ?)",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, user_id);
    sqlite3_bind_text(stmt, 4, now_buf, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, exp_buf, -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::optional<UserInfo> SqliteUserManager::get_web_session(const std::string& token) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT username, user_id, expires FROM web_sessions WHERE token = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<UserInfo> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        int user_id = sqlite3_column_int(stmt, 1);
        std::string expires = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        sqlite3_finalize(stmt);

        // Check expiry
        struct std::tm tm = {};
        std::istringstream ss(expires);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        auto exp_time = std::chrono::system_clock::from_time_t(timegm(&tm));
        if (std::chrono::system_clock::now() > exp_time) {
            delete_web_session(token);
            return std::nullopt;
        }
        return UserInfo{user_id, username, "", ""};
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

void SqliteUserManager::delete_web_session(const std::string& token) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM web_sessions WHERE token = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

int SqliteUserManager::cleanup_web_sessions() {
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_t));

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM web_sessions WHERE expires < ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, buf, -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    int count = sqlite3_changes(pImpl->db);
    sqlite3_finalize(stmt);
    return count;
}

} // namespace ragger
