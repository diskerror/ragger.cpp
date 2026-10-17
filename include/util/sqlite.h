/**
 * Stmt — a thin RAII wrapper over sqlite3_stmt.
 *
 * Prepares on construction and finalizes in the destructor, so a statement
 * can never leak on an early return (the hand-rolled
 * prepare/bind/step/finalize pattern had to remember to finalize on every
 * path). Binds are chainable and 1-based (SQLite convention); column
 * accessors are 0-based.
 *
 *   Stmt s(db, "SELECT id, name FROM users WHERE username = ?");
 *   s.bind(1, username);
 *   if (s.step()) { int id = s.column_int(0); auto name = s.column_text(1); }
 *
 * A prepare failure throws std::runtime_error — the SQL in this backend is
 * static, so that only happens on a genuine schema/DB fault, which the old
 * silent-failure paths would have turned into subtle wrong results anyway.
 */
#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ragger {

class Stmt {
public:
    Stmt(sqlite3* db, std::string_view sql) {
        if (sqlite3_prepare_v2(db, sql.data(), static_cast<int>(sql.size()),
                               &stmt_, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("sqlite prepare failed: ") +
                                     sqlite3_errmsg(db));
        }
    }
    ~Stmt() { if (stmt_) sqlite3_finalize(stmt_); }

    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
    Stmt(Stmt&& o) noexcept : stmt_(o.stmt_) { o.stmt_ = nullptr; }

    // ---- binds (1-based). Text/blob use TRANSIENT so SQLite copies. -------
    Stmt& bind(int i, std::string_view v) {
        sqlite3_bind_text(stmt_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
        return *this;
    }
    Stmt& bind(int i, const char* v) {
        sqlite3_bind_text(stmt_, i, v, -1, SQLITE_TRANSIENT);
        return *this;
    }
    Stmt& bind(int i, int v)     { sqlite3_bind_int(stmt_, i, v); return *this; }
    Stmt& bind(int i, int64_t v) { sqlite3_bind_int64(stmt_, i, v); return *this; }
    // int64_t is `long` on Linux but `long long` on macOS; accept the other
    // spelling so `long long` args aren't ambiguous on Linux.
    template<class T>
        requires (std::is_same_v<T, long long> && !std::is_same_v<long long, int64_t>)
    Stmt& bind(int i, T v) { sqlite3_bind_int64(stmt_, i, v); return *this; }
    Stmt& bind(int i, double v)  { sqlite3_bind_double(stmt_, i, v); return *this; }
    Stmt& bind_null(int i)       { sqlite3_bind_null(stmt_, i); return *this; }
    Stmt& bind_blob(int i, const void* data, int n) {
        sqlite3_bind_blob(stmt_, i, data, n, SQLITE_TRANSIENT);
        return *this;
    }

    // ---- step -------------------------------------------------------------
    /// Advance one row of a SELECT. Returns true for SQLITE_ROW, false for
    /// SQLITE_DONE. (Do NOT use the bool to detect success of a write — for
    /// INSERT/UPDATE/DELETE use exec(), which returns true on SQLITE_DONE.)
    bool step() { return sqlite3_step(stmt_) == SQLITE_ROW; }

    /// Same as step(), but throws on an actual error instead of reporting it
    /// as end-of-rows. step() cannot distinguish SQLITE_DONE from
    /// SQLITE_ERROR/BUSY/ABORT, so `while (s.step())` silently truncates when
    /// a scan fails partway — a read loop that half-finishes and reports
    /// success. Use this in loops where a short read is a correctness bug
    /// (bulk rebuilds, exports, migrations) rather than a cosmetic one.
    bool step_checked() {
        int rc = sqlite3_step(stmt_);
        if (rc == SQLITE_ROW)  return true;
        if (rc == SQLITE_DONE) return false;
        throw std::runtime_error(
            std::string("sqlite step failed (rc=") + std::to_string(rc) + "): " +
            sqlite3_errmsg(sqlite3_db_handle(stmt_)));
    }

    /// Run an INSERT/UPDATE/DELETE to completion. Returns true on SQLITE_DONE
    /// (success), false on any error code — so `if (!s.exec()) throw ...`
    /// mirrors the old `if (sqlite3_step(stmt) != SQLITE_DONE) throw ...`.
    bool exec() { return sqlite3_step(stmt_) == SQLITE_DONE; }

    // ---- column accessors (0-based) ---------------------------------------
    bool    is_null(int c) const { return sqlite3_column_type(stmt_, c) == SQLITE_NULL; }
    int     column_int(int c) const { return sqlite3_column_int(stmt_, c); }
    int64_t column_int64(int c) const { return sqlite3_column_int64(stmt_, c); }
    double  column_double(int c) const { return sqlite3_column_double(stmt_, c); }

    /// Column as string ("" if NULL).
    std::string column_text(int c) const {
        auto* p = reinterpret_cast<const char*>(sqlite3_column_text(stmt_, c));
        return p ? std::string(p) : std::string();
    }
    /// Column as string, nullopt if the column is NULL.
    std::optional<std::string> column_text_opt(int c) const {
        if (is_null(c)) return std::nullopt;
        return column_text(c);
    }

    const void* column_blob(int c) const { return sqlite3_column_blob(stmt_, c); }
    int         column_bytes(int c) const { return sqlite3_column_bytes(stmt_, c); }
    int column_count() const { return sqlite3_column_count(stmt_); }
    int column_type(int c) const { return sqlite3_column_type(stmt_, c); }

    /// Escape hatch for the rare call that needs the raw handle.
    sqlite3_stmt* raw() { return stmt_; }

private:
    sqlite3_stmt* stmt_ = nullptr;
};

/**
 * Transaction — RAII wrapper for BEGIN IMMEDIATE / COMMIT / ROLLBACK.
 *
 * The constructor runs BEGIN IMMEDIATE and owns the transaction. commit()
 * ends it; if the destructor runs with the transaction still open it issues
 * ROLLBACK so an early return or a thrown exception can never leave a half-
 * applied write or a dangling open transaction.
 *
 * Nesting rule: several store paths are called from callers that open their
 * own transaction. A nested BEGIN would throw "cannot start a transaction
 * within a transaction", so when the constructor finds autocommit already
 * off it does NOTHING and does not own: commit() and the destructor are both
 * no-ops and the outer owner decides. The inner scope thereby simply joins
 * the outer transaction.
 */
class Transaction {
public:
    explicit Transaction(sqlite3* db) : db_(db) {
        if (!sqlite3_get_autocommit(db)) {
            // Already inside a transaction opened by an outer caller: join it,
            // do not own it — the outer owner decides commit vs rollback.
            owns_ = false;
            return;
        }
        char* err = nullptr;
        if (sqlite3_exec(db_, "BEGIN IMMEDIATE", nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "BEGIN IMMEDIATE failed";
            sqlite3_free(err);
            throw std::runtime_error("sqlite BEGIN failed: " + msg);
        }
        owns_ = true;
    }
    ~Transaction() noexcept {
        if (owns_ && !committed_) {
            // Errors ignored: nothing useful can be done from a destructor.
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        }
    }

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

    void commit() {
        if (!owns_) return;  // joined an outer transaction: it decides
        char* err = nullptr;
        if (sqlite3_exec(db_, "COMMIT", nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "COMMIT failed";
            sqlite3_free(err);
            throw std::runtime_error("sqlite COMMIT failed: " + msg);
        }
        committed_ = true;
    }

private:
    sqlite3* db_;
    bool owns_ = false;
    bool committed_ = false;
};

} // namespace ragger
