#include "sqlite/backend_impl.h"
#include "embedder.h"
#include "config.h"
#include "lang.h"
#include "util/fs.h"
#include "util/time.h"
#include "util/sqlite.h"
#include "double_metaphone.h"
#include "fts_tokenize.h"
#include "sqlite/text_index.h"
#include "vector_codec.h"
#include "Logger.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <iterator>
#include <filesystem>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <mutex>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ragger::sqlite {

using json = nlohmann::json;
namespace fs = std::filesystem;


    /// Users + settings tables — declarative, no in-place migration
    /// (single-user app; pre-v2 data is exported out-of-band). `users` mirrors
    /// the reference DDL plus `password_hash` for credentialed access.
    /// Shared by both constructors.
    void SqliteBackend::Impl::create_user_schema() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                token_hash    TEXT NOT NULL,
                password_hash TEXT,
                created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at    INTEGER NOT NULL DEFAULT (unixepoch())
            )
        )");
        exec(R"(
            CREATE TRIGGER IF NOT EXISTS users_modified
            AFTER UPDATE ON users
            BEGIN
                UPDATE users SET updated_at = unixepoch()
                WHERE id = NEW.id;
            END
        )");
        exec(R"(
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        )");
    }


    // ---- User / settings CRUD (gap closure) --------------------------------

    std::optional<UserInfo> SqliteBackend::Impl::get_user_by_username(const std::string& username) {
        Stmt s(db, "SELECT id, username, token_hash FROM users WHERE username = ?");
        s.bind(1, username);
        if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
        return std::nullopt;
    }


    std::optional<UserInfo> SqliteBackend::Impl::get_user_by_token_hash(const std::string& token_hash) {
        Stmt s(db, "SELECT id, username, token_hash FROM users WHERE token_hash = ?");
        s.bind(1, token_hash);
        if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
        return std::nullopt;
    }


    std::optional<std::string> SqliteBackend::Impl::get_user_password(const std::string& username) {
        Stmt s(db, "SELECT password_hash FROM users WHERE username = ?");
        s.bind(1, username);
        if (s.step()) return s.column_text_opt(0);
        return std::nullopt;
    }


    void SqliteBackend::Impl::set_user_password(const std::string& username, const std::string& pw_hash) {
        Stmt s(db, "UPDATE users SET password_hash = ? WHERE username = ?");
        s.bind(1, pw_hash).bind(2, username).step();
    }


    int SqliteBackend::Impl::create_user(const std::string& username, const std::string& token_hash) {
        int64_t ts = db_epoch();
        Stmt s(db,
            "INSERT INTO users (username, token_hash, created_at, updated_at) VALUES (?,?,?,?)");
        s.bind(1, username).bind(2, token_hash).bind(3, ts).bind(4, ts).step();
        return sqlite3_changes(db) > 0
            ? static_cast<int>(sqlite3_last_insert_rowid(db))
            : -1;
    }


    bool SqliteBackend::Impl::delete_user(const std::string& username) {
        Stmt s(db, "DELETE FROM users WHERE username = ?");
        s.bind(1, username).step();
        return sqlite3_changes(db) > 0;
    }


    void SqliteBackend::Impl::update_user_token(const std::string& username, const std::string& new_hash) {
        Stmt s(db, "UPDATE users SET token_hash = ? WHERE username = ?");
        s.bind(1, new_hash).bind(2, username).step();
    }


    std::optional<std::string> SqliteBackend::Impl::get_setting(const std::string& key) {
        Stmt s(db, "SELECT value FROM settings WHERE key = ?");
        s.bind(1, key);
        if (s.step()) return s.column_text_opt(0);
        return std::nullopt;
    }


    void SqliteBackend::Impl::set_setting(const std::string& key, const std::string& value) {
        Stmt s(db, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)");
        s.bind(1, key).bind(2, value).step();
    }


    // ---- Schema introspection (gap closure) --------------------------------

    std::vector<SchemaObject> SqliteBackend::Impl::list_schema_objects() {
        std::vector<SchemaObject> result;
        Stmt s(db,
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY CASE type "
            "  WHEN 'table'   THEN 0 "
            "  WHEN 'index'   THEN 1 "
            "  WHEN 'trigger' THEN 2 "
            "  ELSE 3 END, name");
        while (s.step_checked()) {
            SchemaObject obj;
            obj.type     = s.column_text(0);
            obj.name     = s.column_text(1);
            obj.tbl_name = s.column_text(2);
            obj.sql      = s.column_text(3);
            // Filter FTS5 shadow tables and other internal tables
            if (obj.name.find("_fts") != std::string::npos) continue;
            result.push_back(std::move(obj));
        }
        return result;
    }


    std::vector<std::string> SqliteBackend::Impl::table_column_names(const std::string& table) {
        std::vector<std::string> names;
        Stmt s(db, "PRAGMA table_info(" + table + ")");
        while (s.step_checked()) {
            names.push_back(s.column_text(1));  // col 1 = column name
        }
        return names;
    }


    int SqliteBackend::Impl::iterate_table_rows(const std::string& table,
                           const std::function<void(const ExportRow&)>& cb) {
        Stmt s(db, "SELECT * FROM " + table);
        int count = 0;
        while (s.step_checked()) {
            int ncols = sqlite3_column_count(s.raw());
            ExportRow row(ncols);
            for (int i = 0; i < ncols; ++i) {
                int coltype = sqlite3_column_type(s.raw(), i);
                switch (coltype) {
                    case SQLITE_NULL:
                        row[i].type = ExportCell::Type::Null;
                        break;
                    case SQLITE_INTEGER:
                        row[i].type    = ExportCell::Type::Integer;
                        row[i].int_val = sqlite3_column_int64(s.raw(), i);
                        break;
                    case SQLITE_FLOAT:
                        row[i].type      = ExportCell::Type::Float;
                        row[i].float_val = sqlite3_column_double(s.raw(), i);
                        break;
                    case SQLITE_BLOB: {
                        row[i].type = ExportCell::Type::Blob;
                        const auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(s.raw(), i));
                        int nb = sqlite3_column_bytes(s.raw(), i);
                        row[i].blob_val.assign(p, p + nb);
                        break;
                    }
                    case SQLITE_TEXT:
                    default: {
                        row[i].type     = ExportCell::Type::Text;
                        const char* txt = reinterpret_cast<const char*>(
                            sqlite3_column_text(s.raw(), i));
                        row[i].text_val = txt ? txt : "";
                        break;
                    }
                }
            }
            cb(row);
            ++count;
        }
        return count;
    }


    /// Convenience views: epoch ints rendered as local datetime, embedding
    /// BLOBs collapsed to a has_embedding flag, and term_ids resolved to the
    /// readable term. These exist purely for human inspection (sqlite3 CLI,
    /// GUI browsers) -- no application code reads them.
    ///
    /// DROP-then-CREATE, deliberately NOT "CREATE VIEW IF NOT EXISTS". A view
    /// holds no data, so recreating is free -- and IF NOT EXISTS meant an
    /// existing DB kept a stale view definition FOREVER after this DDL changed
    /// (exactly how the v0.16 views were left still selecting the retired
    /// `phon` column). Unconditional recreation keeps every DB's views in
    /// lockstep with this code on every open.
    void SqliteBackend::Impl::create_views() {
        // Column order mirrors the base tables: narrow, readable columns first,
        // wide/derived ones last, so `SELECT *` stays legible on one screen.
        auto view = [&](const std::string& name, const std::string& body) {
            exec(std::format("DROP VIEW IF EXISTS {}", name));
            exec(std::format("CREATE VIEW {} AS {}", name, body));
        };

        view("users_view", R"(
            SELECT id, username, token_hash, password_hash,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
            FROM users
        )");
        view("models_view", R"(
            SELECT model_id, name,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM models
        )");
        view("sessions_view", R"(
            SELECT session_id, guid, name, name_source,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM sessions
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name)");
        view("turns_view", R"(
            SELECT turn_id, user_text, assistant_text, model_id, session_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turns
        )");
        view("turn_summaries_view", R"(
            SELECT
                turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
                datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
                datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
                unigram_count, bigram_count,
                embedding_version,
                CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turn_summaries
            WHERE text != ''
        )");
        view("summaries_view", R"(
            SELECT summary_id, text, level, tags, session_id, model_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM summaries
        )");
        view("decisions_view", R"(
            SELECT decision_id, text, status, tags,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM decisions
        )");
        view("document_sources_view", R"(
            SELECT document_source_id, title, path, year, tags,
                   datetime(imported_at, 'unixepoch', 'localtime') AS imported_at
            FROM document_sources
        )");
        view("documents_view", R"(
            SELECT document_id, text, tags, chunk_index, document_source_id,
                   datetime(modified_on, 'unixepoch', 'localtime') AS modified_on,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM documents
        )");

        // Junction views (v0.16): the raw <t>_terms tables store term_id, which
        // is unreadable on inspection. Each view resolves it to the actual term
        // string. An INNER JOIN is correct here -- the FK to terms(term_id) is
        // NOT NULL, so a missing parent would be corruption, and surfacing zero
        // rows for it is the right signal. Ordered most-frequent-first within a
        // record, which is what you almost always want when eyeballing why
        // something did or didn't match.
        struct J { const char* view; const char* junc; const char* pk; };
        for (const J& j : {
                 J{"turns_terms_view",          "turns_terms",          "turn_id"},
                 J{"turn_summaries_terms_view", "turn_summaries_terms", "turn_summary_id"},
                 J{"summaries_terms_view",      "summaries_terms",      "summary_id"},
                 J{"documents_terms_view",      "documents_terms",      "document_id"},
                 J{"decisions_terms_view",      "decisions_terms",      "decision_id"},
             }) {
            view(j.view, std::format(
                "SELECT j.{} AS {}, t.term, j.count "
                "FROM {} j JOIN terms t ON t.term_id = j.term_id "
                "ORDER BY j.{}, j.count DESC, t.term",
                j.pk, j.pk, j.junc, j.pk));
        }
    }


} // namespace ragger::sqlite
