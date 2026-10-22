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


    /// `settings` keys whose value lives in `semqlite_config` under
    /// SemanticSQLite's own key name (embedding identity stamps: what the stored
    /// vectors are). The `desired_*` staging keys and `embedding_version` are
    /// Ragger's change-management state and deliberately stay in `settings`.
    struct SemqliteSettingKey { const char* setting; const char* semqlite; };
    static constexpr SemqliteSettingKey kSemqliteSettingKeys[] = {
        {"embedding_model", "embedding_model"},
        {"vector_type",     "embedding_vector_type"},
        {"dimensions",      "embedding_dims"},
    };


    /// Users + settings tables — declarative, no in-place migration
    /// (single-user app; pre-v2 data is exported out-of-band). `users` mirrors
    /// the reference DDL plus `password_hash` for credentialed access.
    /// Shared by both constructors.
    void Backend::Impl::create_user_schema() {
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

        // semqlite_config: the facts-about-the-file table SemanticSQLite reads
        // when it opens this database (same shape as the table its
        // SEMQLITE_SAVE() writes). It records what the stored vectors ARE
        // (model, dtype, dimensions, offset, embedder) in SemanticSQLite's own
        // key names so a user can write EMBED() queries against this file with
        // no setup. Everything Ragger uses to MANAGE a change (desired_*,
        // embedding_version, reembedding*) stays in `settings`.
        exec(R"(
            CREATE TABLE IF NOT EXISTS semqlite_config (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        )");

        // One-time, idempotent move of the legacy identity stamps out of
        // `settings`. Gated on the rows existing (not on db_version) because a
        // DB already stamped 0.17 never re-runs the 0.16 -> 0.17 leg.
        // INSERT OR IGNORE keeps a value already in semqlite_config.
        for (const auto& m : kSemqliteSettingKeys) {
            exec(std::format(
                "INSERT OR IGNORE INTO semqlite_config (key, value) "
                "SELECT '{1}', value FROM settings WHERE key = '{0}'",
                m.setting, m.semqlite));
            exec(std::format("DELETE FROM settings WHERE key = '{0}'", m.setting));
        }
    }


    // ---- User / settings CRUD (gap closure) --------------------------------

    std::optional<UserInfo> Backend::Impl::get_user_by_username(const std::string& username) {
        Stmt s(db, "SELECT id, username, token_hash FROM users WHERE username = ?");
        s.bind(1, username);
        if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
        return std::nullopt;
    }


    std::optional<UserInfo> Backend::Impl::get_user_by_token_hash(const std::string& token_hash) {
        Stmt s(db, "SELECT id, username, token_hash FROM users WHERE token_hash = ?");
        s.bind(1, token_hash);
        if (s.step()) return UserInfo{s.column_int(0), s.column_text(1), s.column_text(2)};
        return std::nullopt;
    }


    std::optional<std::string> Backend::Impl::get_user_password(const std::string& username) {
        Stmt s(db, "SELECT password_hash FROM users WHERE username = ?");
        s.bind(1, username);
        if (s.step()) return s.column_text_opt(0);
        return std::nullopt;
    }


    void Backend::Impl::set_user_password(const std::string& username, const std::string& pw_hash) {
        Stmt s(db, "UPDATE users SET password_hash = ? WHERE username = ?");
        s.bind(1, pw_hash).bind(2, username).step();
    }


    int Backend::Impl::create_user(const std::string& username, const std::string& token_hash) {
        int64_t ts = db_epoch();
        Stmt s(db,
            "INSERT INTO users (username, token_hash, created_at, updated_at) VALUES (?,?,?,?)");
        s.bind(1, username).bind(2, token_hash).bind(3, ts).bind(4, ts).step();
        return sqlite3_changes(db) > 0
            ? static_cast<int>(sqlite3_last_insert_rowid(db))
            : -1;
    }


    bool Backend::Impl::delete_user(const std::string& username) {
        Stmt s(db, "DELETE FROM users WHERE username = ?");
        s.bind(1, username).step();
        return sqlite3_changes(db) > 0;
    }


    void Backend::Impl::update_user_token(const std::string& username, const std::string& new_hash) {
        Stmt s(db, "UPDATE users SET token_hash = ? WHERE username = ?");
        s.bind(1, new_hash).bind(2, username).step();
    }


    /// The semqlite_config key that now holds a `settings` key's value, or
    /// nullptr when the setting still lives in `settings`.
    static const char* semqlite_key_for(const std::string& setting_key) {
        for (const auto& m : kSemqliteSettingKeys)
            if (setting_key == m.setting) return m.semqlite;
        return nullptr;
    }


    std::optional<std::string> Backend::Impl::get_semqlite_config(const std::string& key) {
        // A read-only connection (export) never ran create_user_schema().
        if (!table_exists("semqlite_config")) return std::nullopt;
        Stmt s(db, "SELECT value FROM semqlite_config WHERE key = ?");
        s.bind(1, key);
        if (s.step()) return s.column_text_opt(0);
        return std::nullopt;
    }


    void Backend::Impl::set_semqlite_config(const std::string& key, const std::string& value) {
        Stmt s(db, "INSERT OR REPLACE INTO semqlite_config (key, value) VALUES (?, ?)");
        s.bind(1, key).bind(2, value).step();
    }


    std::optional<std::string> Backend::Impl::get_setting(const std::string& key) {
        // The embedding identity stamps live in semqlite_config now. Fall back
        // to a leftover `settings` row only for a connection that never ran the
        // move (read-only export of a not-yet-opened older file).
        if (const char* sk = semqlite_key_for(key)) {
            if (auto v = get_semqlite_config(sk)) return v;
        }
        Stmt s(db, "SELECT value FROM settings WHERE key = ?");
        s.bind(1, key);
        if (s.step()) return s.column_text_opt(0);
        return std::nullopt;
    }


    void Backend::Impl::set_setting(const std::string& key, const std::string& value) {
        if (const char* sk = semqlite_key_for(key)) {
            set_semqlite_config(sk, value);
            return;
        }
        Stmt s(db, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)");
        s.bind(1, key).bind(2, value).step();
    }


    // ---- Schema introspection (gap closure) --------------------------------

    std::vector<SchemaObject> Backend::Impl::list_schema_objects() {
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


    std::vector<std::string> Backend::Impl::table_column_names(const std::string& table) {
        std::vector<std::string> names;
        Stmt s(db, "PRAGMA table_info(" + table + ")");
        while (s.step_checked()) {
            names.push_back(s.column_text(1));  // col 1 = column name
        }
        return names;
    }


    int Backend::Impl::iterate_table_rows(const std::string& table,
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


    /// Convenience views: epoch ints rendered as local datetime, and embedding
    /// BLOBs collapsed to a has_embedding flag. One per table that holds a
    /// date integer. These exist purely for human inspection (semqlite, GUI
    /// browsers) -- no application code reads them.
    ///
    /// DROP-then-CREATE, deliberately NOT "CREATE VIEW IF NOT EXISTS". A view
    /// holds no data, so recreating is free -- and IF NOT EXISTS meant an
    /// existing DB kept a stale view definition FOREVER after this DDL changed
    /// (exactly how the v0.16 views were left still selecting the retired
    /// `phon` column). Unconditional recreation keeps every DB's views in
    /// lockstep with this code on every open.
    void Backend::Impl::create_views() {
        // Column order mirrors the base tables: narrow, readable columns first,
        // wide/derived ones last, so `SELECT *` stays legible on one screen.
        auto view = [&](const std::string& name, const std::string& body) {
            exec(std::format("DROP VIEW IF EXISTS {}", name));
            exec(std::format("CREATE VIEW {} AS {}", name, body));
        };

        // PHASE9: hashes are deliberately NOT exposed — these views exist
        // for human inspection (semqlite), and a view that prints a
        // token/password hash turns any casual read of the DB into a
        // credential leak.
        view("users_view", R"(
            SELECT id, username,
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
                   datetime(started_on, 'unixepoch', 'localtime') AS started_on,
                   datetime(ended_on, 'unixepoch', 'localtime') AS ended_on,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM sessions
        )");
        view("turns_view", R"(
            SELECT turn_id, user_text, assistant_text, model_id, session_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turns
        )");
        // PHASE5: gains episode_id. The view shows ALL rows including ''
        // marker rows (the raw table is the admin truth); Phase 9 decides
        // whether a filtered `_live_view` joins the set.
        view("turn_summaries_view", R"(
            SELECT
                turn_summary_id, text, turn_id, session_id, episode_id,
                turn_model_id, summary_model_id,
                datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
                datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
                unigram_count, bigram_count,
                embedding_version,
                CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turn_summaries
        )");
        view("episodes_view", R"(
            SELECT episode_id, text, tags, session_id, model_id, supersedes,
                   datetime(span_start, 'unixepoch', 'localtime') AS span_start,
                   datetime(span_end, 'unixepoch', 'localtime') AS span_end,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM episodes
        )");
        // the old summaries view is gone with its table (PHASE8); knowledge_view
        // covers notes now.
        // knowledge_view (PHASE7): replaces decisions_view + insights_view.
        view("knowledge_view", R"(
            SELECT knowledge_id, kind, text, status, tags,
                   trigger, rationale, scope, supersedes, session_id, model_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM knowledge
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

        // Retired: the evidence passthrough views and the <t>_terms junction
        // views (term_id resolved to the term). Only the views that render a
        // date integer as a date string remain; anything else is a join the
        // inspector can write. They are dropped by NAME, never "every view not
        // listed above", so a view a user made in their own copy survives.
        for (const char* name : {"knowledge_turn_evidence_view",
                                 "knowledge_document_evidence_view"})
            exec(std::format("DROP VIEW IF EXISTS {}", name));
        for (const auto& t : kTextTables)
            exec(std::format("DROP VIEW IF EXISTS {}_view", t.junction));
    }


} // namespace ragger::sqlite
