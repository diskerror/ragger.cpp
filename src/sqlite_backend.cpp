/**
 * SQLite backend for Ragger Memory (C++ port)
 */
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "ragger/lang.h"
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
#include <iomanip>
#include <filesystem>
#include <numeric>
#include <set>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ragger {

using json = nlohmann::json;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// IEEE half-precision (f16) helpers — embeddings are stored as f16 to
// halve blob size; all in-memory math stays float32.
// _Float16 is native on Apple Silicon.
// -----------------------------------------------------------------------
static inline uint16_t f32_to_f16(float f) {
    _Float16 h = static_cast<_Float16>(f);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}
static inline float f16_to_f32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}
// Convert a float vector to f16 and bind it as the embedding blob.
// SQLITE_TRANSIENT makes sqlite copy immediately, so the temporary is safe.
static inline void bind_embedding(sqlite3_stmt* s, int idx,
                                  const std::vector<float>& emb) {
    std::vector<uint16_t> h(emb.size());
    for (size_t i = 0; i < emb.size(); ++i) h[i] = f32_to_f16(emb[i]);
    sqlite3_bind_blob(s, idx, h.data(),
                      static_cast<int>(h.size() * sizeof(uint16_t)),
                      SQLITE_TRANSIENT);
}

// -----------------------------------------------------------------------
// Impl
// -----------------------------------------------------------------------
struct SqliteBackend::Impl {
    sqlite3*    db       = nullptr;
    Embedder*   embedder = nullptr;    // nullable — null for DB-only (user mgmt) mode
    std::string db_path;

    // Embedding cache for the summaries table — invalidated on writes.
    // Vector scores come from here; keyword scores come from FTS5
    // (summaries_fts) at query time. The two are blended in search().
    bool                           cache_valid = false;
    std::vector<int>               cached_ids;
    std::vector<std::string>       cached_texts;
    Eigen::MatrixXf                cached_embeddings;   // rows × 384
    std::vector<json>              cached_metadata;
    std::vector<std::string>       cached_timestamps;

    Impl(Embedder& emb, const std::string& path)
        : embedder(&emb)
    {
        const auto& cfg = config();
        db_path = path.empty() ? cfg.resolved_db_path() : expand_path(path);

        // Create parent dirs
        fs::create_directories(fs::path(db_path).parent_path());

        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }

        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys = ON");
        create_schema();
    }

    /// DB-only constructor — no embedder, only user management ops work.
    explicit Impl(const std::string& path)
        : embedder(nullptr)
    {
        db_path = expand_path(path);
        fs::create_directories(fs::path(db_path).parent_path());

        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }

        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys = ON");
        // Only ensure users table + user columns exist (skip memories/BM25)
        create_users_schema();
    }

    ~Impl() { close(); }

    // ---- helpers -------------------------------------------------------
    void exec(const char* sql) {
        char* errmsg = nullptr;
        int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            std::string err = errmsg ? errmsg : "unknown error";
            sqlite3_free(errmsg);
            throw std::runtime_error(std::format(lang::ERR_SQL, err));
        }
    }

    void create_schema() {
        // Users table — token-based auth
        exec(R"(
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL,
                created    TEXT NOT NULL,
                modified   TEXT NOT NULL
            )
        )");
        exec(R"(
            CREATE TRIGGER IF NOT EXISTS users_modified
            AFTER UPDATE ON users
            BEGIN
                UPDATE users SET modified = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE id = NEW.id;
            END
        )");

        // ---- v2 fading-memory schema (issue #33): turns / summaries /
        //      decisions / documents / models, with FTS5 (issue #49).
        //      See scripts/schema_v2_fading_memory.sql for the reference DDL.
        //      Pre-v2 data is exported out-of-band, not migrated in place.

        // models — lookup table; turns + summaries reference it.
        exec(R"(
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL UNIQUE
            )
        )");

        // turns (L1) — raw verbatim exchanges. embedding nullable for the
        // deferred-embedding path (partial row written, backfilled later).
        exec(R"(
            CREATE TABLE IF NOT EXISTS turns (
                turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id       INTEGER REFERENCES models(model_id),
                user_text      TEXT NOT NULL,
                assistant_text TEXT,
                embedding      BLOB,
                timestamp      TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp)");

        // summaries (L2/L3/L4) — level discriminates turn/session/project.
        // Lean per scripts/schema_v2_fading_memory.sql (the reference DDL).
        // The generic Role-1 store() also lands here (level='session',
        // status='complete'); keyword search is FTS5, not a metadata blob.
        exec(R"(
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id   INTEGER REFERENCES models(model_id),
                text       TEXT NOT NULL,
                embedding  BLOB,
                level      TEXT NOT NULL,
                status     TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                timestamp  TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level     ON summaries(level)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_status    ON summaries(status)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_timestamp ON summaries(timestamp)");

        // decisions (L6).
        exec(R"(
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                embedding   BLOB,
                status      TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '',
                timestamp   TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)");

        // documents (L5) — user-curated RAG; the only sharable table.
        // Lean per the reference DDL: title identifies the publication, tags
        // group/describe (subject), year is the publish year, path is the
        // origin. Only `text` is embedded. Keyword search via documents_fts.
        exec(R"(
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                embedding   BLOB,
                path        TEXT,
                title       TEXT,
                tags        TEXT NOT NULL DEFAULT '',
                year        INTEGER,
                chunk_index INTEGER,
                imported_at TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_documents_imported_at ON documents(imported_at)");

        exec(R"(
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        )");
        // FTS5 — external-content virtual tables + sync triggers replace
        // the old hand-rolled bm25_* sidecars (issue #49).
        create_fts_schema();

        // Schema upgrades for the users/sessions tables (idempotent).
        // The v2 memory tables are created fresh — no in-place migration.
        migrate_add_token_rotated_at();
        migrate_add_preferred_model();
        migrate_add_password_hash();
        migrate_add_web_sessions();
        migrate_add_chat_sessions();
    }

    /// FTS5 external-content virtual tables + sync triggers for the four
    /// searchable content tables (turns, summaries, decisions, documents).
    /// Idempotent — safe to call on every open.
    void create_fts_schema() {
        exec(R"(CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
            user_text, assistant_text,
            content='turns', content_rowid='turn_id'))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
            INSERT INTO turns_fts(rowid, user_text, assistant_text)
            VALUES (new.turn_id, new.user_text, new.assistant_text);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
            INSERT INTO turns_fts(turns_fts, rowid, user_text, assistant_text)
            VALUES ('delete', old.turn_id, old.user_text, old.assistant_text);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
            INSERT INTO turns_fts(turns_fts, rowid, user_text, assistant_text)
            VALUES ('delete', old.turn_id, old.user_text, old.assistant_text);
            INSERT INTO turns_fts(rowid, user_text, assistant_text)
            VALUES (new.turn_id, new.user_text, new.assistant_text);
        END)");

        exec(R"(CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
            text, tags,
            content='summaries', content_rowid='summary_id'))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
            INSERT INTO summaries_fts(rowid, text, tags)
            VALUES (new.summary_id, new.text, new.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
            INSERT INTO summaries_fts(summaries_fts, rowid, text, tags)
            VALUES ('delete', old.summary_id, old.text, old.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries BEGIN
            INSERT INTO summaries_fts(summaries_fts, rowid, text, tags)
            VALUES ('delete', old.summary_id, old.text, old.tags);
            INSERT INTO summaries_fts(rowid, text, tags)
            VALUES (new.summary_id, new.text, new.tags);
        END)");

        exec(R"(CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
            text, tags,
            content='decisions', content_rowid='decision_id'))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
            INSERT INTO decisions_fts(rowid, text, tags)
            VALUES (new.decision_id, new.text, new.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions BEGIN
            INSERT INTO decisions_fts(decisions_fts, rowid, text, tags)
            VALUES ('delete', old.decision_id, old.text, old.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions BEGIN
            INSERT INTO decisions_fts(decisions_fts, rowid, text, tags)
            VALUES ('delete', old.decision_id, old.text, old.tags);
            INSERT INTO decisions_fts(rowid, text, tags)
            VALUES (new.decision_id, new.text, new.tags);
        END)");

        exec(R"(CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            title, text, tags,
            content='documents', content_rowid='document_id'))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, title, text, tags)
            VALUES (new.document_id, new.title, new.text, new.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, text, tags)
            VALUES ('delete', old.document_id, old.title, old.text, old.tags);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, text, tags)
            VALUES ('delete', old.document_id, old.title, old.text, old.tags);
            INSERT INTO documents_fts(rowid, title, text, tags)
            VALUES (new.document_id, new.title, new.text, new.tags);
        END)");
    }

    void migrate_add_token_rotated_at() {
        // Check if column exists
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "PRAGMA table_info(users)", -1, &stmt, nullptr);
        bool has_token_rotated_at = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "token_rotated_at") has_token_rotated_at = true;
        }
        sqlite3_finalize(stmt);

        if (!has_token_rotated_at) {
            exec("ALTER TABLE users ADD COLUMN token_rotated_at TEXT");
            std::cerr << ragger::lang::MSG_MIGRATE_TOKEN_ROTATED_AT << "\n";
        }
    }

    void migrate_add_preferred_model() {
        // Check if column exists
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "PRAGMA table_info(users)", -1, &stmt, nullptr);
        bool has_preferred_model = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "preferred_model") has_preferred_model = true;
        }
        sqlite3_finalize(stmt);

        if (!has_preferred_model) {
            exec("ALTER TABLE users ADD COLUMN preferred_model TEXT");
            std::cerr << ragger::lang::MSG_MIGRATE_PREFERRED_MODEL << "\n";
        }
    }

    void migrate_add_password_hash() {
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "PRAGMA table_info(users)", -1, &stmt, nullptr);
        bool has_password_hash = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "password_hash") has_password_hash = true;
        }
        sqlite3_finalize(stmt);

        if (!has_password_hash) {
            exec("ALTER TABLE users ADD COLUMN password_hash TEXT");
            std::cerr << ragger::lang::MSG_MIGRATE_PASSWORD_HASH << "\n";
        }
    }

    void migrate_add_web_sessions() {
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

    void migrate_add_chat_sessions() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                web_token  TEXT,
                username   TEXT NOT NULL,
                messages   TEXT NOT NULL,
                created    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                updated    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY (web_token) REFERENCES web_sessions(token) ON DELETE SET NULL
            )
        )");
    }

    /// Minimal schema for user management only (no memories/BM25).
    void create_users_schema() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL,
                created    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                modified   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        )");
        
        exec(R"(
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        )");
        
        migrate_add_token_rotated_at();
        migrate_add_preferred_model();
        migrate_add_password_hash();
        migrate_add_web_sessions();
        migrate_add_chat_sessions();
    }

    // ---- path normalization -------------------------------------------
    static std::string normalize_path(const std::string& text) {
        if (!config().normalize_home_path) return text;
        const char* home = std::getenv("HOME");
        if (!home) return text;
        std::string prefix = std::string(home) + "/";
        std::string out = text;
        size_t pos = 0;
        while ((pos = out.find(prefix, pos)) != std::string::npos) {
            out.replace(pos, prefix.size(), "~/");
            pos += 2;
        }
        return out;
    }

    // ---- local timestamp ("%F %T" == "YYYY-MM-DD HH:MM:SS") ----------
    static std::string local_timestamp(std::time_t tt) {
        std::tm local{};
        localtime_r(&tt, &local);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%F %T", &local);
        return std::string(buf);
    }
    static std::string local_timestamp() {
        return local_timestamp(
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    }

    // ---- cache --------------------------------------------------------
    void invalidate_cache() { cache_valid = false; }

    // Loads the summaries table into the vector cache. Keyword scores come
    // from FTS5 (summaries_fts) at query time — see keyword_scores().
    void ensure_cache() {
        if (cache_valid) return;

        cached_ids.clear();
        cached_texts.clear();
        cached_metadata.clear();
        cached_timestamps.clear();

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT summary_id, text, embedding, level, status, tags, timestamp "
            "FROM summaries",
            -1, &stmt, nullptr);

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            auto col_text = [&](int i) -> std::string {
                const char* s = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
                return s ? s : "";
            };
            cached_ids.push_back(sqlite3_column_int(stmt, 0));
            cached_texts.push_back(col_text(1));

            const void* blob = sqlite3_column_blob(stmt, 2);
            int blob_bytes   = sqlite3_column_bytes(stmt, 2);
            int n_halfs      = blob_bytes / static_cast<int>(sizeof(uint16_t));
            // NULL / wrong-sized (e.g. a legacy f32 blob, or a deferred row
            // before backfill) → zero vector. Cosine yields 0; FTS5 still
            // matches text. Old f32 DBs need `rebuild-embeddings`.
            std::vector<float> emb(expected_dims, 0.0f);
            if (n_halfs == expected_dims && blob != nullptr) {
                const uint16_t* h = static_cast<const uint16_t*>(blob);
                for (int i = 0; i < expected_dims; ++i) emb[i] = f16_to_f32(h[i]);
            }
            emb_rows.push_back(std::move(emb));

            // Lean v2 summaries: surface level/status/tags as metadata so the
            // generic API and keep-protection (via tags) have what they need.
            json meta = json::object();
            meta["level"]  = col_text(3);
            meta["status"] = col_text(4);
            std::string tags = col_text(5);
            if (!tags.empty()) meta["tags"] = tags;
            cached_metadata.push_back(std::move(meta));

            cached_timestamps.push_back(col_text(6));
        }
        sqlite3_finalize(stmt);

        // Pack into Eigen matrix (rows × dims). Every row is padded/zeroed to
        // expected_dims so NULL-embedding rows don't poison the matrix shape.
        int n = static_cast<int>(emb_rows.size());
        cached_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            cached_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
        }

        cache_valid = true;
    }

    // ---- FTS5 keyword scoring -----------------------------------------
    // Build a safe MATCH expression from arbitrary user text: extract
    // alphanumeric tokens and OR together quoted terms. Returns "" when the
    // query has no usable tokens (caller then skips the keyword pass).
    static std::string fts_match_expr(const std::string& query) {
        std::string expr, tok;
        auto flush = [&]() {
            if (tok.empty()) return;
            if (!expr.empty()) expr += " OR ";
            expr += "\"" + tok + "\"";
            tok.clear();
        };
        for (char c : query) {
            if (std::isalnum(static_cast<unsigned char>(c))) tok += c;
            else flush();
        }
        flush();
        return expr;
    }

    // bm25(summaries_fts) over a MATCH expression → summary_id → score, where
    // higher = more relevant (FTS5 bm25() is lower-is-better, so the sign is
    // flipped). Non-matching rows are absent (treated as 0 by the caller).
    std::unordered_map<int, float> keyword_scores(const std::string& match_expr) {
        std::unordered_map<int, float> out;
        if (match_expr.empty()) return out;
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db,
                "SELECT rowid, bm25(summaries_fts) FROM summaries_fts "
                "WHERE summaries_fts MATCH ?",
                -1, &stmt, nullptr) != SQLITE_OK)
            return out;   // malformed expr etc. → no keyword contribution
        sqlite3_bind_text(stmt, 1, match_expr.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id   = sqlite3_column_int(stmt, 0);
            double s = sqlite3_column_double(stmt, 1);
            out[id]  = static_cast<float>(-s);
        }
        sqlite3_finalize(stmt);
        return out;
    }

    // ---- public API ---------------------------------------------------

    // v2: the generic store API writes a summary (L2/L3/L4). A memory-only
    // install records agent memories here. `level`/`status` come from
    // metadata when supplied; defaults suit a settled session-level note.
    // collection/category and the free-form metadata blob are gone in v2 —
    // metadata fields are either promoted to columns or dropped. FTS5 sync
    // triggers index the row, so there is no explicit BM25 step.
    std::string store(const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        std::string level  = metadata.value("level",  std::string("session"));
        std::string status = metadata.value("status", std::string("complete"));

        // Optional historical timestamp override (imports of past
        // conversations). Must be an ISO-8601 UTC string; else falls to now.
        std::string ts_override;
        if (metadata.contains("timestamp") && metadata["timestamp"].is_string()) {
            ts_override = metadata["timestamp"].get<std::string>();
        }

        // tags: accept a JSON array or a plain string. `keep`/`bad` are
        // flag-tags folded into the tags column — a row tagged "keep" is
        // protected from delete/update (see delete_memory / update_text).
        std::string tags_str;
        if (metadata.contains("tags")) {
            auto& tv = metadata["tags"];
            if (tv.is_array()) {
                for (size_t i = 0; i < tv.size(); ++i) {
                    if (i > 0) tags_str += ",";
                    tags_str += tv[i].get<std::string>();
                }
            } else if (tv.is_string()) {
                tags_str = tv.get<std::string>();
            }
        }
        if (metadata.value("keep", false) &&
            tags_str.find("keep") == std::string::npos)
            tags_str += (tags_str.empty() ? "" : ",") + std::string("keep");
        if (metadata.value("bad", false) &&
            tags_str.find("bad") == std::string::npos)
            tags_str += (tags_str.empty() ? "" : ",") + std::string("bad");

        std::string text = normalize_path(raw_text);

        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        auto ts = ts_override.empty() ? local_timestamp() : ts_override;

        // FTS5 sync triggers index the row from text/tags — no manual step.
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO summaries (text, embedding, level, status, tags, timestamp) "
            "VALUES (?,?,?,?,?,?)",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        if (defer_embedding) {
            sqlite3_bind_null(stmt, 2);
        } else {
            bind_embedding(stmt, 2, emb);
        }
        sqlite3_bind_text(stmt, 3, level.c_str(),    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, status.c_str(),   -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, tags_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, ts.c_str(),       -1, SQLITE_TRANSIENT);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) {
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }

        int summary_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        invalidate_cache();
        return std::to_string(summary_id);
    }

    // ---- store_document: write Level 5 RAG chunk ----------------------
    int store_document(const DocumentChunk& chunk, bool defer_embedding) {
        // Path-normalise the body text for consistency with store().
        std::string text = normalize_path(chunk.text);

        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        std::string imported_at = chunk.imported_at.empty() ? local_timestamp() : chunk.imported_at;

        // Lean documents schema (reference DDL): path/title/tags/year/
        // chunk_index. Only `text` is embedded; documents_fts sync triggers
        // index title/text/tags — no manual keyword step.
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO documents "
            "(text, embedding, path, title, tags, year, chunk_index, imported_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            -1, &stmt, nullptr);

        auto bind_opt = [&](int idx, const std::string& v) {
            if (v.empty()) sqlite3_bind_null(stmt, idx);
            else sqlite3_bind_text(stmt, idx, v.c_str(), -1, SQLITE_TRANSIENT);
        };

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        if (defer_embedding) {
            sqlite3_bind_null(stmt, 2);
        } else {
            bind_embedding(stmt, 2, emb);
        }
        bind_opt(3, chunk.path);
        bind_opt(4, chunk.title);
        sqlite3_bind_text(stmt, 5, chunk.tags.c_str(), -1, SQLITE_TRANSIENT);
        if (chunk.year <= 0) sqlite3_bind_null(stmt, 6);
        else sqlite3_bind_int(stmt, 6, chunk.year);
        if (chunk.chunk_index <= 0) sqlite3_bind_null(stmt, 7);
        else sqlite3_bind_int(stmt, 7, chunk.chunk_index);
        sqlite3_bind_text(stmt, 8, imported_at.c_str(), -1, SQLITE_TRANSIENT);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) {
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }

        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // ---- turns (L1) raw exchange capture ------------------------------
    // Resolve a model name to its models.model_id, creating the row if it
    // doesn't exist. Empty name → 0 (callers bind NULL).
    int get_or_create_model(const std::string& name) {
        if (name.empty()) return 0;
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db, "SELECT model_id FROM models WHERE name = ?", -1, &s, nullptr);
        sqlite3_bind_text(s, 1, name.c_str(), -1, SQLITE_TRANSIENT);
        int id = 0;
        if (sqlite3_step(s) == SQLITE_ROW) id = sqlite3_column_int(s, 0);
        sqlite3_finalize(s);
        if (id) return id;
        sqlite3_prepare_v2(db, "INSERT INTO models (name) VALUES (?)", -1, &s, nullptr);
        sqlite3_bind_text(s, 1, name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(s);
        sqlite3_finalize(s);
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // Embedding for a turn is over the joined exchange (user + assistant),
    // using the same U+001F unit-separator as chat's summary turns.
    static std::string turn_embed_text(const std::string& u, const std::string& a) {
        return a.empty() ? u : (u + "\n\x1F\n" + a);
    }

    // Store a raw L1 turn. An empty assistant_text writes a *partial* row
    // (assistant + embedding NULL) for the prompt-arrival/finalize flow;
    // otherwise the exchange is embedded unless defer_embedding. FTS5 sync
    // triggers index user_text/assistant_text. Returns turn_id.
    int store_turn(const std::string& user_text, const std::string& assistant_text,
                   const std::string& model_name, bool defer_embedding) {
        std::string u = normalize_path(user_text);
        std::string a = normalize_path(assistant_text);
        int model_id  = get_or_create_model(model_name);

        std::vector<float> emb;
        bool have_emb = false;
        if (!defer_embedding && !a.empty()) {
            emb = embedder->encode(turn_embed_text(u, a));
            have_emb = true;
        }

        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO turns (model_id, user_text, assistant_text, embedding, timestamp) "
            "VALUES (?,?,?,?,?)",
            -1, &s, nullptr);
        if (model_id) sqlite3_bind_int(s, 1, model_id); else sqlite3_bind_null(s, 1);
        sqlite3_bind_text(s, 2, u.c_str(), -1, SQLITE_TRANSIENT);
        if (a.empty()) sqlite3_bind_null(s, 3);
        else sqlite3_bind_text(s, 3, a.c_str(), -1, SQLITE_TRANSIENT);
        if (have_emb)
            bind_embedding(s, 4, emb);
        else sqlite3_bind_null(s, 4);
        sqlite3_bind_text(s, 5, local_timestamp().c_str(), -1, SQLITE_TRANSIENT);

        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        if (rc != SQLITE_DONE)
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // Finalize a partial turn: set assistant_text, (re)embed the exchange,
    // and record the model. Returns false if the turn doesn't exist.
    bool finalize_turn(int turn_id, const std::string& assistant_text,
                       const std::string& model_name) {
        sqlite3_stmt* g = nullptr;
        sqlite3_prepare_v2(db, "SELECT user_text FROM turns WHERE turn_id = ?", -1, &g, nullptr);
        sqlite3_bind_int(g, 1, turn_id);
        std::string u;
        bool found = false;
        if (sqlite3_step(g) == SQLITE_ROW) {
            found = true;
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(g, 0));
            u = p ? p : "";
        }
        sqlite3_finalize(g);
        if (!found) return false;

        std::string a = normalize_path(assistant_text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(turn_embed_text(u, a));

        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE turns SET assistant_text = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE turn_id = ?",
            -1, &s, nullptr);
        sqlite3_bind_text(s, 1, a.c_str(), -1, SQLITE_TRANSIENT);
        bind_embedding(s, 2, emb);
        if (model_id) sqlite3_bind_int(s, 3, model_id); else sqlite3_bind_null(s, 3);
        sqlite3_bind_int(s, 4, turn_id);
        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        return rc == SQLITE_DONE;
    }

    // ---- summaries (L2/L3) pipeline primitives (issue #22) ------------
    // Insert a summary row. level: 'turn' (L2) | 'session' (L3) | 'project'.
    // status: 'current' (running L3) | 'complete'. Embeds text, records model.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& status, const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO summaries (model_id, text, embedding, level, status, tags, timestamp) "
            "VALUES (?,?,?,?,?,'',?)",
            -1, &s, nullptr);
        if (model_id) sqlite3_bind_int(s, 1, model_id); else sqlite3_bind_null(s, 1);
        sqlite3_bind_text(s, 2, t.c_str(), -1, SQLITE_TRANSIENT);
        bind_embedding(s, 3, emb);
        sqlite3_bind_text(s, 4, level.c_str(),  -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(s, 5, status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(s, 6, local_timestamp().c_str(), -1, SQLITE_TRANSIENT);
        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        if (rc != SQLITE_DONE)
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // The current running L3 session summary, if any: (summary_id, text).
    std::optional<std::pair<int, std::string>> current_session_summary() {
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT summary_id, text FROM summaries "
            "WHERE level='session' AND status='current' "
            "ORDER BY summary_id DESC LIMIT 1",
            -1, &s, nullptr);
        std::optional<std::pair<int, std::string>> out;
        if (sqlite3_step(s) == SQLITE_ROW) {
            int id = sqlite3_column_int(s, 0);
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s, 1));
            out = std::make_pair(id, std::string(p ? p : ""));
        }
        sqlite3_finalize(s);
        return out;
    }

    // Recipe ingredients (issue #23): recent summaries of a given level, and
    // current decisions — fetched by recency (not semantic search) for the
    // default tiered payload. Returned newest-first.
    std::vector<std::string> recent_summaries(const std::string& level, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT text FROM summaries WHERE level = ? "
            "ORDER BY timestamp DESC, summary_id DESC LIMIT ?",
            -1, &s, nullptr);
        sqlite3_bind_text(s, 1, level.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(s, 2, limit);
        while (sqlite3_step(s) == SQLITE_ROW) {
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s, 0));
            if (p) out.emplace_back(p);
        }
        sqlite3_finalize(s);
        return out;
    }

    std::vector<std::string> current_decisions(int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT text FROM decisions WHERE status = 'current' "
            "ORDER BY timestamp DESC, decision_id DESC LIMIT ?",
            -1, &s, nullptr);
        sqlite3_bind_int(s, 1, limit);
        while (sqlite3_step(s) == SQLITE_ROW) {
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s, 0));
            if (p) out.emplace_back(p);
        }
        sqlite3_finalize(s);
        return out;
    }

    // Replace a summary's text + embedding, update its model. False if absent.
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE summaries SET text = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE summary_id = ?",
            -1, &s, nullptr);
        sqlite3_bind_text(s, 1, t.c_str(), -1, SQLITE_TRANSIENT);
        bind_embedding(s, 2, emb);
        if (model_id) sqlite3_bind_int(s, 3, model_id); else sqlite3_bind_null(s, 3);
        sqlite3_bind_int(s, 4, summary_id);
        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        bool ok = (rc == SQLITE_DONE && sqlite3_changes(db) > 0);
        if (ok) invalidate_cache();
        return ok;
    }

    // Set a summary's status (e.g. mark a session summary 'complete').
    bool set_summary_status(int summary_id, const std::string& status) {
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE summaries SET status = ? WHERE summary_id = ?",
            -1, &s, nullptr);
        sqlite3_bind_text(s, 1, status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(s, 2, summary_id);
        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        return rc == SQLITE_DONE && sqlite3_changes(db) > 0;
    }

    bool update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        // Refuse to mutate protected rows for parity with delete_memory.
        sqlite3_stmt* check_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT tags FROM summaries WHERE summary_id = ?",
                           -1, &check_stmt, nullptr);
        sqlite3_bind_int(check_stmt, 1, memory_id);
        bool exists = false;
        bool protected_row = false;
        if (sqlite3_step(check_stmt) == SQLITE_ROW) {
            exists = true;
            const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt, 0));
            if (tags && std::string(tags).find("keep") != std::string::npos) {
                protected_row = true;
            }
        }
        sqlite3_finalize(check_stmt);
        if (!exists || protected_row) return false;

        // Lean v2 summaries: only text/embedding/tags are mutable here.
        // keep/bad flags fold into tags (parity with store()).
        std::string tags_str;
        if (metadata.contains("tags")) {
            auto& tv = metadata["tags"];
            if (tv.is_array()) {
                for (size_t i = 0; i < tv.size(); ++i) {
                    if (i > 0) tags_str += ",";
                    tags_str += tv[i].get<std::string>();
                }
            } else if (tv.is_string()) {
                tags_str = tv.get<std::string>();
            }
        }
        if (metadata.value("keep", false) &&
            tags_str.find("keep") == std::string::npos)
            tags_str += (tags_str.empty() ? "" : ",") + std::string("keep");
        if (metadata.value("bad", false) &&
            tags_str.find("bad") == std::string::npos)
            tags_str += (tags_str.empty() ? "" : ",") + std::string("bad");

        std::string text = normalize_path(raw_text);
        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        // FTS5 sync triggers re-index from the new text/tags automatically.
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE summaries SET text = ?, embedding = ?, tags = ? "
            "WHERE summary_id = ?",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        if (defer_embedding) {
            sqlite3_bind_null(stmt, 2);
        } else {
            bind_embedding(stmt, 2, emb);
        }
        sqlite3_bind_text(stmt, 3, tags_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 4, memory_id);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) return false;

        invalidate_cache();
        return true;
    }

    // Hybrid search over summaries: vector cosine (cached embeddings) blended
    // with FTS5 keyword relevance (bm25(summaries_fts)). Both are min-max
    // normalized and combined with vector_weight/bm25_weight. The result
    // score reported is the raw cosine; ranking uses the blended score.
    //
    // NOTE: the lean v2 summaries table has no collection column, so the
    // `collections` filter is currently a no-op (kept for API/source compat).
    // Documents (L5) are not yet merged into general search — that returns
    // with the documents-search recipe work; this function is structured so
    // adding a documents pass is a localized change.
    SearchResponse search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> /*collections*/) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        ensure_cache();
        int n = static_cast<int>(cached_ids.size());
        if (n == 0) return {{}, {{"corpus_size", 0}}};

        // ---- query embedding ------------------------------------------
        auto t_embed_start = clock::now();
        auto q_vec = embedder->encode(query);
        auto t_embed_end = clock::now();

        // ---- vector cosine --------------------------------------------
        auto t_search_start = clock::now();
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(), static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        Eigen::MatrixXf emb = cached_embeddings;
        Eigen::VectorXf norms = emb.rowwise().norm();
        for (int i = 0; i < n; ++i)
            if (norms(i) > 1e-12f) emb.row(i) /= norms(i);
        Eigen::VectorXf similarities = emb * q_norm;   // raw cosine, reported

        // ---- FTS5 keyword + hybrid blend ------------------------------
        Eigen::VectorXf combined = similarities;
        if (config().bm25_enabled) {
            auto kw = keyword_scores(fts_match_expr(query));
            if (!kw.empty()) {
                Eigen::VectorXf kw_vec(n);
                for (int i = 0; i < n; ++i) {
                    auto it = kw.find(cached_ids[i]);
                    kw_vec(i) = (it == kw.end()) ? 0.0f : it->second;
                }
                auto norm_minmax = [](Eigen::VectorXf& v) {
                    float mn = v.minCoeff(), mx = v.maxCoeff();
                    if (mx > mn) v = (v.array() - mn) / (mx - mn);
                };
                Eigen::VectorXf vec_norm = similarities, kw_norm = kw_vec;
                norm_minmax(vec_norm);
                norm_minmax(kw_norm);
                combined = config().vector_weight * vec_norm +
                           config().bm25_weight   * kw_norm;
            }
        }
        auto t_search_end = clock::now();

        // ---- top-k selection ------------------------------------------
        int top_k = std::min(limit, n);
        std::vector<int> ranking(n);
        std::iota(ranking.begin(), ranking.end(), 0);
        std::partial_sort(ranking.begin(), ranking.begin() + top_k, ranking.end(),
                          [&](int a, int b) { return combined(a) > combined(b); });

        std::vector<SearchResult> results;
        for (int k = 0; k < top_k; ++k) {
            int i = ranking[k];
            float score = similarities(i);   // report raw cosine
            if (score < min_score) continue;
            results.push_back({cached_ids[i], cached_texts[i], score,
                               cached_metadata[i], cached_timestamps[i]});
        }

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"embedding_ms", ms(t_embed_start, t_embed_end)},
            {"search_ms",    ms(t_search_start, t_search_end)},
            {"total_ms",     ms(t_start, clock::now())},
            {"corpus_size",  n}
        };
        return {std::move(results), std::move(timing)};
    }

    int count() const {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM summaries",
                           -1, &stmt, nullptr);
        int c = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            c = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        return c;
    }

    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). Returns every summary, score 0.
    std::vector<SearchResult> load_all(const std::string& /*collection*/) {
        std::vector<SearchResult> results;
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT summary_id, text, level, status, tags, timestamp "
            "FROM summaries ORDER BY summary_id",
            -1, &stmt, nullptr);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            auto col = [&](int i) -> const char* {
                return reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            };
            int id = sqlite3_column_int(stmt, 0);
            const char* text = col(1);
            const char* lvl  = col(2);
            const char* st   = col(3);
            const char* tag  = col(4);
            const char* ts   = col(5);

            json meta = json::object();
            if (lvl && lvl[0]) meta["level"]  = lvl;
            if (st  && st[0])  meta["status"] = st;
            if (tag && tag[0]) meta["tags"]   = tag;

            results.push_back({id, text ? text : "", 0.0f, std::move(meta),
                               ts ? ts : ""});
        }
        sqlite3_finalize(stmt);
        return results;
    }

    // Rebuild the FTS5 keyword indexes from the content tables. (Kept under
    // the rebuild_bm25 name for CLI/source compatibility; the underlying
    // index is now FTS5, not the hand-rolled BM25.) Returns the summary count.
    int rebuild_bm25() {
        exec("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')");
        exec("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')");
        exec("INSERT INTO decisions_fts(decisions_fts) VALUES('rebuild')");
        exec("INSERT INTO turns_fts(turns_fts) VALUES('rebuild')");
        invalidate_cache();
        return count();
    }

    int rebuild_embeddings(Embedder& emb_ref) {
        // Get total count first
        sqlite3_stmt* count_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM summaries",
                           -1, &count_stmt, nullptr);
        int total_count = 0;
        if (sqlite3_step(count_stmt) == SQLITE_ROW) {
            total_count = sqlite3_column_int(count_stmt, 0);
        }
        sqlite3_finalize(count_stmt);
        
        // Re-embed all documents
        sqlite3_stmt* select_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT summary_id AS id, text FROM summaries",
                           -1, &select_stmt, nullptr);

        sqlite3_stmt* update_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE summaries SET embedding = ? WHERE summary_id = ?",
            -1, &update_stmt, nullptr);

        int doc_count = 0;
        while (sqlite3_step(select_stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(select_stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt, 1));
            if (!text) continue;

            // Generate new embedding
            auto emb = emb_ref.encode(text);

            // Update database
            bind_embedding(update_stmt, 1, emb);
            sqlite3_bind_int(update_stmt, 2, id);
            sqlite3_step(update_stmt);
            sqlite3_reset(update_stmt);

            ++doc_count;

            // Print progress counter
            std::cout << std::format(ragger::lang::MSG_REBUILD_EMBEDDINGS_PROGRESS,
                                     doc_count, total_count);
            std::cout.flush();
        }

        sqlite3_finalize(select_stmt);
        sqlite3_finalize(update_stmt);
        
        // Print final newline
        std::cout << "\n";
        
        invalidate_cache();
        return doc_count;
    }

    int backfill_embeddings(Embedder& emb_ref) {
        sqlite3_stmt* select_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT summary_id AS id, text FROM summaries WHERE embedding IS NULL",
            -1, &select_stmt, nullptr);

        sqlite3_stmt* update_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE summaries SET embedding = ? WHERE summary_id = ?",
            -1, &update_stmt, nullptr);

        int updated = 0;
        while (sqlite3_step(select_stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(select_stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt, 1));
            if (!text) continue;

            auto emb = emb_ref.encode(text);
            bind_embedding(update_stmt, 1, emb);
            sqlite3_bind_int(update_stmt, 2, id);
            sqlite3_step(update_stmt);
            sqlite3_reset(update_stmt);
            ++updated;
        }
        sqlite3_finalize(select_stmt);
        sqlite3_finalize(update_stmt);

        if (updated > 0) invalidate_cache();
        return updated;
    }

    // Set a document's embedding (used by the import path after embedding
    // chunks via the subprocess executor). Returns true on a row update.
    bool update_document_embedding(int document_id, const std::vector<float>& emb) {
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE documents SET embedding = ? WHERE document_id = ?",
            -1, &s, nullptr);
        bind_embedding(s, 1, emb);
        sqlite3_bind_int(s, 2, document_id);
        int rc = sqlite3_step(s);
        sqlite3_finalize(s);
        if (rc == SQLITE_DONE && sqlite3_changes(db) > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }

    std::vector<std::string> collections() const {
        std::vector<std::string> result;
        // Lean v2 summaries have no collection column — collections are not a
        // v2 concept. Return empty (kept for API compat).
        return result;
    }

    bool delete_memory(int memory_id) {
        // Check if memory has keep tag in dedicated column
        sqlite3_stmt* check_stmt;
        sqlite3_prepare_v2(db, "SELECT tags FROM summaries WHERE summary_id = ?", -1, &check_stmt, nullptr);
        sqlite3_bind_int(check_stmt, 1, memory_id);
        if (sqlite3_step(check_stmt) == SQLITE_ROW) {
            const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt, 0));
            if (tags && std::string(tags).find("keep") != std::string::npos) {
                sqlite3_finalize(check_stmt);
                return false;  // protected
            }
        }
        sqlite3_finalize(check_stmt);
        
        // Not protected, proceed with deletion
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, "DELETE FROM summaries WHERE summary_id = ?",
                           -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, memory_id);
        sqlite3_step(stmt);
        int changes = sqlite3_changes(db);
        sqlite3_finalize(stmt);
        
        if (changes > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }

    int delete_batch(const std::vector<int>& memory_ids) {
        if (memory_ids.empty()) return 0;

        // Filter out IDs with keep tag
        std::vector<int> deletable_ids;
        for (int memory_id : memory_ids) {
            sqlite3_stmt* check_stmt;
            sqlite3_prepare_v2(db, "SELECT tags FROM summaries WHERE summary_id = ?", -1, &check_stmt, nullptr);
            sqlite3_bind_int(check_stmt, 1, memory_id);
            bool has_keep = false;
            if (sqlite3_step(check_stmt) == SQLITE_ROW) {
                const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt, 0));
                if (tags && std::string(tags).find("keep") != std::string::npos) {
                    has_keep = true;
                }
            }
            sqlite3_finalize(check_stmt);
            
            if (!has_keep) {
                deletable_ids.push_back(memory_id);
            }
        }
        
        if (deletable_ids.empty()) return 0;

        // Build SQL: DELETE FROM summaries WHERE summary_id IN (?,?,...)
        std::string sql = "DELETE FROM summaries WHERE summary_id IN (";
        for (size_t i = 0; i < deletable_ids.size(); ++i) {
            if (i > 0) sql += ",";
            sql += "?";
        }
        sql += ")";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        for (size_t i = 0; i < deletable_ids.size(); ++i) {
            sqlite3_bind_int(stmt, static_cast<int>(i + 1), deletable_ids[i]);
        }
        sqlite3_step(stmt);
        int changes = sqlite3_changes(db);
        sqlite3_finalize(stmt);

        if (changes > 0) {
            invalidate_cache();
        }
        return changes;
    }

    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit,
                                                 const std::string& after = "",
                                                 const std::string& before = "") {
        std::vector<SearchResult> results;

        // Lean v2 summaries expose level / status / tags / timestamp. Filter
        // on those columns; any other requested key matches nothing (there is
        // no free-form metadata blob in the lean schema).
        std::string sql = "SELECT summary_id, text, level, status, tags, timestamp "
                          "FROM summaries";
        std::string where;
        std::vector<std::string> binds;

        for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
            const std::string& k = it.key();
            if (k == "level" || k == "status") {
                where += (where.empty() ? " WHERE " : " AND ") + k + " = ?";
                binds.push_back(it.value().get<std::string>());
            } else if (k == "tags") {
                where += (where.empty() ? " WHERE " : " AND ") + std::string("tags LIKE ?");
                binds.push_back("%" + it.value().get<std::string>() + "%");
            } else {
                // Unsupported key under the lean schema → no rows match.
                return results;
            }
        }

        if (!after.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("timestamp >= ?");
            binds.push_back(after);
        }
        if (!before.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("timestamp < ?");
            binds.push_back(before);
        }

        sql += where + " ORDER BY timestamp DESC";
        if (limit > 0) sql += " LIMIT " + std::to_string(limit);

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        for (size_t i = 0; i < binds.size(); ++i) {
            sqlite3_bind_text(stmt, static_cast<int>(i + 1), binds[i].c_str(), -1, SQLITE_TRANSIENT);
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            auto col = [&](int i) -> const char* {
                return reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            };
            int id = sqlite3_column_int(stmt, 0);
            const char* text = col(1);
            const char* lvl  = col(2);
            const char* st   = col(3);
            const char* tag  = col(4);
            const char* ts   = col(5);

            json metadata = json::object();
            if (lvl && lvl[0]) metadata["level"]  = lvl;
            if (st  && st[0])  metadata["status"] = st;
            if (tag && tag[0]) metadata["tags"]   = tag;

            results.push_back({id, text ? text : "", 0.0f, std::move(metadata),
                               ts ? ts : ""});
        }
        sqlite3_finalize(stmt);
        return results;
    }

    void close() {
        if (db) {
            sqlite3_close(db);
            db = nullptr;
        }
    }
};

// -----------------------------------------------------------------------
// Public API (delegates to Impl)
// -----------------------------------------------------------------------
SqliteBackend::SqliteBackend(Embedder& embedder, const std::string& db_path)
    : pImpl(std::make_unique<Impl>(embedder, db_path)) {}

SqliteBackend::SqliteBackend(const std::string& db_path)
    : pImpl(std::make_unique<Impl>(db_path)) {}

SqliteBackend::~SqliteBackend() = default;

std::string SqliteBackend::db_path() const { return pImpl->db_path; }

std::string SqliteBackend::store(const std::string& text, json metadata, bool defer_embedding) {
    return pImpl->store(text, std::move(metadata), defer_embedding);
}

int SqliteBackend::store_document(const DocumentChunk& chunk, bool defer_embedding) {
    return pImpl->store_document(chunk, defer_embedding);
}

int SqliteBackend::store_turn(const std::string& user_text,
                              const std::string& assistant_text,
                              const std::string& model_name, bool defer_embedding) {
    return pImpl->store_turn(user_text, assistant_text, model_name, defer_embedding);
}

bool SqliteBackend::finalize_turn(int turn_id, const std::string& assistant_text,
                                  const std::string& model_name) {
    return pImpl->finalize_turn(turn_id, assistant_text, model_name);
}

int SqliteBackend::store_summary(const std::string& text, const std::string& level,
                                 const std::string& status, const std::string& model_name) {
    return pImpl->store_summary(text, level, status, model_name);
}

std::optional<std::pair<int, std::string>>
SqliteBackend::current_session_summary() {
    return pImpl->current_session_summary();
}

bool SqliteBackend::update_summary_text(int summary_id, const std::string& text,
                                        const std::string& model_name) {
    return pImpl->update_summary_text(summary_id, text, model_name);
}

bool SqliteBackend::set_summary_status(int summary_id, const std::string& status) {
    return pImpl->set_summary_status(summary_id, status);
}

std::vector<std::string> SqliteBackend::recent_summaries(const std::string& level,
                                                         int limit) {
    return pImpl->recent_summaries(level, limit);
}

std::vector<std::string> SqliteBackend::current_decisions(int limit) {
    return pImpl->current_decisions(limit);
}

bool SqliteBackend::update_text(int memory_id, const std::string& text, json metadata, bool defer_embedding) {
    return pImpl->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse SqliteBackend::search(const std::string& query, int limit,
                                     float min_score,
                                     std::vector<std::string> collections) {
    return pImpl->search(query, limit, min_score, std::move(collections));
}

int SqliteBackend::count() const { return pImpl->count(); }

std::vector<SearchResult> SqliteBackend::load_all(const std::string& collection) {
    return pImpl->load_all(collection);
}

int SqliteBackend::rebuild_bm25() { return pImpl->rebuild_bm25(); }

int SqliteBackend::rebuild_embeddings(Embedder& embedder) {
    return pImpl->rebuild_embeddings(embedder);
}

int SqliteBackend::backfill_embeddings(Embedder& embedder) {
    return pImpl->backfill_embeddings(embedder);
}

bool SqliteBackend::update_document_embedding(int document_id,
                                              const std::vector<float>& emb) {
    return pImpl->update_document_embedding(document_id, emb);
}

std::vector<std::string> SqliteBackend::collections() const {
    return pImpl->collections();
}

void SqliteBackend::close() { pImpl->close(); }

bool SqliteBackend::delete_memory(int memory_id) {
    return pImpl->delete_memory(memory_id);
}

int SqliteBackend::delete_batch(const std::vector<int>& memory_ids) {
    return pImpl->delete_batch(memory_ids);
}

std::vector<SearchResult> SqliteBackend::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    return pImpl->search_by_metadata(metadata_filter, limit, after, before);
}

// --- Chat sessions ---

void SqliteBackend::save_chat_session(const std::string& session_id, const std::string& username,
                                     const std::string& messages_json, const std::string& web_token) {
    std::string buf = pImpl->local_timestamp();

    // Check if session exists
    sqlite3_stmt* check_stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT session_id FROM chat_sessions WHERE session_id = ?",
        -1, &check_stmt, nullptr);
    sqlite3_bind_text(check_stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
    bool exists = (sqlite3_step(check_stmt) == SQLITE_ROW);
    sqlite3_finalize(check_stmt);

    sqlite3_stmt* stmt;
    if (exists) {
        // Update
        sqlite3_prepare_v2(pImpl->db,
            "UPDATE chat_sessions SET messages = ?, updated = ?, web_token = ? WHERE session_id = ?",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, messages_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, buf.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, web_token.empty() ? nullptr : web_token.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, session_id.c_str(), -1, SQLITE_TRANSIENT);
    } else {
        // Insert
        sqlite3_prepare_v2(pImpl->db,
            "INSERT INTO chat_sessions (session_id, username, messages, web_token, created, updated) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, messages_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, web_token.empty() ? nullptr : web_token.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, buf.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, buf.c_str(), -1, SQLITE_TRANSIENT);
    }
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteBackend::get_chat_session(const std::string& session_id) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT messages FROM chat_sessions WHERE session_id = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<std::string> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* msgs = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (msgs) {
            result = std::string(msgs);
        }
    }
    sqlite3_finalize(stmt);
    return result;
}

void SqliteBackend::delete_chat_session(const std::string& session_id) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM chat_sessions WHERE session_id = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

int SqliteBackend::cleanup_old_conversations(int max_age_hours) {
    auto cutoff = std::chrono::system_clock::now() -
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::duration<double, std::ratio<3600>>(max_age_hours));
    std::string cutoff_str = pImpl->local_timestamp(
        std::chrono::system_clock::to_time_t(cutoff));

    // v2: raw verbatim exchanges (L1 turns) are what age out by retention;
    // their gist is preserved in the L2/L3 summaries. Purge old turns by
    // timestamp. (Pre-v2 this deleted summaries tagged collection='conversation';
    // that column is gone in the lean schema.)
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM turns WHERE timestamp < ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, cutoff_str.c_str(), -1, SQLITE_TRANSIENT);

    int deleted = 0;
    if (sqlite3_step(stmt) == SQLITE_DONE) {
        deleted = static_cast<int>(sqlite3_changes(pImpl->db));
    }
    sqlite3_finalize(stmt);
    return deleted;
}

// --- User management methods (single-user mode) ---
std::optional<UserInfo> SqliteBackend::get_user_by_username(const std::string& username) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT id, username, token_hash FROM users WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo user;
        user.id = sqlite3_column_int(stmt, 0);
        user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        user.token_hash = hash ? std::string(hash) : "";
        sqlite3_finalize(stmt);
        return user;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::optional<std::string> SqliteBackend::get_user_password(const std::string& username) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT password_hash FROM users WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string result = hash ? std::string(hash) : "";
        sqlite3_finalize(stmt);
        return !result.empty() ? std::make_optional(result) : std::nullopt;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

void SqliteBackend::update_user_token(const std::string& username, const std::string& new_hash) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "UPDATE users SET token_hash = ? WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, new_hash.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

void SqliteBackend::create_web_session(const std::string& token, const std::string& username,
                                        int user_id, int ttl_seconds) {
    auto now = std::chrono::system_clock::now();
    std::string created = pImpl->local_timestamp();
    std::string expires = pImpl->local_timestamp(
        std::chrono::system_clock::to_time_t(now + std::chrono::seconds(ttl_seconds)));
    
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT OR REPLACE INTO web_sessions (token, username, user_id, created, expires) VALUES (?,?,?,?,?)";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, user_id);
        sqlite3_bind_text(stmt, 4, created.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, expires.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteBackend::get_user_preferred_model(const std::string& username) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT preferred_model FROM users WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* model = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        auto result = model ? std::make_optional(std::string(model)) : std::nullopt;
        sqlite3_finalize(stmt);
        return result;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

void SqliteBackend::update_user_preferred_model(const std::string& username, const std::string& model) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT OR REPLACE INTO users (username, preferred_model) VALUES (?,?)";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, model.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteBackend::get_user_token_rotated_at(const std::string& username) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT token_rotated_at FROM users WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* rotated = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        auto result = rotated ? std::make_optional(std::string(rotated)) : std::nullopt;
        sqlite3_finalize(stmt);
        return result;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

void SqliteBackend::update_user_token_rotated_at(const std::string& username, const std::string& timestamp) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "UPDATE users SET token_rotated_at = ? WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, timestamp.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

int SqliteBackend::create_user(const std::string& username, const std::string& token_hash) {
    std::string timestamp = pImpl->local_timestamp();

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT INTO users (username, token_hash, created, modified) VALUES (?,?,?,?)";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return -1;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, timestamp.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_DONE) {
        int id = static_cast<int>(sqlite3_last_insert_rowid(pImpl->db));
        sqlite3_finalize(stmt);
        return id;
    }
    sqlite3_finalize(stmt);
    return -1;
}

bool SqliteBackend::delete_user(const std::string& username) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "DELETE FROM users WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    sqlite3_step(stmt);
    int changes = sqlite3_changes(pImpl->db);
    sqlite3_finalize(stmt);
    return changes > 0;
}

void SqliteBackend::set_user_password(const std::string& username, const std::string& password_hash) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "UPDATE users SET password_hash = ? WHERE username = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, password_hash.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

std::optional<std::string> SqliteBackend::get_setting(const std::string& key) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT value FROM settings WHERE key = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    
    std::optional<std::string> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (v) result = std::string(v);
    }
    sqlite3_finalize(stmt);
    return result;
}

void SqliteBackend::set_setting(const std::string& key, const std::string& value) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }
    sqlite3_finalize(stmt);
}

std::optional<UserInfo> SqliteBackend::get_user_by_token_hash(const std::string& token_hash) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT id, username, token_hash FROM users WHERE token_hash = ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo user;
        user.id = sqlite3_column_int(stmt, 0);
        user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        user.token_hash = hash ? std::string(hash) : "";
        sqlite3_finalize(stmt);
        return user;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::optional<UserInfo> SqliteBackend::get_web_session(const std::string& token) {
    std::string now_str = pImpl->local_timestamp();

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT u.id, u.username, u.token_hash FROM users u JOIN web_sessions ws ON u.id = ws.user_id WHERE ws.token = ? AND ws.expires > ?";
    if (sqlite3_prepare_v2(pImpl->db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return std::nullopt;
    sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, now_str.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo user;
        user.id = sqlite3_column_int(stmt, 0);
        user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        user.token_hash = hash ? std::string(hash) : "";
        sqlite3_finalize(stmt);
        return user;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

} // namespace ragger
