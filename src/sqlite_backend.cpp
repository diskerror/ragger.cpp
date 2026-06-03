/**
 * SQLite backend for Ragger Memory (C++ port)
 */
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include "ragger/util/fs.h"
#include "ragger/util/time.h"
#include "ragger/util/sqlite.h"
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
// (Embedding blobs are written by Impl::bind_embedding, which honours the
//  configured storage dtype — f16 or f32. SQLITE_TRANSIENT makes sqlite copy
//  immediately, so the temporary buffer is safe.)

// -----------------------------------------------------------------------
// Impl
// -----------------------------------------------------------------------
struct SqliteBackend::Impl {
    sqlite3*    db       = nullptr;
    Embedder*   embedder = nullptr;    // nullable — null for DB-only (user mgmt) mode
    std::string db_path;
    bool        store_f16_ = true;     // on-disk vector dtype (config vector_type)

    // Bind a float vector as the embedding blob in the configured storage
    // dtype: f16 (2 bytes/dim, default) or f32 (4 bytes/dim). The decode side
    // (ensure_cache) distinguishes them by blob size, so the dtype is
    // self-describing on read.
    void bind_embedding(sqlite3_stmt* s, int idx,
                        const std::vector<float>& emb) const {
        if (store_f16_) {
            std::vector<uint16_t> h(emb.size());
            for (size_t i = 0; i < emb.size(); ++i) h[i] = f32_to_f16(emb[i]);
            sqlite3_bind_blob(s, idx, h.data(),
                              static_cast<int>(h.size() * sizeof(uint16_t)),
                              SQLITE_TRANSIENT);
        } else {
            sqlite3_bind_blob(s, idx, emb.data(),
                              static_cast<int>(emb.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
        }
    }

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
        store_f16_ = (cfg.embedding_vector_type != "f32");

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
        // Only ensure users + session tables exist (skip memory tables/FTS).
        create_user_schema();
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

    /// True if `table` has a column named `col`. Used to guard one-time
    /// ADD COLUMN migrations (SQLite has no ADD COLUMN IF NOT EXISTS).
    /// The table name is inlined (PRAGMA table_info doesn't take a bound
    /// parameter reliably); callers pass internal constants, never user input.
    bool column_exists(const std::string& table, const std::string& col) {
        Stmt s(db, "PRAGMA table_info(" + table + ")");
        while (s.step()) {
            if (s.column_text(1) == col) return true;  // col 1 = column name
        }
        return false;
    }

    void create_schema() {
        // Users, settings, and session tables — credentials (for read-only
        // document access) plus web/chat session persistence. These are a
        // separate concern from the v2 memory tables ("out of scope" in
        // scripts/schema_v2_fading_memory.sql); one declarative definition,
        // shared with the DB-only constructor.
        create_user_schema();

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

        // sessions — lookup table; turns + summaries reference it. Normalizes
        // the long conversation GUID (from the agent turn hook) to a compact
        // integer id, mirroring `models`. Grouping key for session summaries.
        exec(R"(
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid       TEXT NOT NULL UNIQUE
            )
        )");

        // turns (L1) — raw verbatim exchanges. embedding nullable for the
        // deferred-embedding path (partial row written, backfilled later).
        exec(R"(
            CREATE TABLE IF NOT EXISTS turns (
                turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text      TEXT NOT NULL,
                assistant_text TEXT,
                embedding      BLOB,
                model_id       INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id     INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
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
                text       TEXT NOT NULL,
                embedding  BLOB,
                level      TEXT NOT NULL,
                model_id   INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                status     TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                timestamp  TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level     ON summaries(level)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_status    ON summaries(status)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_timestamp ON summaries(timestamp)");

        // In-place migration for pre-sessions databases: add the session_id
        // FK column to turns/summaries if an existing DB predates it. SQLite
        // has no ADD COLUMN IF NOT EXISTS, so guard on pragma_table_info.
        // (Fresh DBs already have the column from the CREATE above.)
        if (!column_exists("turns", "session_id"))
            exec("ALTER TABLE turns ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        if (!column_exists("summaries", "session_id"))
            exec("ALTER TABLE summaries ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_session     ON turns(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id)");

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

        // FTS5 — external-content virtual tables + sync triggers replace
        // the old hand-rolled bm25_* sidecars (issue #49).
        create_fts_schema();
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

    /// Users + settings tables — declarative, no in-place migration
    /// (single-user app; pre-v2 data is exported out-of-band). `users` mirrors
    /// the reference DDL (scripts/schema_v2_fading_memory.sql) plus
    /// `password_hash` for credentialed access. Shared by both constructors.
    void create_user_schema() {
        exec(R"(
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                token_hash    TEXT NOT NULL,
                password_hash TEXT,
                created       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                modified      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
            )
        )");
        exec(R"(
            CREATE TRIGGER IF NOT EXISTS users_modified
            AFTER UPDATE ON users
            BEGIN
                UPDATE users SET modified = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
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

    // ---- path normalization -------------------------------------------
    static std::string normalize_path(const std::string& text) {
        if (!config().normalize_home_path) return text;
        std::string home = home_dir();
        if (home.empty()) return text;
        std::string prefix = home + "/";
        std::string out = text;
        size_t pos = 0;
        while ((pos = out.find(prefix, pos)) != std::string::npos) {
            out.replace(pos, prefix.size(), "~/");
            pos += 2;
        }
        return out;
    }

    // ---- local timestamp ("%F %T" == "YYYY-MM-DD HH:MM:SS") ----------
    // Thin aliases over the shared formatter (ragger/util/time.h) so every
    // DB timestamp uses one local-time format.
    static std::string local_timestamp(std::time_t tt) { return db_timestamp(tt); }
    static std::string local_timestamp() { return db_timestamp(); }

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

        Stmt s(db,
            "SELECT summary_id, text, embedding, level, status, tags, timestamp "
            "FROM summaries");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string {
                const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s.raw(), i));
                return p ? p : "";
            };
            cached_ids.push_back(s.column_int(0));
            cached_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            // Self-describing by size: f16 blobs are dims*2 bytes, f32 are
            // dims*4. Decode whichever the row actually holds, so an f16 and an
            // f32 DB both read correctly (and a mixed/legacy DB degrades
            // per-row). NULL / wrong-sized (e.g. a deferred row before
            // backfill) → zero vector: cosine yields 0, FTS5 still matches text.
            std::vector<float> emb(expected_dims, 0.0f);
            if (blob != nullptr &&
                blob_bytes == expected_dims * static_cast<int>(sizeof(uint16_t))) {
                const uint16_t* h = static_cast<const uint16_t*>(blob);
                for (int i = 0; i < expected_dims; ++i) emb[i] = f16_to_f32(h[i]);
            } else if (blob != nullptr &&
                       blob_bytes == expected_dims * static_cast<int>(sizeof(float))) {
                const float* f = static_cast<const float*>(blob);
                for (int i = 0; i < expected_dims; ++i) emb[i] = f[i];
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
        Stmt s(db,
                "SELECT rowid, bm25(summaries_fts) FROM summaries_fts "
                "WHERE summaries_fts MATCH ?");
        s.bind(1, match_expr);
        while (s.step()) {
            int id   = s.column_int(0);
            double val = s.column_double(1);
            out[id]  = static_cast<float>(-val);
        }
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
        Stmt s(db,
            "INSERT INTO summaries (text, embedding, level, status, tags, timestamp) "
            "VALUES (?,?,?,?,?,?)");

        s.bind(1, text);
        if (defer_embedding) {
            s.bind_null(2);
        } else {
            bind_embedding(s.raw(), 2, emb);
        }
        s.bind(3, level).bind(4, status).bind(5, tags_str).bind(6, ts);

        if (!s.exec()) {
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
        Stmt s(db,
            "INSERT INTO documents "
            "(text, embedding, path, title, tags, year, chunk_index, imported_at) "
            "VALUES (?,?,?,?,?,?,?,?)");

        auto bind_opt = [&](int idx, const std::string& v) {
            if (v.empty()) s.bind_null(idx);
            else s.bind(idx, v);
        };

        s.bind(1, text);
        if (defer_embedding) {
            s.bind_null(2);
        } else {
            bind_embedding(s.raw(), 2, emb);
        }
        bind_opt(3, chunk.path);
        bind_opt(4, chunk.title);
        s.bind(5, chunk.tags);
        if (chunk.year <= 0) s.bind_null(6);
        else s.bind(6, chunk.year);
        if (chunk.chunk_index <= 0) s.bind_null(7);
        else s.bind(7, chunk.chunk_index);
        s.bind(8, imported_at);

        if (!s.exec()) {
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }

        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // ---- turns (L1) raw exchange capture ------------------------------
    // Resolve a model name to its models.model_id, creating the row if it
    // doesn't exist. Empty name → 0 (callers bind NULL).
    int get_or_create_model(const std::string& name) {
        if (name.empty()) return 0;
        Stmt s(db, "SELECT model_id FROM models WHERE name = ?");
        s.bind(1, name);
        int id = 0;
        if (s.step()) id = s.column_int(0);
        if (id) return id;
        Stmt ins(db, "INSERT INTO models (name) VALUES (?)");
        ins.bind(1, name);
        ins.step();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    /// Resolve a session GUID to its compact integer id, inserting on first
    /// sighting. Empty guid → 0 (NULL session_id), mirroring get_or_create_model.
    int get_or_create_session(const std::string& guid) {
        if (guid.empty()) return 0;
        Stmt s(db, "SELECT session_id FROM sessions WHERE guid = ?");
        s.bind(1, guid);
        int id = 0;
        if (s.step()) id = s.column_int(0);
        if (id) return id;
        Stmt ins(db, "INSERT INTO sessions (guid) VALUES (?)");
        ins.bind(1, guid);
        ins.step();
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
                   const std::string& model_name, bool defer_embedding,
                   const std::string& session_guid) {
        std::string u = normalize_path(user_text);
        std::string a = normalize_path(assistant_text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);

        std::vector<float> emb;
        bool have_emb = false;
        if (!defer_embedding && !a.empty()) {
            emb = embedder->encode(turn_embed_text(u, a));
            have_emb = true;
        }

        Stmt s(db,
            "INSERT INTO turns (model_id, session_id, user_text, assistant_text, embedding, timestamp) "
            "VALUES (?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, u);
        if (a.empty()) s.bind_null(4);
        else s.bind(4, a);
        if (have_emb)
            bind_embedding(s.raw(), 5, emb);
        else s.bind_null(5);
        s.bind(6, local_timestamp());

        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    /// All turns belonging to a session GUID, oldest first. Empty if the
    /// session is unknown. Powers session-scoped summarization/recipes.
    std::vector<TurnRecord> turns_by_session(const std::string& session_guid) {
        std::vector<TurnRecord> out;
        if (session_guid.empty()) return out;
        Stmt s(db,
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, t.timestamp "
            "FROM turns t "
            "JOIN sessions ss ON t.session_id = ss.session_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "WHERE ss.guid = ? ORDER BY t.turn_id ASC");
        s.bind(1, session_guid);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3),
                s.column_text(4)
            });
        }
        return out;
    }

    // Finalize a partial turn: set assistant_text, (re)embed the exchange,
    // and record the model. Returns false if the turn doesn't exist.
    bool finalize_turn(int turn_id, const std::string& assistant_text,
                       const std::string& model_name) {
        Stmt g(db, "SELECT user_text FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        std::string u;
        bool found = false;
        if (g.step()) {
            found = true;
            u = g.column_text(0);
        }
        if (!found) return false;

        std::string a = normalize_path(assistant_text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(turn_embed_text(u, a));

        Stmt s(db,
            "UPDATE turns SET assistant_text = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE turn_id = ?");
        s.bind(1, a);
        bind_embedding(s.raw(), 2, emb);
        if (model_id) s.bind(3, model_id); else s.bind_null(3);
        s.bind(4, turn_id);
        return s.exec();
    }

    // ---- summaries (L2/L3) pipeline primitives (issue #22) ------------
    // Insert a summary row. level: 'turn' (L2) | 'session' (L3) | 'project'.
    // status: 'current' (running L3) | 'complete'. Embeds text, records model.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& status, const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, text, embedding, level, status, tags, timestamp) "
            "VALUES (?,?,?,?,?,'',?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        s.bind(2, t);
        bind_embedding(s.raw(), 3, emb);
        s.bind(4, level).bind(5, status).bind(6, local_timestamp());
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // The current running L3 session summary, if any: (summary_id, text).
    std::optional<std::pair<int, std::string>> current_session_summary() {
        Stmt s(db,
            "SELECT summary_id, text FROM summaries "
            "WHERE level='session' AND status='current' "
            "ORDER BY summary_id DESC LIMIT 1");
        std::optional<std::pair<int, std::string>> out;
        if (s.step()) {
            int id = s.column_int(0);
            out = std::make_pair(id, s.column_text(1));
        }
        return out;
    }

    // Recipe ingredients (issue #23): recent summaries of a given level, and
    // current decisions — fetched by recency (not semantic search) for the
    // default tiered payload. Returned newest-first.
    std::vector<std::string> recent_summaries(const std::string& level, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM summaries WHERE level = ? "
            "ORDER BY timestamp DESC, summary_id DESC LIMIT ?");
        s.bind(1, level).bind(2, limit);
        while (s.step()) {
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s.raw(), 0));
            if (p) out.emplace_back(p);
        }
        return out;
    }

    std::vector<std::string> current_decisions(int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM decisions WHERE status = 'current' "
            "ORDER BY timestamp DESC, decision_id DESC LIMIT ?");
        s.bind(1, limit);
        while (s.step()) {
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(s.raw(), 0));
            if (p) out.emplace_back(p);
        }
        return out;
    }

    // Replace a summary's text + embedding, update its model. False if absent.
    bool update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "UPDATE summaries SET text = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE summary_id = ?");
        s.bind(1, t);
        bind_embedding(s.raw(), 2, emb);
        if (model_id) s.bind(3, model_id); else s.bind_null(3);
        s.bind(4, summary_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_cache();
        return ok;
    }

    // Set a summary's status (e.g. mark a session summary 'complete').
    bool set_summary_status(int summary_id, const std::string& status) {
        Stmt s(db,
            "UPDATE summaries SET status = ? WHERE summary_id = ?");
        s.bind(1, status).bind(2, summary_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }

    bool update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        // Refuse to mutate protected rows for parity with delete_memory.
        Stmt check_stmt(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        check_stmt.bind(1, memory_id);
        bool exists = false;
        bool protected_row = false;
        if (check_stmt.step()) {
            exists = true;
            const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt.raw(), 0));
            if (tags && std::string(tags).find("keep") != std::string::npos) {
                protected_row = true;
            }
        }
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
        Stmt stmt(db,
            "UPDATE summaries SET text = ?, embedding = ?, tags = ? "
            "WHERE summary_id = ?");

        stmt.bind(1, text);
        if (defer_embedding) {
            stmt.bind_null(2);
        } else {
            bind_embedding(stmt.raw(), 2, emb);
        }
        stmt.bind(3, tags_str);
        stmt.bind(4, memory_id);

        if (!stmt.exec()) return false;

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
        Stmt s(db, "SELECT COUNT(*) FROM summaries");
        int c = 0;
        if (s.step())
            c = s.column_int(0);
        return c;
    }

    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). Returns every summary, score 0.
    std::vector<SearchResult> load_all(const std::string& /*collection*/) {
        std::vector<SearchResult> results;
        Stmt s(db,
            "SELECT summary_id, text, level, status, tags, timestamp "
            "FROM summaries ORDER BY summary_id");

        while (s.step()) {
            auto col = [&](int i) -> const char* {
                return reinterpret_cast<const char*>(sqlite3_column_text(s.raw(), i));
            };
            int id = s.column_int(0);
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
        return results;
    }

    int rebuild_embeddings(Embedder& emb_ref) {
        struct TableSpec {
            const char* table;
            const char* id_col;
            const char* text_col;
            const char* extra_col;  // if set, combined with text_col via turn_embed_text
        };
        static constexpr TableSpec tables[] = {
            { "turns",     "turn_id",     "user_text", "assistant_text" },
            { "summaries", "summary_id",  "text",      nullptr          },
            { "decisions", "decision_id", "text",      nullptr          },
            { "documents", "document_id", "text",      nullptr          },
        };

        int total_count = 0;
        for (auto& t : tables) {
            Stmt s(db, std::format("SELECT COUNT(*) FROM {}", t.table));
            if (s.step())
                total_count += s.column_int(0);
        }

        int doc_count = 0;
        for (auto& t : tables) {
            std::string sel = t.extra_col
                ? std::format("SELECT {} AS id, {}, {} FROM {}", t.id_col, t.text_col, t.extra_col, t.table)
                : std::format("SELECT {} AS id, {} AS text FROM {}", t.id_col, t.text_col, t.table);
            std::string upd = std::format("UPDATE {} SET embedding = ? WHERE {} = ?", t.table, t.id_col);

            Stmt select_stmt(db, sel);

            while (select_stmt.step()) {
                int id = select_stmt.column_int(0);
                const char* col1 = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt.raw(), 1));
                if (!col1) continue;

                std::string embed_text;
                if (t.extra_col) {
                    const char* col2 = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt.raw(), 2));
                    embed_text = turn_embed_text(col1, col2 ? col2 : "");
                } else {
                    embed_text = col1;
                }

                auto emb = emb_ref.encode(embed_text);
                Stmt update(db, upd);
                bind_embedding(update.raw(), 1, emb);
                update.bind(2, id);
                update.step();

                ++doc_count;
                std::cout << std::format(ragger::lang::MSG_REBUILD_EMBEDDINGS_PROGRESS,
                                         doc_count, total_count);
                std::cout.flush();
            }
        }

        std::cout << "\n";
        invalidate_cache();
        return doc_count;
    }

    int backfill_embeddings(Embedder& emb_ref) {
        Stmt select_stmt(db,
            "SELECT summary_id AS id, text FROM summaries WHERE embedding IS NULL");

        int updated = 0;
        while (select_stmt.step()) {
            int id = select_stmt.column_int(0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt.raw(), 1));
            if (!text) continue;

            auto emb = emb_ref.encode(text);
            Stmt update(db, "UPDATE summaries SET embedding = ? WHERE summary_id = ?");
            bind_embedding(update.raw(), 1, emb);
            update.bind(2, id);
            update.step();
            ++updated;
        }

        if (updated > 0) invalidate_cache();
        return updated;
    }

    // Set a document's embedding (used by the import path after embedding
    // chunks via the subprocess executor). Returns true on a row update.
    bool update_document_embedding(int document_id, const std::vector<float>& emb) {
        Stmt s(db,
            "UPDATE documents SET embedding = ? WHERE document_id = ?");
        bind_embedding(s.raw(), 1, emb);
        s.bind(2, document_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
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
        Stmt check_stmt(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        check_stmt.bind(1, memory_id);
        if (check_stmt.step()) {
            const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt.raw(), 0));
            if (tags && std::string(tags).find("keep") != std::string::npos) {
                return false;  // protected
            }
        }

        // Not protected, proceed with deletion
        Stmt stmt(db, "DELETE FROM summaries WHERE summary_id = ?");
        stmt.bind(1, memory_id);
        stmt.step();
        int changes = sqlite3_changes(db);

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
            Stmt check_stmt(db, "SELECT tags FROM summaries WHERE summary_id = ?");
            check_stmt.bind(1, memory_id);
            bool has_keep = false;
            if (check_stmt.step()) {
                const char* tags = reinterpret_cast<const char*>(sqlite3_column_text(check_stmt.raw(), 0));
                if (tags && std::string(tags).find("keep") != std::string::npos) {
                    has_keep = true;
                }
            }

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

        Stmt stmt(db, sql);
        for (size_t i = 0; i < deletable_ids.size(); ++i) {
            stmt.bind(static_cast<int>(i + 1), deletable_ids[i]);
        }
        stmt.step();
        int changes = sqlite3_changes(db);

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

        Stmt stmt(db, sql);
        for (size_t i = 0; i < binds.size(); ++i) {
            stmt.bind(static_cast<int>(i + 1), binds[i]);
        }

        while (stmt.step()) {
            auto col = [&](int i) -> const char* {
                return reinterpret_cast<const char*>(sqlite3_column_text(stmt.raw(), i));
            };
            int id = stmt.column_int(0);
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
                              const std::string& model_name, bool defer_embedding,
                              const std::string& session_guid) {
    return pImpl->store_turn(user_text, assistant_text, model_name,
                             defer_embedding, session_guid);
}

std::vector<TurnRecord> SqliteBackend::turns_by_session(
        const std::string& session_guid) {
    return pImpl->turns_by_session(session_guid);
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
    Stmt stmt(pImpl->db,
        "DELETE FROM turns WHERE timestamp < ?");
    stmt.bind(1, cutoff_str);

    int deleted = 0;
    if (stmt.exec()) {
        deleted = static_cast<int>(sqlite3_changes(pImpl->db));
    }
    return deleted;
}

// --- User management methods (single-user mode) ---
std::optional<UserInfo> SqliteBackend::get_user_by_username(const std::string& username) {
    Stmt stmt(pImpl->db, "SELECT id, username, token_hash FROM users WHERE username = ?");
    stmt.bind(1, username);

    if (stmt.step()) {
        UserInfo user;
        user.id = stmt.column_int(0);
        user.username = stmt.column_text(1);
        user.token_hash = stmt.column_text(2);
        return user;
    }
    return std::nullopt;
}

std::optional<std::string> SqliteBackend::get_user_password(const std::string& username) {
    Stmt stmt(pImpl->db, "SELECT password_hash FROM users WHERE username = ?");
    stmt.bind(1, username);

    if (stmt.step()) {
        std::string result = stmt.column_text(0);
        return !result.empty() ? std::make_optional(result) : std::nullopt;
    }
    return std::nullopt;
}

void SqliteBackend::update_user_token(const std::string& username, const std::string& new_hash) {
    Stmt stmt(pImpl->db, "UPDATE users SET token_hash = ? WHERE username = ?");
    stmt.bind(1, new_hash).bind(2, username);
    stmt.step();
}

int SqliteBackend::create_user(const std::string& username, const std::string& token_hash) {
    std::string timestamp = pImpl->local_timestamp();
    Stmt s(pImpl->db,
           "INSERT INTO users (username, token_hash, created, modified) VALUES (?,?,?,?)");
    s.bind(1, username).bind(2, token_hash).bind(3, timestamp).bind(4, timestamp);
    s.step();
    // changes()==0 means the INSERT failed (e.g. duplicate username) — keep
    // the original -1 contract rather than returning a stale rowid.
    return sqlite3_changes(pImpl->db) > 0
        ? static_cast<int>(sqlite3_last_insert_rowid(pImpl->db))
        : -1;
}

bool SqliteBackend::delete_user(const std::string& username) {
    Stmt s(pImpl->db, "DELETE FROM users WHERE username = ?");
    s.bind(1, username);
    s.step();
    return sqlite3_changes(pImpl->db) > 0;
}

void SqliteBackend::set_user_password(const std::string& username, const std::string& password_hash) {
    Stmt s(pImpl->db, "UPDATE users SET password_hash = ? WHERE username = ?");
    s.bind(1, password_hash).bind(2, username).step();
}

std::optional<std::string> SqliteBackend::get_setting(const std::string& key) {
    Stmt s(pImpl->db, "SELECT value FROM settings WHERE key = ?");
    s.bind(1, key);
    if (s.step()) return s.column_text_opt(0);
    return std::nullopt;
}

void SqliteBackend::set_setting(const std::string& key, const std::string& value) {
    Stmt s(pImpl->db, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)");
    s.bind(1, key).bind(2, value).step();
}

std::optional<UserInfo> SqliteBackend::get_user_by_token_hash(const std::string& token_hash) {
    Stmt s(pImpl->db, "SELECT id, username, token_hash FROM users WHERE token_hash = ?");
    s.bind(1, token_hash);
    if (s.step()) {
        UserInfo user;
        user.id = s.column_int(0);
        user.username = s.column_text(1);
        user.token_hash = s.column_text(2);
        return user;
    }
    return std::nullopt;
}

} // namespace ragger
