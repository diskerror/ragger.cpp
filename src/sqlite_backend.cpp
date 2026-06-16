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
#include "diskerror/logger.h"
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
#include <mutex>
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
// Free helpers (no db access needed)
// -----------------------------------------------------------------------

// Build the tags string from a metadata JSON object. Accepts metadata["tags"]
// as a comma-separated string or a JSON array. `keep`/`bad` flag fields are
// folded in (they protect a row from deletion / mark it as low-quality).
static std::string tags_from_metadata(const json& metadata) {
    std::string tags_str;
    if (metadata.contains("tags")) {
        const auto& tv = metadata["tags"];
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
    return tags_str;
}

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

    // Parallel embedding cache for the documents (L5) table — invalidated on
    // document writes. Mirrors the summaries cache above; vector scores come
    // from here, keyword scores from FTS5 (documents_fts) at query time. Both
    // corpora are merged into a single ranked top-k by search().
    bool                           doc_cache_valid = false;
    std::vector<int>               doc_ids;
    std::vector<std::string>       doc_texts;
    Eigen::MatrixXf                doc_embeddings;      // rows × dims
    std::vector<json>              doc_metadata;
    std::vector<std::string>       doc_timestamps;

    // Parallel embedding cache for the decisions (L6) table — invalidated on
    // decision writes. Mirrors the caches above; vector scores come from here,
    // keyword scores from FTS5 (decisions_fts) at query time. All three corpora
    // (summaries + documents + decisions) are merged into one ranked top-k by
    // search().
    bool                           dec_cache_valid = false;
    std::vector<int>               dec_ids;
    std::vector<std::string>       dec_texts;
    Eigen::MatrixXf                dec_embeddings;      // rows × dims
    std::vector<json>              dec_metadata;
    std::vector<std::string>       dec_timestamps;

    // Serializes all public-API access (H1): two httplib thread pools, the
    // housekeeping timer, and the SummarizerService worker share one backend.
    // The non-thread-safe Embedder and the three embedding caches are guarded
    // by this. Locked once at the SqliteBackend:: public boundary; Impl methods
    // never re-lock, so no recursive deadlock is possible.
    mutable std::mutex mu;

    // Decode an embedding BLOB into a float vector of `dims`, choosing f16 vs
    // f32 by the blob's actual byte length (self-describing), NOT by the
    // store_f16_ config flag. This makes a DB readable even if its stored dtype
    // differs from the current config (e.g. dtype changed without a rebuild).
    //   dims*2 bytes  -> f16   |   dims*4 bytes -> f32
    // Anything else (NULL, deferred-but-unbackfilled row, corruption, or a
    // dimension mismatch) yields a zero vector; the first such row per cache
    // load is logged once as a warning (table + row id).
    std::vector<float> decode_embedding_blob(const void* blob, int blob_bytes,
                                             int dims, const char* table,
                                             int row_id, bool& warned) {
        std::vector<float> emb(dims, 0.0f);
        if (blob == nullptr) return emb;  // deferred row; silent (expected)
        if (blob_bytes == dims * static_cast<int>(sizeof(uint16_t))) {
            const uint16_t* h = static_cast<const uint16_t*>(blob);
            for (int i = 0; i < dims; ++i) emb[i] = f16_to_f32(h[i]);
        } else if (blob_bytes == dims * static_cast<int>(sizeof(float))) {
            const float* f = static_cast<const float*>(blob);
            for (int i = 0; i < dims; ++i) emb[i] = f[i];
        } else if (!warned) {
            warned = true;
            Diskerror::logger::warn(std::format(
                "[cache] {} row {}: embedding blob is {} bytes, expected {} (f16) "
                "or {} (f32); using zero vector",
                table, row_id, blob_bytes, dims * 2, dims * 4));
        }
        return emb;
    }

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
    void invalidate_doc_cache() { doc_cache_valid = false; }
    void invalidate_dec_cache() { dec_cache_valid = false; }

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
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            cached_ids.push_back(s.column_int(0));
            cached_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "summaries",
                cached_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            // Lean v2 summaries: surface level/status/tags as metadata so the
            // generic API and keep-protection (via tags) have what they need.
            json meta = json::object();
            meta["source"] = "summary";
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
            float nrm = cached_embeddings.row(i).norm();
            if (nrm > 1e-12f) cached_embeddings.row(i) /= nrm;
        }

        cache_valid = true;
    }

    // Loads the documents (L5) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (documents_fts) at query time. Metadata carries source="document" plus
    // the title so search() consumers can distinguish documents from summaries.
    void ensure_doc_cache() {
        if (doc_cache_valid) return;

        doc_ids.clear();
        doc_texts.clear();
        doc_metadata.clear();
        doc_timestamps.clear();

        Stmt s(db,
            "SELECT document_id, text, embedding, title, tags, imported_at "
            "FROM documents");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            doc_ids.push_back(s.column_int(0));
            doc_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "documents",
                doc_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "document";
            meta["title"]  = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            doc_metadata.push_back(std::move(meta));

            doc_timestamps.push_back(col_text(5));
        }

        int n = static_cast<int>(emb_rows.size());
        doc_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            doc_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = doc_embeddings.row(i).norm();
            if (nrm > 1e-12f) doc_embeddings.row(i) /= nrm;
        }

        doc_cache_valid = true;
    }

    // Loads the decisions (L6) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (decisions_fts) at query time. Metadata carries source="decision" plus
    // the status so search() consumers can distinguish decisions from the
    // other corpora.
    void ensure_dec_cache() {
        if (dec_cache_valid) return;

        dec_ids.clear();
        dec_texts.clear();
        dec_metadata.clear();
        dec_timestamps.clear();

        Stmt s(db,
            "SELECT decision_id, text, embedding, status, tags, timestamp "
            "FROM decisions");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            dec_ids.push_back(s.column_int(0));
            dec_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "decisions",
                dec_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "decision";
            meta["status"] = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            dec_metadata.push_back(std::move(meta));

            dec_timestamps.push_back(col_text(5));
        }

        int n = static_cast<int>(emb_rows.size());
        dec_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            dec_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = dec_embeddings.row(i).norm();
            if (nrm > 1e-12f) dec_embeddings.row(i) /= nrm;
        }

        dec_cache_valid = true;
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

    // bm25(documents_fts) over a MATCH expression → document_id → score.
    // Analogous to keyword_scores() but against the documents FTS5 index.
    std::unordered_map<int, float> doc_keyword_scores(const std::string& match_expr) {
        std::unordered_map<int, float> out;
        if (match_expr.empty()) return out;
        Stmt s(db,
                "SELECT rowid, bm25(documents_fts) FROM documents_fts "
                "WHERE documents_fts MATCH ?");
        s.bind(1, match_expr);
        while (s.step()) {
            int id   = s.column_int(0);
            double val = s.column_double(1);
            out[id]  = static_cast<float>(-val);
        }
        return out;
    }

    // bm25(decisions_fts) over a MATCH expression → decision_id → score.
    // Analogous to keyword_scores() but against the decisions FTS5 index.
    std::unordered_map<int, float> dec_keyword_scores(const std::string& match_expr) {
        std::unordered_map<int, float> out;
        if (match_expr.empty()) return out;
        Stmt s(db,
                "SELECT rowid, bm25(decisions_fts) FROM decisions_fts "
                "WHERE decisions_fts MATCH ?");
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
        std::string tags_str = tags_from_metadata(metadata);

        std::string text = normalize_path(raw_text);

        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        auto ts = ts_override.empty() ? local_timestamp() : ts_override;

        // Every summary should record the model that produced it. Callers pass
        // the live model via metadata["model"]; empty → 0 → NULL (the raw-turn
        // sentinel), which a non-turn summary should never be. get_or_create_model
        // interns the name in the models table.
        std::string model_name = metadata.value("model", std::string(""));
        int model_id = get_or_create_model(model_name);

        // FTS5 sync triggers index the row from text/tags — no manual step.
        Stmt s(db,
            "INSERT INTO summaries (text, embedding, level, status, tags, timestamp, model_id) "
            "VALUES (?,?,?,?,?,?,?)");

        s.bind(1, text);
        if (defer_embedding) {
            s.bind_null(2);
        } else {
            bind_embedding(s.raw(), 2, emb);
        }
        s.bind(3, level).bind(4, status).bind(5, tags_str).bind(6, ts);
        if (model_id) s.bind(7, model_id); else s.bind_null(7);

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

        invalidate_doc_cache();
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

    // Read back a turn's committed session_id/timestamp (post-COALESCE/dedup)
    // and (re)write its raw placeholder summary. Centralizes the read-back so
    // both store_turn insert paths link the placeholder to the row's *actual*
    // slot, even when dedup moved the timestamp.
    void place_turn_placeholder(int turn_id, const std::string& u,
                                const std::string& a) {
        Stmt g(db, "SELECT session_id, timestamp FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        if (!g.step()) return;
        bool sess_null = g.is_null(0);
        int  session_id = sess_null ? 0 : g.column_int(0);
        std::string ts  = g.column_text(1);
        upsert_turn_placeholder(session_id, sess_null, ts, u, a);
    }

    // Write a raw-text placeholder L2 (level='turn') row for a freshly
    // captured turn so a memory lookup surfaces it *immediately*, before the
    // background summarizer runs (general search does not read the `turns`
    // table). model_id stays NULL — that is the sentinel for "raw, not yet
    // summarized"; handle_l2 later rewrites the text in place and stamps the
    // summarizer model_id. Idempotent: removes any prior NULL-model placeholder
    // for this (session_id, ts) first, then inserts — so a regeneration that
    // moves the turn's timestamp does not strand an orphan placeholder.
    // No-op when a *summarized* row (model_id NOT NULL) already exists.
    void upsert_turn_placeholder(int session_id, bool sess_null,
                                 const std::string& ts,
                                 const std::string& u, const std::string& a) {
        if (a.empty()) return;  // partial turn: nothing to surface yet

        // Already summarized at this slot? leave it.
        {
            Stmt e(db,
                "SELECT 1 FROM summaries WHERE level='turn' AND timestamp=? "
                "AND session_id IS ? AND model_id IS NOT NULL LIMIT 1");
            e.bind(1, ts);
            if (sess_null) e.bind_null(2); else e.bind(2, session_id);
            if (e.step()) return;
        }
        // Drop any stale NULL-model placeholder for this slot (re-capture).
        {
            Stmt d(db,
                "DELETE FROM summaries WHERE level='turn' AND timestamp=? "
                "AND session_id IS ? AND model_id IS NULL");
            d.bind(1, ts);
            if (sess_null) d.bind_null(2); else d.bind(2, session_id);
            d.exec();
        }

        std::string text = normalize_path("User: " + u + "\n\nAssistant: " + a);
        auto emb = embedder->encode(turn_embed_text(u, a));
        Stmt s(db,
            "INSERT INTO summaries "
            "(model_id, session_id, text, embedding, level, status, tags, timestamp) "
            "VALUES (NULL,?,?,?,'turn','complete','',?)");
        if (sess_null) s.bind_null(1); else s.bind(1, session_id);
        s.bind(2, text);
        bind_embedding(s.raw(), 3, emb);
        s.bind(4, ts);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
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

        // Retry/regeneration dedup. When the agent re-answers the *same* user
        // prompt without an intervening new prompt, we get a second turn whose
        // user_text duplicates the immediately-preceding turn. In a healthy
        // agent a regeneration stays in one session; a session change between
        // two identical prompts is itself a breakage signal (observed when the
        // TUI crash-restarted and minted a fresh session GUID per turn). So the
        // match is intentionally cross-session: identical user_text in the most
        // recent turn, within a short time window. We keep-latest — update that
        // row's assistant_text (+ re-embed) in place rather than inserting a
        // duplicate. The FTS au-trigger keeps the index consistent. Empty
        // assistant_text (partial/prompt-arrival rows) never dedups.
        if (!a.empty()) {
            Stmt p(db,
                "SELECT turn_id, user_text, timestamp FROM turns "
                "ORDER BY turn_id DESC LIMIT 1");
            if (p.step()) {
                const int prev_id          = p.column_int(0);
                const std::string prev_u   = p.column_text(1);
                const std::string prev_ts  = p.column_text(2);
                // 5-minute window: wide enough for a slow regeneration, narrow
                // enough that a genuinely re-typed identical prompt much later
                // is still recorded as its own turn.
                bool within_window = false;
                if (prev_u == u) {
                    auto prev_tt = parse_db_timestamp(prev_ts);
                    auto now_tt  = parse_db_timestamp(local_timestamp());
                    if (prev_tt && now_tt)
                        within_window = std::difftime(*now_tt, *prev_tt) <= 300.0;
                }
                if (within_window) {
                    std::vector<float> emb2 = embedder->encode(turn_embed_text(u, a));
                    Stmt up(db,
                        "UPDATE turns SET assistant_text = ?, embedding = ?, "
                        "model_id = COALESCE(?, model_id), "
                        "session_id = COALESCE(?, session_id), timestamp = ? "
                        "WHERE turn_id = ?");
                    up.bind(1, a);
                    bind_embedding(up.raw(), 2, emb2);
                    if (model_id) up.bind(3, model_id); else up.bind_null(3);
                    if (session_id) up.bind(4, session_id); else up.bind_null(4);
                    up.bind(5, local_timestamp());
                    up.bind(6, prev_id);
                    if (!up.exec())
                        throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
                    place_turn_placeholder(prev_id, u, a);
                    return prev_id;
                }
            }
        }

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
        int new_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        place_turn_placeholder(new_id, u, a);
        return new_id;
    }

    // Shared implementation for turn queries over a session GUID.
    // asc=true → oldest-first (ORDER BY turn_id ASC, no limit).
    // asc=false → newest-first (ORDER BY timestamp DESC, turn_id DESC) with optional limit.
    std::vector<TurnRecord> turns_by_session_impl(
            const std::string& session_guid, bool asc, int limit = 0) {
        std::vector<TurnRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, t.timestamp "
            "FROM turns t "
            "JOIN sessions ss ON t.session_id = ss.session_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "WHERE ss.guid = ? ";
        sql += asc ? "ORDER BY t.turn_id ASC"
                   : "ORDER BY t.timestamp DESC, t.turn_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (limit > 0) s.bind(2, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3),
                s.column_text(4),
                session_guid
            });
        }
        return out;
    }

    /// All turns belonging to a session GUID, oldest first.
    std::vector<TurnRecord> turns_by_session(const std::string& session_guid) {
        return turns_by_session_impl(session_guid, true);
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
    // source_timestamp (non-empty) overrides the row's timestamp: L2 turn
    // summaries inherit the source turn's timestamp so the (session_id,
    // timestamp) pair links a turn to its summary (no FK column needed) and
    // stays stable across embedding-model changes.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& status, const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& source_timestamp,
                      const std::string& tags) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding, level, status, tags, timestamp) "
            "VALUES (?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding(s.raw(), 4, emb);
        s.bind(5, level).bind(6, status).bind(7, tags);
        s.bind(8, source_timestamp.empty() ? local_timestamp() : source_timestamp);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // ---- catch-up / recipe helpers ------------------------------------
    // Turns lacking an L2 summary, linked by (session_id, timestamp).
    // LEFT JOIN keeps turns whose summary row doesn't exist; ordered
    // oldest-first so the worker processes in capture order.
    std::vector<TurnRecord> unsummarized_turns(int limit) {
        std::vector<TurnRecord> out;
        std::string sql =
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "       t.timestamp, COALESCE(ss.guid, '') "
            "FROM turns t "
            "LEFT JOIN summaries s "
            "  ON s.level = 'turn' "
            " AND s.session_id IS t.session_id "
            " AND s.timestamp = t.timestamp "
            " AND s.model_id IS NOT NULL "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "LEFT JOIN sessions ss ON t.session_id = ss.session_id "
            "WHERE s.summary_id IS NULL "
            "  AND t.assistant_text IS NOT NULL "
            "ORDER BY t.timestamp ASC, t.turn_id ASC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        if (limit > 0) s.bind(1, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3),
                s.column_text(4),
                s.column_text(5)
            });
        }
        return out;
    }

    // True if a *summarized* 'turn' row exists for (session_id, timestamp) —
    // i.e. model_id IS NOT NULL. A raw NULL-model placeholder does NOT count:
    // the worker must still summarize it (rewrite in place). Mirrors the
    // unsummarized_turns() join so the linkage matches exactly. Anonymous
    // turns (session_id NULL) carry an empty guid here too.
    bool turn_summary_exists(const std::string& session_guid,
                             const std::string& source_timestamp) {
        Stmt s(db,
            "SELECT 1 FROM summaries s "
            "LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE s.level = 'turn' "
            "  AND s.timestamp = ? "
            "  AND COALESCE(ss.guid, '') = ? "
            "  AND s.model_id IS NOT NULL "
            "LIMIT 1");
        s.bind(1, source_timestamp);
        s.bind(2, session_guid);
        return s.step();
    }

    // Draft-tagged summary rows for re-summarization (housekeeping retry).
    std::vector<DraftSummary> draft_summaries(int limit) {
        std::vector<DraftSummary> out;
        std::string sql =
            "SELECT s.summary_id, s.level, COALESCE(ss.guid, ''), s.timestamp "
            "FROM summaries s "
            "LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE s.tags LIKE '%draft%' "
            "ORDER BY s.timestamp ASC, s.summary_id ASC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        if (limit > 0) s.bind(1, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3)
            });
        }
        return out;
    }

    // Sessions whose newest turn is older than (now - pause_minutes) AND
    // that have at least one non-draft L2 summary but no complete L3 yet.
    // The summarizer's pause timer treats this set as "ready to finalize."
    // Returns session GUIDs; anonymous (session_id NULL) turns are skipped.
    std::vector<std::string> sessions_needing_close(int pause_minutes) {
        std::vector<std::string> out;
        if (pause_minutes <= 0) return out;
        const std::string cutoff =
            std::format("datetime('now','localtime','-{} minutes')", pause_minutes);
        std::string sql =
            "SELECT ss.guid "
            "FROM sessions ss "
            "WHERE EXISTS (SELECT 1 FROM summaries s2 "
            "               JOIN sessions ss2 ON s2.session_id = ss2.session_id "
            "               WHERE ss2.session_id = ss.session_id "
            "                 AND s2.level = 'turn' AND s2.tags != 'draft') "
            "  AND (SELECT MAX(t.timestamp) FROM turns t "
            "        WHERE t.session_id = ss.session_id) < " + cutoff + " "
            "  AND NOT EXISTS (SELECT 1 FROM summaries s "
            "                   WHERE s.session_id = ss.session_id "
            "                     AND s.level = 'session' "
            "                     AND s.status = 'complete') "
            "ORDER BY ss.session_id ASC";
        Stmt s(db, sql);
        while (s.step()) {
            const auto g = s.column_text(0);
            if (!g.empty()) out.push_back(g);
        }
        return out;
    }

    // Session turns newest-first (recipe walk-back direction).
    std::vector<TurnRecord> turns_by_session_desc(
            const std::string& session_guid, int limit) {
        return turns_by_session_impl(session_guid, false, limit);
    }

    // Shared query: summaries for a session filtered by level, newest-first.
    std::vector<SummaryRecord> summaries_by_level_desc(
            const std::string& session_guid, const std::string& level, int limit) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT s.summary_id, s.text, s.status, s.timestamp "
            "FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = ? "
            "ORDER BY s.timestamp DESC, s.summary_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid).bind(2, level);
        if (limit > 0) s.bind(3, limit);
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           s.column_text(2), s.column_text(3)});
        }
        return out;
    }

    // L2 (turn) summaries for a session, newest-first.
    std::vector<SummaryRecord> turn_summaries_by_session_desc(
            const std::string& session_guid, int limit) {
        return summaries_by_level_desc(session_guid, "turn", limit);
    }

    // L3 (session) summaries for a session, newest-first.
    std::vector<SummaryRecord> session_summaries_desc(
            const std::string& session_guid, int limit) {
        return summaries_by_level_desc(session_guid, "session", limit);
    }

    // The current running L3 session summary, if any: (summary_id, text).
    std::optional<std::pair<int, std::string>> current_session_summary(
            const std::string& session_guid) {
        // Per-session when a guid is given; legacy global when empty.
        std::string sql =
            "SELECT summary_id, text FROM summaries "
            "WHERE level='session' AND status='current'";
        if (!session_guid.empty())
            sql += " AND session_id = (SELECT session_id FROM sessions WHERE guid = ?)";
        sql += " ORDER BY summary_id DESC LIMIT 1";
        Stmt s(db, sql);
        if (!session_guid.empty()) s.bind(1, session_guid);
        std::optional<std::pair<int, std::string>> out;
        if (s.step()) {
            int id = s.column_int(0);
            out = std::make_pair(id, s.column_text(1));
        }
        return out;
    }

    // The current running L4 project summary, if any: (summary_id, text).
    // Project summaries are session-unscoped (session_id NULL).
    std::optional<std::pair<int, std::string>> current_project_summary() {
        Stmt s(db,
            "SELECT summary_id, text FROM summaries "
            "WHERE level='project' AND status='current' "
            "ORDER BY summary_id DESC LIMIT 1");
        std::optional<std::pair<int, std::string>> out;
        if (s.step())
            out = std::make_pair(s.column_int(0), s.column_text(1));
        return out;
    }

    // All L2 (turn) summary texts for a session, oldest-first.
    // Used by handle_l3_update to build the input for L3.
    std::vector<std::string> l2_summary_texts(const std::string& session_guid) {
        std::vector<std::string> out;
        if (session_guid.empty()) return out;
        Stmt s(db,
            "SELECT s.text FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'turn' AND s.tags != 'draft' "
            "ORDER BY s.timestamp ASC, s.summary_id ASC");
        s.bind(1, session_guid);
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }

    // All complete L3 (session) summary texts, oldest-first.
    // Used by handle_l4_update to build the input for L4.
    std::vector<std::string> complete_l3_summary_texts() {
        std::vector<std::string> out;
        Stmt s(db,
            "SELECT text FROM summaries "
            "WHERE level='session' AND status='complete' "
            "ORDER BY timestamp ASC, summary_id ASC");
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
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
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
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
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
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

    // Summarizer write-back: replace a turn placeholder's raw text with the
    // real summary and stamp the *summarizer* model. Matched by (guid, ts) on
    // the level='turn' row. Updates the row regardless of its current model_id
    // (so a re-summarization is allowed), but is normally called on a NULL-
    // model placeholder. Returns false if no such row exists.
    bool finalize_turn_summary(const std::string& session_guid,
                               const std::string& source_timestamp,
                               const std::string& text,
                               const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);
        Stmt s(db,
            "UPDATE summaries SET text = ?, embedding = ?, model_id = ?, tags = '' "
            "WHERE summary_id = ("
            "  SELECT s.summary_id FROM summaries s "
            "  LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "  WHERE s.level='turn' AND s.timestamp = ? "
            "    AND COALESCE(ss.guid,'') = ? "
            "  ORDER BY s.summary_id DESC LIMIT 1)");
        s.bind(1, t);
        bind_embedding(s.raw(), 2, emb);
        if (model_id) s.bind(3, model_id); else s.bind_null(3);
        s.bind(4, source_timestamp);
        s.bind(5, session_guid);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_cache();
        return ok;
    }

    // Trivial-turn done-marker: stamp a placeholder's model_id without
    // touching its raw text. The "summary" of a 5-word turn is the turn
    // itself; stamping the model removes it from unsummarized_turns() so it
    // isn't re-enqueued forever (analysis M7). Returns false if absent.
    bool mark_turn_summarized(const std::string& session_guid,
                              const std::string& source_timestamp,
                              const std::string& model_name) {
        int model_id = get_or_create_model(model_name);
        if (!model_id) return false;  // need a real model to stamp
        Stmt s(db,
            "UPDATE summaries SET model_id = ? "
            "WHERE summary_id = ("
            "  SELECT s.summary_id FROM summaries s "
            "  LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "  WHERE s.level='turn' AND s.timestamp = ? "
            "    AND COALESCE(ss.guid,'') = ? AND s.model_id IS NULL "
            "  ORDER BY s.summary_id DESC LIMIT 1)");
        s.bind(1, model_id);
        s.bind(2, source_timestamp);
        s.bind(3, session_guid);
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

    // Replace a summary's tags column. Used by the summarizer to clear
    // "draft" once a row has been rewritten with a real summary.
    bool set_summary_tags(int summary_id, const std::string& tags) {
        Stmt s(db,
            "UPDATE summaries SET tags = ? WHERE summary_id = ?");
        s.bind(1, tags).bind(2, summary_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }

    bool update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        // Refuse to mutate protected rows for parity with delete_memory.
        Stmt check_stmt(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        check_stmt.bind(1, memory_id);
        if (!check_stmt.step()) return false;  // row not found
        if (check_stmt.column_text(0).find("keep") != std::string::npos) return false;

        // Lean v2 summaries: only text/embedding/tags are mutable here.
        // keep/bad flags fold into tags (parity with store()).
        std::string tags_str = tags_from_metadata(metadata);

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
    // Documents (L5) and decisions (L6) ARE merged into search: parallel passes
    // (ensure_doc_cache / ensure_dec_cache + their FTS scorers) are scored
    // identically and merged with the summaries results into the single ranked
    // top-k returned here. Each SearchResult's metadata["source"] is "summary",
    // "document", or "decision" so callers can tell the corpora apart.
    SearchResponse search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> /*collections*/) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        ensure_cache();
        ensure_doc_cache();
        ensure_dec_cache();
        int n_sum = static_cast<int>(cached_ids.size());
        int n_doc = static_cast<int>(doc_ids.size());
        int n_dec = static_cast<int>(dec_ids.size());
        if (n_sum == 0 && n_doc == 0 && n_dec == 0) return {{}, {{"corpus_size", 0}}};

        // ---- query embedding ------------------------------------------
        auto t_embed_start = clock::now();
        auto q_vec = embedder->encode(query);
        auto t_embed_end = clock::now();

        auto t_search_start = clock::now();
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(), static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        // A scored candidate: raw cosine is reported as the score, the blended
        // value drives ranking. Both corpora produce these and are merged.
        struct Candidate {
            float        blended;
            SearchResult result;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(n_sum + n_doc + n_dec));

        // Score one corpus (vector cosine blended with FTS5 bm25), pushing its
        // rows as Candidates. Shared by the summaries and documents passes so
        // the blend logic stays in exactly one place.
        auto score_corpus =
            [&](const std::vector<int>& ids,
                const std::vector<std::string>& texts,
                const Eigen::MatrixXf& cache_emb,
                const std::vector<json>& meta,
                const std::vector<std::string>& ts,
                const std::unordered_map<int, float>& kw) {
            int n = static_cast<int>(ids.size());
            if (n == 0) return;

            // cache_emb rows are L2-normalized at cache-build time, and q_norm
            // is the normalized query vector, so this product is raw cosine.
            Eigen::VectorXf similarities = cache_emb * q_norm;   // raw cosine, reported

            Eigen::VectorXf combined = similarities;
            if (config().bm25_enabled && !kw.empty()) {
                Eigen::VectorXf kw_vec(n);
                for (int i = 0; i < n; ++i) {
                    auto it = kw.find(ids[i]);
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

            for (int i = 0; i < n; ++i) {
                candidates.push_back({combined(i),
                    SearchResult{ids[i], texts[i], similarities(i),
                                 meta[i], ts[i]}});
            }
        };

        std::string match_expr = fts_match_expr(query);
        score_corpus(cached_ids, cached_texts, cached_embeddings,
                     cached_metadata, cached_timestamps,
                     config().bm25_enabled ? keyword_scores(match_expr)
                                           : std::unordered_map<int, float>{});
        score_corpus(doc_ids, doc_texts, doc_embeddings,
                     doc_metadata, doc_timestamps,
                     config().bm25_enabled ? doc_keyword_scores(match_expr)
                                           : std::unordered_map<int, float>{});
        score_corpus(dec_ids, dec_texts, dec_embeddings,
                     dec_metadata, dec_timestamps,
                     config().bm25_enabled ? dec_keyword_scores(match_expr)
                                           : std::unordered_map<int, float>{});
        auto t_search_end = clock::now();

        // ---- merged top-k selection -----------------------------------
        // Filter by min_score (raw cosine) FIRST, then rank survivors by
        // blended score and emit up to `limit`. Filtering before top-k
        // (rather than after) prevents under-filling: a high-blend candidate
        // with sub-threshold cosine no longer steals a slot from a valid one.
        std::vector<int> ranking;
        ranking.reserve(candidates.size());
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            if (candidates[i].result.score >= min_score)
                ranking.push_back(i);
        }
        int total = static_cast<int>(ranking.size());
        int top_k = std::min(limit, total);
        std::partial_sort(ranking.begin(),
                          ranking.begin() + top_k,
                          ranking.end(),
                          [&](int a, int b) {
                              return candidates[a].blended > candidates[b].blended;
                          });

        std::vector<SearchResult> results;
        for (int k = 0; k < top_k; ++k) {
            results.push_back(candidates[ranking[k]].result);
        }

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"embedding_ms", ms(t_embed_start, t_embed_end)},
            {"search_ms",    ms(t_search_start, t_search_end)},
            {"total_ms",     ms(t_start, clock::now())},
            {"corpus_size",  n_sum + n_doc + n_dec}
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

    // Total rows across the four embedded tables (turns, summaries, decisions,
    // documents) — i.e. how many rows `rebuild_embeddings()` will re-encode.
    // (count() alone is just summaries, which understates the rebuild scope.)
    int count_embeddable_rows() const {
        int total = 0;
        for (const char* tbl : {"turns", "summaries", "decisions", "documents"}) {
            Stmt s(db, std::format("SELECT COUNT(*) FROM {}", tbl));
            if (s.step()) total += s.column_int(0);
        }
        return total;
    }

    // True if any embedded table holds a non-NULL embedding. EXISTS short-
    // circuits on the first hit. Deferred (NULL-embedding) rows don't count —
    // they get the current model on backfill, so they aren't incompatible.
    bool has_embeddings() const {
        Stmt s(db,
            "SELECT EXISTS("
            "  SELECT 1 FROM summaries  WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM turns     WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM documents WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM decisions WHERE embedding IS NOT NULL "
            "  LIMIT 1)");
        return s.step() && s.column_int(0) != 0;
    }

    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). Returns every summary, score 0.
    std::vector<SearchResult> load_all(const std::string& /*collection*/) {
        std::vector<SearchResult> results;
        Stmt s(db,
            "SELECT summary_id, text, level, status, tags, timestamp "
            "FROM summaries ORDER BY summary_id");

        while (s.step()) {
            int id        = s.column_int(0);
            auto text     = s.column_text(1);
            auto lvl      = s.column_text(2);
            auto st       = s.column_text(3);
            auto tag      = s.column_text(4);
            auto ts       = s.column_text(5);

            json meta = json::object();
            if (!lvl.empty()) meta["level"]  = lvl;
            if (!st.empty())  meta["status"] = st;
            if (!tag.empty()) meta["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(meta), std::move(ts)});
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
                auto col1 = select_stmt.column_text(1);
                if (col1.empty()) continue;

                std::string embed_text;
                if (t.extra_col) {
                    embed_text = turn_embed_text(col1, select_stmt.column_text(2));
                } else {
                    embed_text = std::move(col1);
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
        invalidate_doc_cache();
        invalidate_dec_cache();
        return doc_count;
    }

    int backfill_embeddings(Embedder& emb_ref) {
        // Cover all four embedded tables (not just summaries).
        // Same TableSpec structure as rebuild_embeddings, but filters to
        // WHERE embedding IS NULL so only unembedded rows are touched.
        struct TableSpec {
            const char* table;
            const char* id_col;
            const char* text_col;
            const char* extra_col;
        };
        static constexpr TableSpec tables[] = {
            { "turns",     "turn_id",     "user_text", "assistant_text" },
            { "summaries", "summary_id",  "text",      nullptr          },
            { "decisions", "decision_id", "text",      nullptr          },
            { "documents", "document_id", "text",      nullptr          },
        };

        int updated = 0;
        for (auto& t : tables) {
            std::string sel = t.extra_col
                ? std::format("SELECT {} AS id, {}, {} FROM {} WHERE embedding IS NULL",
                              t.id_col, t.text_col, t.extra_col, t.table)
                : std::format("SELECT {} AS id, {} AS text FROM {} WHERE embedding IS NULL",
                              t.id_col, t.text_col, t.table);
            std::string upd = std::format("UPDATE {} SET embedding = ? WHERE {} = ?",
                                          t.table, t.id_col);
            Stmt select_stmt(db, sel);
            while (select_stmt.step()) {
                int id = select_stmt.column_int(0);
                auto col1 = select_stmt.column_text(1);
                if (col1.empty()) continue;
                std::string embed_text = t.extra_col
                    ? turn_embed_text(col1, select_stmt.column_text(2))
                    : std::move(col1);
                auto emb = emb_ref.encode(embed_text);
                Stmt update(db, upd);
                bind_embedding(update.raw(), 1, emb);
                update.bind(2, id);
                update.step();
                ++updated;
            }
        }

        if (updated > 0) {
            invalidate_cache();
            invalidate_doc_cache();
            invalidate_dec_cache();
        }
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
            invalidate_doc_cache();
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

    // Returns true if the summaries row exists and has a "keep" tag.
    bool has_keep_tag(int summary_id) {
        Stmt s(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        s.bind(1, summary_id);
        if (!s.step()) return false;
        return s.column_text(0).find("keep") != std::string::npos;
    }

    bool delete_memory(int memory_id) {
        if (has_keep_tag(memory_id)) return false;
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
        // Single atomic DELETE filtered by keep-tag, entirely in SQL.
        std::string sql = "DELETE FROM summaries WHERE summary_id IN (";
        for (size_t i = 0; i < memory_ids.size(); ++i) {
            if (i > 0) sql += ",";
            sql += "?";
        }
        sql += ") AND (tags IS NULL OR tags NOT LIKE '%keep%')";
        Stmt(db, "BEGIN").exec();
        Stmt stmt(db, sql);
        for (size_t i = 0; i < memory_ids.size(); ++i)
            stmt.bind(static_cast<int>(i + 1), memory_ids[i]);
        stmt.step();
        int changes = sqlite3_changes(db);
        Stmt(db, "COMMIT").exec();
        if (changes > 0) invalidate_cache();
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
            int id        = stmt.column_int(0);
            auto text     = stmt.column_text(1);
            auto lvl      = stmt.column_text(2);
            auto st       = stmt.column_text(3);
            auto tag      = stmt.column_text(4);
            auto ts       = stmt.column_text(5);

            json metadata = json::object();
            if (!lvl.empty()) metadata["level"]  = lvl;
            if (!st.empty())  metadata["status"] = st;
            if (!tag.empty()) metadata["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(metadata), std::move(ts)});
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
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store(text, std::move(metadata), defer_embedding);
}

int SqliteBackend::store_document(const DocumentChunk& chunk, bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_document(chunk, defer_embedding);
}

int SqliteBackend::store_turn(const std::string& user_text,
                              const std::string& assistant_text,
                              const std::string& model_name, bool defer_embedding,
                              const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_turn(user_text, assistant_text, model_name,
                             defer_embedding, session_guid);
}

std::vector<TurnRecord> SqliteBackend::turns_by_session(
        const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turns_by_session(session_guid);
}

bool SqliteBackend::finalize_turn(int turn_id, const std::string& assistant_text,
                                  const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->finalize_turn(turn_id, assistant_text, model_name);
}

int SqliteBackend::store_summary(const std::string& text, const std::string& level,
                                 const std::string& status, const std::string& model_name,
                                 const std::string& session_guid,
                                 const std::string& source_timestamp,
                                 const std::string& tags) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_summary(text, level, status, model_name, session_guid,
                                source_timestamp, tags);
}

std::optional<std::pair<int, std::string>>
SqliteBackend::current_session_summary(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->current_session_summary(session_guid);
}

std::optional<std::pair<int, std::string>>
SqliteBackend::current_project_summary() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->current_project_summary();
}

std::vector<std::string>
SqliteBackend::l2_summary_texts(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->l2_summary_texts(session_guid);
}

std::vector<std::string>
SqliteBackend::complete_l3_summary_texts() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->complete_l3_summary_texts();
}

bool SqliteBackend::update_summary_text(int summary_id, const std::string& text,
                                        const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_summary_text(summary_id, text, model_name);
}

bool SqliteBackend::finalize_turn_summary(const std::string& session_guid,
                                          const std::string& source_timestamp,
                                          const std::string& text,
                                          const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->finalize_turn_summary(session_guid, source_timestamp,
                                        text, model_name);
}

bool SqliteBackend::mark_turn_summarized(const std::string& session_guid,
                                         const std::string& source_timestamp,
                                         const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->mark_turn_summarized(session_guid, source_timestamp, model_name);
}

bool SqliteBackend::set_summary_status(int summary_id, const std::string& status) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_summary_status(summary_id, status);
}

bool SqliteBackend::set_summary_tags(int summary_id, const std::string& tags) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_summary_tags(summary_id, tags);
}

std::vector<std::string> SqliteBackend::recent_summaries(const std::string& level,
                                                         int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->recent_summaries(level, limit);
}

std::vector<std::string> SqliteBackend::current_decisions(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->current_decisions(limit);
}

std::vector<TurnRecord> SqliteBackend::unsummarized_turns(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->unsummarized_turns(limit);
}

bool SqliteBackend::turn_summary_exists(const std::string& session_guid,
                                        const std::string& source_timestamp) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_summary_exists(session_guid, source_timestamp);
}

std::vector<DraftSummary> SqliteBackend::draft_summaries(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->draft_summaries(limit);
}

std::vector<std::string> SqliteBackend::sessions_needing_close(int pause_minutes) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->sessions_needing_close(pause_minutes);
}

std::vector<TurnRecord> SqliteBackend::turns_by_session_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turns_by_session_desc(session_guid, limit);
}

std::vector<SummaryRecord>
SqliteBackend::turn_summaries_by_session_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_summaries_by_session_desc(session_guid, limit);
}

std::vector<SummaryRecord>
SqliteBackend::session_summaries_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->session_summaries_desc(session_guid, limit);
}

bool SqliteBackend::update_text(int memory_id, const std::string& text, json metadata, bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse SqliteBackend::search(const std::string& query, int limit,
                                     float min_score,
                                     std::vector<std::string> collections) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->search(query, limit, min_score, std::move(collections));
}

int SqliteBackend::count() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->count(); }

bool SqliteBackend::has_embeddings() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->has_embeddings(); }

int SqliteBackend::count_embeddable_rows() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->count_embeddable_rows(); }

std::vector<SearchResult> SqliteBackend::load_all(const std::string& collection) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->load_all(collection);
}

int SqliteBackend::rebuild_embeddings(Embedder& embedder) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->rebuild_embeddings(embedder);
}

int SqliteBackend::backfill_embeddings(Embedder& embedder) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->backfill_embeddings(embedder);
}

bool SqliteBackend::update_document_embedding(int document_id,
                                              const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_document_embedding(document_id, emb);
}

std::vector<std::string> SqliteBackend::collections() const {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->collections();
}

void SqliteBackend::close() { pImpl->close(); }

bool SqliteBackend::delete_memory(int memory_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->delete_memory(memory_id);
}

int SqliteBackend::delete_batch(const std::vector<int>& memory_ids) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->delete_batch(memory_ids);
}

std::vector<SearchResult> SqliteBackend::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->search_by_metadata(metadata_filter, limit, after, before);
}

int SqliteBackend::cleanup_old_conversations(float max_age_hours) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
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

} // namespace ragger
