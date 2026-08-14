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
#include "ragger/double_metaphone.h"
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
#include <regex>
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
// (Embedding blobs are written by Impl::bind_embedding, which honors the
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

    // Parallel embedding cache for the turn_summaries (L2) table — invalidated
    // on turn-summary writes. Mirrors the caches above; vector scores come
    // from here, keyword scores from FTS5 (turn_summaries_fts) at query time.
    // Rows with NULL text (unsummarized placeholders) are excluded — they
    // must never appear in search results or feed embedding similarity.
    // Metadata carries source="turn_summary" plus turn_id/session_id so
    // search() consumers can distinguish turn-level results from the other
    // three corpora.
    bool                           turn_cache_valid = false;
    std::vector<int>               turn_ids;
    std::vector<std::string>       turn_texts;
    Eigen::MatrixXf                turn_embeddings;     // rows × dims
    std::vector<json>              turn_metadata;
    std::vector<std::string>       turn_timestamps;

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
        // Wait up to 10s on a locked DB instead of failing instantly —
        // WAL allows concurrent readers, but two writers (daemon +
        // import CLI) still serialize; without this any collision is an
        // immediate "database is locked" error.
        sqlite3_busy_timeout(db, 10000);
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
        sqlite3_busy_timeout(db, 10000);  // see main constructor
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
    void exec(const std::string& sql) { exec(sql.c_str()); }

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

    bool table_exists(const std::string& table) {
        Stmt s(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?");
        s.bind(1, table);
        return s.step();
    }

    void create_schema() {
        // Capture "was this DB already populated?" BEFORE any CREATE TABLE
        // IF NOT EXISTS runs below -- this is the only reliable way to tell
        // a genuinely fresh install (no memory tables at all yet) from an
        // existing pre-0.12 DB that simply never had its db_version row
        // stamped. Only a fresh install should be auto-stamped '0.12' at
        // the bottom of this function; an existing un-migrated DB must
        // fail the startup version-gate check so the user runs
        // scripts/migrate_to_db0.12.sh instead of silently being treated
        // as current.
        bool db_preexisted = table_exists("turns");

        // Users, settings, and session tables — credentials (for read-only
        // document access) plus web/chat session persistence. These are a
        // separate concern from the v2 memory tables; one declarative
        // definition, shared with the DB-only constructor.
        create_user_schema();

        // ---- v2 fading-memory schema (issue #33): turns / summaries /
        //      decisions / documents / models, with FTS5 (issue #49).
        //      Pre-v2 data is exported out-of-band, not migrated in place.

        // models — lookup table; turns + summaries reference it. created_at
        // was added in a later schema revision; the column sits at the end so
        // it maps cleanly to the canonical (id, keys, data, timestamps) order.
        exec(R"(
            CREATE TABLE IF NOT EXISTS models (
                model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )
        )");

        // sessions — lookup table; turns + summaries reference it. Normalizes
        // the long conversation GUID (from the agent turn hook) to a compact
        // integer id, mirroring `models`. Grouping key for session summaries.
        exec(R"(
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid       TEXT NOT NULL UNIQUE,
                name       TEXT,
                name_source TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )
        )");

        // turns (L1) — raw verbatim exchanges. embedding nullable for the
        // deferred-embedding path (partial row written, backfilled later).
        // created_at has NO DEFAULT (matches scripts/schema_db0.12.sql
        // exactly) -- the app always supplies it explicitly on INSERT.
        exec(R"(
            CREATE TABLE IF NOT EXISTS turns (
                turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text      TEXT NOT NULL,
                assistant_text TEXT,
                model_id       INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id     INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                created_at     INTEGER NOT NULL,
                embedding      BLOB,
                phon           TEXT
            )
        )");

        // summaries (L2/L3/L4) — as of db_version 0.12, level is only ever
        // 'episode' | 'session' | 'project'; turn-level rows moved to the
        // dedicated turn_summaries table below.
        // created_at = insert/create time; updated_at = last regenerate time
        // for running rows (session/project) and span-end for episode rows.
        // Both NOT NULL (updated_at = created_at on first insert).
        exec(R"(
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                level      TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                model_id   INTEGER REFERENCES models(model_id),
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                embedding  BLOB,
                phon       TEXT
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level      ON summaries(level)");
        // idx_summaries_created_at created after migration (see note above).

        // turn_summaries (L2) -- NEW at db_version 0.12. One row per
        // summarized turn, split out of `summaries`. turn_id is a real FK,
        // ON DELETE SET NULL: pruning the raw turn leaves the summary behind
        // with turn_id NULL rather than stranding it via a fragile
        // (session_id, created_at) match. See scripts/schema_db0.12.sql.
        exec(R"(
            CREATE TABLE IF NOT EXISTS turn_summaries (
                turn_summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                text             TEXT,
                turn_id          INTEGER REFERENCES turns(turn_id) ON DELETE SET NULL,
                session_id       INTEGER REFERENCES sessions(session_id),
                turn_model_id    INTEGER REFERENCES models(model_id),
                summary_model_id INTEGER REFERENCES models(model_id),
                turn_datetime    INTEGER NOT NULL,
                summarized_on    INTEGER NOT NULL DEFAULT (unixepoch()),
                embedding        BLOB,
                phon             TEXT
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_turn_id    ON turn_summaries(turn_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_session_id ON turn_summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_datetime   ON turn_summaries(turn_datetime)");
        // Upsert target for finalize_turn_summary: one row per turn_id. NULL
        // turn_id (a pruned turn) is excepted -- multiple such rows are fine.
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_summaries_turn_id_uniq "
             "ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL");

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

        // decisions (L6). v4 column order: keys/data, created_at, embedding, phon.
        exec(R"(
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                status      TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '',
                created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                embedding   BLOB,
                phon        TEXT
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)");

        // documents (L5) — user-curated RAG; the only sharable table.
        // Lean per the reference DDL: title identifies the publication, tags
        // group/describe (subject), year is the publish year, path is the
        // origin. Only `text` is embedded. Keyword search via documents_fts.
        // v4 column order: keys/data, imported_at, embedding, phon.
        exec(R"(
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                path        TEXT,
                title       TEXT,
                tags        TEXT NOT NULL DEFAULT '',
                year        INTEGER,
                chunk_index INTEGER,
                imported_at TEXT NOT NULL,
                embedding   BLOB,
                phon        TEXT
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_documents_imported_at ON documents(imported_at)");

        // In-place migration for pre-phon databases (dolphining sounds-like):
        // add the `phon` column to each context table if an existing DB predates
        // it. Placed after ALL four CREATE TABLEs so the ALTERs can't hit a
        // not-yet-created table. Fresh DBs already have the column; existing
        // rows get phon = NULL, backfilled at startup or via `ragger rebuild-phon`.
        if (!column_exists("turns", "phon"))
            exec("ALTER TABLE turns ADD COLUMN phon TEXT");
        if (!column_exists("summaries", "phon"))
            exec("ALTER TABLE summaries ADD COLUMN phon TEXT");
        if (!column_exists("decisions", "phon"))
            exec("ALTER TABLE decisions ADD COLUMN phon TEXT");
        if (!column_exists("documents", "phon"))
            exec("ALTER TABLE documents ADD COLUMN phon TEXT");
        if (!column_exists("turn_summaries", "phon"))
            exec("ALTER TABLE turn_summaries ADD COLUMN phon TEXT");

        // v0.12 schema-version hard gate: no auto-migration in the binary.
        // First, stamp db_version for a genuinely fresh install ONLY --
        // db_preexisted (captured before any CREATE TABLE ran, at the top
        // of this function) distinguishes "no memory tables existed yet"
        // from an existing pre-0.12 DB that simply lacks the version row.
        // An existing un-migrated DB must NOT get auto-stamped here; it
        // needs to fail the check below so the user runs
        // scripts/migrate_to_db0.12.sh instead of silently being treated
        // as current.
        if (!db_preexisted)
            exec("INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '0.12')");
        {
            std::string actual = db_version();
            if (actual != kExpectedDbVersion) {
                std::cerr << std::format(
                    "Ragger: database schema version mismatch (found '{}', need '{}').\n"
                    "This binary requires a database migrated to schema {}.\n"
                    "Run scripts/migrate_to_db{}.sh against your ~/.ragger/memories.db, "
                    "then restart.\n",
                    actual.empty() ? "<none>" : actual,
                    kExpectedDbVersion, kExpectedDbVersion, kExpectedDbVersion);
                std::exit(1);
            }
        }

        // Indexes — (re)created here, after migration, so a table rebuild
        // (which drops the table and with it every index) can't leave them
        // missing. All IF NOT EXISTS, so fresh / migrated / already-current
        // DBs all converge to the same set.
        exec("CREATE INDEX IF NOT EXISTS idx_turns_created_at      ON turns(created_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_session         ON turns(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level       ON summaries(level)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_created_at  ON summaries(created_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_session     ON summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_decisions_status      ON decisions(status)");
        exec("CREATE INDEX IF NOT EXISTS idx_documents_imported_at ON documents(imported_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_turn_id    ON turn_summaries(turn_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_session_id ON turn_summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_datetime   ON turn_summaries(turn_datetime)");
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_summaries_turn_id_uniq "
             "ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL");

        // FTS5 — external-content virtual tables + sync triggers replace
        // the old hand-rolled bm25_* sidecars (issue #49).
        create_fts_schema();

        // Human-readable views (datetime()-rendered timestamps,
        // has_embedding/has_phon booleans) mirroring scripts/schema_db0.12.sql
        // exactly. Always (re)created so fresh/migrated DBs converge.
        create_views();
    }


    // ---- schema version --------------------------------------------------
    // The DB's schema version lives in settings['db_version'] as a string
    // (e.g. "0.12"). Absent means pre-versioning (legacy v3/v4 DB from
    // before this key existed) -- returns "" in that case. Startup hard-gates
    // on this via kExpectedDbVersion below; there is no in-binary migration.
    static constexpr std::string_view kExpectedDbVersion = "0.12";

    // Watermark keys for the boundary-detection housekeeping scans (see
    // sessions_needing_close_boundary()/projects_needing_close_boundary()
    // below). Persisted in the same `settings` key/value table as
    // db_version, so a failed/skipped close never re-scans from scratch.
    static constexpr std::string_view kSessionBoundaryWatermarkKey =
        "session_boundary_watermark_turn_id";
    static constexpr std::string_view kProjectBoundaryWatermarkKey =
        "project_boundary_watermark_turn_id";

    std::string db_version() {
        Stmt s(db, "SELECT value FROM settings WHERE key = 'db_version'");
        if (s.step()) return s.column_text(0);
        return "";  // absent
    }

    void set_db_version(const std::string& v) {
        Stmt s(db,
            "INSERT INTO settings (key, value) VALUES ('db_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, v);
        s.exec();
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

        exec(R"(CREATE VIRTUAL TABLE IF NOT EXISTS turn_summaries_fts USING fts5(
            text,
            content='turn_summaries', content_rowid='turn_summary_id'))");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turn_summaries_ai AFTER INSERT ON turn_summaries BEGIN
            INSERT INTO turn_summaries_fts(rowid, text)
            VALUES (new.turn_summary_id, new.text);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turn_summaries_ad AFTER DELETE ON turn_summaries BEGIN
            INSERT INTO turn_summaries_fts(turn_summaries_fts, rowid, text)
            VALUES ('delete', old.turn_summary_id, old.text);
        END)");
        exec(R"(CREATE TRIGGER IF NOT EXISTS turn_summaries_au AFTER UPDATE ON turn_summaries BEGIN
            INSERT INTO turn_summaries_fts(turn_summaries_fts, rowid, text)
            VALUES ('delete', old.turn_summary_id, old.text);
            INSERT INTO turn_summaries_fts(rowid, text)
            VALUES (new.turn_summary_id, new.text);
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

        create_phon_fts_schema();
    }

    /// Phonetic ("dolphining" sounds-like) FTS5 tables: one external-content
    /// index over each context table's `phon` column, kept in sync by its own
    /// triggers. Separate from the text FTS so phon produces an INDEPENDENT
    /// bm25 score for the three-way search blend (vector + text + phon). Codes
    /// are space-joined Double Metaphone keys (see phonize()); the default
    /// unicode61 tokenizer splits them on spaces exactly like ordinary tokens.
    /// Idempotent — safe on every open.
    void create_phon_fts_schema() {
        struct P { const char* fts; const char* base; const char* rowid; };
        const P tables[] = {
            {"turns_phon_fts",     "turns",     "turn_id"},
            {"summaries_phon_fts", "summaries", "summary_id"},
            {"turn_summaries_phon_fts", "turn_summaries", "turn_summary_id"},
            {"decisions_phon_fts", "decisions", "decision_id"},
            {"documents_phon_fts", "documents", "document_id"},
        };
        for (const auto& t : tables) {
            exec(std::format(
                "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING fts5("
                "phon, content='{}', content_rowid='{}')",
                t.fts, t.base, t.rowid));
            exec(std::format(
                "CREATE TRIGGER IF NOT EXISTS {0}_pai AFTER INSERT ON {1} BEGIN "
                "INSERT INTO {2}(rowid, phon) VALUES (new.{3}, new.phon); END",
                t.base, t.base, t.fts, t.rowid));
            exec(std::format(
                "CREATE TRIGGER IF NOT EXISTS {0}_pad AFTER DELETE ON {1} BEGIN "
                "INSERT INTO {2}({2}, rowid, phon) "
                "VALUES ('delete', old.{3}, old.phon); END",
                t.base, t.base, t.fts, t.rowid));
            exec(std::format(
                "CREATE TRIGGER IF NOT EXISTS {0}_pau AFTER UPDATE ON {1} BEGIN "
                "INSERT INTO {2}({2}, rowid, phon) "
                "VALUES ('delete', old.{3}, old.phon); "
                "INSERT INTO {2}(rowid, phon) VALUES (new.{3}, new.phon); END",
                t.base, t.base, t.fts, t.rowid));

            // External-content FTS built over a table that ALREADY has rows
            // (the phon column was ADD COLUMN-migrated onto an existing DB)
            // starts with an EMPTY index while the base holds content. The
            // delete-triggers then try to remove postings that were never
            // inserted → "database disk image is malformed" on the next UPDATE.
            // 'rebuild' syncs the index to current base.phon content so the
            // per-row delete/insert triggers stay valid.
            //
            // Detecting the desync: count(*) on an external-content FTS reads
            // THROUGH to the base table, so it always "matches" even when the
            // index is empty. The `<fts>_docsize` shadow table, however, holds
            // exactly one row per *indexed* document — so comparing its count
            // to the base row count is a reliable desync probe. If they differ
            // (e.g. index empty after ADD COLUMN, or corrupted by a prior
            // partial run), rebuild to resync. 'rebuild' is a no-op-safe resync.
            long long indexed_rows = 0, base_rows = 0;
            {
                Stmt is(db, std::format("SELECT count(*) FROM {}_docsize", t.fts));
                if (is.step()) indexed_rows = is.column_int(0);
                Stmt bs(db, std::format("SELECT count(*) FROM {}", t.base));
                if (bs.step()) base_rows = bs.column_int(0);
            }
            if (indexed_rows != base_rows) {
                exec(std::format("INSERT INTO {0}({0}) VALUES('rebuild')", t.fts));
            }
        }
    }

    /// Users + settings tables — declarative, no in-place migration
    /// (single-user app; pre-v2 data is exported out-of-band). `users` mirrors
    /// the reference DDL plus `password_hash` for credentialed access.
    /// Shared by both constructors.
    void create_user_schema() {
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

    /// Human-readable views mirroring scripts/schema_db0.12.sql exactly:
    /// datetime()-rendered epoch timestamps, and has_embedding/has_phon
    /// (0/1) collapsing the raw embedding/phon BLOB/TEXT columns. Meant to
    /// be opened directly in a plain SQLite browser by a human. Idempotent
    /// — safe to call on every open.
    void create_views() {
        exec(R"(
            CREATE VIEW IF NOT EXISTS users_view AS
            SELECT id, username, token_hash, password_hash,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
            FROM users
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS models_view AS
            SELECT model_id, name,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM models
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS sessions_view AS
            SELECT session_id, guid, name, name_source,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM sessions
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name)");
        exec(R"(
            CREATE VIEW IF NOT EXISTS turns_view AS
            SELECT turn_id, user_text, assistant_text, model_id, session_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
                   CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
            FROM turns
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS turn_summaries_view AS
            SELECT
                turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
                datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
                datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
                CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
                CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
            FROM turn_summaries
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS summaries_view AS
            SELECT summary_id, text, level, tags, session_id, model_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
                   CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
            FROM summaries
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS decisions_view AS
            SELECT decision_id, text, status, tags,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
                   CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
            FROM decisions
        )");
        exec(R"(
            CREATE VIEW IF NOT EXISTS documents_view AS
            SELECT document_id, text, path, title, tags, year, chunk_index, imported_at,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
                   CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
            FROM documents
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

    // Strip a leading "Decision #NNN", "Design Decision #NNN", or
    // "Decision Log #NNN" label (optionally with a "(context)" aside
    // and/or the word "Logged") from decision text, e.g.
    // "Decision #122 (Vienna, June 2026): Ragger adds ..." ->
    // "Ragger adds ...". The row's own decision_id + created_at already
    // carry this information — a hand/agent-authored numeric label baked
    // into the text is pure duplication that dilutes embedding/FTS search
    // (and drifts out of sync with the real decision_id over time).
    // Markdown emphasis markers ("**"/"*") wrapping the label are also
    // consumed. No-op if the text doesn't start with this pattern.
    static std::string strip_decision_number_prefix(const std::string& text) {
        static const std::regex prefix_re(
            R"(^\s*\*{0,2}(?:Design\s+)?Decision(?:\s+Log)?\s*#\d+\s*)"
            R"((?:Logged)?\s*(?:\([^)]*\))?\s*\*{0,2}\s*:\s*\*{0,2}\s*)",
            std::regex::icase);
        std::smatch m;
        if (!std::regex_search(text, m, prefix_re) || m.position(0) != 0)
            return text;
        std::string out = text.substr(m.length(0));
        // If the consumed prefix opened with "**"/"*" (bold/italic wrapping
        // the whole label), the matching close marker sits at the very end
        // of the string, past what the prefix regex could reach — trim it
        // too so we don't leave a dangling "**"/"*" behind.
        std::string opened = m.str(0);
        size_t nstars = 0;
        for (char c : opened) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
            if (c == '*') { ++nstars; continue; }
            break;
        }
        if (nstars > 0) {
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
                out.pop_back();
            size_t trailing = 0;
            for (size_t i = out.size(); i > 0 && out[i - 1] == '*'; --i) ++trailing;
            if (trailing >= nstars) out.erase(out.size() - nstars);
        }
        return out;
    }

    // ---- local timestamp ("%F %T" == "YYYY-MM-DD HH:MM:SS") ----------
    // Thin aliases over the shared formatter (ragger/util/time.h) so every
    // DB timestamp uses one local-time format.
    static std::string local_timestamp(std::time_t tt) { return db_timestamp(tt); }
    static std::string local_timestamp() { return db_timestamp(); }

    // ---- epoch timestamp for INTEGER columns --------------------------
    // resolve_epoch(caller_ts_string): callers (importers) sometimes supply
    // a specific historical timestamp as a "%F %T" string rather than "now".
    // Parse it via the shared parse_db_timestamp() and convert to epoch
    // seconds; empty input -> current time. A non-empty string that fails
    // to parse is a caller error (malformed import timestamp) -- surfaced
    // as a thrown exception rather than silently binding garbage, matching
    // how other malformed-input cases in this file are handled.
    static int64_t resolve_epoch(const std::string& caller_ts) {
        if (caller_ts.empty()) return db_epoch();
        auto tt = parse_db_timestamp(caller_ts);
        if (!tt) {
            throw std::runtime_error(
                "Ragger: malformed timestamp supplied to store call: '" + caller_ts + "'");
        }
        return db_epoch(*tt);
    }

    // ---- cache --------------------------------------------------------
    void invalidate_cache() { cache_valid = false; }
    void invalidate_doc_cache() { doc_cache_valid = false; }
    void invalidate_dec_cache() { dec_cache_valid = false; }
    void invalidate_turn_cache() { turn_cache_valid = false; }

    // Loads the summaries table into the vector cache. Keyword scores come
    // from FTS5 (summaries_fts) at query time — see keyword_scores().
    void ensure_cache() {
        if (cache_valid) return;

        cached_ids.clear();
        cached_texts.clear();
        cached_metadata.clear();
        cached_timestamps.clear();

        Stmt s(db,
            "SELECT summary_id, text, embedding, level, tags, "
            "       datetime(created_at,'unixepoch','localtime') "
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

            // Lean v2 summaries: surface level/tags as metadata so the
            // generic API and keep-protection (via tags) have what they need.
            json meta = json::object();
            meta["source"] = "summary";
            meta["level"]  = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            cached_metadata.push_back(std::move(meta));

            cached_timestamps.push_back(col_text(5));
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
            "SELECT decision_id, text, embedding, status, tags, "
            "       datetime(created_at,'unixepoch','localtime') "
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

    // Loads the turn_summaries (L2) table into the parallel vector cache.
    // Mirrors ensure_dec_cache(): vector scores come from here, keyword
    // scores from FTS5 (turn_summaries_fts) at query time. WHERE text IS NOT
    // NULL excludes unfinalized placeholder rows (summarizer hasn't run yet)
    // — these must never surface in search results or feed the embedding
    // similarity matrix. Metadata carries source="turn_summary" plus
    // turn_id/session_id (turn_summaries has no tags/status/level columns).
    void ensure_turn_cache() {
        if (turn_cache_valid) return;

        turn_ids.clear();
        turn_texts.clear();
        turn_metadata.clear();
        turn_timestamps.clear();

        Stmt s(db,
            "SELECT turn_summary_id, text, embedding, turn_id, session_id, "
            "       datetime(turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries WHERE text IS NOT NULL");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            turn_ids.push_back(s.column_int(0));
            turn_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "turn_summaries",
                turn_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "turn_summary";
            if (!s.is_null(3)) meta["turn_id"] = s.column_int(3);
            if (!s.is_null(4)) meta["session_id"] = s.column_int(4);
            // Human-readable datetime (mirrors turns.created_at) for temporal
            // ordering; kept alongside turn_id (raw-turn lookup key).
            std::string ts = col_text(5);
            if (!ts.empty()) meta["datetime"] = ts;
            turn_metadata.push_back(std::move(meta));

            turn_timestamps.push_back(std::move(ts));
        }

        int n = static_cast<int>(emb_rows.size());
        turn_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            turn_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = turn_embeddings.row(i).norm();
            if (nrm > 1e-12f) turn_embeddings.row(i) /= nrm;
        }

        turn_cache_valid = true;
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

    // bm25(turn_summaries_fts) over a MATCH expression → turn_summary_id →
    // score. Analogous to keyword_scores() but against the turn_summaries
    // FTS5 index.
    std::unordered_map<int, float> turn_keyword_scores(const std::string& match_expr) {
        std::unordered_map<int, float> out;
        if (match_expr.empty()) return out;
        Stmt s(db,
                "SELECT rowid, bm25(turn_summaries_fts) FROM turn_summaries_fts "
                "WHERE turn_summaries_fts MATCH ?");
        s.bind(1, match_expr);
        while (s.step()) {
            int id   = s.column_int(0);
            double val = s.column_double(1);
            out[id]  = static_cast<float>(-val);
        }
        return out;
    }

    // ---- phonetic ("dolphining" sounds-like) scoring ------------------
    // bm25 over a *_phon_fts index, keyed by the base rowid. `phon_expr` is an
    // FTS5 MATCH expression built from the *phonized* query (Double Metaphone
    // codes OR'd together) — see phon_match_expr(). Non-matching rows are absent
    // (0 to the caller). Higher = better (bm25 sign flipped), mirroring
    // keyword_scores(). One helper per corpus so the FTS table name is fixed.
    std::unordered_map<int, float> phon_scores_for(const char* fts_table,
                                                   const std::string& phon_expr) {
        std::unordered_map<int, float> out;
        if (phon_expr.empty()) return out;
        Stmt s(db, std::format(
            "SELECT rowid, bm25({0}) FROM {0} WHERE {0} MATCH ?", fts_table));
        s.bind(1, phon_expr);
        while (s.step()) {
            int id   = s.column_int(0);
            double val = s.column_double(1);
            out[id]  = static_cast<float>(-val);
        }
        return out;
    }
    std::unordered_map<int, float> sum_phon_scores(const std::string& e) {
        return phon_scores_for("summaries_phon_fts", e);
    }
    std::unordered_map<int, float> doc_phon_scores(const std::string& e) {
        return phon_scores_for("documents_phon_fts", e);
    }
    std::unordered_map<int, float> dec_phon_scores(const std::string& e) {
        return phon_scores_for("decisions_phon_fts", e);
    }
    std::unordered_map<int, float> turn_phon_scores(const std::string& e) {
        return phon_scores_for("turn_summaries_phon_fts", e);
    }

    // Build a phon MATCH expression: phonize the query, then OR the Double
    // Metaphone codes as quoted terms (same shape as fts_match_expr). Empty
    // when the query yields no codes (caller skips the phon pass).
    static std::string phon_match_expr(const std::string& query) {
        std::string phon = phonize(query);   // space-joined DM codes
        std::string expr, tok;
        auto flush = [&]() {
            if (tok.empty()) return;
            if (!expr.empty()) expr += " OR ";
            expr += "\"" + tok + "\"";
            tok.clear();
        };
        for (char c : phon) {
            if (c == ' ') flush(); else tok += c;
        }
        flush();
        return expr;
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

        auto ts = resolve_epoch(ts_override);

        // Every summary should record the model that produced it. Callers pass
        // the live model via metadata["model"]; empty → 0 → NULL (the raw-turn
        // sentinel), which a non-turn summary should never be. get_or_create_model
        // interns the name in the models table.
        std::string model_name = metadata.value("model", std::string(""));
        int model_id = get_or_create_model(model_name);

        // FTS5 sync triggers index the row from text/tags — no manual step.
        // phon = Double Metaphone of the body (dolphining sounds-like); the
        // *_phon_fts triggers index it. Always computed (cheap), even when the
        // embedding is deferred.
        Stmt s(db,
            "INSERT INTO summaries (text, embedding, phon, level, tags, created_at, model_id, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)");

        s.bind(1, text);
        if (defer_embedding) {
            s.bind_null(2);
        } else {
            bind_embedding(s.raw(), 2, emb);
        }
        s.bind(3, phonize(text));
        s.bind(4, level).bind(5, tags_str).bind(6, ts);
        if (model_id) s.bind(7, model_id); else s.bind_null(7);
        s.bind(8, ts);  // updated_at == created_at on insert

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

        // (imported_at is INTENTIONALLY still a TEXT 'YYYYMMDD' date-grouping
        // key, not a real timestamp -- do not convert to epoch. See
        // scripts/schema_db0.12.sql comment on documents.imported_at.)
        std::string imported_at = chunk.imported_at.empty() ? local_timestamp() : chunk.imported_at;

        // Lean documents schema (reference DDL): path/title/tags/year/
        // chunk_index. Only `text` is embedded; documents_fts sync triggers
        // index title/text/tags — no manual keyword step.
        Stmt s(db,
            "INSERT INTO documents "
            "(text, embedding, phon, path, title, tags, year, chunk_index, imported_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");

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
        s.bind(3, phonize(text));
        bind_opt(4, chunk.path);
        bind_opt(5, chunk.title);
        s.bind(6, chunk.tags);
        if (chunk.year <= 0) s.bind_null(7);
        else s.bind(7, chunk.year);
        if (chunk.chunk_index <= 0) s.bind_null(8);
        else s.bind(8, chunk.chunk_index);
        s.bind(9, imported_at);

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
    /// If session_name is non-empty, sets sessions.name (and name_source) on
    /// first sighting or updates it if the title has changed since last capture.
    int get_or_create_session(const std::string& guid,
                              const std::string& session_name = "",
                              const std::string& name_source = "") {
        if (guid.empty()) return 0;
        Stmt s(db, "SELECT session_id FROM sessions WHERE guid = ?");
        s.bind(1, guid);
        int id = 0;
        if (s.step()) id = s.column_int(0);
        if (!id) {
            Stmt ins(db, "INSERT INTO sessions (guid, name, name_source) VALUES (?, ?, ?)");
            ins.bind(1, guid);
            if (session_name.empty()) ins.bind_null(2);
            else                     ins.bind(2, session_name);
            if (session_name.empty()) ins.bind_null(3);
            else                      ins.bind(3, name_source);
            ins.step();
            id = static_cast<int>(sqlite3_last_insert_rowid(db));
        } else if (!session_name.empty()) {
            // Update name if changed (retitled session, or first time a name
            // is available for a previously-unnamed session).
            Stmt u(db, "UPDATE sessions SET name = ?, name_source = ? "
                       "WHERE session_id = ? AND (name IS NULL OR name != ?)");
            u.bind(1, session_name);
            u.bind(2, name_source);
            u.bind(3, id);
            u.bind(4, session_name);
            u.exec();
        }
        return id;
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
    // `source_timestamp` (non-empty, db format) overrides created_at for
    // historical imports; the regeneration-dedup window is skipped in that
    // case (it reasons about "now", which is meaningless for old turns).
    int store_turn(const std::string& user_text, const std::string& assistant_text,
                   const std::string& model_name, bool defer_embedding,
                   const std::string& session_guid,
                   const std::string& source_timestamp = "",
                   const std::string& session_name = "",
                   const std::string& name_source = "") {
        std::string u = normalize_path(user_text);
        std::string a = normalize_path(assistant_text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid, session_name, name_source);

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
        // Skipped entirely for historical imports (source_timestamp set).
        if (!a.empty() && source_timestamp.empty()) {
            Stmt p(db,
                "SELECT turn_id, user_text, created_at, session_id FROM turns "
                "ORDER BY turn_id DESC LIMIT 1");
            if (p.step()) {
                const int prev_id          = p.column_int(0);
                const std::string prev_u   = p.column_text(1);
                const int64_t prev_ts_ep   = p.column_int64(2);
                // 5-minute window: wide enough for a slow regeneration, narrow
                // enough that a genuinely re-typed identical prompt much later
                // is still recorded as its own turn.
                bool within_window = false;
                if (prev_u == u) {
                    int64_t now_ep = db_epoch();
                    within_window = (now_ep - prev_ts_ep) <= 300;
                }
                if (within_window) {
                    std::vector<float> emb2 = embedder->encode(turn_embed_text(u, a));
                    // Single timestamp value shared by the turns UPDATE and the
                    // placeholder cleanup/insert below — computed once, never
                    // re-read from the clock, so the two writes can't drift
                    // apart and strand the old (session_id, prev_ts) slot.
                    const int64_t new_ts = db_epoch();
                    Stmt up(db,
                        "UPDATE turns SET assistant_text = ?, embedding = ?, phon = ?, "
                        "model_id = COALESCE(?, model_id), "
                        "session_id = COALESCE(?, session_id), created_at = ? "
                        "WHERE turn_id = ?");
                    up.bind(1, a);
                    bind_embedding(up.raw(), 2, emb2);
                    up.bind(3, phonize(u + " " + a));
                    if (model_id) up.bind(4, model_id); else up.bind_null(4);
                    if (session_id) up.bind(5, session_id); else up.bind_null(5);
                    up.bind(6, new_ts);
                    up.bind(7, prev_id);
                    if (!up.exec())
                        throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
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
            "INSERT INTO turns (model_id, session_id, user_text, assistant_text, embedding, phon, created_at) "
            "VALUES (?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, u);
        if (a.empty()) s.bind_null(4);
        else s.bind(4, a);
        if (have_emb)
            bind_embedding(s.raw(), 5, emb);
        else s.bind_null(5);
        s.bind(6, phonize(a.empty() ? u : (u + " " + a)));
        s.bind(7, resolve_epoch(source_timestamp));

        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        int new_id = static_cast<int>(sqlite3_last_insert_rowid(db));
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
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "       datetime(t.created_at,'unixepoch','localtime') "
            "FROM turns t "
            "JOIN sessions ss ON t.session_id = ss.session_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "WHERE ss.guid = ? ";
        sql += asc ? "ORDER BY t.turn_id ASC"
                   : "ORDER BY t.created_at DESC, t.turn_id DESC";
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
            "UPDATE turns SET assistant_text = ?, embedding = ?, phon = ?, "
            "model_id = COALESCE(?, model_id) WHERE turn_id = ?");
        s.bind(1, a);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, phonize(u + " " + a));
        if (model_id) s.bind(4, model_id); else s.bind_null(4);
        s.bind(5, turn_id);
        return s.exec();
    }

    // ---- summaries (L2/L3) pipeline primitives (issue #22) ------------
    // Insert a summary row. level: 'turn' (L2) | 'session' (L3) | 'project'.
    // Embeds text, records model.
    // source_timestamp (non-empty) overrides the row's timestamp: L2 turn
    // summaries inherit the source turn's timestamp so the (session_id,
    // timestamp) pair links a turn to its summary (no FK column needed) and
    // stays stable across embedding-model changes.
    int store_summary(const std::string& text, const std::string& level,
                      const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& source_timestamp,
                      const std::string& tags) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding, phon, level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding(s.raw(), 4, emb);
        s.bind(5, phonize(t));
        s.bind(6, level).bind(7, tags);
        {
            const int64_t ts = resolve_epoch(source_timestamp);
            s.bind(8, ts);
            s.bind(9, ts);  // updated_at == created_at on insert
        }
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // ---- store_decision: write Level 6 curated decision/lesson ----------
    // Mirrors store_summary but the decisions table has no level/model/session
    // columns — just text/embedding/status/tags/timestamp. defer_embedding
    // leaves embedding NULL for a later backfill pass.
    int store_decision(const std::string& text, const std::string& status,
                       const std::string& tags,
                       const std::string& source_timestamp,
                       bool defer_embedding) {
        std::string t = normalize_path(strip_decision_number_prefix(text));

        Stmt s(db,
            "INSERT INTO decisions (text, embedding, phon, status, tags, created_at) "
            "VALUES (?,?,?,?,?,?)");
        s.bind(1, t);
        if (defer_embedding) {
            s.bind_null(2);
        } else {
            auto emb = embedder->encode(t);
            bind_embedding(s.raw(), 2, emb);
        }
        s.bind(3, phonize(t));
        s.bind(4, status.empty() ? "current" : status);
        s.bind(5, tags);
        s.bind(6, resolve_epoch(source_timestamp));
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_dec_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // ---- catch-up / recipe helpers ------------------------------------
    // Turns lacking an L2 summary, linked by (session_id, timestamp).
    // LEFT JOIN keeps turns whose summary row doesn't exist; ordered
    // newest-first so the summarizer works on the most recently active
    // turns first. This matters most for a manual resummarize (nulling
    // model_id on a batch of `summaries` rows via direct SQL): the turns
    // the user actually cares about right now are usually the recent
    // ones, and with catch_up_batch_size capping each tick, newest-first
    // means those show up with real summaries soonest instead of waiting
    // behind a long tail of old backlog.
    std::vector<TurnRecord> unsummarized_turns(int limit) {
        std::vector<TurnRecord> out;
        std::string sql =
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "       datetime(t.created_at,'unixepoch','localtime'), COALESCE(ss.guid, '') "
            "FROM turns t "
            "LEFT JOIN turn_summaries ts ON ts.turn_id = t.turn_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "LEFT JOIN sessions ss ON t.session_id = ss.session_id "
            "WHERE ts.turn_id IS NULL "
            "  AND t.assistant_text IS NOT NULL "
            "ORDER BY t.created_at DESC, t.turn_id DESC";
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

    // ---- turn_id-based turn_summaries helpers (v0.12.0) ----------------
    bool turn_summary_exists(int turn_id) {
        Stmt s(db, "SELECT 1 FROM turn_summaries WHERE turn_id = ? LIMIT 1");
        s.bind(1, turn_id);
        return s.step();
    }

    bool finalize_turn_summary(int turn_id, const std::string& text,
                                const std::string& summary_model_name) {
        std::string t = normalize_path(text);
        int model_id = get_or_create_model(summary_model_name);
        auto emb = embedder->encode(t);

        Stmt g(db, "SELECT created_at, model_id, session_id FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        if (!g.step()) return false;  // turn_id must exist
        int64_t turn_dt = g.column_int64(0);
        bool turn_model_null = g.is_null(1);
        int turn_model_id = turn_model_null ? 0 : g.column_int(1);
        bool session_null = g.is_null(2);
        int session_id = session_null ? 0 : g.column_int(2);

        // Plain insert — no placeholder row exists to conflict with anymore
        // (Reid's design decision). The unique partial index
        // idx_turn_summaries_turn_id_uniq still guards against an accidental
        // duplicate finalize for the same turn_id: Stmt::exec() just checks
        // sqlite3_step() == SQLITE_DONE, so a UNIQUE-constraint violation on
        // a second finalize for the same turn_id naturally falls through to
        // `return false` below (no exception, no special-case needed) —
        // matching this file's existing convention of surfacing insert
        // failures via the bool return rather than throwing.
        Stmt s(db,
            "INSERT INTO turn_summaries "
            "(turn_id, text, embedding, phon, session_id, turn_model_id, "
            " summary_model_id, turn_datetime, summarized_on) "
            "VALUES (?,?,?,?,?,?,?,?,unixepoch())");
        s.bind(1, turn_id);
        s.bind(2, t);
        bind_embedding(s.raw(), 3, emb);
        s.bind(4, phonize(t));
        if (session_null) s.bind_null(5); else s.bind(5, session_id);
        if (turn_model_null) s.bind_null(6); else s.bind(6, turn_model_id);
        if (model_id) s.bind(7, model_id); else s.bind_null(7);
        s.bind(8, turn_dt);
        bool ok = s.exec();
        if (ok) invalidate_turn_cache();
        return ok;
    }

    // Create a turn_summaries row marking a turn as "done" with no summary
    // text (trivial-turn skip, or poison-turn abandonment) — mirrors
    // finalize_turn_summary's turns lookup, just with text left NULL.
    // Reid's design decision: no placeholder mechanism, so a turn_summaries
    // row now always represents completed work (real summary, trivial-skip,
    // or poison-abandon), never an in-progress sentinel.
    bool mark_turn_summarized(int turn_id, const std::string& model_name) {
        int model_id = get_or_create_model(model_name);
        if (!model_id) return false;

        Stmt g(db, "SELECT created_at, model_id, session_id FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        if (!g.step()) return false;  // turn_id must exist
        int64_t turn_dt = g.column_int64(0);
        bool turn_model_null = g.is_null(1);
        int turn_model_id = turn_model_null ? 0 : g.column_int(1);
        bool session_null = g.is_null(2);
        int session_id = session_null ? 0 : g.column_int(2);

        Stmt s(db,
            "INSERT INTO turn_summaries "
            "(turn_id, session_id, turn_model_id, summary_model_id, turn_datetime, summarized_on) "
            "VALUES (?,?,?,?,?,unixepoch())");
        s.bind(1, turn_id);
        if (session_null) s.bind_null(2); else s.bind(2, session_id);
        if (turn_model_null) s.bind_null(3); else s.bind(3, turn_model_id);
        s.bind(4, model_id);
        s.bind(5, turn_dt);
        bool ok = s.exec();
        if (ok) invalidate_turn_cache();
        return ok;
    }

    // Draft-tagged summary rows for re-summarization (housekeeping retry).
    std::vector<DraftSummary> draft_summaries(int limit) {
        std::vector<DraftSummary> out;
        std::string sql =
            "SELECT s.summary_id, s.level, COALESCE(ss.guid, ''), "
            "       datetime(s.created_at,'unixepoch','localtime') "
            "FROM summaries s "
            "LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE s.tags LIKE '%draft%' "
            "ORDER BY s.created_at ASC, s.summary_id ASC";
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
            // NOTE: sessions_needing_close() has no live callers as of the
            // Phase-2 episode/session-rollup rework (see summarizer_service.cpp's
            // "sessions_needing_close() path is retired" comment) -- this query
            // is dead code today. Redirected to turn_summaries anyway (rather
            // than left pointing at the now-empty summaries/level='turn' rows)
            // so the function stays correct if it's ever revived.
            "WHERE EXISTS (SELECT 1 FROM turn_summaries s2 "
            "               WHERE s2.session_id = ss.session_id) "
            "  AND (SELECT MAX(t.created_at) FROM turns t "
            "        WHERE t.session_id = ss.session_id) < " + cutoff + " "
            "  AND NOT EXISTS (SELECT 1 FROM summaries s "
            "                   WHERE s.session_id = ss.session_id "
            "                     AND s.level = 'session') "
            "ORDER BY ss.session_id ASC";
        Stmt s(db, sql);
        while (s.step()) {
            const auto g = s.column_text(0);
            if (!g.empty()) out.push_back(g);
        }
        return out;
    }

    // ---- episode layer (EPISODE_PLAN Phase 1) --------------------------
    // Most recent episode's span-end (stored in updated_at) for a session.
    // Empty when the session has no episode rows yet.
    std::string last_episode_end(const std::string& session_guid) {
        if (session_guid.empty()) return {};
        Stmt s(db,
            "SELECT datetime(COALESCE(s.updated_at, s.created_at),'unixepoch','localtime') "
            "FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "ORDER BY s.created_at DESC, s.summary_id DESC LIMIT 1");
        s.bind(1, session_guid);
        if (s.step()) return s.column_text(0);
        return {};
    }

    // Non-draft L2 turn summaries for a session with timestamp > since_ts
    // (empty since_ts = all), oldest-first. Composes the closing episode.
    // since_ts arrives as a "%F %T" string (from last_episode_end(), which
    // renders the epoch column back to that format) -- parse it back to
    // epoch seconds to compare against the INTEGER created_at column.
    std::vector<SummaryRecord> l2_summaries_since(
            const std::string& session_guid, const std::string& since_ts) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        // L2 turn summaries now live entirely in turn_summaries (db_version
        // 0.12) -- redirected from the old summaries/level='turn' rows.
        // turn_summaries has no `status`/`tags` columns (no draft state for
        // turn-level rows under the new design), so `status` is left empty
        // and the old `tags != 'draft'` filter is simply dropped.
        std::string sql =
            "SELECT s.turn_summary_id, s.text, '', "
            "       datetime(s.turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? ";
        if (!since_ts.empty()) sql += "AND s.turn_datetime > ? ";
        sql += "ORDER BY s.turn_datetime ASC, s.turn_summary_id ASC";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (!since_ts.empty()) s.bind(2, resolve_epoch(since_ts));
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           s.column_text(2), s.column_text(3)});
        }
        return out;
    }

    // Candidate turns for similarity-based episode detection: joins
    // turn_summaries against turns to fetch both embeddings (turn + summary)
    // plus the turn-summary text. Rows where EITHER embedding is NULL are
    // skipped (boundary detection requires both signals).
    std::vector<EpisodeCandidateTurn> episode_candidate_turns(
            const std::string& session_guid, const std::string& since_ts) {
        std::vector<EpisodeCandidateTurn> out;
        if (session_guid.empty()) return out;
        const int dims = config().embedding_dimensions;

        std::string sql =
            "SELECT ts.turn_summary_id, ts.text, "
            "       datetime(ts.turn_datetime,'unixepoch','localtime'), "
            "       ts.turn_datetime, "
            "       t.embedding,  "  // raw turn embedding
            "       ts.embedding "   // turn-summary embedding
            "FROM turn_summaries ts "
            "JOIN turns t ON ts.turn_id = t.turn_id "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? "
            "  AND t.embedding IS NOT NULL "
            "  AND ts.embedding IS NOT NULL ";
        if (!since_ts.empty()) sql += "AND ts.turn_datetime > ? ";
        sql += "ORDER BY ts.turn_datetime ASC, ts.turn_summary_id ASC";

        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (!since_ts.empty()) s.bind(2, resolve_epoch(since_ts));

        bool warned = false;
        while (s.step()) {
            EpisodeCandidateTurn ct;
            ct.turn_summary_id   = s.column_int(0);
            ct.summary_text      = s.column_text(1);
            ct.timestamp         = s.column_text(2);
            ct.epoch             = s.column_int64(3);
            ct.turn_embedding    = decode_embedding_blob(
                s.column_blob(4), s.column_bytes(4), dims, "turns", ct.turn_summary_id, warned);
            ct.summary_embedding = decode_embedding_blob(
                s.column_blob(5), s.column_bytes(5), dims, "turn_summaries", ct.turn_summary_id, warned);
            out.push_back(std::move(ct));
        }
        return out;
    }

    // Insert one immutable level='episode' row: timestamp=first_ts (span
    // start), updated_at=last_ts (span end). Mirrors store_summary.
    int store_episode(const std::string& text, const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& first_ts, const std::string& last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding, phon, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding(s.raw(), 4, emb);
        s.bind(5, phonize(t));
        s.bind(6, std::string("episode"))
         .bind(7, std::string(""));
        s.bind(8, resolve_epoch(first_ts));
        s.bind(9, resolve_epoch(last_ts.empty() ? first_ts : last_ts));
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // Sessions whose open episode is ready to close: they have >=1 non-draft
    // L2 turn summary past the last episode's end, and their newest turn is
    // older than idle_minutes. Returns session GUIDs.
    std::vector<std::string> episodes_needing_close(int idle_minutes) {
        std::vector<std::string> out;
        if (idle_minutes <= 0) return out;
        const std::string cutoff =
            std::format("(unixepoch('now','-{} minutes'))", idle_minutes);
        // For each session: last episode end (updated_at of newest episode, or
        // '' when none). "Turns needing an episode" = L2 turn_summaries rows
        // with timestamp > that end (turn_summaries has no tags/'draft'
        // state under the new design, so the old tags != 'draft' filter is
        // simply dropped here). Idle = MAX(turn.timestamp) < cutoff.
        std::string sql =
            "SELECT ss.guid "
            "FROM sessions ss "
            "WHERE EXISTS ("
            "  SELECT 1 FROM turn_summaries s "
            "  WHERE s.session_id = ss.session_id "
            "    AND s.turn_datetime > COALESCE(("
            "        SELECT COALESCE(e.updated_at, e.created_at) FROM summaries e "
            "        WHERE e.session_id = ss.session_id AND e.level = 'episode' "
            "        ORDER BY e.created_at DESC, e.summary_id DESC LIMIT 1), 0)) "
            "  AND (SELECT MAX(t.created_at) FROM turns t "
            "        WHERE t.session_id = ss.session_id) < " + cutoff + " "
            "ORDER BY ss.session_id ASC";
        Stmt s(db, sql);
        while (s.step()) {
            const auto g = s.column_text(0);
            if (!g.empty()) out.push_back(g);
        }
        return out;
    }

    // ---- Phase 2: boundary-triggered session/project rollups -----------
    // All episode texts for a session, oldest-first. Corpus for the session
    // rollup (episodes + any tail L2 turns are combined by the caller).
    std::vector<std::string> episode_texts(const std::string& session_guid) {
        std::vector<std::string> out;
        if (session_guid.empty()) return out;
        Stmt s(db,
            "SELECT s.text FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "ORDER BY s.created_at ASC, s.summary_id ASC");
        s.bind(1, session_guid);
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }

    // Stamp updated_at = now for a running rollup row.
    bool set_summary_updated_at(int summary_id) {
        Stmt s(db, "UPDATE summaries SET updated_at = ? WHERE summary_id = ?");
        s.bind(1, db_epoch()).bind(2, summary_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }

    // ---- boundary-detection watermark (settings key/value table) --------
    int64_t get_watermark(const std::string& key) {
        Stmt s(db, "SELECT value FROM settings WHERE key = ?");
        s.bind(1, key);
        if (s.step()) {
            try { return std::stoll(s.column_text(0)); } catch (...) { return 0; }
        }
        return 0;
    }

    void set_watermark(const std::string& key, int64_t turn_id) {
        Stmt s(db, "INSERT INTO settings (key, value) VALUES (?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, key);
        s.bind(2, std::to_string(turn_id));
        s.exec();
    }

    // Maximal runs of consecutive (by turn_id) turns sharing the same
    // non-NULL session_id, edge-detected via a running group id that
    // increments whenever session_id changes from the previous row. The
    // final/currently-open group (grp == MAX(grp)) is always excluded —
    // it's still open because a future turn could extend it.
    std::vector<ClosedRun> sessions_needing_close_boundary() {
        std::vector<ClosedRun> out;
        int64_t watermark = get_watermark(std::string(kSessionBoundaryWatermarkKey));
        // Structural floor (Reid's invariant: "no automatic lookback to do
        // any session summary before the last session summary"). A run is
        // eligible only if NO level='session' summary already covers it --
        // i.e. none exists for this session whose stored span end
        // (updated_at == the run's last_ts at store time) reaches this run's
        // last turn timestamp. This makes correctness independent of the
        // watermark: even with watermark=0 (e.g. a migration that never
        // stamped the key -> get_watermark() defaults to 0), an
        // already-summarized run is never re-closed. The watermark remains as
        // a cheap first-pass filter (`last_turn_id > ?`) so the common
        // steady-state scan stays bounded, but it is no longer the sole
        // guard against re-summarizing history.
        Stmt s(db, R"(
            WITH lagged AS (
                SELECT t.turn_id, t.session_id, t.created_at,
                    LAG(t.session_id) OVER (ORDER BY t.turn_id) AS prev_sid
                FROM turns t
            ),
            numbered AS (
                SELECT turn_id, session_id, created_at,
                    SUM(CASE WHEN session_id IS prev_sid THEN 0 ELSE 1 END)
                        OVER (ORDER BY turn_id) AS grp
                FROM lagged
            ),
            runs AS (
                SELECT session_id, grp,
                       MIN(turn_id) AS first_turn_id, MAX(turn_id) AS last_turn_id,
                       MIN(created_at) AS first_ts, MAX(created_at) AS last_ts
                FROM numbered
                GROUP BY grp
            )
            SELECT ss.guid, r.first_ts, r.last_ts, r.first_turn_id, r.last_turn_id
            FROM runs r
            JOIN sessions ss ON r.session_id = ss.session_id
            WHERE r.session_id IS NOT NULL
              AND r.last_turn_id > ?
              AND r.grp < (SELECT MAX(grp) FROM numbered)
              AND NOT EXISTS (
                  SELECT 1 FROM summaries sm
                  WHERE sm.level = 'session'
                    AND sm.session_id = r.session_id
                    AND sm.updated_at >= r.last_ts
              )
            ORDER BY r.first_turn_id ASC
        )");
        s.bind(1, watermark);
        while (s.step()) {
            out.push_back({s.column_text(0), s.column_int64(1), s.column_int64(2),
                            s.column_int(3), s.column_int(4)});
        }
        return out;
    }

    // Same shape as sessions_needing_close_boundary(), but grouped by a
    // time gap (>= gap_days days) between consecutive turns (by turn_id)
    // rather than a session_id change, and NOT restricted to non-NULL
    // session_id (project runs span all turns, anonymous included).
    std::vector<ClosedRun> projects_needing_close_boundary(int gap_days) {
        std::vector<ClosedRun> out;
        if (gap_days <= 0) return out;
        int64_t gap_seconds = static_cast<int64_t>(gap_days) * 86400;
        int64_t watermark = get_watermark(std::string(kProjectBoundaryWatermarkKey));
        Stmt s(db, R"(
            WITH lagged AS (
                SELECT t.turn_id, t.created_at,
                    LAG(t.created_at) OVER (ORDER BY t.turn_id) AS prev_ts
                FROM turns t
            ),
            numbered AS (
                SELECT turn_id, created_at,
                    SUM(CASE
                            WHEN prev_ts IS NULL THEN 1
                            WHEN (created_at - prev_ts) >= ? THEN 1
                            ELSE 0
                        END) OVER (ORDER BY turn_id) AS grp
                FROM lagged
            ),
            runs AS (
                SELECT grp,
                       MIN(turn_id) AS first_turn_id, MAX(turn_id) AS last_turn_id,
                       MIN(created_at) AS first_ts, MAX(created_at) AS last_ts
                FROM numbered
                GROUP BY grp
            )
            SELECT r.first_ts, r.last_ts, r.first_turn_id, r.last_turn_id
            FROM runs r
            WHERE r.last_turn_id > ?
              AND r.grp < (SELECT MAX(grp) FROM numbered)
            ORDER BY r.first_turn_id ASC
        )");
        s.bind(1, gap_seconds);
        s.bind(2, watermark);
        while (s.step()) {
            out.push_back({"", s.column_int64(0), s.column_int64(1),
                            s.column_int(2), s.column_int(3)});
        }
        return out;
    }

    void advance_session_boundary_watermark(int turn_id) {
        set_watermark(std::string(kSessionBoundaryWatermarkKey), turn_id);
    }

    void advance_project_boundary_watermark(int turn_id) {
        set_watermark(std::string(kProjectBoundaryWatermarkKey), turn_id);
    }

    // Bounded text-gathering for a closed session run: episodes whose full
    // span [created_at, COALESCE(updated_at,created_at)] lies entirely
    // within [first_ts, last_ts] for this session_guid, plus trailing
    // turn_summaries after the newest such episode's end (or from
    // first_ts if there are none) up to last_ts. Unlike episode_texts()/
    // l2_summaries_since() (unbounded above), this never reaches into a
    // LATER run of the same session_guid.
    std::vector<std::string> bounded_session_rollup_texts(
            const std::string& session_guid, int64_t first_ts, int64_t last_ts) {
        std::vector<std::string> out;
        if (session_guid.empty()) return out;

        // Episodes whose full span lies within the run, oldest-first.
        Stmt e(db,
            "SELECT s.text, COALESCE(s.updated_at, s.created_at) FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "  AND s.created_at >= ? AND COALESCE(s.updated_at, s.created_at) <= ? "
            "ORDER BY s.created_at ASC, s.summary_id ASC");
        e.bind(1, session_guid);
        e.bind(2, first_ts);
        e.bind(3, last_ts);
        int64_t newest_episode_end = first_ts - 1;  // sentinel: no episodes yet
        while (e.step()) {
            auto t = e.column_text(0);
            int64_t end_ts = e.column_int64(1);
            if (!t.empty()) out.push_back(std::move(t));
            if (end_ts > newest_episode_end) newest_episode_end = end_ts;
        }

        // Trailing turn_summaries strictly after the newest episode's end
        // (or from first_ts-1 if no episodes), up to and including last_ts.
        Stmt ts_q(db,
            "SELECT ts.text FROM turn_summaries ts "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? AND ts.text IS NOT NULL "
            "  AND ts.turn_datetime > ? AND ts.turn_datetime <= ? "
            "ORDER BY ts.turn_datetime ASC, ts.turn_summary_id ASC");
        ts_q.bind(1, session_guid);
        ts_q.bind(2, newest_episode_end);
        ts_q.bind(3, last_ts);
        while (ts_q.step()) {
            auto t = ts_q.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }

    // Bounded text-gathering for a closed project run: every level='session'
    // row whose full span [created_at, COALESCE(updated_at,created_at)]
    // lies entirely within [first_ts, last_ts], session-unscoped (a project
    // run spans sessions), oldest-first.
    std::vector<std::string> bounded_project_rollup_texts(
            int64_t first_ts, int64_t last_ts) {
        std::vector<std::string> out;
        Stmt s(db,
            "SELECT text FROM summaries "
            "WHERE level = 'session' "
            "  AND created_at >= ? AND COALESCE(updated_at, created_at) <= ? "
            "ORDER BY created_at ASC, summary_id ASC");
        s.bind(1, first_ts);
        s.bind(2, last_ts);
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }

    // Insert one immutable level='session' row spanning a closed run.
    // Mirrors store_episode's shape exactly, except created_at/updated_at
    // are bound directly as epoch ints (ClosedRun already carries real
    // epoch seconds — no resolve_epoch() round-trip needed).
    int store_session_summary(const std::string& text, const std::string& model_name,
                              const std::string& session_guid,
                              int64_t first_ts, int64_t last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding, phon, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding(s.raw(), 4, emb);
        s.bind(5, phonize(t));
        s.bind(6, std::string("session"))
         .bind(7, std::string(""));
        s.bind(8, first_ts);
        s.bind(9, last_ts);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

    // Insert one immutable level='project' row spanning a closed run.
    // Session-unscoped (session_id NULL). Otherwise mirrors store_episode.
    int store_project_summary(const std::string& text, const std::string& model_name,
                              int64_t first_ts, int64_t last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding, phon, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        s.bind_null(2);
        s.bind(3, t);
        bind_embedding(s.raw(), 4, emb);
        s.bind(5, phonize(t));
        s.bind(6, std::string("project"))
         .bind(7, std::string(""));
        s.bind(8, first_ts);
        s.bind(9, last_ts);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }

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
            "SELECT s.summary_id, s.text, s.created_at "
            "FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = ? "
            "ORDER BY s.created_at DESC, s.summary_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid).bind(2, level);
        if (limit > 0) s.bind(3, limit);
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           std::string(), s.column_text(2)});
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

    // Exact-match existence check for import idempotency: same text +
    // same created_at timestamp already present in summaries. Used by
    // `ragger import conversations`/`import summaries` so re-running an
    // import (or feeding overlapping exports) doesn't duplicate rows.
    // Normalizes `text` the same way store_summary() does before storing
    // (absolute home-dir paths -> "~/") — otherwise a chunk containing
    // e.g. "/Users/reid/..." never matches what's actually in the table
    // (already normalized to "~/...") and gets re-inserted on every run.
    bool summary_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(text);
        Stmt s(db, "SELECT 1 FROM summaries WHERE text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, created_at);
        return s.step();
    }

    // Mirrors summary_exists_exact for the decisions table (same
    // normalize-before-compare fix applies here too — plus the
    // decision-number-prefix strip, since store_decision() strips that
    // before storing too).
    bool decision_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(strip_decision_number_prefix(text));
        Stmt s(db, "SELECT 1 FROM decisions WHERE text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, created_at);
        return s.step();
    }

    // Fuzzy dedup against live-captured turns: exact user_text, timestamp
    // within ±window_seconds. Catches import overlap with turns Ragger
    // captured live (identical content, a few seconds of clock skew —
    // message send time vs. capture time, plus any tz-conversion rounding).
    bool turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                           int window_seconds) {
        Stmt s(db,
            "SELECT 1 FROM turns WHERE user_text = ? "
            "AND created_at BETWEEN datetime(?, ?) AND datetime(?, ?) LIMIT 1");
        std::string neg = "-" + std::to_string(window_seconds) + " seconds";
        std::string pos = "+" + std::to_string(window_seconds) + " seconds";
        s.bind(1, user_text);
        s.bind(2, ts).bind(3, neg);
        s.bind(4, ts).bind(5, pos);
        return s.step();
    }

    // Collapse whitespace, drop zero-width/variation-selector/control
    // characters, and case-fold so the same exchange pasted through two
    // different clients (Telegram markdown escaping vs. Claude.ai's raw
    // text) compares equal. Comparison-only — never used for storage.
    static std::string normalize_for_compare(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool prev_space = false;
        for (unsigned char c : s) {
            // Drop ASCII control chars outright (keep the rest of UTF-8
            // multibyte sequences as-is; this is a coarse cross-source
            // comparison, not a full Unicode normalizer).
            if (c < 0x20 && c != '\n') continue;
            if (c == '\n' || c == '\t' || c == ' ') {
                if (!prev_space && !out.empty()) out.push_back(' ');
                prev_space = true;
                continue;
            }
            prev_space = false;
            out.push_back(static_cast<char>(std::tolower(c)));
        }
        while (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    }

    // Cross-source overlap lookup: scan turns for a user_text that
    // normalizes-equal (see normalize_for_compare), ignoring timestamp
    // entirely. Used to reconcile the same exchange captured from two
    // different importers whose timestamps have no reason to agree.
    std::optional<TurnRecord> find_turn_by_text(const std::string& user_text) {
        std::string target = normalize_for_compare(user_text);
        if (target.empty()) return std::nullopt;
        Stmt s(db,
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "t.created_at, ss.guid "
            "FROM turns t "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "LEFT JOIN sessions ss ON t.session_id = ss.session_id "
            "WHERE t.user_text LIKE '%' || substr(?, 1, 40) || '%'");
        // Narrow with a cheap substring prefilter (avoids a full table
        // scan through normalize_for_compare on every row), then confirm
        // with the real normalized comparison in C++.
        s.bind(1, user_text.substr(0, 40));
        while (s.step()) {
            std::string cand = s.column_text(1);
            if (normalize_for_compare(cand) == target) {
                return TurnRecord{
                    s.column_int(0), cand, s.column_text(2),
                    s.column_text(3), s.column_text(4), s.column_text(5)
                };
            }
        }
        return std::nullopt;
    }

    // Upgrade an existing turn's timestamp/session_guid in place. Empty
    // args leave that field untouched. session_guid resolves/creates a
    // sessions row same as store_turn.
    bool update_turn_meta(int turn_id, const std::string& timestamp,
                          const std::string& session_guid) {
        if (timestamp.empty() && session_guid.empty()) return true;
        std::string sql = "UPDATE turns SET ";
        bool first = true;
        if (!timestamp.empty()) { sql += "created_at = ?"; first = false; }
        if (!session_guid.empty()) {
            if (!first) sql += ", ";
            sql += "session_id = ?";
        }
        sql += " WHERE turn_id = ?";
        Stmt s(db, sql);
        int idx = 1;
        if (!timestamp.empty()) s.bind(idx++, timestamp);
        if (!session_guid.empty())
            s.bind(idx++, get_or_create_session(session_guid));
        s.bind(idx, turn_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }

    // Recipe ingredients (issue #23): recent summaries of a given level, and
    // current decisions — fetched by recency (not semantic search) for the
    // default tiered payload. Returned newest-first.
    std::vector<std::string> recent_summaries(const std::string& level, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM summaries WHERE level = ? "
            "ORDER BY created_at DESC, summary_id DESC LIMIT ?");
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
            "ORDER BY created_at DESC, decision_id DESC LIMIT ?");
        s.bind(1, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }

    // Set a decision's status (e.g. "roadmap" -> "current" once planned
    // work is done, or -> "superseded"/"deprecated" once stale).
    bool set_decision_status(int decision_id, const std::string& status) {
        Stmt s(db, "UPDATE decisions SET status = ? WHERE decision_id = ?");
        s.bind(1, status).bind(2, decision_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_dec_cache();
        return ok;
    }

    // Decisions with an arbitrary status, most recent first — e.g.
    // status="roadmap" to list planned/future work explicitly (roadmap
    // entries are deliberately excluded from current_decisions()'s
    // recall-pipeline query so unfinished plans don't clutter every
    // session's context).
    std::vector<std::string> decisions_by_status(const std::string& status, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM decisions WHERE status = ? "
            "ORDER BY created_at DESC, decision_id DESC LIMIT ?");
        s.bind(1, status);
        s.bind(2, limit);
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
            "UPDATE summaries SET text = ?, embedding = ?, phon = ?, "
            "model_id = COALESCE(?, model_id) WHERE summary_id = ?");
        s.bind(1, t);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, phonize(t));
        if (model_id) s.bind(4, model_id); else s.bind_null(4);
        s.bind(5, summary_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_cache();
        return ok;
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
            "UPDATE summaries SET text = ?, embedding = ?, phon = ?, tags = ? "
            "WHERE summary_id = ?");

        stmt.bind(1, text);
        if (defer_embedding) {
            stmt.bind_null(2);
        } else {
            bind_embedding(stmt.raw(), 2, emb);
        }
        stmt.bind(3, phonize(text));
        stmt.bind(4, tags_str);
        stmt.bind(5, memory_id);

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
        ensure_turn_cache();
        int n_sum = static_cast<int>(cached_ids.size());
        int n_doc = static_cast<int>(doc_ids.size());
        int n_dec = static_cast<int>(dec_ids.size());
        int n_turn = static_cast<int>(turn_ids.size());
        if (n_sum == 0 && n_doc == 0 && n_dec == 0 && n_turn == 0) return {{}, {{"corpus_size", 0}}};

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
        candidates.reserve(static_cast<size_t>(n_sum + n_doc + n_dec + n_turn));

        // Score one corpus: vector cosine blended with FTS5 bm25 (keyword) and
        // the phonetic "sounds-like" signal. Each signal is min-max normalized
        // to [0,1] then weighted-summed. Shared by the summaries/documents/
        // decisions passes so the blend logic lives in exactly one place.
        auto score_corpus =
            [&](const std::vector<int>& ids,
                const std::vector<std::string>& texts,
                const Eigen::MatrixXf& cache_emb,
                const std::vector<json>& meta,
                const std::vector<std::string>& ts,
                const std::unordered_map<int, float>& kw,
                const std::unordered_map<int, float>& ph) {
            int n = static_cast<int>(ids.size());
            if (n == 0) return;

            // cache_emb rows are L2-normalized at cache-build time, and q_norm
            // is the normalized query vector, so this product is raw cosine.
            Eigen::VectorXf similarities = cache_emb * q_norm;   // raw cosine, reported

            auto norm_minmax = [](Eigen::VectorXf& v) {
                float mn = v.minCoeff(), mx = v.maxCoeff();
                if (mx > mn) v = (v.array() - mn) / (mx - mn);
            };
            auto gather = [&](const std::unordered_map<int, float>& m) {
                Eigen::VectorXf v(n);
                for (int i = 0; i < n; ++i) {
                    auto it = m.find(ids[i]);
                    v(i) = (it == m.end()) ? 0.0f : it->second;
                }
                return v;
            };

            bool use_kw   = config().bm25_enabled && !kw.empty();
            bool use_phon = config().phon_weight > 0.0f && !ph.empty();

            // Normalized [0,1] signal vectors. vec is always computed; kw/ph
            // only when their signal is active. When neither keyword nor phon
            // contributes, ranking falls back to raw cosine (combined) so the
            // vec-only path behaves exactly as before.
            Eigen::VectorXf vec_norm = similarities;
            Eigen::VectorXf kw_norm, ph_norm;
            Eigen::VectorXf combined = similarities;
            if (use_kw || use_phon) {
                norm_minmax(vec_norm);
                combined = config().vector_weight * vec_norm;
                if (use_kw) {
                    kw_norm = gather(kw);
                    norm_minmax(kw_norm);
                    combined += config().bm25_weight * kw_norm;
                }
                if (use_phon) {
                    ph_norm = gather(ph);
                    norm_minmax(ph_norm);
                    combined += config().phon_weight * ph_norm;
                }
            }

            for (int i = 0; i < n; ++i) {
                SearchResult sr{ids[i], texts[i], similarities(i), meta[i], ts[i]};
                // Per-signal breakdown for RAGGER_STATS. -1 marks an inactive
                // signal so the analysis can tell "0 contribution" apart from
                // "not part of this search". vec_score is the normalized cosine
                // that actually fed the blend (== raw cosine on the vec-only
                // path, since no min-max is applied there).
                sr.vec_score  = vec_norm(i);
                sr.bm25_score = use_kw   ? kw_norm(i) : -1.0f;
                sr.phon_score = use_phon ? ph_norm(i) : -1.0f;
                sr.blended    = combined(i);
                candidates.push_back({combined(i), std::move(sr)});
            }
        };

        std::string match_expr = fts_match_expr(query);
        std::string phon_expr  = config().phon_weight > 0.0f
                               ? phon_match_expr(query) : std::string{};
        auto no_scores = std::unordered_map<int, float>{};
        score_corpus(cached_ids, cached_texts, cached_embeddings,
                     cached_metadata, cached_timestamps,
                     config().bm25_enabled ? keyword_scores(match_expr) : no_scores,
                     phon_expr.empty() ? no_scores : sum_phon_scores(phon_expr));
        score_corpus(doc_ids, doc_texts, doc_embeddings,
                     doc_metadata, doc_timestamps,
                     config().bm25_enabled ? doc_keyword_scores(match_expr) : no_scores,
                     phon_expr.empty() ? no_scores : doc_phon_scores(phon_expr));
        score_corpus(dec_ids, dec_texts, dec_embeddings,
                     dec_metadata, dec_timestamps,
                     config().bm25_enabled ? dec_keyword_scores(match_expr) : no_scores,
                     phon_expr.empty() ? no_scores : dec_phon_scores(phon_expr));
        score_corpus(turn_ids, turn_texts, turn_embeddings,
                     turn_metadata, turn_timestamps,
                     config().bm25_enabled ? turn_keyword_scores(match_expr) : no_scores,
                     phon_expr.empty() ? no_scores : turn_phon_scores(phon_expr));
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
            {"corpus_size",  n_sum + n_doc + n_dec + n_turn}
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

    std::vector<std::pair<std::string, int64_t>> table_row_counts() const {
        // User-facing tables for the dashboard status pane, in display order.
        static const char* kTables[] = {
            "turns", "summaries", "turn_summaries", "sessions",
            "documents", "decisions", "statements", "models",
        };
        std::vector<std::pair<std::string, int64_t>> out;
        for (const char* t : kTables) {
            try {
                Stmt s(db, std::string("SELECT COUNT(*) FROM ") + t);
                if (s.step())
                    out.emplace_back(t, static_cast<int64_t>(s.column_int(0)));
            } catch (...) {
                // Table absent in this schema version — just skip it.
            }
        }
        return out;
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
            "SELECT summary_id, text, level, tags, created_at "
            "FROM summaries ORDER BY summary_id");

        while (s.step()) {
            int id        = s.column_int(0);
            auto text     = s.column_text(1);
            auto lvl      = s.column_text(2);
            auto tag      = s.column_text(3);
            auto ts       = s.column_text(4);

            json meta = json::object();
            if (!lvl.empty()) meta["level"]  = lvl;
            if (!tag.empty()) meta["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(meta), std::move(ts)});
        }
        return results;
    }

    // Re-encode embeddings across all four embedded tables (turns, summaries,
    // decisions, documents). Two modes, selected by `only_missing`:
    //   * false → re-encode every row (full rebuild; used by CLI
    //             `ragger rebuild-embeddings` after a model/dtype change)
    //   * true  → only rows WHERE embedding IS NULL (cheap backfill of
    //             deferred-embed rows; used at server startup)
    // When `progress` is set, prints a running "n/total" line to stdout
    // (interactive CLI). Returns the number of rows (re-)embedded.
    int embed_tables(Embedder& emb_ref, bool only_missing, bool progress) {
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

        const char* where = only_missing ? " WHERE embedding IS NULL" : "";

        // Total is only needed for the progress display.
        int total_count = 0;
        if (progress) {
            for (auto& t : tables) {
                Stmt s(db, std::format("SELECT COUNT(*) FROM {}{}", t.table, where));
                if (s.step())
                    total_count += s.column_int(0);
            }
        }

        int done = 0;
        for (auto& t : tables) {
            std::string sel = t.extra_col
                ? std::format("SELECT {} AS id, {}, {} FROM {}{}",
                              t.id_col, t.text_col, t.extra_col, t.table, where)
                : std::format("SELECT {} AS id, {} AS text FROM {}{}",
                              t.id_col, t.text_col, t.table, where);
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

                ++done;
                if (progress) {
                    std::cout << std::format(ragger::lang::MSG_REBUILD_EMBEDDINGS_PROGRESS,
                                             done, total_count);
                    std::cout.flush();
                }
            }
        }

        if (progress)
            std::cout << "\n";

        if (done > 0 || !only_missing) {
            invalidate_cache();
            invalidate_doc_cache();
            invalidate_dec_cache();
        }
        return done;
    }

    // Full re-encode of every embedded row (interactive, with progress).
    int rebuild_embeddings(Embedder& emb_ref) {
        return embed_tables(emb_ref, /*only_missing=*/false, /*progress=*/true);
    }

    // Cheap backfill: embed only rows left NULL by deferred-embed stores.
    int backfill_embeddings(Embedder& emb_ref) {
        return embed_tables(emb_ref, /*only_missing=*/true, /*progress=*/false);
    }

    // (Re)compute the phon (Double Metaphone) column for every context-table
    // row from its text. No embedder needed — pure string work. `only_missing`
    // limits to rows WHERE phon IS NULL (cheap backfill after a migration adds
    // the column); false recomputes all rows (e.g. after a phonize() change).
    // The *_phon_fts sync triggers reindex each UPDATE automatically. Returns
    // the number of rows rewritten.
    int rebuild_phon(bool only_missing, bool progress) {
        struct TableSpec {
            const char* table;
            const char* id_col;
            const char* text_col;
            const char* extra_col;  // joined with text_col for turns
        };
        static constexpr TableSpec tables[] = {
            { "turns",     "turn_id",     "user_text", "assistant_text" },
            { "summaries", "summary_id",  "text",      nullptr          },
            { "decisions", "decision_id", "text",      nullptr          },
            { "documents", "document_id", "text",      nullptr          },
        };
        const char* where = only_missing ? " WHERE phon IS NULL" : "";

        int total_count = 0;
        if (progress) {
            for (auto& t : tables) {
                Stmt s(db, std::format("SELECT COUNT(*) FROM {}{}", t.table, where));
                if (s.step()) total_count += s.column_int(0);
            }
        }

        int done = 0;
        for (auto& t : tables) {
            std::string sel = t.extra_col
                ? std::format("SELECT {} AS id, {}, {} FROM {}{}",
                              t.id_col, t.text_col, t.extra_col, t.table, where)
                : std::format("SELECT {} AS id, {} AS text FROM {}{}",
                              t.id_col, t.text_col, t.table, where);
            std::string upd = std::format("UPDATE {} SET phon = ? WHERE {} = ?",
                                          t.table, t.id_col);
            Stmt select_stmt(db, sel);
            while (select_stmt.step()) {
                int id = select_stmt.column_int(0);
                std::string text = select_stmt.column_text(1);
                if (t.extra_col) {
                    std::string a = select_stmt.column_text(2);
                    if (!a.empty()) text += " " + a;
                }
                Stmt update(db, upd);
                update.bind(1, phonize(text));
                update.bind(2, id);
                update.step();
                ++done;
                if (progress) {
                    std::cout << std::format("\rComputing phonetic codes: {}/{}",
                                             done, total_count);
                    std::cout.flush();
                }
            }
        }
        if (progress) std::cout << "\n";
        return done;
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

    // Per-row embedding write-back for the other three context tables.
    // Mirrors update_document_embedding; each invalidates the cache that
    // backs its table's vector search.
    bool update_decision_embedding(int decision_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE decisions SET embedding = ? WHERE decision_id = ?");
        bind_embedding(s.raw(), 1, emb);
        s.bind(2, decision_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_dec_cache();
            return true;
        }
        return false;
    }

    bool update_summary_embedding(int summary_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE summaries SET embedding = ? WHERE summary_id = ?");
        bind_embedding(s.raw(), 1, emb);
        s.bind(2, summary_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }

    bool update_turn_embedding(int turn_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE turns SET embedding = ? WHERE turn_id = ?");
        bind_embedding(s.raw(), 1, emb);
        s.bind(2, turn_id);
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

        // Lean v2 summaries expose level / tags / timestamp. Filter
        // on those columns; any other requested key matches nothing (there is
        // no free-form metadata blob in the lean schema).
        std::string sql = "SELECT summary_id, text, level, tags, created_at "
                          "FROM summaries";
        std::string where;
        std::vector<std::string> binds;

        for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
            const std::string& k = it.key();
            if (k == "level") {
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
            where += (where.empty() ? " WHERE " : " AND ") + std::string("created_at >= ?");
            binds.push_back(after);
        }
        if (!before.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("created_at < ?");
            binds.push_back(before);
        }

        sql += where + " ORDER BY created_at DESC";
        if (limit > 0) sql += " LIMIT " + std::to_string(limit);

        Stmt stmt(db, sql);
        for (size_t i = 0; i < binds.size(); ++i) {
            stmt.bind(static_cast<int>(i + 1), binds[i]);
        }

        while (stmt.step()) {
            int id        = stmt.column_int(0);
            auto text     = stmt.column_text(1);
            auto lvl      = stmt.column_text(2);
            auto tag      = stmt.column_text(3);
            auto ts       = stmt.column_text(4);

            json metadata = json::object();
            if (!lvl.empty()) metadata["level"]  = lvl;
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
                              const std::string& session_guid,
                              const std::string& source_timestamp,
                              const std::string& session_name,
                              const std::string& name_source) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_turn(user_text, assistant_text, model_name,
                             defer_embedding, session_guid, source_timestamp,
                             session_name, name_source);
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
                                 const std::string& model_name,
                                 const std::string& session_guid,
                                 const std::string& source_timestamp,
                                 const std::string& tags) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_summary(text, level, model_name, session_guid,
                                source_timestamp, tags);
}

bool SqliteBackend::summary_exists_exact(const std::string& text, const std::string& created_at) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->summary_exists_exact(text, created_at);
}

bool SqliteBackend::decision_exists_exact(const std::string& text, const std::string& created_at) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->decision_exists_exact(text, created_at);
}

bool SqliteBackend::turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                                      int window_seconds) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_exists_fuzzy(user_text, ts, window_seconds);
}

std::optional<TurnRecord> SqliteBackend::find_turn_by_text(const std::string& user_text) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->find_turn_by_text(user_text);
}

bool SqliteBackend::update_turn_meta(int turn_id, const std::string& timestamp,
                                     const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_turn_meta(turn_id, timestamp, session_guid);
}

bool SqliteBackend::update_summary_text(int summary_id, const std::string& text,
                                        const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_summary_text(summary_id, text, model_name);
}

bool SqliteBackend::turn_summary_exists(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_summary_exists(turn_id);
}

bool SqliteBackend::finalize_turn_summary(int turn_id, const std::string& text,
                                          const std::string& summary_model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->finalize_turn_summary(turn_id, text, summary_model_name);
}

bool SqliteBackend::mark_turn_summarized(int turn_id, const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->mark_turn_summarized(turn_id, model_name);
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

bool SqliteBackend::set_decision_status(int decision_id, const std::string& status) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_decision_status(decision_id, status);
}

std::vector<std::string> SqliteBackend::decisions_by_status(const std::string& status, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->decisions_by_status(status, limit);
}

std::vector<TurnRecord> SqliteBackend::unsummarized_turns(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->unsummarized_turns(limit);
}

std::vector<DraftSummary> SqliteBackend::draft_summaries(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->draft_summaries(limit);
}

std::vector<std::string> SqliteBackend::sessions_needing_close(int pause_minutes) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->sessions_needing_close(pause_minutes);
}

std::string SqliteBackend::last_episode_end(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->last_episode_end(session_guid);
}

std::vector<SummaryRecord> SqliteBackend::l2_summaries_since(
        const std::string& session_guid, const std::string& since_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->l2_summaries_since(session_guid, since_ts);
}

std::vector<EpisodeCandidateTurn> SqliteBackend::episode_candidate_turns(
        const std::string& session_guid, const std::string& since_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episode_candidate_turns(session_guid, since_ts);
}

int SqliteBackend::store_episode(const std::string& text,
        const std::string& model_name, const std::string& session_guid,
        const std::string& first_ts, const std::string& last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_episode(text, model_name, session_guid, first_ts, last_ts);
}

std::vector<std::string> SqliteBackend::episodes_needing_close(int idle_minutes) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episodes_needing_close(idle_minutes);
}

std::vector<std::string> SqliteBackend::episode_texts(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episode_texts(session_guid);
}

bool SqliteBackend::set_summary_updated_at(int summary_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_summary_updated_at(summary_id);
}

std::vector<ClosedRun> SqliteBackend::sessions_needing_close_boundary() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->sessions_needing_close_boundary();
}

std::vector<ClosedRun> SqliteBackend::projects_needing_close_boundary(int gap_days) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->projects_needing_close_boundary(gap_days);
}

void SqliteBackend::advance_session_boundary_watermark(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->advance_session_boundary_watermark(turn_id);
}

void SqliteBackend::advance_project_boundary_watermark(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->advance_project_boundary_watermark(turn_id);
}

std::vector<std::string> SqliteBackend::bounded_session_rollup_texts(
        const std::string& session_guid, int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->bounded_session_rollup_texts(session_guid, first_ts, last_ts);
}

std::vector<std::string> SqliteBackend::bounded_project_rollup_texts(
        int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->bounded_project_rollup_texts(first_ts, last_ts);
}

int SqliteBackend::store_session_summary(const std::string& text,
        const std::string& model_name, const std::string& session_guid,
        int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_session_summary(text, model_name, session_guid, first_ts, last_ts);
}

int SqliteBackend::store_project_summary(const std::string& text,
        const std::string& model_name, int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_project_summary(text, model_name, first_ts, last_ts);
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

std::vector<std::pair<std::string, int64_t>> SqliteBackend::table_row_counts() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->table_row_counts(); }

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

int SqliteBackend::rebuild_phon(bool only_missing, bool progress) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->rebuild_phon(only_missing, progress);
}

bool SqliteBackend::update_document_embedding(int document_id,
                                              const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_document_embedding(document_id, emb);
}

int SqliteBackend::store_decision(const std::string& text,
                                  const std::string& status,
                                  const std::string& tags,
                                  const std::string& source_timestamp,
                                  bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_decision(text, status, tags, source_timestamp, defer_embedding);
}

bool SqliteBackend::update_decision_embedding(int decision_id,
                                              const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_decision_embedding(decision_id, emb);
}

bool SqliteBackend::update_summary_embedding(int summary_id,
                                             const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_summary_embedding(summary_id, emb);
}

bool SqliteBackend::update_turn_embedding(int turn_id,
                                          const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_turn_embedding(turn_id, emb);
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
    int64_t cutoff_epoch = db_epoch(std::chrono::system_clock::to_time_t(cutoff));

    // v2: raw verbatim exchanges (L1 turns) are what age out by retention;
    // their gist is preserved in the L2/L3 summaries. Purge old turns by
    // timestamp. (Pre-v2 this deleted summaries tagged collection='conversation';
    // that column is gone in the lean schema.)
    Stmt stmt(pImpl->db,
        "DELETE FROM turns WHERE created_at < ?");
    stmt.bind(1, cutoff_epoch);

    int deleted = 0;
    if (stmt.exec()) {
        deleted = static_cast<int>(sqlite3_changes(pImpl->db));
    }
    return deleted;
}

} // namespace ragger
