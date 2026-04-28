/**
 * SQLite backend for Ragger Memory (C++ port)
 */
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/bm25.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
// Impl
// -----------------------------------------------------------------------
struct SqliteBackend::Impl {
    sqlite3*    db       = nullptr;
    Embedder*   embedder = nullptr;    // nullable — null for DB-only (user mgmt) mode
    BM25Index   bm25;  // initialized after config loaded
    std::string db_path;

    // Embedding cache — invalidated on writes
    bool                           cache_valid = false;
    std::vector<int>               cached_ids;
    std::vector<std::string>       cached_texts;
    Eigen::MatrixXf                cached_embeddings;   // rows × 384
    std::vector<json>              cached_metadata;
    std::vector<std::string>       cached_collections;
    std::vector<std::string>       cached_timestamps;

    Impl(Embedder& emb, const std::string& path)
        : embedder(&emb), bm25(config().bm25_k1, config().bm25_b)
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
            throw std::runtime_error(std::string(lang::ERR_SQLITE_OPEN) + err);
        }

        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys = ON");
        create_schema();
    }

    /// DB-only constructor — no embedder, only user management ops work.
    explicit Impl(const std::string& path)
        : embedder(nullptr), bm25(0, 0)
    {
        db_path = expand_path(path);
        fs::create_directories(fs::path(db_path).parent_path());

        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error(std::string(lang::ERR_SQLITE_OPEN) + err);
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
            throw std::runtime_error(std::string(lang::ERR_SQL) + err);
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

        // embedding is nullable: deferred-embedding writes (chat partial
        // turns, future bulk imports) insert with NULL; a backfill pass
        // fills them in afterwards.
        exec(R"(
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT NOT NULL,
                embedding BLOB,
                metadata  TEXT,
                timestamp TEXT NOT NULL,
                user_id   INTEGER REFERENCES users(id)
            )
        )");
        exec(R"(
            CREATE TABLE IF NOT EXISTS memory_usage (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL
                    REFERENCES memories(id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                timestamp TEXT NOT NULL
            )
        )");
        
        exec(R"(
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_memory_usage_memory_id ON memory_usage(memory_id)");
        exec(R"(
            CREATE TABLE IF NOT EXISTS bm25_index (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL
                    REFERENCES memories(id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                token     TEXT NOT NULL,
                term_freq INTEGER NOT NULL
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_bm25_memory_id ON bm25_index(memory_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_bm25_token ON bm25_index(token)");

        // Migrations
        migrate_add_user_id();
        migrate_dedicated_columns();
        migrate_embedding_nullable();
        migrate_add_token_rotated_at();
        migrate_add_preferred_model();
        migrate_add_password_hash();
        migrate_add_web_sessions();
        migrate_add_chat_sessions();
    }

    void migrate_embedding_nullable() {
        // Detect whether memories.embedding still has the old NOT NULL
        // constraint. SQLite doesn't support ALTER COLUMN to drop it, so
        // the standard fix is to rebuild the table.
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, "PRAGMA table_info(memories)", -1, &stmt, nullptr);
        bool embedding_notnull = false;
        bool found_embedding = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "embedding") {
                found_embedding = true;
                embedding_notnull = sqlite3_column_int(stmt, 3) != 0;  // notnull flag
                break;
            }
        }
        sqlite3_finalize(stmt);
        if (!found_embedding || !embedding_notnull) return;

        std::cerr << ragger::lang::MSG_MIGRATE_EMBEDDING_NULLABLE << "\n";
        exec("PRAGMA foreign_keys = OFF");
        exec("BEGIN");
        exec(R"(
            CREATE TABLE memories_new (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                embedding  BLOB,
                metadata   TEXT,
                timestamp  TEXT NOT NULL,
                user_id    INTEGER REFERENCES users(id),
                collection TEXT NOT NULL DEFAULT 'memory',
                category   TEXT NOT NULL DEFAULT '',
                tags       TEXT NOT NULL DEFAULT ''
            )
        )");
        exec(R"(
            INSERT INTO memories_new (id, text, embedding, metadata, timestamp,
                                      user_id, collection, category, tags)
            SELECT id, text, embedding, metadata, timestamp,
                   user_id, collection, category, tags
            FROM memories
        )");
        exec("DROP TABLE memories");
        exec("ALTER TABLE memories_new RENAME TO memories");
        exec("CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection)");
        exec("CREATE INDEX IF NOT EXISTS idx_memories_category   ON memories(category)");
        exec("COMMIT");
        exec("PRAGMA foreign_keys = ON");
    }

    void migrate_add_user_id() {
        // Check if column exists
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "PRAGMA table_info(memories)", -1, &stmt, nullptr);
        bool has_user_id = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "user_id") has_user_id = true;
        }
        sqlite3_finalize(stmt);

        if (!has_user_id) {
            exec("ALTER TABLE memories ADD COLUMN user_id INTEGER REFERENCES users(id)");
            std::cerr << ragger::lang::MSG_MIGRATE_USER_ID << "\n";
        }
    }

    void migrate_dedicated_columns() {
        // Check if collection column already exists
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "PRAGMA table_info(memories)", -1, &stmt, nullptr);
        bool has_collection = false;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (col == "collection") has_collection = true;
        }
        sqlite3_finalize(stmt);

        if (has_collection) return;

        std::cerr << ragger::lang::MSG_MIGRATE_DEDICATED_COLUMNS << "\n";

        exec("ALTER TABLE memories ADD COLUMN collection TEXT NOT NULL DEFAULT 'memory'");
        exec("ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT ''");
        exec("ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT ''");
        exec("CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection)");
        exec("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)");

        // Backfill from JSON metadata
        stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT id, metadata FROM memories WHERE metadata IS NOT NULL",
            -1, &stmt, nullptr);

        sqlite3_stmt* update_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE memories SET collection = ?, category = ?, tags = ?, metadata = ? WHERE id = ?",
            -1, &update_stmt, nullptr);

        int updated = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const char* meta_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (!meta_str) continue;

            json meta;
            try { meta = json::parse(meta_str); }
            catch (...) { continue; }

            // Extract dedicated fields
            std::string collection = meta.value("collection", "memory");
            std::string category = meta.value("category", "");
            meta.erase("collection");
            meta.erase("category");

            // Tags: array → comma-separated, string kept as-is
            std::vector<std::string> tag_list;
            if (meta.contains("tags")) {
                auto& tags_val = meta["tags"];
                if (tags_val.is_array()) {
                    for (auto& t : tags_val) tag_list.push_back(t.get<std::string>());
                } else if (tags_val.is_string() && !tags_val.get<std::string>().empty()) {
                    // Split comma-separated
                    std::string s = tags_val.get<std::string>();
                    size_t pos = 0;
                    while (pos < s.size()) {
                        size_t comma = s.find(',', pos);
                        if (comma == std::string::npos) comma = s.size();
                        std::string t = s.substr(pos, comma - pos);
                        // trim
                        size_t start = t.find_first_not_of(" \t");
                        size_t end = t.find_last_not_of(" \t");
                        if (start != std::string::npos)
                            tag_list.push_back(t.substr(start, end - start + 1));
                        pos = comma + 1;
                    }
                }
                meta.erase("tags");
            }

            // Boolean flags → tags
            if (meta.value("keep", false)) {
                if (std::find(tag_list.begin(), tag_list.end(), "keep") == tag_list.end())
                    tag_list.push_back("keep");
            }
            meta.erase("keep");
            if (meta.value("bad", false)) {
                if (std::find(tag_list.begin(), tag_list.end(), "bad") == tag_list.end())
                    tag_list.push_back("bad");
            }
            meta.erase("bad");

            // Join tags
            std::string tags_str;
            for (size_t i = 0; i < tag_list.size(); ++i) {
                if (i > 0) tags_str += ",";
                tags_str += tag_list[i];
            }

            // Clean metadata JSON
            std::string cleaned = meta.empty() ? "" : meta.dump();

            sqlite3_bind_text(update_stmt, 1, collection.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(update_stmt, 2, category.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(update_stmt, 3, tags_str.c_str(), -1, SQLITE_TRANSIENT);
            if (cleaned.empty())
                sqlite3_bind_null(update_stmt, 4);
            else
                sqlite3_bind_text(update_stmt, 4, cleaned.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(update_stmt, 5, id);
            sqlite3_step(update_stmt);
            sqlite3_reset(update_stmt);
            ++updated;
        }
        sqlite3_finalize(stmt);
        sqlite3_finalize(update_stmt);

        std::cerr << std::format(ragger::lang::MSG_MIGRATE_DEDICATED_BACKFILL, updated) << "\n";
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

    // ---- iso timestamp ------------------------------------------------
    static std::string now_iso() {
        auto now = std::chrono::system_clock::now();
        auto tt  = std::chrono::system_clock::to_time_t(now);
        std::tm gm{};
        gmtime_r(&tt, &gm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &gm);
        return std::string(buf) + "Z";
    }

    // ---- BM25 token indexing ------------------------------------------
    void index_bm25_tokens(int memory_id, const std::string& text) {
        auto tokens = BM25Index::tokenize(text);
        if (tokens.empty()) return;

        // Count term frequencies
        std::unordered_map<std::string, int> tf;
        for (auto& t : tokens) ++tf[t];

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO bm25_index (memory_id, token, term_freq) VALUES (?,?,?)",
            -1, &stmt, nullptr);

        for (auto& [token, freq] : tf) {
            sqlite3_bind_int(stmt, 1, memory_id);
            sqlite3_bind_text(stmt, 2, token.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 3, freq);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
    }

    // ---- load BM25 from storage ---------------------------------------
    void load_bm25_from_storage() {
        if (!config().bm25_enabled) return;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT memory_id, token, term_freq FROM bm25_index",
            -1, &stmt, nullptr);

        std::vector<int> doc_ids;
        std::vector<std::string> tokens;
        std::vector<int> freqs;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            doc_ids.push_back(sqlite3_column_int(stmt, 0));
            tokens.emplace_back(
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
            freqs.push_back(sqlite3_column_int(stmt, 2));
        }
        sqlite3_finalize(stmt);

        if (!doc_ids.empty()) {
            bm25.load(doc_ids, tokens, freqs);
        }
    }

    // ---- cache --------------------------------------------------------
    void invalidate_cache() { cache_valid = false; }

    void ensure_cache() {
        if (cache_valid) return;

        cached_ids.clear();
        cached_texts.clear();
        cached_metadata.clear();
        cached_timestamps.clear();
        cached_collections.clear();

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT id, text, embedding, metadata, timestamp, collection, category, tags FROM memories",
            -1, &stmt, nullptr);

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            cached_ids.push_back(sqlite3_column_int(stmt, 0));
            cached_texts.emplace_back(
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));

            const void* blob = sqlite3_column_blob(stmt, 2);
            int blob_bytes   = sqlite3_column_bytes(stmt, 2);
            int n_floats     = blob_bytes / sizeof(float);
            // NULL or wrong-sized embedding (e.g. row written by deferred-
            // embedding path before backfill ran) → zero vector. Cosine
            // against any query yields 0; BM25 still scores from text.
            std::vector<float> emb(expected_dims, 0.0f);
            if (n_floats == expected_dims && blob != nullptr) {
                std::memcpy(emb.data(), blob, blob_bytes);
            }
            emb_rows.push_back(std::move(emb));

            // Reconstruct full metadata from columns + JSON blob
            const char* meta_str =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            json meta = meta_str ? json::parse(meta_str) : json::object();

            const char* col_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            const char* cat_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
            const char* tag_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));

            std::string collection = col_str ? col_str : "memory";
            std::string category = cat_str ? cat_str : "";
            std::string tags = tag_str ? tag_str : "";

            meta["collection"] = collection;
            if (!category.empty()) meta["category"] = category;
            if (!tags.empty()) meta["tags"] = tags;

            cached_metadata.push_back(std::move(meta));
            cached_collections.push_back(collection);

            cached_timestamps.emplace_back(
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
        }
        sqlite3_finalize(stmt);

        // Pack into Eigen matrix (rows × dims). Dimensions come from config
        // — every emb_rows entry is already padded/zeroed to expected_dims
        // so NULL-embedding rows don't poison the matrix shape.
        int n = static_cast<int>(emb_rows.size());
        cached_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            cached_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
        }

        // Build / load BM25 index
        if (config().bm25_enabled) {
            load_bm25_from_storage();
            if (!bm25.is_built() && !cached_texts.empty()) {
                bm25.build(cached_texts);
            }
        }

        cache_valid = true;
    }

    // ---- usage tracking -----------------------------------------------
    void track_usage(const std::vector<int>& ids) {
        if (ids.empty()) return;  // usage tracking always on
        auto ts = now_iso();
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO memory_usage (memory_id, timestamp) VALUES (?,?)",
            -1, &stmt, nullptr);
        for (int id : ids) {
            sqlite3_bind_int(stmt, 1, id);
            sqlite3_bind_text(stmt, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
    }

    // ---- public API ---------------------------------------------------

    std::string store(const std::string& raw_text, json metadata, bool defer_embedding) {
        // Ensure metadata is an object (default param may be null)
        if (metadata.is_null()) metadata = json::object();

        // Extract dedicated columns from metadata
        std::string collection = metadata.value("collection", std::string("memory"));
        std::string category = metadata.value("category", "");
        // Optional historical timestamp override (used by imports of past
        // conversations — Claude Code JSONL, claude.ai export, etc.).
        // Must be an ISO-8601 UTC string; otherwise falls back to now.
        std::string ts_override;
        if (metadata.contains("timestamp") && metadata["timestamp"].is_string()) {
            ts_override = metadata["timestamp"].get<std::string>();
        }
        metadata.erase("collection");
        metadata.erase("category");
        metadata.erase("timestamp");

        // Extract tags
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
            metadata.erase("tags");
        }
        // Boolean flags → tags
        if (metadata.value("keep", false)) {
            if (tags_str.find("keep") == std::string::npos)
                tags_str += (tags_str.empty() ? "" : ",") + std::string("keep");
        }
        metadata.erase("keep");
        if (metadata.value("bad", false)) {
            if (tags_str.find("bad") == std::string::npos)
                tags_str += (tags_str.empty() ? "" : ",") + std::string("bad");
        }
        metadata.erase("bad");

        // Normalize paths
        std::string text = normalize_path(raw_text);

        // Compute embedding unless deferred — caller (or startup backfill)
        // will fill it in later.
        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        // Timestamp — use explicit override from metadata if present
        auto ts = ts_override.empty() ? now_iso() : ts_override;

        // Insert with dedicated columns
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO memories (text, embedding, metadata, timestamp, collection, category, tags) "
            "VALUES (?,?,?,?,?,?,?)",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        if (defer_embedding) {
            sqlite3_bind_null(stmt, 2);
        } else {
            sqlite3_bind_blob(stmt, 2, emb.data(),
                              static_cast<int>(emb.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
        }
        std::string meta_str = metadata.empty() ? "" : metadata.dump();
        if (meta_str.empty())
            sqlite3_bind_null(stmt, 3);
        else
            sqlite3_bind_text(stmt, 3, meta_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, ts.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, collection.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, category.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, tags_str.c_str(), -1, SQLITE_TRANSIENT);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) {
            throw std::runtime_error(std::string(lang::ERR_STORE_FAILED) +
                                     sqlite3_errmsg(db));
        }

        int memory_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        index_bm25_tokens(memory_id, text);
        invalidate_cache();

        return std::to_string(memory_id);
    }

    bool update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        // Refuse to mutate protected rows for parity with delete_memory.
        sqlite3_stmt* check_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT tags FROM memories WHERE id = ?",
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

        // Mirror store()'s metadata extraction so the dedicated columns
        // stay in sync if the caller changed collection/category/tags.
        std::string collection = metadata.value("collection", std::string("memory"));
        std::string category = metadata.value("category", "");
        metadata.erase("collection");
        metadata.erase("category");
        metadata.erase("timestamp");  // never overwrite original timestamp here

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
            metadata.erase("tags");
        }
        if (metadata.value("keep", false)) {
            if (tags_str.find("keep") == std::string::npos)
                tags_str += (tags_str.empty() ? "" : ",") + std::string("keep");
        }
        metadata.erase("keep");
        if (metadata.value("bad", false)) {
            if (tags_str.find("bad") == std::string::npos)
                tags_str += (tags_str.empty() ? "" : ",") + std::string("bad");
        }
        metadata.erase("bad");

        std::string text = normalize_path(raw_text);
        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE memories "
            "SET text = ?, embedding = ?, metadata = ?, "
            "    collection = ?, category = ?, tags = ? "
            "WHERE id = ?",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        if (defer_embedding) {
            sqlite3_bind_null(stmt, 2);
        } else {
            sqlite3_bind_blob(stmt, 2, emb.data(),
                              static_cast<int>(emb.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
        }
        std::string meta_str = metadata.empty() ? "" : metadata.dump();
        if (meta_str.empty())
            sqlite3_bind_null(stmt, 3);
        else
            sqlite3_bind_text(stmt, 3, meta_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, collection.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, category.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, tags_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 7, memory_id);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) return false;

        // Rebuild this row's BM25 tokens against the new text.
        sqlite3_stmt* del = nullptr;
        sqlite3_prepare_v2(db, "DELETE FROM bm25_index WHERE memory_id = ?",
                           -1, &del, nullptr);
        sqlite3_bind_int(del, 1, memory_id);
        sqlite3_step(del);
        sqlite3_finalize(del);
        index_bm25_tokens(memory_id, text);
        invalidate_cache();
        return true;
    }

    SearchResponse search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> collections) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        ensure_cache();
        int n = static_cast<int>(cached_ids.size());
        if (n == 0) return {{}, {{"corpus_size", 0}}};

        // ---- collection filter ----------------------------------------
        std::vector<int> filtered_idx;  // indices into cache arrays
        if (collections.empty()) {
            // Default: search all
            filtered_idx.resize(n);
            std::iota(filtered_idx.begin(), filtered_idx.end(), 0);
        } else {
            bool search_all = false;
            for (auto& c : collections)
                if (c == "*" || c == "all") { search_all = true; break; }

            if (search_all) {
                filtered_idx.resize(n);
                std::iota(filtered_idx.begin(), filtered_idx.end(), 0);
            } else {
                for (int i = 0; i < n; ++i) {
                    for (auto& c : collections)
                        if (cached_collections[i] == c) { filtered_idx.push_back(i); break; }
                }
            }
        }

        int filtered_n = static_cast<int>(filtered_idx.size());
        if (filtered_n == 0) {
            return {{}, {{"corpus_size", n}, {"filtered_size", 0}}};
        }

        // ---- query embedding ------------------------------------------
        auto t_embed_start = clock::now();
        auto q_vec = embedder->encode(query);
        auto t_embed_end = clock::now();

        // ---- cosine similarity ----------------------------------------
        auto t_search_start = clock::now();

        // Map query into Eigen
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(),
                                       static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        // Build filtered embedding matrix
        Eigen::MatrixXf filt_emb(filtered_n, config().embedding_dimensions);
        for (int i = 0; i < filtered_n; ++i) {
            filt_emb.row(i) = cached_embeddings.row(filtered_idx[i]);
        }

        // Row-wise normalize
        Eigen::VectorXf norms = filt_emb.rowwise().norm();
        for (int i = 0; i < filtered_n; ++i) {
            if (norms(i) > 1e-12f)
                filt_emb.row(i) /= norms(i);
        }

        // Cosine similarities
        Eigen::VectorXf similarities = filt_emb * q_norm;

        // ---- BM25 hybrid -----------------------------------------------
        Eigen::VectorXf combined = similarities;

        if (config().bm25_enabled && bm25.is_built()) {
            // Map filtered_idx to doc IDs for BM25
            std::vector<int> bm25_ids;
            bm25_ids.reserve(filtered_n);
            for (int i : filtered_idx) bm25_ids.push_back(cached_ids[i]);

            auto bm25_scores_vec = bm25.score(query, bm25_ids);
            Eigen::Map<Eigen::VectorXf> bm25_scores(
                bm25_scores_vec.data(), filtered_n);

            // Min-max normalize both
            auto norm_minmax = [](Eigen::VectorXf& v) {
                float mn = v.minCoeff();
                float mx = v.maxCoeff();
                if (mx > mn) v = (v.array() - mn) / (mx - mn);
            };

            Eigen::VectorXf vec_norm = similarities;
            Eigen::VectorXf bm25_norm = bm25_scores;
            norm_minmax(vec_norm);
            norm_minmax(bm25_norm);

            combined = config().vector_weight * vec_norm + config().bm25_weight * bm25_norm;
        }

        auto t_search_end = clock::now();

        // ---- top-k selection -------------------------------------------
        int top_k = std::min(limit, filtered_n);
        std::vector<int> ranking(filtered_n);
        std::iota(ranking.begin(), ranking.end(), 0);
        std::partial_sort(ranking.begin(), ranking.begin() + top_k,
                          ranking.end(),
                          [&](int a, int b) {
                              return combined(a) > combined(b);
                          });

        std::vector<SearchResult> results;
        std::vector<int> usage_ids;
        for (int k = 0; k < top_k; ++k) {
            int local_i  = ranking[k];
            int cache_i  = filtered_idx[local_i];
            float score  = similarities(local_i);  // report raw cosine
            if (score < min_score) continue;

            results.push_back({
                cached_ids[cache_i],
                cached_texts[cache_i],
                score,
                cached_metadata[cache_i],
                cached_timestamps[cache_i]
            });
            usage_ids.push_back(cached_ids[cache_i]);
        }

        track_usage(usage_ids);

        // ---- timing ----------------------------------------------------
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        json timing = {
            {"embedding_ms", ms(t_embed_start, t_embed_end)},
            {"search_ms",    ms(t_search_start, t_search_end)},
            {"total_ms",     ms(t_start, clock::now())},
            {"corpus_size",  n},
            {"filtered_size", filtered_n}
        };

        return {std::move(results), std::move(timing)};
    }

    int count() const {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM memories",
                           -1, &stmt, nullptr);
        int c = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            c = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        return c;
    }

    std::vector<SearchResult> load_all(const std::string& collection) {
        std::vector<SearchResult> results;
        std::string sql = "SELECT id, text, metadata, timestamp, collection, category, tags FROM memories";
        if (!collection.empty()) {
            sql += " WHERE collection = ?";
        }
        sql += " ORDER BY id";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        if (!collection.empty()) {
            sqlite3_bind_text(stmt, 1, collection.c_str(), -1, SQLITE_TRANSIENT);
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* meta_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            const char* col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            const char* cat = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            const char* tag = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));

            json meta = meta_str ? json::parse(meta_str) : json::object();
            meta["collection"] = col ? col : "memory";
            if (cat && cat[0]) meta["category"] = cat;
            if (tag && tag[0]) meta["tags"] = tag;

            results.push_back({
                id,
                text ? text : "",
                0.0f,
                std::move(meta),
                ts ? ts : ""
            });
        }
        sqlite3_finalize(stmt);
        return results;
    }

    int rebuild_bm25() {
        // Clear existing index
        exec("DELETE FROM bm25_index");

        // Re-index all documents
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT id, text FROM memories",
                           -1, &stmt, nullptr);

        int doc_count = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (text) {
                index_bm25_tokens(id, text);
                ++doc_count;
            }
        }
        sqlite3_finalize(stmt);
        invalidate_cache();
        return doc_count;
    }

    int rebuild_embeddings(Embedder& emb_ref) {
        // Get total count first
        sqlite3_stmt* count_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM memories",
                           -1, &count_stmt, nullptr);
        int total_count = 0;
        if (sqlite3_step(count_stmt) == SQLITE_ROW) {
            total_count = sqlite3_column_int(count_stmt, 0);
        }
        sqlite3_finalize(count_stmt);
        
        // Re-embed all documents
        sqlite3_stmt* select_stmt = nullptr;
        sqlite3_prepare_v2(db, "SELECT id, text FROM memories",
                           -1, &select_stmt, nullptr);

        sqlite3_stmt* update_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE memories SET embedding = ? WHERE id = ?",
            -1, &update_stmt, nullptr);

        int doc_count = 0;
        while (sqlite3_step(select_stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(select_stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt, 1));
            if (!text) continue;

            // Generate new embedding
            auto emb = emb_ref.encode(text);

            // Update database
            sqlite3_bind_blob(update_stmt, 1, emb.data(),
                            static_cast<int>(emb.size() * sizeof(float)),
                            SQLITE_TRANSIENT);
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
            "SELECT id, text FROM memories WHERE embedding IS NULL",
            -1, &select_stmt, nullptr);

        sqlite3_stmt* update_stmt = nullptr;
        sqlite3_prepare_v2(db,
            "UPDATE memories SET embedding = ? WHERE id = ?",
            -1, &update_stmt, nullptr);

        int updated = 0;
        while (sqlite3_step(select_stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(select_stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt, 1));
            if (!text) continue;

            auto emb = emb_ref.encode(text);
            sqlite3_bind_blob(update_stmt, 1, emb.data(),
                              static_cast<int>(emb.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
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

    std::vector<std::string> collections() const {
        std::vector<std::string> result;
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT DISTINCT collection FROM memories",
            -1, &stmt, nullptr);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (col) result.emplace_back(col);
        }
        sqlite3_finalize(stmt);
        return result;
    }

    bool delete_memory(int memory_id) {
        // Check if memory has keep tag in dedicated column
        sqlite3_stmt* check_stmt;
        sqlite3_prepare_v2(db, "SELECT tags FROM memories WHERE id = ?", -1, &check_stmt, nullptr);
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
        sqlite3_prepare_v2(db, "DELETE FROM memories WHERE id = ?",
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
            sqlite3_prepare_v2(db, "SELECT tags FROM memories WHERE id = ?", -1, &check_stmt, nullptr);
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

        // Build SQL: DELETE FROM memories WHERE id IN (?,?,...)
        std::string sql = "DELETE FROM memories WHERE id IN (";
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

        // Build SQL WHERE using dedicated columns where possible
        std::string sql = "SELECT id, text, metadata, timestamp, collection, category, tags FROM memories";
        std::string where;
        std::vector<std::string> binds;
        // Track which keys are handled by SQL so we skip them in C++ filtering
        std::set<std::string> sql_keys;

        for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
            if (it.key() == "collection" || it.key() == "category") {
                where += (where.empty() ? " WHERE " : " AND ") + it.key() + " = ?";
                binds.push_back(it.value().get<std::string>());
                sql_keys.insert(it.key());
            } else if (it.key() == "tags") {
                where += (where.empty() ? " WHERE " : " AND ") + std::string("tags LIKE ?");
                binds.push_back("%" + it.value().get<std::string>() + "%");
                sql_keys.insert(it.key());
            }
        }

        // Add temporal filtering
        if (!after.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("timestamp >= ?");
            binds.push_back(after);
        }
        if (!before.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("timestamp < ?");
            binds.push_back(before);
        }

        sql += where + " ORDER BY timestamp DESC";
        if (limit > 0) sql += " LIMIT " + std::to_string(limit * 2);  // over-fetch for C++ filtering

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        for (size_t i = 0; i < binds.size(); ++i) {
            sqlite3_bind_text(stmt, static_cast<int>(i + 1), binds[i].c_str(), -1, SQLITE_TRANSIENT);
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* meta_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            const char* ts = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            const char* col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            const char* cat = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            const char* tag = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));

            // Reconstruct full metadata
            json metadata = meta_str ? json::parse(meta_str) : json::object();
            metadata["collection"] = col ? col : "memory";
            if (cat && cat[0]) metadata["category"] = cat;
            if (tag && tag[0]) metadata["tags"] = tag;

            // Check remaining (non-SQL) filter fields in C++
            bool match = true;
            for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
                if (sql_keys.count(it.key())) continue;  // already filtered by SQL
                if (!metadata.contains(it.key()) || metadata[it.key()] != it.value()) {
                    match = false;
                    break;
                }
            }

            if (match) {
                results.push_back({
                    id,
                    text ? text : "",
                    0.0f,
                    std::move(metadata),
                    ts ? ts : ""
                });
                if (limit > 0 && (int)results.size() >= limit) break;
            }
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
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_t));

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
        sqlite3_bind_text(stmt, 2, buf, -1, SQLITE_TRANSIENT);
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
        sqlite3_bind_text(stmt, 5, buf, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, buf, -1, SQLITE_TRANSIENT);
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
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - std::chrono::duration_cast<std::chrono::system_clock::duration>(
        std::chrono::duration<double, std::ratio<3600>>(max_age_hours));
    auto cutoff_t = std::chrono::system_clock::to_time_t(cutoff);
    char cutoff_str[32];
    std::strftime(cutoff_str, sizeof(cutoff_str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&cutoff_t));

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "DELETE FROM memories WHERE collection = 'conversation' AND timestamp < ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, cutoff_str, -1, SQLITE_STATIC);

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
    auto now_tt = std::chrono::system_clock::to_time_t(now);
    std::tm now_gm{};
    gmtime_r(&now_tt, &now_gm);
    char now_buf[32];
    std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%dT%H:%M:%SZ", &now_gm);
    std::string created(now_buf);
    
    auto expires_tp = now + std::chrono::seconds(ttl_seconds);
    auto expires_tt = std::chrono::system_clock::to_time_t(expires_tp);
    std::tm expires_gm{};
    gmtime_r(&expires_tt, &expires_gm);
    char expires_buf[32];
    std::strftime(expires_buf, sizeof(expires_buf), "%Y-%m-%dT%H:%M:%SZ", &expires_gm);
    std::string expires(expires_buf);
    
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
    auto now = std::chrono::system_clock::now();
    auto now_tt = std::chrono::system_clock::to_time_t(now);
    std::tm now_gm{};
    gmtime_r(&now_tt, &now_gm);
    char now_buf[32];
    std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%dT%H:%M:%SZ", &now_gm);
    std::string timestamp(now_buf);
    
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
    auto now = std::chrono::system_clock::now();
    auto now_tt = std::chrono::system_clock::to_time_t(now);
    std::tm now_gm{};
    gmtime_r(&now_tt, &now_gm);
    char now_buf[32];
    std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%dT%H:%M:%SZ", &now_gm);
    std::string now_str(now_buf);
    
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
