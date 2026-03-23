/**
 * SQLite backend for Ragger Memory (C++ port)
 */
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/bm25.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
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
    Embedder&   embedder;
    BM25Index   bm25;  // initialized after config loaded
    std::string db_path;

    // Embedding cache — invalidated on writes
    bool                           cache_valid = false;
    std::vector<int>               cached_ids;
    std::vector<std::string>       cached_texts;
    Eigen::MatrixXf                cached_embeddings;   // rows × 384
    std::vector<json>              cached_metadata;
    std::vector<std::string>       cached_timestamps;

    Impl(Embedder& emb, const std::string& path)
        : embedder(emb), bm25(config().bm25_k1, config().bm25_b)
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
                is_admin   INTEGER NOT NULL DEFAULT 0,
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

        exec(R"(
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT NOT NULL,
                embedding BLOB NOT NULL,
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

        // Migration: add user_id to existing memories table
        migrate_add_user_id();
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
            std::cerr << "Migrated memories: added user_id column\n";
        }
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

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT id, text, embedding, metadata, timestamp FROM memories",
            -1, &stmt, nullptr);

        std::vector<std::vector<float>> emb_rows;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            cached_ids.push_back(sqlite3_column_int(stmt, 0));
            cached_texts.emplace_back(
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));

            const void* blob = sqlite3_column_blob(stmt, 2);
            int blob_bytes   = sqlite3_column_bytes(stmt, 2);
            int n_floats     = blob_bytes / sizeof(float);
            std::vector<float> emb(n_floats);
            std::memcpy(emb.data(), blob, blob_bytes);
            emb_rows.push_back(std::move(emb));

            const char* meta_str =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            cached_metadata.push_back(
                meta_str ? json::parse(meta_str) : json::object());

            cached_timestamps.emplace_back(
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
        }
        sqlite3_finalize(stmt);

        // Pack into Eigen matrix (rows × dims)
        int n = static_cast<int>(emb_rows.size());
        int dims = n > 0 ? static_cast<int>(emb_rows[0].size()) : config().embedding_dimensions;
        cached_embeddings.resize(n, dims);
        for (int i = 0; i < n; ++i) {
            cached_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), dims);
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

    std::string store(const std::string& raw_text, json metadata) {
        // Ensure collection
        if (!metadata.contains("collection")) {
            metadata["collection"] = config().default_collection;
        }

        // Normalize paths
        std::string text = normalize_path(raw_text);

        // Compute embedding
        auto emb = embedder.encode(text);

        // Timestamp
        auto ts = now_iso();

        // Insert
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO memories (text, embedding, metadata, timestamp) VALUES (?,?,?,?)",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt, 2, emb.data(),
                          static_cast<int>(emb.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        std::string meta_str = metadata.dump();
        sqlite3_bind_text(stmt, 3, meta_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, ts.c_str(), -1, SQLITE_TRANSIENT);

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
                    auto col = cached_metadata[i].value("collection",
                                                        config().default_collection);
                    for (auto& c : collections)
                        if (col == c) { filtered_idx.push_back(i); break; }
                }
            }
        }

        int filtered_n = static_cast<int>(filtered_idx.size());
        if (filtered_n == 0) {
            return {{}, {{"corpus_size", n}, {"filtered_size", 0}}};
        }

        // ---- query embedding ------------------------------------------
        auto t_embed_start = clock::now();
        auto q_vec = embedder.encode(query);
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
        std::string sql = "SELECT id, text, metadata, timestamp FROM memories";
        if (!collection.empty()) {
            sql += " WHERE json_extract(metadata, '$.collection') = ?";
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

            results.push_back({
                id,
                text ? text : "",
                0.0f,
                meta_str ? json::parse(meta_str) : json::object(),
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

    std::vector<std::string> collections() const {
        std::vector<std::string> result;
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT DISTINCT json_extract(metadata, '$.collection') FROM memories",
            -1, &stmt, nullptr);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* col = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (col) result.emplace_back(col);
        }
        sqlite3_finalize(stmt);
        return result;
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

SqliteBackend::~SqliteBackend() = default;

std::string SqliteBackend::store(const std::string& text, json metadata) {
    return pImpl->store(text, std::move(metadata));
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

std::vector<std::string> SqliteBackend::collections() const {
    return pImpl->collections();
}

void SqliteBackend::close() { pImpl->close(); }

int SqliteBackend::create_user(const std::string& username,
                                const std::string& token_hash,
                                bool is_admin) {
    auto now = pImpl->now_iso();
    std::string sql = "INSERT INTO users (username, token_hash, is_admin, created, modified) "
                      "VALUES (?, ?, ?, ?, ?)";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, is_admin ? 1 : 0);
    sqlite3_bind_text(stmt, 4, now.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, now.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return static_cast<int>(sqlite3_last_insert_rowid(pImpl->db));
}

std::optional<SqliteBackend::UserInfo> SqliteBackend::get_user_by_token_hash(
        const std::string& token_hash) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT id, username, is_admin FROM users WHERE token_hash = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo u;
        u.id = sqlite3_column_int(stmt, 0);
        u.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.is_admin = sqlite3_column_int(stmt, 2) != 0;
        u.token_hash = token_hash;
        sqlite3_finalize(stmt);
        return u;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::optional<SqliteBackend::UserInfo> SqliteBackend::get_user_by_username(
        const std::string& username) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(pImpl->db,
        "SELECT id, username, is_admin, token_hash FROM users WHERE username = ?",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        UserInfo u;
        u.id = sqlite3_column_int(stmt, 0);
        u.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.is_admin = sqlite3_column_int(stmt, 2) != 0;
        u.token_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        sqlite3_finalize(stmt);
        return u;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}

} // namespace ragger
