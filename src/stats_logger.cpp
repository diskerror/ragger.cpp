/**
 * StatsLogger implementation — see stats_logger.h.
 *
 * Compiled ONLY under -DRAGGER_STATS (CMake option RAGGER_STATS, default OFF).
 * Best-effort, non-throwing: every method guards its own errors so a stats
 * failure can never reach the search path.
 */
#ifdef RAGGER_STATS

#include "ragger/stats_logger.h"

#include <sqlite3.h>

#include <algorithm>

#include "ragger/config.h"      // expand_path
#include "ragger/util/time.h"   // db_timestamp
#include "ragger/util/sqlite.h" // Stmt (RAII)
#include "diskerror/logger.h"

namespace ragger {

namespace {
constexpr const char* kSchema = R"sql(
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS lookups (
    id           INTEGER PRIMARY KEY,
    ts           TEXT    NOT NULL,
    query        TEXT    NOT NULL,
    limit_n      INTEGER NOT NULL,
    min_score    REAL    NOT NULL,
    result_count INTEGER NOT NULL,
    elapsed_ms   REAL
);
CREATE TABLE IF NOT EXISTS hits (
    id         INTEGER PRIMARY KEY,
    lookup_id  INTEGER NOT NULL REFERENCES lookups(id) ON DELETE CASCADE,
    rank       INTEGER NOT NULL,
    memory_id  INTEGER NOT NULL,
    score      REAL    NOT NULL,
    collection TEXT,
    snippet    TEXT
);
CREATE INDEX IF NOT EXISTS idx_hits_lookup ON hits(lookup_id);
)sql";

// Pull a human-readable collection/source label out of a result's metadata,
// tolerating whatever shape the row carries (best-effort, never throws).
std::string collection_of(const SearchResult& r) {
    try {
        if (r.metadata.is_object()) {
            for (const char* key : {"collection", "kind", "level", "source"}) {
                auto it = r.metadata.find(key);
                if (it != r.metadata.end() && it->is_string())
                    return it->get<std::string>();
            }
        }
    } catch (...) {}
    return {};
}

std::string snippet_of(const SearchResult& r, std::size_t max = 160) {
    std::string s = r.text;
    if (s.size() > max) s.resize(max);
    return s;
}
} // namespace

StatsLogger::StatsLogger(const std::string& db_path) {
    const std::string path =
        db_path.empty() ? expand_path("~/.ragger/stats.db") : expand_path(db_path);
    try {
        if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
            Diskerror::logger::warn(std::string("stats: open failed: ") +
                                    (db_ ? sqlite3_errmsg(db_) : "unknown"));
            if (db_) { sqlite3_close(db_); db_ = nullptr; }
            return;
        }
        sqlite3_busy_timeout(db_, 2000);
        char* err = nullptr;
        if (sqlite3_exec(db_, kSchema, nullptr, nullptr, &err) != SQLITE_OK) {
            Diskerror::logger::warn(std::string("stats: schema failed: ") +
                                    (err ? err : "unknown"));
            sqlite3_free(err);
            sqlite3_close(db_);
            db_ = nullptr;
            return;
        }
        Diskerror::logger::debug("stats: logging enabled → " + path);
    } catch (...) {
        if (db_) { sqlite3_close(db_); db_ = nullptr; }
    }
}

StatsLogger::~StatsLogger() {
    if (db_) sqlite3_close(db_);
}

void StatsLogger::log_lookup(const std::string& query,
                             int limit,
                             float min_score,
                             const std::vector<SearchResult>& results,
                             double elapsed_ms,
                             int top_n) {
    if (!db_) return;
    try {
        // One transaction per lookup keeps the lookup row + its hits atomic
        // and cheap. A failure rolls back and is swallowed.
        if (sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr) != SQLITE_OK)
            return;

        sqlite3_int64 lookup_id = 0;
        {
            Stmt s(db_,
                "INSERT INTO lookups(ts,query,limit_n,min_score,result_count,elapsed_ms) "
                "VALUES(?,?,?,?,?,?)");
            s.bind(1, db_timestamp())
             .bind(2, query)
             .bind(3, limit)
             .bind(4, static_cast<double>(min_score))
             .bind(5, static_cast<int>(results.size()));
            if (elapsed_ms < 0.0) s.bind_null(6); else s.bind(6, elapsed_ms);
            if (!s.exec()) { sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr); return; }
            lookup_id = sqlite3_last_insert_rowid(db_);
        }

        const int cap = (top_n <= 0)
            ? static_cast<int>(results.size())
            : std::min<int>(top_n, static_cast<int>(results.size()));
        for (int i = 0; i < cap; ++i) {
            const SearchResult& r = results[i];
            Stmt s(db_,
                "INSERT INTO hits(lookup_id,rank,memory_id,score,collection,snippet) "
                "VALUES(?,?,?,?,?,?)");
            s.bind(1, static_cast<int64_t>(lookup_id))
             .bind(2, i + 1)
             .bind(3, r.id)
             .bind(4, static_cast<double>(r.score));
            const std::string coll = collection_of(r);
            if (coll.empty()) s.bind_null(5); else s.bind(5, coll);
            s.bind(6, snippet_of(r));
            s.exec();
        }

        sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr);
    } catch (const std::exception& e) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        Diskerror::logger::debug(std::string("stats: log_lookup swallowed: ") + e.what());
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
    }
}

} // namespace ragger

#endif // RAGGER_STATS
