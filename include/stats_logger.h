/**
 * StatsLogger — opt-in retrieval instrumentation (issue: RAGGER_STATS).
 *
 * Compiled in ONLY when the build defines RAGGER_STATS (CMake option
 * RAGGER_STATS, default OFF). When the flag is off this header is never
 * included by the search path and the binary is byte-for-byte unaffected.
 *
 * Purpose: study how retrieval behaves (focused vs. associative "dolphining",
 * density effects) by logging, per lookup, the query and its top-N ranked
 * hits to a SEPARATE, discardable database (~/.ragger/stats.db). It never
 * touches memories.db.
 *
 * Contract: best-effort and non-throwing. A stats failure (locked DB, disk
 * full, schema fault) must NEVER propagate into the search path — every
 * public method swallows errors and degrades to a no-op. Losing a stats row
 * is acceptable; breaking a user's memory lookup is not.
 */
#pragma once

#ifdef RAGGER_STATS

#include <string>
#include <vector>

#include "storage_types.h"

struct sqlite3;

namespace ragger {

class StatsLogger {
public:
    /// Open (and create/migrate) the stats DB. `db_path` empty →
    /// ~/.ragger/stats.db (sibling of memories.db). On any failure the logger
    /// is left disabled; log_lookup() becomes a no-op.
    explicit StatsLogger(const std::string& db_path = "");
    ~StatsLogger();

    StatsLogger(const StatsLogger&)            = delete;
    StatsLogger& operator=(const StatsLogger&) = delete;

    /// Record one lookup and its ranked results. `top_n` caps how many hits
    /// are written per lookup (<=0 means "all results"). Best-effort: any
    /// error is swallowed. `elapsed_ms` is the backend search time if known
    /// (negative → store NULL).
    void log_lookup(const std::string& query,
                    int limit,
                    float min_score,
                    const std::vector<SearchResult>& results,
                    double elapsed_ms = -1.0,
                    int top_n = 3);

    bool enabled() const { return db_ != nullptr; }

private:
    sqlite3* db_ = nullptr;
};

} // namespace ragger

#endif // RAGGER_STATS
