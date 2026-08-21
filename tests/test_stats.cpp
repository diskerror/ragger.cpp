/**
 * StatsLogger tests — opt-in retrieval instrumentation (RAGGER_STATS).
 *
 * The ENTIRE test body is wrapped in `#ifdef RAGGER_STATS`. When the flag is
 * off, stats_logger.{h,cpp} compile to nothing and there is no StatsLogger to
 * test, so main() prints a skip notice and returns success. The CMake target
 * is also only built+registered when RAGGER_STATS is ON (see CMakeLists.txt),
 * so in a default build this file is never even compiled — the guard here is
 * belt-and-suspenders so the file is always self-consistent.
 *
 * Strategy: point the logger at a THROWAWAY db (/tmp), log fabricated lookups,
 * then reopen the db read-only and assert the rows landed correctly. No live
 * data, no embedder, no daemon — pure unit test of the logging path.
 */

#ifdef RAGGER_STATS

#include "stats_logger.h"
#include "storage_types.h"

#include <sqlite3.h>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <limits>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ragger::SearchResult;
using ragger::StatsLogger;

static const std::string DB = "/tmp/ragger_test_stats.db";

static void cleanup() {
    fs::remove(DB);
    fs::remove(DB + "-wal");
    fs::remove(DB + "-shm");
}

// Tiny read-only helper: run a single-int aggregate query against the db.
static int scalar(const std::string& sql) {
    sqlite3* db = nullptr;
    assert(sqlite3_open(DB.c_str(), &db) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    assert(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    int out = -1;
    if (sqlite3_step(st) == SQLITE_ROW) out = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
    sqlite3_close(db);
    return out;
}

// Read a single text cell.
static std::string scalar_text(const std::string& sql) {
    sqlite3* db = nullptr;
    assert(sqlite3_open(DB.c_str(), &db) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    assert(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    std::string out;
    if (sqlite3_step(st) == SQLITE_ROW) {
        auto* p = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        if (p) out = p;
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
    return out;
}

// Read a single REAL cell; returns NaN when the cell is NULL.
static double scalar_real(const std::string& sql) {
    sqlite3* db = nullptr;
    assert(sqlite3_open(DB.c_str(), &db) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    assert(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    double out = std::numeric_limits<double>::quiet_NaN();
    if (sqlite3_step(st) == SQLITE_ROW) {
        if (sqlite3_column_type(st, 0) != SQLITE_NULL)
            out = sqlite3_column_double(st, 0);
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
    return out;
}

static SearchResult mk(int id, const std::string& text, float score,
                       ragger::json meta = {}) {
    SearchResult r;
    r.id = id;
    r.text = text;
    r.score = score;
    r.metadata = std::move(meta);
    r.timestamp = "";
    return r;
}

int main() {
    std::println("Running stats logger tests:");

    // --- 1. Construction creates the db + schema -----------------------
    std::println("  test_open_creates_schema...");
    cleanup();
    {
        StatsLogger s(DB);
        assert(s.enabled());
    }
    assert(fs::exists(DB));
    // Both tables exist.
    assert(scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                  "AND name IN ('lookups','hits')") == 2);
    std::println("    OK");

    // --- 2. A lookup writes one lookups row + top-N hits ---------------
    std::println("  test_log_lookup_writes_rows...");
    cleanup();
    {
        StatsLogger s(DB);
        std::vector<SearchResult> results = {
            mk(101, "the white whale breached", 0.91f, {{"collection", "doc"}}),
            mk(102, "obsession and the sea",    0.77f, {{"kind", "summary"}}),
            mk(103, "a quiet grief",            0.55f, {{"level", "decision"}}),
            mk(104, "unrelated chatter",        0.40f),
            mk(105, "more noise",               0.33f),
        };
        s.log_lookup("the cost of obsession", 5, 0.3f, results, 12.5, /*top_n=*/3);
    }
    assert(scalar("SELECT COUNT(*) FROM lookups") == 1);
    // top_n=3 → only 3 hits stored even though 5 results were returned.
    assert(scalar("SELECT COUNT(*) FROM hits") == 3);
    // result_count records the FULL set (5), not the capped hit count.
    assert(scalar("SELECT result_count FROM lookups") == 5);
    // Query text round-trips.
    assert(scalar_text("SELECT query FROM lookups") == "the cost of obsession");
    // Ranks are 1..3 in score order.
    assert(scalar("SELECT memory_id FROM hits WHERE rank=1") == 101);
    assert(scalar("SELECT memory_id FROM hits WHERE rank=3") == 103);
    // Collection label extracted from metadata (collection/kind/level fallback).
    assert(scalar_text("SELECT collection FROM hits WHERE rank=1") == "doc");
    assert(scalar_text("SELECT collection FROM hits WHERE rank=2") == "summary");
    assert(scalar_text("SELECT collection FROM hits WHERE rank=3") == "decision");
    std::println("    OK");

    // --- 3. top_n<=0 stores ALL results --------------------------------
    std::println("  test_top_n_unlimited...");
    cleanup();
    {
        StatsLogger s(DB);
        std::vector<SearchResult> results = {
            mk(1, "a", 0.9f), mk(2, "b", 0.8f), mk(3, "c", 0.7f), mk(4, "d", 0.6f),
        };
        s.log_lookup("q", 10, 0.0f, results, -1.0, /*top_n=*/0);
    }
    assert(scalar("SELECT COUNT(*) FROM hits") == 4);
    // elapsed_ms negative → stored as NULL.
    assert(scalar("SELECT COUNT(*) FROM lookups WHERE elapsed_ms IS NULL") == 1);
    std::println("    OK");

    // --- 4. Empty result set: lookup row, zero hits --------------------
    std::println("  test_empty_results...");
    cleanup();
    {
        StatsLogger s(DB);
        s.log_lookup("no hits", 5, 0.9f, {}, 3.0);
    }
    assert(scalar("SELECT COUNT(*) FROM lookups") == 1);
    assert(scalar("SELECT result_count FROM lookups") == 0);
    assert(scalar("SELECT COUNT(*) FROM hits") == 0);
    std::println("    OK");

    // --- 5. Multiple lookups accumulate & FK links hold ----------------
    std::println("  test_multiple_lookups...");
    cleanup();
    {
        StatsLogger s(DB);
        s.log_lookup("first",  3, 0.0f, {mk(1, "x", 0.5f)}, 1.0);
        s.log_lookup("second", 3, 0.0f, {mk(2, "y", 0.6f), mk(3, "z", 0.4f)}, 2.0);
    }
    assert(scalar("SELECT COUNT(*) FROM lookups") == 2);
    assert(scalar("SELECT COUNT(*) FROM hits") == 3);
    // Every hit points at a real lookup row (no orphans).
    assert(scalar("SELECT COUNT(*) FROM hits h "
                  "LEFT JOIN lookups l ON h.lookup_id = l.id "
                  "WHERE l.id IS NULL") == 0);
    std::println("    OK");

    // --- 6. Per-signal breakdown columns (vec/bm25/phon/blended) --------
    std::println("  test_per_signal_breakdown...");
    cleanup();
    {
        StatsLogger s(DB);
        // Two hits: one with all three signals active, one where phon was
        // inactive (-1 sentinel → NULL) to prove the active/inactive distinction.
        SearchResult a = mk(201, "all three active", 0.88f, {{"collection", "doc"}});
        a.vec_score = 0.9f; a.bm25_score = 0.5f; a.phon_score = 0.2f; a.blended = 8.9f;
        SearchResult b = mk(202, "phon inactive", 0.60f, {{"collection", "doc"}});
        b.vec_score = 0.4f; b.bm25_score = 0.3f; b.phon_score = -1.0f; b.blended = 4.4f;
        s.log_lookup("signal probe", 5, 0.0f, {a, b}, 5.0, /*top_n=*/3);
    }
    // Active signals round-trip as their stored values.
    assert(std::abs(scalar_real("SELECT vec_score  FROM hits WHERE memory_id=201") - 0.9) < 1e-6);
    assert(std::abs(scalar_real("SELECT bm25_score FROM hits WHERE memory_id=201") - 0.5) < 1e-6);
    assert(std::abs(scalar_real("SELECT phon_score FROM hits WHERE memory_id=201") - 0.2) < 1e-6);
    assert(std::abs(scalar_real("SELECT blended    FROM hits WHERE memory_id=201") - 8.9) < 1e-6);
    // Inactive phon (-1) is stored as NULL, not -1.
    assert(scalar("SELECT phon_score IS NULL FROM hits WHERE memory_id=202") == 1);
    // ...while its active vec/bm25 still land.
    assert(std::abs(scalar_real("SELECT vec_score FROM hits WHERE memory_id=202") - 0.4) < 1e-6);
    std::println("    OK");

    cleanup();
    std::println("All stats logger tests passed.");
    return 0;
}

#else  // !RAGGER_STATS

#include <print>
int main() {
    std::println("stats logger tests skipped (built without RAGGER_STATS)");
    return 0;
}

#endif // RAGGER_STATS
