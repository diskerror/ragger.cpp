/**
 * test_rebuild_check.cpp — S111: scripts/rebuild-check.sql is SELECT-only
 * and counts exactly the defects that exist.
 *
 * Three parts:
 *   1. Text check: with comments stripped, the script contains no data-
 *      changing or PRAGMA keyword.
 *   2. Empty DB: every check reads 0 (schema from scripts/schema_db0.17.sql,
 *      which verify-schema-sql.sh pins to the binary's own schema).
 *   3. Seeded fixture: one deliberate row per defect kind; each count must
 *      equal the value set up below — including the two checks that seed
 *      interacts with (episodes_without_members counts 2, embedding counts
 *      equal the seeded table sizes).
 *
 * The script runs through the sqlite3 C API with the DB opened READ-ONLY,
 * so a stray write would fail the run even if the text check missed it.
 * No TestEnv: nothing here touches config or ~/.ragger — raw sqlite3 in a
 * mkdtemp directory.
 */
#include "check.h"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

// X14: the mkdtemp scratch dir must not outlive the run. RAII covers normal
// exit and exceptions; CHECK failures abort(), which skips destructors, so a
// SIGABRT handler removes the dir too and then re-raises the default action.
namespace {
fs::path g_scratch_dir;

void remove_scratch() {
    if (g_scratch_dir.empty()) return;
    std::error_code ec;
    fs::remove_all(g_scratch_dir, ec);
    g_scratch_dir.clear();
}

void scratch_on_signal(int sig) {
    remove_scratch();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

struct ScratchCleanup {
    ~ScratchCleanup() { remove_scratch(); }
};
ScratchCleanup g_scratch_cleanup;
}  // namespace

static std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    CHECK(in.is_open());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void exec_sql(sqlite3* db, const std::string& sql) {
    char* err = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::fprintf(stderr, "exec failed (%s): %s\n", sqlite3_errmsg(db),
                     sql.substr(0, 200).c_str());
        CHECK(rc == SQLITE_OK);
    }
}

/// Run the check script and return check_name -> count.
static std::map<std::string, int> run_checks(sqlite3* db, const std::string& script) {
    std::map<std::string, int> out;
    char* err = nullptr;
    const int rc = sqlite3_exec(
        db, script.c_str(),
        [](void* ctx, int argc, char** argv, char**) -> int {
            auto* m = static_cast<std::map<std::string, int>*>(ctx);
            if (argc != 2) return 1;
            (*m)[argv[0]] = std::atoi(argv[1]);
            return 0;
        },
        &out, &err);
    CHECK(rc == SQLITE_OK);
    return out;
}

/// Strip `--` line comments so the keyword scan sees statement text only.
static std::string strip_comments(const std::string& sql) {
    std::string out;
    std::istringstream in(sql);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find("--");
        if (pos != std::string::npos) line = line.substr(0, pos);
        out += line;
        out += '\n';
    }
    return out;
}

static bool has_word(const std::string& hay_upper, const std::string& word) {
    const auto isalnum_ = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    size_t pos = 0;
    while ((pos = hay_upper.find(word, pos)) != std::string::npos) {
        const bool left = pos > 0 && isalnum_(hay_upper[pos - 1]);
        const bool right = isalnum_(hay_upper[pos + word.size()]);
        if (!left && !right) return true;
        pos += 1;
    }
    return false;
}

int main() {
    std::signal(SIGABRT, scratch_on_signal);  // X14: clean the scratch dir on CHECK-abort
    const fs::path repo = fs::path(__FILE__).parent_path().parent_path();
    const std::string script = slurp(repo / "scripts/rebuild-check.sql");
    const std::string schema = slurp(repo / "scripts/schema_db0.17.sql");

    // 1. Text check: no data-changing statement, no PRAGMA assignment.
    {
        const std::string stmts = strip_comments(script);
        std::string upper = stmts;
        std::transform(upper.begin(), upper.end(), upper.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        for (const char* banned : {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                                   "CREATE", "PRAGMA", "ATTACH", "DETACH",
                                   "TRUNCATE", "VACUUM", "REPLACE", "GRANT"})
            CHECK(!has_word(upper, banned));
        // Every statement must be a SELECT (the WITH ... SELECT spine).
        const auto body = upper.find_first_not_of(" \t\n\r");
        CHECK(body != std::string::npos);
        const std::string head = upper.substr(body, 6);
        CHECK(upper.compare(body, 4, "WITH") == 0 || head == "SELECT");
    }

    char tmpl[] = "/tmp/ragger_rebuild_check_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    g_scratch_dir = dir;  // X14: removed by ScratchCleanup at exit (and the abort handler)
    const std::string db_path = std::string(dir) + "/check.db";

    auto build_db = [&]() {
        sqlite3* db = nullptr;
        CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
        exec_sql(db, schema);
        sqlite3_close(db);
    };

    auto open_readonly = [&]() {
        sqlite3* db = nullptr;
        const int rc = sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
        CHECK(rc == SQLITE_OK);
        return db;
    };

    // 2. Empty DB: all counts 0, and the read-only open runs the script.
    {
        std::remove(db_path.c_str());
        build_db();
        sqlite3* db = open_readonly();
        const auto counts = run_checks(db, script);
        sqlite3_close(db);
        CHECK(counts.size() == 17);
        for (const auto& [name, count] : counts)
            if (count != 0)
                std::fprintf(stderr, "empty DB: %s = %d (expected 0)\n", name.c_str(), count);
        bool all_zero = std::all_of(counts.begin(), counts.end(),
                                    [](const auto& kv) { return kv.second == 0; });
        CHECK(all_zero);
    }

    // 3. Seeded fixture: one deliberate row per defect kind.
    {
        std::remove(db_path.c_str());
        build_db();
        sqlite3* db = nullptr;
        CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
        exec_sql(db,
            // models
            "INSERT INTO models(model_id, name) VALUES (1, 'm-one'), (2, 'bad');"
            // sessions: s1 named, s2 unnamed (unnamed_sessions_with_l2 target)
            "INSERT INTO sessions(session_id, guid, name) VALUES "
            "  (1, 'g1', 'Alpha'), (2, 'g2', NULL);"
            // turns: T1 has no summary; T2 exists to carry the time mismatch
            "INSERT INTO turns(turn_id, user_text, created_at, session_id) VALUES "
            "  (1, 'u1', 900, 1), (2, 'u2', 1000, 1);"
            // episodes: E1 clean span covering 1000..1020; E2 wrong created_at
            // and no members; E3 empty text (evidence target), no members
            "INSERT INTO episodes(episode_id, text, session_id, span_start, span_end, created_at) VALUES "
            "  (1, 'E1', 1, 1000, 1020, 1020), "
            "  (2, 'E2', 1, 2000, 2010, 1999), "
            "  (3, '',    1, 3000, 3010, 3010);"
            // L2 rows: S1 NULL text; S2 empty not-bad; S3 empty bad (marker);
            // S4 mismatched time attached to E1; S5 stranded inside E1's span;
            // S6 open tail in the unnamed session
            "INSERT INTO turn_summaries(turn_summary_id, text, turn_id, session_id, episode_id, "
            "                            turn_model_id, summary_model_id, turn_datetime) VALUES "
            "  (1, NULL,        NULL, 1, NULL, 1, 1, 500), "
            "  (2, '',          NULL, 1, NULL, 1, 1, 600), "
            "  (3, '',          NULL, 1, NULL, 1, 2, 700), "
            "  (4, 'sum ok',    2,    1, 1,    1, 1, 1010), "
            "  (5, 'stranded',  NULL, 1, NULL, 1, 1, 1005), "
            "  (6, 'open tail', NULL, 2, NULL, 1, 1, 9999), "
            // S7: empty text INSIDE E1's span — a stranded candidate only if
            // the check ever loses its text != '' guard (NC-B).
            "  (7, '',          NULL, 1, NULL, 1, 1, 1015);"
            // knowledge: K1 decision without trigger; K2 insight with evidence
            // pointing at the empty episode E3
            "INSERT INTO knowledge(knowledge_id, kind, text, \"trigger\") VALUES "
            "  (1, 'decision', 'd', NULL); "
            "INSERT INTO knowledge(knowledge_id, kind, text) VALUES (2, 'insight', 'i'); "
            "INSERT INTO knowledge_episode_evidence(knowledge_id, episode_id) VALUES (2, 3);");
        sqlite3_close(db);

        sqlite3* ro = open_readonly();
        const auto counts = run_checks(ro, script);
        sqlite3_close(ro);

        const std::map<std::string, int> expected = {
            {"turns_without_summary", 1},
            {"l2_null_text", 1},
            {"l2_empty_text_not_bad", 2},          // S2 and S7
            {"l2_bad_rows", 1},
            {"l2_time_mismatch", 1},
            {"episodes_created_at_not_span_end", 1},
            {"episodes_without_members", 2},          // E2 and E3
            {"l2_stranded", 1},
            {"l2_open_tail", 1},
            {"decisions_missing_trigger", 1},
            {"unnamed_sessions_with_l2", 1},
            {"embeddings_missing_or_stale_turns", 2},               // T1, T2
            {"embeddings_missing_or_stale_turn_summaries", 7},      // S1..S7
            {"embeddings_missing_or_stale_documents", 0},
            {"embeddings_missing_or_stale_knowledge", 2},           // K1, K2
            {"embeddings_missing_or_stale_episodes", 3},            // E1..E3
            {"evidence_empty_episode_text", 1},
        };
        CHECK(counts.size() == expected.size());
        for (const auto& [name, want] : expected) {
            auto it = counts.find(name);
            if (it == counts.end() || it->second != want)
                std::fprintf(stderr, "seeded: %s = %d (expected %d)\n", name.c_str(),
                             it == counts.end() ? -1 : it->second, want);
            CHECK(it != counts.end());
            CHECK(it->second == want);
        }
        // The two runs must agree on the check-name set.
        std::set<std::string> names;
        for (const auto& [name, _] : expected) names.insert(name);
        CHECK(std::all_of(counts.begin(), counts.end(),
                          [&](const auto& kv) { return names.count(kv.first) == 1; }));
    }

    std::printf("test_rebuild_check: all passed\n");
    return 0;
}
