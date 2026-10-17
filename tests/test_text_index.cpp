// test_text_index — sqlite::TextIndex over an in-memory DB: schema, index, score.
#include <sqlite3.h>

#include "check.h"
#include <iostream>

#include "sqlite/text_index.h"

int main() {
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);

    // Minimal decisions table (id + text + count columns the index writes).
    const char* ddl =
        "CREATE TABLE decisions ("
        "  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  text TEXT,"
        "  trigger TEXT,"   /* db 0.17: TextIndex reads trigger + text */
        "  unigram_count INTEGER NOT NULL DEFAULT 0,"
        "  bigram_count  INTEGER NOT NULL DEFAULT 0);";
    CHECK(sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(sqlite3_exec(db,
        "INSERT INTO decisions(text) VALUES "
        "('The quick brown fox jumps'),"
        "('A lazy brown dog sleeps'),"
        "('Nothing about animals here');",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();

    // terms table must now exist.
    {
        sqlite3_stmt* s = nullptr;
        CHECK(sqlite3_prepare_v2(db,
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='terms'",
            -1, &s, nullptr) == SQLITE_OK);
        CHECK(sqlite3_step(s) == SQLITE_ROW);
        CHECK(sqlite3_column_int(s, 0) == 1);
        sqlite3_finalize(s);
    }

    // Full reindex: all 3 records processed.
    int n = idx.reindex_table("decisions");
    CHECK(n == 3);

    // decisions_terms must be populated and idempotent on a second pass.
    auto term_rows = [&]() {
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db, "SELECT count(*) FROM decisions_terms", -1, &s, nullptr);
        sqlite3_step(s);
        int c = sqlite3_column_int(s, 0);
        sqlite3_finalize(s);
        return c;
    };
    int rows1 = term_rows();
    CHECK(rows1 > 0);
    int n2 = idx.reindex_table("decisions");
    CHECK(n2 == 3);
    CHECK(term_rows() == rows1);  // idempotent: no doubling

    // Scoring: "brown" appears in rows 1 and 2 -> both should score > 0,
    // and row 3 (no shared terms) should be absent.
    auto scores = idx.score_query("decisions", "brown fox");
    bool saw1 = false, saw2 = false, saw3 = false;
    for (const auto& r : scores) {
        if (r.id == 1) saw1 = (r.score > 0.0f);
        if (r.id == 2) saw2 = (r.score > 0.0f);
        if (r.id == 3) saw3 = true;
    }
    CHECK(saw1 && "row 1 (quick brown fox) should score");
    CHECK(saw2 && "row 2 (lazy brown dog) should score on 'brown'");
    CHECK(!saw3 && "row 3 shares no query terms");

    // "fox" is rarer than "brown" -> row 1 should outrank row 2.
    float s1 = 0, s2 = 0;
    for (const auto& r : scores) {
        if (r.id == 1) s1 = r.score;
        if (r.id == 2) s2 = r.score;
    }
    CHECK(s1 > s2 && "row with rarer 'fox' term should rank higher");

    // ---- PHASE1 Step 3b: index writes must fail loudly -------------------
    // FKs must be ON for the junction-table constraint to fire (explicit here
    // even though our linked SQLite defaults them ON — the test proves the
    // throw regardless of the default).
    CHECK(sqlite3_exec(db, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(sqlite3_exec(db,
        "CREATE TABLE turn_summaries ("
        "  turn_summary_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0,"
        "  bigram_count  INTEGER NOT NULL DEFAULT 0);",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    // A record id that does not exist in the base table must throw (the FK
    // failure used to be swallowed — the exact mechanism that hid the
    // turn_summaries mis-keying bug). replace_record_terms/set_counts are
    // protected on a final class, so the path is exercised through the
    // public index_record(), which calls both.
    bool threw = false;
    try {
        idx.index_record("turn_summaries", 999999, "brown fox");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw && "indexing a nonexistent record id must throw");

    // set_counts on an existing row still works (counts land on the row).
    CHECK(sqlite3_exec(db, "INSERT INTO turn_summaries(turn_summary_id, text) VALUES (7, 'seven brown foxes')",
                       nullptr, nullptr, nullptr) == SQLITE_OK);
    idx.index_record("turn_summaries", 7, "seven brown foxes");
    {
        sqlite3_stmt* s = nullptr;
        CHECK(sqlite3_prepare_v2(db,
            "SELECT unigram_count FROM turn_summaries WHERE turn_summary_id = 7",
            -1, &s, nullptr) == SQLITE_OK);
        CHECK(sqlite3_step(s) == SQLITE_ROW);
        CHECK(sqlite3_column_int(s, 0) > 0);
        sqlite3_finalize(s);
    }

    sqlite3_close(db);
    std::cout << "test_text_index: OK\n";
    return 0;
}
