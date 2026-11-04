// test_text_index — sqlite::TextIndex over an in-memory DB: schema, index, score.
#include <sqlite3.h>

#include "check.h"
#include <iostream>
#include <print>
#include <string>
#include <vector>

#include "sqlite/text_index.h"
#include "sqlite/tables.h"
#include "fts_tokenize.h"

#include <algorithm>

namespace {

// PHASE3 Step 1: registry self-consistency.
void p3_registry_checks() {
    using namespace ragger::sqlite;
    CHECK(kTextTableCount == 5);  // PHASE8: summaries gone (turns, turn_summaries, episodes, knowledge, documents)
    for (const auto& t : kTextTables) {
        CHECK(std::string(t.junction) == std::string(t.table) + "_terms");
        CHECK(find_text_table(t.table) == &t);
    }
    CHECK(find_text_table("nope") == nullptr);
    CHECK(find_text_table("") == nullptr);
    std::cout << "  registry OK\n";
}

// Raw-SQL mirrors of the index queries (TextIndex is `final` and its hooks
// are protected; the pin asserts DB state directly, which is stronger).
int p3_scalar(sqlite3* db, const char* sql) {
    sqlite3_stmt* s = nullptr;
    CHECK(sqlite3_prepare_v2(db, sql, -1, &s, nullptr) == SQLITE_OK);
    int v = -1;
    if (sqlite3_step(s) == SQLITE_ROW) v = sqlite3_column_int(s, 0);
    sqlite3_finalize(s);
    return v;
}
int p3_term_id(sqlite3* db, const std::string& term) {
    std::string sql = "SELECT term_id FROM terms WHERE term = '" + term + "'";
    return p3_scalar(db, sql.c_str());
}
int p3_doc_freq(sqlite3* db, const std::string& junction, int term_id) {
    std::string sql = "SELECT COUNT(*) FROM " + junction +
                      " WHERE term_id = " + std::to_string(term_id);
    return p3_scalar(db, sql.c_str());
}
std::vector<int> p3_posting_ids(sqlite3* db, const std::string& junction,
                                const std::string& pk, int term_id) {
    std::string sql = "SELECT " + pk + " FROM " + junction +
                      " WHERE term_id = " + std::to_string(term_id) +
                      " ORDER BY " + pk;
    std::vector<int> ids;
    sqlite3_stmt* s = nullptr;
    CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &s, nullptr) == SQLITE_OK);
    while (sqlite3_step(s) == SQLITE_ROW) ids.push_back(sqlite3_column_int(s, 0));
    sqlite3_finalize(s);
    return ids;
}

// One raw INSERT per known text table (each has different required columns).
// An unknown table name must fail loudly so later phases that add or remove
// a table are forced to update this helper (PHASE3 Step 1 intent).
void p3_insert_row(sqlite3* db, const std::string& table, int id,
                   const std::string& text) {
    const std::string i = std::to_string(id);
    std::string sql;
    if (table == "turns")
        sql = "INSERT INTO turns(turn_id, user_text, assistant_text) VALUES ("
              + i + ", '" + text + "', 'ack " + i + "')";
    else if (table == "turn_summaries")
        sql = "INSERT INTO turn_summaries(turn_summary_id, text) VALUES ("
              + i + ", '" + text + "')";
    else if (table == "episodes")
        sql = "INSERT INTO episodes(episode_id, text) VALUES ("
              + i + ", '" + text + "')";
    else if (table == "knowledge")
        sql = "INSERT INTO knowledge(knowledge_id, kind, text) VALUES ("
              + i + ", 'note', '" + text + "')";
    else if (table == "documents") {
        // The real store path always creates a document_sources row; the
        // doc text formula's subquery needs one (no row => NULL fold).
        std::string src = "INSERT INTO document_sources(document_source_id, title) VALUES ("
                          + i + ", NULL);";
        CHECK(sqlite3_exec(db, src.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        sql = "INSERT INTO documents(document_id, text, document_source_id) VALUES ("
              + i + ", '" + text + "', " + i + ")";
    }
    else
        CHECK(!"p3_insert_row: unknown table");
    CHECK(sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
}

// PHASE3 Step 0 pin: every text table round-trips through the index —
// insert 2 rows, index, corpus_size/doc_freq/postings, re-index changes
// postings, reindex_table counts both rows. Safety net for the registry
// refactor (Steps 1–3): behavior must be identical before and after.
void p3_pin_all_tables() {
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr) == SQLITE_OK);

    const char* ddl =
        "CREATE TABLE document_sources (document_source_id INTEGER PRIMARY KEY, title TEXT);"
        "CREATE TABLE turns (turn_id INTEGER PRIMARY KEY, user_text TEXT, assistant_text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE turn_summaries (turn_summary_id INTEGER PRIMARY KEY, text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE episodes (episode_id INTEGER PRIMARY KEY, text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE knowledge (knowledge_id INTEGER PRIMARY KEY, kind TEXT, text TEXT, trigger TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE documents (document_id INTEGER PRIMARY KEY, text TEXT,"
        "  document_source_id INTEGER,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);";
    CHECK(sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();

    // The pin iterates the registry itself: any phase that adds or removes
    // a registry row must extend p3_insert_row (unknown name = loud fail).
    for (const auto& t : ragger::sqlite::kTextTables) {
        const std::string table = t.table;
        const std::string pk = t.pk;
        const std::string junction = t.junction;
        CHECK(junction == table + "_terms");
        p3_insert_row(db, table, 1, "qwertyuiop asdfghjkl");
        p3_insert_row(db, table, 2, "zxcvbnm asdfghjkl");

        CHECK(idx.reindex_table(table) == 2);
        CHECK(p3_scalar(db, ("SELECT COUNT(*) FROM " + table).c_str()) == 2);

        const int t_one = p3_term_id(db, "qwertyuiop");
        CHECK(t_one != -1 && t_one != 0);
        CHECK(p3_doc_freq(db, junction, t_one) == 1);
        auto posts = p3_posting_ids(db, junction, pk, t_one);
        CHECK(posts.size() == 1 && posts[0] == 1);

        // Re-index row 1 with different text: the old word's posting goes
        // away, the new word's appears.
        idx.index_record(table, 1, "vbnmzxcv asdfghjkl");
        CHECK(p3_doc_freq(db, junction, t_one) == 0);
        CHECK(p3_posting_ids(db, junction, pk, t_one).empty());
        CHECK(p3_doc_freq(db, junction, p3_term_id(db, "vbnmzxcv")) == 1);

        CHECK(idx.reindex_table(table) == 2);
        std::cout << "  pin " << table << " OK\n";
    }

    sqlite3_close(db);
}

} // namespace

// PHASE9 Step 1: the WITHOUT ROWID junction shape must give the hot paths
// index-backed plans — term->postings and doc_freq ride the PK B-tree (no
// SCAN of the junction), and delete-by-row uses the secondary
// idx_<table>_terms_row index.
void p9_junction_plans() {
    std::println("  p9_junction_plans...");
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr) == SQLITE_OK);
    const char* ddl =
        "CREATE TABLE knowledge (knowledge_id INTEGER PRIMARY KEY, text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);";
    CHECK(sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);
    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();

    auto plan_of = [&](const std::string& sql) {
        sqlite3_stmt* st = nullptr;
        const std::string eqp = "EXPLAIN QUERY PLAN " + sql;
        if (sqlite3_prepare_v2(db, eqp.c_str(), -1, &st, nullptr) != SQLITE_OK) {
            std::println("    plan prepare failed: {} ({})", eqp, sqlite3_errmsg(db));
            CHECK(false);
            return std::string();
        }
        std::string plan;
        while (sqlite3_step(st) == SQLITE_ROW)
            plan += std::string(reinterpret_cast<const char*>(sqlite3_column_text(st, 3))) + " | ";
        sqlite3_finalize(st);
        std::println("    {}", plan);
        return plan;
    };

    // postings: term -> rows (the score_query hot path)
    auto p = plan_of("SELECT j.knowledge_id, j.count, b.unigram_count, b.bigram_count "
                     "FROM knowledge_terms j JOIN knowledge b ON b.knowledge_id = j.knowledge_id "
                     "WHERE j.term_id = 1");
    CHECK(p.find("PRIMARY KEY") != std::string::npos);
    CHECK(p.find("SCAN knowledge_terms") == std::string::npos);

    // delete by row id (write path)
    p = plan_of("DELETE FROM knowledge_terms WHERE knowledge_id = 1");
    CHECK(p.find("idx_knowledge_terms_row") != std::string::npos);
    CHECK(p.find("SCAN knowledge_terms") == std::string::npos);

    // doc_freq
    p = plan_of("SELECT COUNT(*) FROM knowledge_terms WHERE term_id = 1");
    CHECK(p.find("PRIMARY KEY") != std::string::npos || p.find("COVERING") != std::string::npos);

    sqlite3_close(db);
    std::println(" OK");
}


// S49 (B §6 item 12): the breadth the suite was missing — the `turns`
// two-column index path, the literal_enabled=false branch, reset_terms_table()
// against already-cached statements, and score ordering.
//
// Each gets its own in-memory DB: reset_terms_table() drops and recreates every
// junction table, so sharing a handle with the other tests would make the order
// of main() matter.
namespace {

sqlite3* p10_open(const char* ddl) {
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);
    return db;
}

// Every text table create_schema()'s junctions reference. A DELETE on `terms`
// with foreign_keys=ON walks the CASCADE clause of ALL five junction tables, so
// a fixture that deletes a term must have all five parents present (the error
// is 'no such table: main.episodes', far from the statement's own subject).
const char* kP10FullDDL =
    "CREATE TABLE document_sources (document_source_id INTEGER PRIMARY KEY, title TEXT);"
    "CREATE TABLE turns (turn_id INTEGER PRIMARY KEY, user_text TEXT, assistant_text TEXT,"
    "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE turn_summaries (turn_summary_id INTEGER PRIMARY KEY, text TEXT,"
    "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE episodes (episode_id INTEGER PRIMARY KEY, text TEXT,"
    "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE knowledge (knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, text TEXT, trigger TEXT,"
    "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);"
    "CREATE TABLE documents (document_id INTEGER PRIMARY KEY, text TEXT, document_source_id INTEGER,"
    "  unigram_count INTEGER NOT NULL DEFAULT 0, bigram_count INTEGER NOT NULL DEFAULT 0);";

// Ids returned by score_query(), sorted by score descending (the function
// returns an unsorted map dump — sorting here is what makes an ordering
// assertion meaningful).
std::vector<ragger::TextScore> p10_sorted(ragger::sqlite::TextIndex& idx,
                                          const std::string& table,
                                          const std::string& query) {
    auto out = idx.score_query(table, query);
    std::sort(out.begin(), out.end(), [](const ragger::TextScore& a, const ragger::TextScore& b) {
        return a.score > b.score;
    });
    return out;
}

bool p10_has(const std::vector<ragger::TextScore>& v, int id) {
    for (const auto& s : v) if (s.id == id) return true;
    return false;
}

// (a) `turns` is the one text table whose indexed text is built from TWO
// columns (user_text + assistant_text). Terms must be findable from either.
void p10_turns_two_column_path() {
    std::println("  p10_turns_two_column_path...");
    sqlite3* db = p10_open(
        "CREATE TABLE turns (turn_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  user_text TEXT, assistant_text TEXT,"
        "  unigram_count INTEGER NOT NULL DEFAULT 0,"
        "  bigram_count  INTEGER NOT NULL DEFAULT 0);");
    CHECK(sqlite3_exec(db,
        "INSERT INTO turns(user_text, assistant_text) VALUES "
        "('zebra quantum flux','acknowledged'),"
        "('a plain sentence','the reply explains quantum explicitly'),"
        "('nothing relevant at all','nope');",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();
    CHECK(idx.reindex_table("turns") == 3);

    // 'quantum' sits in user_text of row 1 and assistant_text of row 2: both
    // must be found, which is only possible if both columns were indexed.
    auto hits = p10_sorted(idx, "turns", "quantum");
    CHECK(p10_has(hits, 1) && "term in user_text must be indexed");
    CHECK(p10_has(hits, 2) && "term in assistant_text must be indexed");
    CHECK(!p10_has(hits, 3) && "row sharing no term must be absent");

    // Column-exclusive terms resolve to exactly their own row.
    auto zebra = p10_sorted(idx, "turns", "zebra");
    CHECK(zebra.size() == 1 && zebra[0].id == 1);
    auto expl = p10_sorted(idx, "turns", "explicitly");
    CHECK(expl.size() == 1 && expl[0].id == 2);

    // An empty assistant_text (the partial-turn shape) must not break the
    // join: the row keeps indexing from user_text alone.
    CHECK(sqlite3_exec(db, "INSERT INTO turns(turn_id, user_text, assistant_text) "
                           "VALUES (4, 'kappa partial', NULL)",
                       nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(idx.reindex_table("turns") == 4);
    auto kappa = p10_sorted(idx, "turns", "kappa");
    CHECK(kappa.size() == 1 && kappa[0].id == 4);

    sqlite3_close(db);
    std::println(" OK");
}

// (b) literal_enabled = false => DMP-only scoring (the weights reach TextIndex
// through set_weights(); search.cpp's push_text_weights() is the production
// caller). The literal-only query is built honestly rather than guessed: the
// test computes the SAME primary metaphone code the indexer stored and deletes
// that term (CASCADE empties its junction rows), so 'accuracy' is left findable
// only through its literal stem. (A symbol word looks tempting but is not
// literal-only: Double Metaphone does give '+' a code.)
void p10_literal_disabled_scoring() {
    std::println("  p10_literal_disabled_scoring...");
    sqlite3* db = p10_open(kP10FullDDL);
    CHECK(sqlite3_exec(db,
        "INSERT INTO knowledge(kind, text) VALUES "
        "('note', 'speed accuracy'),('note', 'unrelated prose');",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();
    CHECK(idx.reindex_table("knowledge") == 2);

    const std::string stem = ragger::fts::stem("accuracy");
    const std::string meta = ragger::fts::metaphone(stem);
    CHECK(!stem.empty() && !meta.empty());
    CHECK(p3_term_id(db, stem) != -1 && p3_term_id(db, meta) != -1);

    // Default weights: both paths live, so the word hits.
    CHECK(p10_has(p10_sorted(idx, "knowledge", "accuracy"), 1));

    // Drop the phonetic term: what remains for 'accuracy' is literal-only.
    sqlite3_stmt* s = nullptr;
    CHECK(sqlite3_prepare_v2(db, "DELETE FROM terms WHERE term = ?", -1, &s, nullptr) == SQLITE_OK);
    sqlite3_bind_text(s, 1, meta.c_str(), -1, SQLITE_TRANSIENT);
    CHECK(sqlite3_step(s) == SQLITE_DONE);
    sqlite3_finalize(s);

    // Literal path on: still hits (through the stem).
    CHECK(p10_has(p10_sorted(idx, "knowledge", "accuracy"), 1));

    ragger::TextIndexWeights off{};
    off.literal_enabled = false;
    idx.set_weights(off);

    // Literal path off: the literal-only query finds nothing...
    CHECK(p10_sorted(idx, "knowledge", "accuracy").empty());
    // ...while an ordinary term (its metaphone intact) still hits.
    CHECK(p10_has(p10_sorted(idx, "knowledge", "prose"), 2));

    sqlite3_close(db);
    std::println(" OK");
}

// (c) reset_terms_table() DROPs terms and every junction table and recreates
// them. The hot paths hold prepared statements across that DDL; this pins the
// documented "sqlite3_prepare_v2 reprepares transparently" claim.
void p10_reset_terms_with_cached_stmts() {
    std::println("  p10_reset_terms_with_cached_stmts...");
    sqlite3* db = p10_open(kP10FullDDL);
    CHECK(sqlite3_exec(db,
        "INSERT INTO knowledge(kind, text) VALUES "
        "('note', 'alpha bravo'),('note', 'charlie delta');",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();
    CHECK(idx.reindex_table("knowledge") == 2);

    // Warm the caches: lookup_term / doc_freq / postings / corpus_size all run.
    CHECK(p10_has(p10_sorted(idx, "knowledge", "alpha"), 1));
    CHECK(p10_has(p10_sorted(idx, "knowledge", "charlie"), 2));

    idx.reset_terms_table();

    // terms and every junction row are gone: the same query finds nothing,
    // through the same (now reprepared) statements.
    CHECK(p10_sorted(idx, "knowledge", "alpha").empty());
    CHECK(p3_scalar(db, "SELECT COUNT(*) FROM knowledge_terms") == 0);

    // Re-index one row: it becomes findable again, and only it. term_ids
    // restart fresh from 1 (AUTOINCREMENT counter went with the DROP).
    idx.index_record("knowledge", 1, "alpha bravo");
    auto after = p10_sorted(idx, "knowledge", "alpha");
    CHECK(after.size() == 1 && after[0].id == 1);
    CHECK(p3_scalar(db, "SELECT MIN(term_id) FROM terms") == 1);   // counter restarted fresh
    CHECK(p10_sorted(idx, "knowledge", "charlie").empty());

    sqlite3_close(db);
    std::println(" OK");
}

// (d) Score ordering: TF is count/unigram_count, so the row that repeats the
// query term must rank first, and the returned scores must be strictly
// descending once sorted (the old assertion only checked non-emptiness).
void p10_score_ordering_strict() {
    std::println("  p10_score_ordering_strict...");
    sqlite3* db = p10_open(kP10FullDDL);
    CHECK(sqlite3_exec(db,
        "INSERT INTO knowledge(kind, text) VALUES "
        "('note', 'widget widget widget'),('note', 'widget alone'),('note', 'no matching prose here');",
        nullptr, nullptr, nullptr) == SQLITE_OK);

    ragger::sqlite::TextIndex idx(db);
    idx.create_schema();
    CHECK(idx.reindex_table("knowledge") == 3);

    auto hits = p10_sorted(idx, "knowledge", "widget");
    CHECK(hits.size() == 2);                       // row 3 shares no term
    CHECK(hits[0].id == 1);                        // the repeating row ranks first
    CHECK(hits[0].score > hits[1].score);          // strictly, not merely >=
    for (size_t i = 1; i < hits.size(); ++i)
        CHECK(hits[i - 1].score > hits[i].score);

    sqlite3_close(db);
    std::println(" OK");
}

} // namespace


int main() {
    p3_registry_checks();
    p9_junction_plans();
    p3_pin_all_tables();
    p10_turns_two_column_path();
    p10_literal_disabled_scoring();
    p10_reset_terms_with_cached_stmts();
    p10_score_ordering_strict();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open(":memory:", &db) == SQLITE_OK);

    // Minimal knowledge table (id + text + count columns the index writes).
    const char* ddl =
        "CREATE TABLE knowledge ("
        "  knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  kind TEXT,"
        "  text TEXT,"
        "  trigger TEXT,"   /* db 0.17: TextIndex reads trigger + text */
        "  unigram_count INTEGER NOT NULL DEFAULT 0,"
        "  bigram_count  INTEGER NOT NULL DEFAULT 0);";
    CHECK(sqlite3_exec(db, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(sqlite3_exec(db,
        "INSERT INTO knowledge(kind, text) VALUES "
        "('note', 'The quick brown fox jumps'),"
        "('note', 'A lazy brown dog sleeps'),"
        "('note', 'Nothing about animals here');",
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
    int n = idx.reindex_table("knowledge");
    CHECK(n == 3);

    // knowledge_terms must be populated and idempotent on a second pass.
    auto term_rows = [&]() {
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db, "SELECT count(*) FROM knowledge_terms", -1, &s, nullptr);
        sqlite3_step(s);
        int c = sqlite3_column_int(s, 0);
        sqlite3_finalize(s);
        return c;
    };
    int rows1 = term_rows();
    CHECK(rows1 > 0);
    int n2 = idx.reindex_table("knowledge");
    CHECK(n2 == 3);
    CHECK(term_rows() == rows1);  // idempotent: no doubling

    // Scoring: "brown" appears in rows 1 and 2 -> both should score > 0,
    // and row 3 (no shared terms) should be absent.
    auto scores = idx.score_query("knowledge", "brown fox");
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
