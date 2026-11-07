/**
 * SQLite backend integration tests
 *
 * Requires ONNX model files at the configured model_dir.
 * Uses a temp DB that's deleted after the test.
 */
#include "config.h"
#include "backfill.h"
#include "embedder.h"
#include "inference.h"
#include "sqlite/backend.h"
#include "store_kind.h"
#include "sqlite/status.h"
#include "sqlite/tables.h"
#include "user_store.h"
#include "auth.h"
#include "util/time.h"
#include "util/fs.h"
#include "util/sqlite.h"
#include "test_env.h"
#include <sqlite3.h>
#include "check.h"
#include <filesystem>
#include <iostream>
#include <print>
#include <regex>
#include <thread>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <set>
#include <vector>
#include <algorithm>
#include <atomic>
#include <optional>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// LR S57: the shared same-exchange merge rule lives in src/sqlite/write.cpp
// (deliberately not in a header); declared here so the test pins its values.
namespace ragger::sqlite {
std::optional<std::string> merge_assistant_text(const std::string& prev,
                                                const std::string& next);
}

namespace fs = std::filesystem;

static const std::string TEMP_DB = "/tmp/ragger_test_backend.db";

// Clean up temp DB and WAL/SHM files
static void cleanup() {
    fs::remove(TEMP_DB);
    fs::remove(TEMP_DB + "-wal");
    fs::remove(TEMP_DB + "-shm");
    // BM25 index table lives in same DB, no separate file
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

void test_store_and_count(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    CHECK(db.count() == 0);

    db.store("Hello world, this is a test memory.");
    CHECK(db.count() == 1);

    db.store("Another memory about something else.");
    CHECK(db.count() == 2);

    db.close();
    cleanup();
}

void test_store_with_metadata(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // PHASE8: notes are knowledge rows; a level other than "note" becomes
    // a `level:<value>` tag (there is no level column any more).
    ragger::json meta = {
        {"level", "session"},
        {"tags", {"test", "unit"}}
    };
    std::string id = db.store("Metadata round-trip test.", meta);
    CHECK(!id.empty());

    auto all = db.load_all();
    CHECK(all.size() == 1);
    // Tags stored as comma-separated string in the tags column, with the
    // level folded in.
    CHECK(all[0].metadata["tags"] == "test,unit,level:session");
    CHECK(all[0].metadata["kind"] == "note");

    db.close();
    cleanup();
}

void test_search_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("The capital of France is Paris.");
    db.store("SQLite is a lightweight database engine.");
    db.store("Machine learning uses neural networks for prediction.");

    auto resp = db.search("What is the capital of France?", 3, 0.0f);
    CHECK(!resp.results.empty());
    // The France/Paris doc should be the top result
    CHECK(resp.results[0].text.find("Paris") != std::string::npos);

    db.close();
    cleanup();
}

// FTS5 MATCH is built from arbitrary user text; punctuation/quotes/operators
// must not produce a syntax error. The keyword pass should degrade to "no
// keyword contribution" rather than throw, and vector search still returns.
void test_search_fts_special_chars(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("SQLite is a lightweight embedded database engine.");
    db.store("The Eiffel Tower stands in Paris.");

    // Queries laden with FTS5-significant characters — must not throw.
    const char* nasty[] = {
        "database \"engine\" AND (lightweight)",
        "what is SQLite?  -- punctuation: *,^,:",
        "NEAR(\"paris",          // unbalanced quote / bare operator
        "***",                   // no usable tokens → keyword pass skipped
    };
    for (const char* q : nasty) {
        auto resp = db.search(q, 5, 0.0f);   // must not throw
        (void)resp;
    }

    // A normal keyword still resolves through the hybrid path.
    auto resp = db.search("lightweight database", 5, 0.0f);
    CHECK(!resp.results.empty());
    CHECK(resp.results[0].text.find("SQLite") != std::string::npos);

    db.close();
    cleanup();
}

void test_search_min_score(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("The weather in Los Angeles is sunny.");
    db.store("Quantum physics studies subatomic particles.");

    // Very high min_score should return no results for a vague query
    auto resp = db.search("random unrelated gibberish xyzzy", 5, 0.99f);
    CHECK(resp.results.empty());

    db.close();
    cleanup();
}

// Regression for bug M4: min_score must be applied BEFORE top-k truncation,
// not after. Previously, partial_sort picked the top `limit` candidates by
// blended score and only then dropped any whose raw cosine was below
// min_score — so a high-blend/low-cosine candidate could occupy a slot and
// get filtered out, leaving fewer results than qualifying candidates existed.
// Here we assert two invariants that the post-filter ordering can violate:
//   (1) every returned result's score is >= min_score, and
//   (2) results are sorted by blended score with no min_score "holes" — i.e.
//       we never return fewer than min(limit, #qualifying) items.
void test_search_min_score_before_topk(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // A cluster of clearly on-topic docs (high cosine to the query) plus some
    // off-topic ones. With a moderate threshold the on-topic docs qualify.
    db.store("Paris is the capital city of France.");
    db.store("France's capital, Paris, sits on the Seine river.");
    db.store("The capital of France is the city of Paris.");
    db.store("Lyon and Marseille are large French cities, not the capital.");
    db.store("Bananas are a tropical fruit rich in potassium.");
    db.store("The mitochondria is the powerhouse of the cell.");

    const float thr = 0.25f;
    auto resp = db.search("What is the capital of France?", 5, thr);

    // (1) Invariant: nothing below threshold is ever returned.
    for (const auto& r : resp.results)
        CHECK(r.score >= thr);

    // (2) Count how many candidates actually clear the threshold by asking for
    // a huge limit; with the fix, a smaller limit returns exactly
    // min(limit, qualifying), never fewer due to top-k stealing.
    auto all = db.search("What is the capital of France?", 100, thr);
    size_t qualifying = all.results.size();
    size_t expected = std::min<size_t>(5, qualifying);
    CHECK(resp.results.size() == expected);

    db.close();
    cleanup();
}

void test_search_limit(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    for (int i = 0; i < 10; ++i) {
        db.store("Test document number " + std::to_string(i) + " with some content.");
    }

    auto resp = db.search("test document", 3, 0.0f);
    CHECK((int)resp.results.size() <= 3);

    db.close();
    cleanup();
}

void test_load_all(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("First memory.");
    db.store("Second memory.");
    db.store("Third memory.");

    auto all = db.load_all();
    CHECK(all.size() == 3);

    db.close();
    cleanup();
}

void test_rebuild_embeddings(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Store some test documents
    db.store("Embedding rebuild test document one.");
    db.store("Embedding rebuild test document two.");
    db.store("Something completely different here.");

    // Rebuild all embeddings
    int count = db.rebuild_embeddings(emb);
    CHECK(count == 3);

    // Search should still work after rebuild
    auto resp = db.search("embedding test", 5, 0.0f);
    CHECK(!resp.results.empty());

    db.close();
    cleanup();
}

void test_rebuild_embeddings_empty_db(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Rebuild on empty DB should return 0
    int count = db.rebuild_embeddings(emb);
    CHECK(count == 0);

    db.close();
    cleanup();
}

void test_rebuild_embeddings_count_matches(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Store various numbers of documents
    for (int i = 0; i < 7; ++i) {
        db.store("Test document number " + std::to_string(i));
    }

    int total_count = db.count();
    int rebuild_count = db.rebuild_embeddings(emb);
    
    // Rebuild count should match total count
    CHECK(rebuild_count == total_count);
    CHECK(rebuild_count == 7);

    db.close();
    cleanup();
}

// search_text_only() is the fallback used when vector search is unusable —
// embedding drift at startup, and while a re-embed is running. It queries every
// FTS corpus including turn_summaries, which has no created_at column; naming
// the wrong timestamp column there threw out of the whole function, so the
// fallback returned nothing at all and the failure only showed up in a log line
// nobody was reading. Exercise every corpus so a column rename can't do it again.
void test_search_text_only(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("A summary about harpsichord tuning.");
    ragger::DocumentChunk doc;
    doc.text  = "A document about harpsichord maintenance.";
    doc.title = "Harpsichords";
    doc.path  = "harpsichord.md";
    db.store_document(doc);
    db.store_decision("Decided to buy a harpsichord.", "current", "music");
    int turn_id = db.store_turn("What is a harpsichord?",
                                "A plucked keyboard instrument.", "test-model");
    // An L2 turn summary is the corpus that carried the bug. It only matters
    // if the query actually HITS it — the bad column sat in the per-row fetch,
    // which never runs when a corpus has no matches, so an empty
    // turn_summaries table would let the bug through.
    db.finalize_turn_summary(turn_id, "User asked about the harpsichord.",
                             "test-model");
    CHECK(db.turn_summary_exists(turn_id));

    // Must not throw, and must actually find the keyword across corpora.
    auto resp = db.search_text_only("harpsichord", 10);
    CHECK(resp.timing.contains("text_only"));
    CHECK(!resp.results.empty());

    // Every hit carries a source label and no vector signal.
    bool saw_turn_summary = false;
    for (const auto& r : resp.results) {
        CHECK(r.metadata.contains("source"));
        CHECK(r.vec_score < 0.0f);
        if (r.metadata["source"] == "turn_summary") {
            saw_turn_summary = true;
            CHECK(!r.timestamp.empty());   // resolved from turn_datetime
        }
    }
    CHECK(saw_turn_summary);

    // A query that matches nothing returns cleanly rather than throwing.
    auto none = db.search_text_only("zzzzznotawordzzzzz", 5);
    CHECK(none.results.empty());

    db.close();
    cleanup();
}

void test_search_timing(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("Timing test document.");
    auto resp = db.search("timing", 5, 0.0f);
    // Timing JSON should have keys
    CHECK(resp.timing.contains("embedding_ms") || resp.timing.contains("total_ms") ||
           !resp.timing.empty());

    db.close();
    cleanup();
}

void test_search_by_metadata(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Lean v2: filterable columns are level / status / tags (+ time window).
    db.store("Apple note.",  {{"level", "turn"},    {"tags", {"fruit"}}});
    db.store("Banana note.", {{"level", "turn"},    {"tags", {"fruit", "tropical"}}});
    db.store("Car note.",    {{"level", "session"}, {"tags", {"vehicle"}}});

    // Filter by level (PHASE8: level lives as a `level:<value>` tag)
    auto results = db.search_by_metadata({{"level", "turn"}});
    CHECK(results.size() == 2);
    for (auto& r : results)
        CHECK(r.metadata["tags"].get<std::string>().find("level:turn") != std::string::npos);

    // Filter by tag (LIKE on the tags column)
    results = db.search_by_metadata({{"tags", "vehicle"}});
    CHECK(results.size() == 1);
    CHECK(results[0].text == "Car note.");

    // AND across columns
    results = db.search_by_metadata({{"level", "turn"}, {"tags", "tropical"}});
    CHECK(results.size() == 1);
    CHECK(results[0].text == "Banana note.");

    // Limit respected
    results = db.search_by_metadata({{"level", "turn"}}, 1);
    CHECK(results.size() == 1);

    // No matches → empty
    results = db.search_by_metadata({{"level", "project"}});
    CHECK(results.empty());

    // Unsupported key under lean schema → empty
    results = db.search_by_metadata({{"color", "red"}});
    CHECK(results.empty());

    db.close();
    cleanup();
}

void test_user_management(ragger::Embedder& emb) {
    cleanup();
    // User management now uses sqlite::Backend (separate from storage backend)
    ragger::UserStore umgr(TEMP_DB);

    // create_user → returns valid ID
    int user_id = umgr.create_user("testuser", "abc123hash");
    CHECK(user_id > 0);

    int admin_id = umgr.create_user("adminuser", "def456hash");
    CHECK(admin_id > 0);
    CHECK(admin_id != user_id);

    // get_user_by_token_hash → finds created user
    auto user_opt = umgr.get_user_by_token_hash("abc123hash");
    CHECK(user_opt.has_value());
    CHECK(user_opt->username == "testuser");
    CHECK(user_opt->token_hash == "abc123hash");

    // get_user_by_username → finds created user
    user_opt = umgr.get_user_by_username("adminuser");
    CHECK(user_opt.has_value());
    CHECK(user_opt->username == "adminuser");
    CHECK(user_opt->token_hash == "def456hash");

    // get_user_by_token_hash with wrong hash → nullopt
    user_opt = umgr.get_user_by_token_hash("wronghash");
    CHECK(!user_opt.has_value());

    // get_user_by_username with wrong name → nullopt
    user_opt = umgr.get_user_by_username("nonexistent");
    CHECK(!user_opt.has_value());

    cleanup();
}

void test_timestamp_format(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("Timestamp format test.");
    auto all = db.load_all();
    CHECK(all.size() == 1);

    // PHASE8: load_all renders knowledge.created_at as local
    // "YYYY-MM-DD HH:MM:SS" (same rendering as every other read path).
    auto& ts = all[0].timestamp;
    CHECK(!ts.empty());
    std::tm tm{};
    CHECK(strptime(ts.c_str(), "%Y-%m-%d %H:%M:%S", &tm) != nullptr);
    tm.tm_isdst = -1;   // let mktime resolve DST (strptime leaves it 0)
    int64_t epoch_val = static_cast<int64_t>(mktime(&tm));
    int64_t now_epoch = static_cast<int64_t>(std::time(nullptr));
    CHECK(epoch_val > 0);
    CHECK(epoch_val <= now_epoch + 5);       // not in the future
    CHECK(now_epoch - epoch_val < 300);      // stored "just now"

    db.close();
    cleanup();
}

// Guards the read-path wiring: the generic store/read API operates on the
// unified `knowledge` table as notes (PHASE8; formerly the lean `summaries`
// table), not the dropped pre-v2 `memories` table. tags round-trip with the
// level folded in as a tag; there is no metadata blob.
void test_v2_summaries_backing(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        db.store("v2 summaries backing check.",
                 {{"level", "session"}, {"tags", {"note"}}});

        auto all = db.load_all();
        CHECK(all.size() == 1);
        CHECK(all[0].metadata["kind"] == "note");
        CHECK(all[0].metadata["tags"] == "note,level:session");
        db.close();
    }

    // Inspect the raw DB: the row must be in `knowledge`, and the legacy
    // `memories` table must not exist in a fresh database.
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM knowledge WHERE kind='note'", -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);

    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='memories'",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 0);   // no legacy table in fresh DB
    sqlite3_finalize(st);

    sqlite3_close(raw);
    cleanup();
}

// store_document() must write Level 5 RAG chunks to the lean `documents`
// table with title / tags (subject) / year / path in dedicated columns for
// grouping — none of which are folded into the embedded `text`.
void test_store_document(ragger::Embedder& emb) {
    cleanup();
    int doc_id = -1;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        ragger::DocumentChunk doc;
        doc.text        = "The mitochondria is the powerhouse of the cell.";
        doc.title       = "Cell Biology Primer";
        doc.tags        = "biology,cells";
        doc.year        = 2021;
        doc.path        = "/tmp/primer.md";
        doc.chunk_index = 1;

        doc_id = db.store_document(doc);
        CHECK(doc_id > 0);
        db.close();
    }

    // Verify the dedicated columns round-trip. Per-document metadata
    // (title/tags/year/path) lives in document_sources since the v0.15
    // normalization migration; documents (chunks) hold only text/chunk_index
    // + a document_source_id FK.
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT d.text, ds.title, ds.tags, ds.year, ds.path, d.chunk_index "
        "FROM documents d JOIN document_sources ds "
        "  ON ds.document_source_id = d.document_source_id "
        "WHERE d.document_id = ?",
        -1, &st, nullptr);
    sqlite3_bind_int(st, 1, doc_id);
    CHECK(sqlite3_step(st) == SQLITE_ROW);

    auto col = [&](int i) {
        const char* s = reinterpret_cast<const char*>(sqlite3_column_text(st, i));
        return std::string(s ? s : "");
    };
    std::string text = col(0);
    CHECK(col(1) == "Cell Biology Primer");
    CHECK(col(2) == "biology,cells");
    CHECK(sqlite3_column_int(st, 3) == 2021);
    CHECK(col(4) == "/tmp/primer.md");
    CHECK(sqlite3_column_int(st, 5) == 1);

    // Grouping metadata must NOT be embedded into the body text.
    CHECK(text == "The mitochondria is the powerhouse of the cell.");
    CHECK(text.find("Cell Biology Primer") == std::string::npos);

    sqlite3_finalize(st);
    sqlite3_close(raw);
    cleanup();
}

// Uniform turn capture (issue #41): store_turn writes a raw L1 exchange into
// the `turns` table with a model reference; the partial/finalize flow embeds
// the exchange once the assistant reply arrives.
void test_store_turn(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        // Complete turn in one call — embedded immediately.
        int t1 = db.store_turn("What is the capital of France?",
                               "The capital of France is Paris.", "test-model");
        CHECK(t1 > 0);

        // Partial turn (no assistant yet) → embedding NULL → finalize fills it.
        int t2 = db.store_turn("Tell me about photosynthesis.", "", "test-model",
                               /*defer_embedding=*/true);
        CHECK(t2 > 0 && t2 != t1);
        CHECK(db.finalize_turn(t2, "Plants convert sunlight into energy.",
                                "test-model"));
        db.close();
    }

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    // Both rows present, both have assistant_text and a non-NULL embedding,
    // and a model_id resolved to the 'test-model' models row.
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT t.user_text, t.assistant_text, "
        "       length(t.embedding), m.name "
        "FROM turns t JOIN models m ON m.model_id = t.model_id "
        "ORDER BY t.turn_id", -1, &st, nullptr);
    int rows = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
        const char* u = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        const char* a = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        int emb_bytes = sqlite3_column_int(st, 2);
        const char* model = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
        CHECK(u && a && a[0] != '\0');          // assistant filled in
        // Embedded as f16, payload-only (db_version 0.15+; no version byte —
        // that tag lives in the sibling embedding_version column now):
        // 384 dims * 2 bytes/dim.
        CHECK(emb_bytes == 384 * 2);
        CHECK(model && std::string(model) == "test-model");
        ++rows;
    }
    sqlite3_finalize(st);
    CHECK(rows == 2);

    // The model row was created exactly once, not duplicated.
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM models WHERE name='test-model'",
                       -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);

    // finalize_turn on a non-existent id returns false.
    {
        ragger::sqlite::Backend db2(emb, TEMP_DB);
        CHECK(!db2.finalize_turn(99999, "x", "test-model"));
        db2.close();
    }

    sqlite3_close(raw);
    cleanup();
}

// PHASE9B step 10: store_turn keeps only the FINAL assistant text of a turn
// and never rewrites history. Probe = newest turn of the SAME session, same
// user text, within 5 minutes. See Backend::Impl::store_turn.
static std::string p9b_turn_field(const std::string& db_path, int turn_id, const char* col) {
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    std::string sql = std::string("SELECT ") + col + " FROM turns WHERE turn_id = ?";
    sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr);
    sqlite3_bind_int(st, 1, turn_id);
    std::string out = "<missing>";
    if (sqlite3_step(st) == SQLITE_ROW) {
        const char* t = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        out = t ? t : "<null>";
    }
    sqlite3_finalize(st);
    sqlite3_close(raw);
    return out;
}

static int p9b_count_turns(const std::string& db_path) {
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turns", -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    int n = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
    sqlite3_close(raw);
    return n;
}

void test_store_turn_dedup(ragger::Embedder& emb) {
    // (a) identical exchange, same session -> same id, one row, nothing changes.
    // (c) identical exchange, different sessions -> two rows.
    cleanup();
    int t1 = 0, t2 = 0, t3 = 0;
    std::string created1;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        t1 = db.store_turn("Explain recipes.", "They pick context.", "m", false, "sessionA");
        CHECK(t1 > 0);
        created1 = p9b_turn_field(TEMP_DB, t1, "created_at");
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        t2 = db.store_turn("Explain recipes.", "They pick context.", "m", false, "sessionA");
        CHECK(t2 == t1);
        t3 = db.store_turn("Explain recipes.", "They pick context.", "m", false, "sessionB");
        CHECK(t3 != t1);
        db.close();
    }
    CHECK(p9b_count_turns(TEMP_DB) == 2);
    CHECK(p9b_turn_field(TEMP_DB, t1, "created_at") == created1);
    CHECK(p9b_turn_field(TEMP_DB, t1, "assistant_text") == "They pick context.");
    cleanup();
}

void test_store_turn_dedup_same_session(ragger::Embedder& emb) {
    const std::string S = "p9b-turn-guid";
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        // (b) same user text, different assistant text: cannot be told from a
        // continuation, so it is APPENDED to the same row (no second turn, no
        // overwrite).
        int t1 = db.store_turn("U1", "Part one.", "m", false, S);
        std::string created1 = p9b_turn_field(TEMP_DB, t1, "created_at");
        int t2 = db.store_turn("U1", "Part two.", "m", false, S);
        CHECK(t2 == t1);
        CHECK(p9b_turn_field(TEMP_DB, t1, "assistant_text") == "Part one.\n\nPart two.");
        CHECK(p9b_turn_field(TEMP_DB, t1, "created_at") == created1);
        CHECK(p9b_turn_field(TEMP_DB, t1, "embedding IS NOT NULL") == "1");

        // A repeat of an already-held part (second capture hook) is a no-op.
        int t3 = db.store_turn("U1", "Part two.", "m", false, S);
        CHECK(t3 == t1);
        CHECK(p9b_turn_field(TEMP_DB, t1, "assistant_text") == "Part one.\n\nPart two.");

        // Cumulative resend (starts with the held text) replaces, not appends.
        int t4 = db.store_turn("U1", "Part one.\n\nPart two.\n\nPart three.", "m", false, S);
        CHECK(t4 == t1);
        CHECK(p9b_turn_field(TEMP_DB, t1, "assistant_text") ==
              "Part one.\n\nPart two.\n\nPart three.");

        // (d) partial row (no assistant text) completed in place.
        int p1 = db.store_turn("U2", "", "m", false, S);
        CHECK(p1 > t1);
        CHECK(p9b_turn_field(TEMP_DB, p1, "assistant_text") == "<null>");
        std::string pc = p9b_turn_field(TEMP_DB, p1, "created_at");
        int p2 = db.store_turn("U2", "Final.", "m", false, S);
        CHECK(p2 == p1);
        CHECK(p9b_turn_field(TEMP_DB, p1, "assistant_text") == "Final.");
        CHECK(p9b_turn_field(TEMP_DB, p1, "created_at") == pc);

        // (e) deferred embedding on an append leaves the embedding NULL (never a
        // stale vector for the old text).
        int d1 = db.store_turn("U3", "First.", "m", false, S);
        int d2 = db.store_turn("U3", "Second.", "m", /*defer_embedding=*/true, S);
        CHECK(d2 == d1);
        CHECK(p9b_turn_field(TEMP_DB, d1, "embedding IS NULL") == "1");

        // A different prompt is its own turn; unknown session never probes.
        int n1 = db.store_turn("U4", "A", "m", false, S);
        CHECK(n1 != d1);
        int e1 = db.store_turn("U5", "A", "m", false, "");
        int e2 = db.store_turn("U5", "B", "m", false, "");
        CHECK(e1 != e2);
        db.close();
    }
    cleanup();
}

// --- v0.12.0 turn_summaries regression tests -------------------------------
// The turn_id-FK-based turn_summaries table replaces the old
// (session_guid, source_timestamp)-keyed linkage. Per Reid's explicit design
// decision, there is no placeholder-row mechanism: store_turn() never
// creates a turn_summaries row. finalize_turn_summary()/mark_turn_summarized()
// are plain INSERTs (no upsert) — the unique partial index on turn_id
// guards against an accidental duplicate insert for the same turn_id.
// unsummarized_turns() finds turns to process via a LEFT JOIN "no matching
// row" check, not a NULL-text sentinel.

// store_turn() must NOT create any turn_summaries row, even for a complete
// (non-partial) turn — the row only appears once finalize_turn_summary() or
// mark_turn_summarized() is explicitly called.
void test_store_turn_creates_no_turn_summaries_row(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t1 = db.store_turn("What is the capital of Spain?",
                           "The capital of Spain is Madrid.", "test-model");
    CHECK(t1 > 0);
    CHECK(!db.turn_summary_exists(t1));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 0);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// A partial turn (no assistant_text yet) also gets no turn_summaries row —
// same as a complete turn, since store_turn() never creates one at all now.
void test_store_turn_partial_turn_creates_no_turn_summaries_row(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t1 = db.store_turn("Tell me about volcanoes.", "", "test-model",
                          /*defer_embedding=*/true);
    CHECK(t1 > 0);
    CHECK(!db.turn_summary_exists(t1));

    db.close();
    cleanup();
}

// finalize_turn_summary() is a plain INSERT now (no upsert): calling it
// twice on the same turn_id must fail on the second call (unique partial
// index on turn_id rejects the duplicate), leaving exactly ONE row with the
// FIRST call's text in place — not the second.
void test_finalize_turn_summary_rejects_duplicate(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t1 = db.store_turn("Summarize this exchange.",
                           "First finalize target.", "test-model");
    CHECK(t1 > 0);

    CHECK(db.finalize_turn_summary(t1, "First summary text.", "summarizer-model"));
    CHECK(!db.finalize_turn_summary(t1, "Second summary text.", "summarizer-model"));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 1);   // no duplicate row
    sqlite3_finalize(st);

    sqlite3_prepare_v2(raw, "SELECT text FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        CHECK(txt && std::string(txt) == "First summary text.");
    }
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// finalize_turn_summary() against a turn_id that doesn't exist in `turns`
// must fail cleanly (false) and must not insert a row.
void test_finalize_turn_summary_missing_turn_returns_false(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const int missing_turn_id = 999999;
    CHECK(!db.finalize_turn_summary(missing_turn_id, "orphan text", "test-model"));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, missing_turn_id);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 0);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// Deleting the parent `turns` row must SET NULL on turn_summaries.turn_id
// (ON DELETE SET NULL FK), not cascade-delete the summary row, and must
// leave text/turn_datetime unchanged.
void test_turn_summary_survives_turn_deletion(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t1 = db.store_turn("A turn destined for deletion.",
                           "Its summary should survive.", "test-model");
    CHECK(t1 > 0);
    CHECK(db.finalize_turn_summary(t1, "Survivor summary text.", "summarizer-model"));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    int64_t turn_datetime_before = 0;
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT turn_datetime FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    turn_datetime_before = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);

    // Raw delete of the parent turns row -- exercises the FK trigger, not a
    // backend method (there's no public "delete_turn" primitive to call).
    sqlite3_prepare_v2(raw, "PRAGMA foreign_keys = ON", -1, &st, nullptr);
    sqlite3_step(st);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw, "DELETE FROM turns WHERE turn_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_DONE);
    sqlite3_finalize(st);

    // The turn_summaries row still exists, turn_id is now NULL, and the
    // text/turn_datetime are unchanged from before the deletion.
    sqlite3_prepare_v2(raw,
        "SELECT text, turn_id, turn_datetime FROM turn_summaries "
        "WHERE turn_datetime = ?", -1, &st, nullptr);
    sqlite3_bind_int64(st, 1, turn_datetime_before);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        CHECK(txt && std::string(txt) == "Survivor summary text.");
    }
    CHECK(sqlite3_column_type(st, 1) == SQLITE_NULL);   // turn_id now NULL
    CHECK(sqlite3_column_int64(st, 2) == turn_datetime_before);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// unsummarized_turns() must exclude a turn that already has a finalized
// turn_summaries row and include one that doesn't.
void test_unsummarized_turns_excludes_summarized(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t_summarized = db.store_turn("This turn will be summarized.",
                                     "Summarized turn's reply.", "test-model");
    CHECK(t_summarized > 0);
    int t_pending = db.store_turn("This turn stays unsummarized.",
                                  "Pending turn's reply.", "test-model");
    CHECK(t_pending > 0);

    CHECK(db.finalize_turn_summary(t_summarized, "Already done.", "summarizer-model"));

    auto pending = db.unsummarized_turns(0);
    bool found_pending = false, found_summarized = false;
    for (auto& t : pending) {
        if (t.turn_id == t_pending) found_pending = true;
        if (t.turn_id == t_summarized) found_summarized = true;
    }
    CHECK(found_pending);
    CHECK(!found_summarized);

    db.close();
    cleanup();
}

// mark_turn_summarized() marks a trivial/skip-worthy turn as done WITHOUT
// real summary text -- text must be set to '' (empty string, unambiguously
// "intentionally blank" rather than looking like a data error) while
// summary_model_id gets resolved to the given model name.
void test_mark_turn_summarized_trivial_turn(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // store_turn() no longer creates any turn_summaries row automatically.
    int t1 = db.store_turn("ok", "ok", "test-model");
    CHECK(t1 > 0);
    CHECK(!db.turn_summary_exists(t1));

    CHECK(db.mark_turn_summarized(t1, "trivial-skip-model"));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT ts.text, m.name FROM turn_summaries ts "
        "JOIN models m ON m.model_id = ts.summary_model_id "
        "WHERE ts.turn_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_type(st, 0) == SQLITE_TEXT);   // text is '' (not NULL)
    {
        const char* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        CHECK(txt && std::string(txt).empty());
    }
    {
        const char* model = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        CHECK(model && std::string(model) == "trivial-skip-model");
    }
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// reset_abandoned_turn_summaries() must delete ONLY "bad"-tagged rows
// (poison-abandon markers, text = '') and leave legitimate trivial-turn
// skips (any other model name, also text = '') untouched. Deleting a
// poison-abandon row must free the turn_id to reappear in
// unsummarized_turns() so housekeeping can retry it for real.
void test_reset_abandoned_turn_summaries(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // A genuine trivial-turn skip: must survive the reset untouched.
    int t_trivial = db.store_turn("hi", "hi", "test-model");
    CHECK(db.mark_turn_summarized(t_trivial, "trivial-skip-model"));

    // A poison-abandoned turn: tagged with the "bad" sentinel model, text
    // NULL. This is what summarizer_service.cpp writes when a turn hits
    // max_turn_failures.
    int t_bad = db.store_turn(
        "this turn kept failing to summarize before the fix landed",
        "and this is a long enough assistant reply to not be trivial-skipped",
        "test-model");
    CHECK(db.mark_turn_summarized(t_bad, "bad"));
    CHECK(db.turn_summary_exists(t_bad));

    // Both turns currently have a turn_summaries row, so neither shows up
    // as unsummarized yet.
    {
        auto pending = db.unsummarized_turns(0);
        for (auto& p : pending) CHECK(p.turn_id != t_trivial && p.turn_id != t_bad);
    }

    int reset_count = db.reset_abandoned_turn_summaries(0);
    CHECK(reset_count == 1);  // only the "bad" row, not the trivial-skip row

    // The trivial-skip row must still exist, untouched.
    CHECK(db.turn_summary_exists(t_trivial));
    // The poison-abandoned row must be gone.
    CHECK(!db.turn_summary_exists(t_bad));

    // t_bad must now be back in the unsummarized-turns candidate set;
    // t_trivial must NOT (it's legitimately done).
    {
        auto pending = db.unsummarized_turns(0);
        bool found_bad = false, found_trivial = false;
        for (auto& p : pending) {
            if (p.turn_id == t_bad) found_bad = true;
            if (p.turn_id == t_trivial) found_trivial = true;
        }
        CHECK(found_bad);
        CHECK(!found_trivial);
    }

    // Re-running with nothing left to reset is a clean no-op.
    CHECK(db.reset_abandoned_turn_summaries(0) == 0);

    db.close();
    cleanup();
}


// PHASE8 note primitives: store() writes knowledge notes (kind='note').
// A level other than "note" becomes a `level:<value>` tag; two stores of
// the same text are two rows (dedup is the importers' job); a non-creatable
// status is rejected.
void test_note_primitives(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string id1 = db.store("First note.", {{"level", "session"}});
    const std::string id2 = db.store("Second note.", {{"tags", "a,b"}});
    CHECK(!id1.empty() && !id2.empty());

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    auto scalar_int = [&](const std::string& sql) -> long long {
        sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr);
        if (st == nullptr) {
            std::println("  note_primitives PREPARE FAILED: {} ({})", sql, sqlite3_errmsg(raw));
            CHECK(false);
            return -1;
        }
        long long v = (sqlite3_step(st) == SQLITE_ROW) ? sqlite3_column_int64(st, 0) : -1;
        sqlite3_finalize(st); st = nullptr;
        return v;
    };
    auto scalar_text = [&](const std::string& sql) -> std::string {
        sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr);
        if (st == nullptr) {
            std::println("  note_primitives PREPARE FAILED: {} ({})", sql, sqlite3_errmsg(raw));
            CHECK(false);
            return {};
        }
        std::string v = (sqlite3_step(st) == SQLITE_ROW)
            ? reinterpret_cast<const char*>(sqlite3_column_text(st, 0)) : std::string("<none>");
        sqlite3_finalize(st); st = nullptr;
        return v;
    };

    CHECK(scalar_int("SELECT COUNT(*) FROM knowledge WHERE kind='note'") == 2);
    CHECK(scalar_text("SELECT tags FROM knowledge WHERE knowledge_id=" + id1)
          == "level:session");
    CHECK(scalar_text("SELECT tags FROM knowledge WHERE knowledge_id=" + id2) == "a,b");
    CHECK(scalar_text("SELECT status FROM knowledge WHERE knowledge_id=" + id1)
          == "current");

    // Same text twice: two rows (no dedup on the store path).
    db.store("First note.", {});   // no metadata: still a note
    CHECK(scalar_int("SELECT COUNT(*) FROM knowledge WHERE kind='note'") == 3);

    sqlite3_close(raw);

    // 'retired' is not creatable.
    bool threw = false;
    try { db.store("Bad status note.", {{"status", "retired"}}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    CHECK(db.current_decisions(5).empty());      // none created

    db.close();
    cleanup();
}

void test_path_normalization(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const char* home = std::getenv("HOME");
    CHECK(home != nullptr);
    std::string text = std::string("File at ") + home + "/Documents/test.txt is important.";
    db.store(text);

    auto all = db.load_all();
    CHECK(all.size() == 1);
    // Should be normalized to ~/
    CHECK(all[0].text.find("~/Documents/test.txt") != std::string::npos);
    CHECK(all[0].text.find(home) == std::string::npos);

    db.close();
    cleanup();
}

// search() merges three corpora — summaries (L2/L3/L4), documents (L5), and
// decisions (L6) — into one ranked result set, tagging each hit with a
// metadata["source"]. This drives the general_search recipe layer. Decisions
// and documents are embedded via backfill (they may be inserted without an
// embedding), so this also exercises backfill_embeddings() invalidating the
// document + decision caches.
void test_search_merges_three_corpora(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        // (1) Summary via the public store API (embedded immediately).
        db.store("The Voyager probes left the solar system carrying golden records.");

        // (2) Document via store_document (embedded immediately).
        ragger::DocumentChunk doc;
        doc.text  = "Photosynthesis converts sunlight into chemical energy in plants.";
        doc.title = "Botany Notes";
        doc.path  = "/tmp/botany.md";
        db.store_document(doc);

        // (3) Knowledge row inserted directly into the table with NO embedding,
        // mirroring how the external summarizer used to write decisions.
        // backfill then embeds it and must invalidate the knowledge cache.
        {
            sqlite3* raw = nullptr;
            CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
            const char* sql =
                "INSERT INTO knowledge (kind, text, status, tags, created_at) "
                "VALUES ('decision', 'We will migrate the database to PostgreSQL next quarter.', "
                "'current', '', 1767225600)";
            char* err = nullptr;
            CHECK(sqlite3_exec(raw, sql, nullptr, nullptr, &err) == SQLITE_OK);
            sqlite3_close(raw);
        }

        // Embed the unembedded decision row (and any other deferred rows).
        int filled = db.backfill_embeddings(emb);
        CHECK(filled >= 1);

        // Each corpus should surface for its own topical query, tagged by source.
        auto source_of = [&](const std::string& query) -> std::string {
            auto resp = db.search(query, 5, 0.0f, {});
            CHECK(!resp.results.empty());
            return resp.results[0].metadata.value("source", std::string());
        };

        // PHASE8: the note corpus shares the "knowledge" source tag.
        CHECK(source_of("interstellar spacecraft golden record") == "knowledge");
        CHECK(source_of("how plants make energy from light")     == "document");
        CHECK(source_of("plan to switch to PostgreSQL database")  == "knowledge");

        // corpus_size in the timing payload reflects all three tables.
        auto resp = db.search("database", 5, 0.0f, {});
        CHECK(resp.timing.value("corpus_size", 0) == 3);

        db.close();
    }
    cleanup();
}

// A turn_summary (L2) hit carries metadata source="turn_summary", a
// human-readable "datetime" (mirrors turns.created_at), the session_id,
// and the turn_id (raw-turn lookup key). datetime is the addition; keeping
// turn_id follows the "more data is fine, too little is bad" principle.
// The result's primary id is the turn_summary_id.
void test_turn_summary_search_metadata(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        const std::string G = "meta-shape-session";
        const std::string when = "2026-04-15 09:30:00";
        int t1 = db.store_turn("Q1", "A1", "memo", false, G, when);
        CHECK(t1 > 0);
        CHECK(db.finalize_turn_summary(
            t1, "Discussed the migration to a columnar store.", "memo"));

        auto resp = db.search("columnar store migration", 5, 0.0f, {});
        CHECK(!resp.results.empty());

        // Find the turn_summary hit among results.
        const ragger::SearchResult* hit = nullptr;
        for (const auto& r : resp.results) {
            if (r.metadata.value("source", std::string()) == "turn_summary") {
                hit = &r;
                break;
            }
        }
        CHECK(hit && "expected a turn_summary result");

        // datetime present and mirrors the stored turn timestamp.
        CHECK(hit->metadata.contains("datetime"));
        CHECK(hit->metadata["datetime"] == when);
        // The top-level result timestamp is also populated (not blanked).
        CHECK(hit->timestamp == when);
        // session_id and turn_id both present.
        CHECK(hit->metadata.contains("session_id"));
        CHECK(hit->metadata.contains("turn_id"));
        CHECK(hit->metadata["turn_id"] == t1);

        db.close();
    }
    cleanup();
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
// =======================================================================
// Agent API Phase 1 (db 0.17): structured decisions
// =======================================================================

// Raw-sqlite helpers for Phase 1 tests. Named for what they do (never
// "run"/"exec" alone) so they cannot collide with anything else here.
static sqlite3* p1_open_raw() {
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    CHECK(sqlite3_exec(raw, "PRAGMA foreign_keys = ON", nullptr, nullptr, nullptr) == SQLITE_OK);
    return raw;
}
static long long p1_scalar_int(sqlite3* raw, const std::string& sql) {
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) {
        std::println("  p1_scalar_int PREPARE FAILED: {} ({})", sql, sqlite3_errmsg(raw));
        CHECK(false);
    }
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    long long v = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    return v;
}

// PHASE9A: a NULL-text turn_summaries row (legacy, text unset) is reset like a
// poison-abandon row, so its turn gets summarized for real. A '' row (known
// empty / trivial skip) and a row with real text are left alone.
// PHASE9A: a pre-0.16 database is refused before anything is touched.
static void p9a_test_rejects_pre_016_databases(ragger::Embedder& emb) {
    for (const char* ver : {"0.12", "0.15", ""}) {   // "" = existing DB with no version row
        cleanup();
        {
            sqlite3* raw = nullptr;
            CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
            std::string sql =
                "CREATE TABLE turns (turn_id INTEGER PRIMARY KEY);"
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);";
            if (std::string(ver) != "")
                sql += std::string("INSERT INTO settings VALUES ('db_version', '") + ver + "');";
            CHECK(sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
            sqlite3_close(raw);
        }
        bool threw = false;
        std::string what;
        try { ragger::sqlite::Backend db(emb, TEMP_DB); }
        catch (const std::exception& e) { threw = true; what = e.what(); }
        CHECK(threw);
        CHECK(what.find("0.16") != std::string::npos);

        // Untouched: still exactly the two tables we created, and no backup file.
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT count(*) FROM sqlite_master WHERE type = 'table'") == 2);
        sqlite3_close(raw);
        int backups = 0;
        for (auto& e : fs::directory_iterator(fs::path(TEMP_DB).parent_path()))
            if (e.path().filename().string().find("ragger_test_backend.db_BACKUP_") == 0) ++backups;
        CHECK(backups == 0);
    }
    cleanup();
}

static void p9a_test_reset_null_text_rows(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int t_null  = db.store_turn("legacy turn one",
        "a long enough assistant reply for the first legacy turn", "test-model");
    int t_empty = db.store_turn("hi", "hi", "test-model");
    int t_text  = db.store_turn("real turn",
        "a long enough assistant reply that has a real summary", "test-model");
    CHECK(db.mark_turn_summarized(t_null, "legacy-model"));
    CHECK(db.mark_turn_summarized(t_empty, "trivial-skip-model"));
    CHECK(db.finalize_turn_summary(t_text, "A real summary.", "sum-model"));

    sqlite3* raw = p1_open_raw();
    CHECK(sqlite3_exec(raw,
        ("UPDATE turn_summaries SET text = NULL WHERE turn_id = " + std::to_string(t_null)).c_str(),
        nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);

    CHECK(db.reset_abandoned_turn_summaries(0) == 1);   // only the NULL-text row
    CHECK(!db.turn_summary_exists(t_null));
    CHECK(db.turn_summary_exists(t_empty));
    CHECK(db.turn_summary_exists(t_text));

    bool found_null = false;
    for (auto& p : db.unsummarized_turns(0)) if (p.turn_id == t_null) found_null = true;
    CHECK(found_null);                                  // back in the candidate set

    CHECK(db.reset_abandoned_turn_summaries(0) == 0);   // clean no-op afterwards
    db.close();
    cleanup();
}
static std::string p1_scalar_text(sqlite3* raw, const std::string& sql) {
    sqlite3_stmt* st = nullptr;
    CHECK(sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    if (sqlite3_step(st) != SQLITE_ROW) {
        std::println("  p1_scalar_text NO ROW: {}", sql);
        CHECK(false);
    }
    const unsigned char* t = sqlite3_column_text(st, 0);
    std::string v = t ? reinterpret_cast<const char*>(t) : "<NULL>";
    sqlite3_finalize(st);
    return v;
}
// Store a turn and its summary; return the new turn_summary_id.
static int p1_make_turn_summary(ragger::sqlite::Backend& db, const std::string& tag) {
    int t = db.store_turn("Question about " + tag, "Answer about " + tag, "test-model");
    CHECK(t > 0);
    CHECK(db.finalize_turn_summary(t, "Summary about " + tag, "summarizer-model"));
    sqlite3* raw = p1_open_raw();
    int id = static_cast<int>(p1_scalar_int(raw, std::format(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
    sqlite3_close(raw);
    return id;
}
// PHASE5: "%F %T" local wall-time -> epoch seconds (matches resolve_epoch's
// localtime interpretation, which is what store_turn/episodes store).
static long long p5_epoch(const std::string& ts) {
    std::tm tm{};
    strptime(ts.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
    tm.tm_isdst = -1;                              // let mktime pick DST (PDT etc.)
    return static_cast<long long>(mktime(&tm));   // localtime, like resolve_epoch
}

// LR S80 (D2): one closed episode over one fresh turn summary — the unit
// evidence points at now. Own session per tag.
static int p1_make_episode(ragger::sqlite::Backend& db, const std::string& tag) {
    const std::string guid = "ep-" + tag;
    const std::string stamp = "2026-07-01 10:00:00";
    const int t = db.store_turn("Question about " + tag, "Answer about " + tag,
                                "test-model", false, guid, stamp);
    CHECK(t > 0);
    CHECK(db.finalize_turn_summary(t, "Summary about " + tag, "summarizer-model"));
    sqlite3* raw = p1_open_raw();
    const int ts = static_cast<int>(p1_scalar_int(raw, std::format(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
    sqlite3_close(raw);
    ragger::EpisodeClose ec;
    ec.text = "Episode about " + tag;
    ec.model_name = "summarizer-model";
    ec.session_guid = guid;
    ec.span_start = p5_epoch(stamp);
    ec.span_end   = p5_epoch(stamp);
    ec.member_turn_summary_ids = {ts};
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);
    return ep;
}

static bool p1_throws_invalid(ragger::sqlite::Backend& db, const ragger::KnowledgeInput& in) {
    try { db.store_knowledge(in); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_structured_decision_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text      = "  Use a custom inverted index, not FTS5.  ";
    in.trigger   = "Use when choosing a full-text index strategy";
    in.rationale = "FTS5 MATCH is AND-only without query rewriting.";
    in.tags      = "search,sqlite";
    auto r = db.store_knowledge(in);
    CHECK(r.id > 0);
    CHECK(!r.superseded.has_value());

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM knowledge WHERE knowledge_id = {}", r.id);
    CHECK(p1_scalar_text(raw, "SELECT text" + w) == "Use a custom inverted index, not FTS5.");
    CHECK(p1_scalar_text(raw, "SELECT trigger" + w) == "Use when choosing a full-text index strategy");
    CHECK(p1_scalar_text(raw, "SELECT rationale" + w) == "FTS5 MATCH is AND-only without query rewriting.");
    CHECK(p1_scalar_text(raw, "SELECT scope" + w) == "<NULL>");          // "" -> NULL
    CHECK(p1_scalar_text(raw, "SELECT status" + w) == "current");
    CHECK(p1_scalar_text(raw, "SELECT tags" + w) == "search,sqlite");
    CHECK(p1_scalar_int(raw, "SELECT embedding IS NOT NULL" + w) == 1);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count > 0" + w) == 1);
    sqlite3_close(raw);

    // Search surfaces the trigger in metadata.
    auto resp = db.search("full-text index strategy", 5, 0.0f);
    bool found = false;
    for (const auto& h : resp.results)
        if (h.metadata.value("source", "") == "knowledge" &&
            h.metadata.value("trigger", "") == in.trigger) found = true;
    CHECK(found);

    db.close();
    cleanup();
}

// The C++ embed/index string (trigger + "\n" + text) MUST equal
// kDecisionEmbedTextSQL, or a housekeeping rebuild silently changes the
// vector and the term counts. Prove it: store, rebuild both, compare.
void test_structured_decision_rebuild_is_stable(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text    = "Keep 'current' as the active decision status.";
    in.trigger = "Use when choosing decision status values";
    int id = db.store_knowledge(in).id;

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM knowledge WHERE knowledge_id = {}", id);
    const std::string blob_before = p1_scalar_text(raw, "SELECT hex(embedding)" + w);
    const long long   uni_before  = p1_scalar_int(raw, "SELECT unigram_count" + w);
    const long long   bi_before   = p1_scalar_int(raw, "SELECT bigram_count" + w);
    sqlite3_close(raw);

    db.rebuild_embeddings(emb, {}, "knowledge");
    db.reindex_table("knowledge", {});

    raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, "SELECT hex(embedding)" + w) == blob_before);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count" + w) == uni_before);
    CHECK(p1_scalar_int(raw, "SELECT bigram_count" + w) == bi_before);
    sqlite3_close(raw);
    db.close();
    cleanup();
}

void test_structured_decision_validation(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput ok;
    ok.kind = "decision";
    ok.text = "Valid ruling.";
    ok.trigger = "Use when testing validation";

    auto bad = ok; bad.text = "   ";
    CHECK(p1_throws_invalid(db, bad));
    bad = ok; bad.trigger = "";
    CHECK(p1_throws_invalid(db, bad));
    bad = ok; bad.status = "superseded";
    CHECK(p1_throws_invalid(db, bad));
    bad = ok; bad.status = "fact";
    CHECK(p1_throws_invalid(db, bad));
    bad = ok; bad.supersedes = 9999;
    CHECK(p1_throws_invalid(db, bad));
    bad = ok; bad.evidence = {9999};
    CHECK(p1_throws_invalid(db, bad));

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE kind = 'decision'") == 0);   // nothing written
    sqlite3_close(raw);

    // provisional and pending are valid starting statuses.
    auto p = ok; p.status = "provisional";
    CHECK(db.store_knowledge(p).id > 0);
    p.status = "pending";
    CHECK(db.store_knowledge(p).id > 0);
    db.close();
    cleanup();
}

void test_structured_decision_supersedes(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Embed decisions as text only.";
    a.trigger = "Use when choosing what to embed for a decision";
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Embed decisions as trigger then text.";
    b.trigger = a.trigger;
    b.supersedes = a_id;
    auto rb = db.store_knowledge(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT status FROM knowledge WHERE knowledge_id = {}", a_id)) == "superseded");
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT supersedes FROM knowledge WHERE knowledge_id = {}", rb.id)) == a_id);
    // Old row is kept, text untouched (honest history).
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT text FROM knowledge WHERE knowledge_id = {}", a_id)) == "Embed decisions as text only.");
    sqlite3_close(raw);

    // Search no longer surfaces the superseded row; it does surface the new one.
    auto resp = db.search("Embed decisions as text only", 10, 0.0f);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "knowledge") continue;
        if (h.id == a_id)  saw_old = true;
        if (h.id == rb.id) saw_new = true;
    }
    CHECK(!saw_old);
    CHECK(saw_new);

    // A superseded row can't be superseded again.
    ragger::KnowledgeInput c = b;
    c.supersedes = a_id;
    CHECK(p1_throws_invalid(db, c));
    db.close();
    cleanup();
}

void test_structured_decision_evidence(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ep1 = p1_make_episode(db, "evidence one");
    int ep2 = p1_make_episode(db, "evidence two");

    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text = "Link decisions to episodes.";
    in.trigger = "Use when recording why a decision was made";
    in.evidence = {ep1, ep2, ep1};   // duplicate is ignored
    int id = db.store_knowledge(in).id;

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM knowledge_episode_evidence WHERE knowledge_id = {}", id)) == 2);
    // RESTRICT: an episode that is evidence cannot be deleted.
    int rc = sqlite3_exec(raw, std::format(
        "DELETE FROM episodes WHERE episode_id = {}", ep1).c_str(),
        nullptr, nullptr, nullptr);
    CHECK(rc != SQLITE_OK);
    // CASCADE: deleting the decision removes its links.
    CHECK(sqlite3_exec(raw, std::format(
        "DELETE FROM knowledge WHERE knowledge_id = {}", id).c_str(),
        nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_episode_evidence") == 0);
    sqlite3_close(raw);
    db.close();
    cleanup();
}

void test_orphan_turn_summary_protected(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts = p1_make_turn_summary(db, "orphan");
    sqlite3* raw = p1_open_raw();
    // Cull the turn: ON DELETE SET NULL leaves the summary orphaned.
    CHECK(sqlite3_exec(raw, std::format(
        "DELETE FROM turns WHERE turn_id = (SELECT turn_id FROM turn_summaries "
        "WHERE turn_summary_id = {})", ts).c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT turn_id IS NULL FROM turn_summaries WHERE turn_summary_id = {}", ts)) == 1);

    char* err = nullptr;
    int rc = sqlite3_exec(raw, std::format(
        "DELETE FROM turn_summaries WHERE turn_summary_id = {}", ts).c_str(),
        nullptr, nullptr, &err);
    CHECK(rc != SQLITE_OK);
    CHECK(err && std::string(err).find("has no source turn") != std::string::npos);
    sqlite3_free(err);

    // An EMPTY orphan marker row stays deletable.
    CHECK(sqlite3_exec(raw,
        "INSERT INTO turn_summaries (text, turn_id, turn_datetime) VALUES ('', NULL, 0)",
        nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(sqlite3_exec(raw,
        "DELETE FROM turn_summaries WHERE text = '' AND turn_id IS NULL",
        nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);
    db.close();
    cleanup();
}

// A hand-built 0.16-shaped DB must migrate to 0.17: new columns in canonical
// order, stray status words moved into tags, evidence table + trigger present,
// existing decisions_terms rows intact. PHASE5 extends it to cover episodes
// (ids/spans moved out of summaries), the turn_summaries rebuild (episode_id
// backfill incl. cross-session same-second collision), watermarks, and the
// seq_save mechanism.
void test_migrate_0_16_to_0_17(ragger::Embedder& emb) {
    cleanup();
    {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        const char* ddl = R"(
            PRAGMA foreign_keys = OFF;
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO settings VALUES ('db_version', '0.16');
            INSERT INTO settings VALUES ('session_boundary_watermark_turn_id', '99');
            INSERT INTO settings VALUES ('project_boundary_watermark_turn_id', '98');
            INSERT INTO settings VALUES ('phon_weight', '0.1');
            INSERT INTO settings VALUES ('ini_migrated', '1');
            CREATE TABLE turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT NOT NULL,
                assistant_text TEXT,
                model_id INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                created_at INTEGER NOT NULL,
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            -- a real 0.16 DB carries this legacy index; the leg must drop it
            -- (the fresh-vs-migrated equivalence check below guards that).
            CREATE INDEX idx_turns_embedding_version ON turns(embedding_version);
            CREATE TABLE decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, status TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            INSERT INTO decisions (decision_id, text, status, tags, unigram_count) VALUES
                (1, 'first ruling',     'current',    '',  2),
                (3, 'a fact',           'fact',       '',  2),
                (5, 'a lesson',         'lesson',     'x', 2),
                (7, 'roadmap ruling',   'roadmap',    '',  2),
                (9, 'deprecated ruling','deprecated', '',  2);
            CREATE TABLE sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT NOT NULL UNIQUE, name TEXT, name_source TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()));
            INSERT INTO sessions (guid, name, name_source) VALUES
                ('guid-a', 'A', 'test'), ('guid-b', NULL, NULL);
            INSERT INTO turns (user_text, session_id, created_at) VALUES
                ('t1', 1, 1700000500), ('t2', 1, 1700000100), ('t3', 1, 1700000300);
            CREATE TABLE models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()));
            INSERT INTO models (model_id, name) VALUES (1, 'summarizer-x');
            CREATE TABLE summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, level TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                session_id INTEGER, model_id INTEGER,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            -- Two episodes with OVERLAPPING wall-clock spans, one per
            -- session: the backfill must never cross-assign. The five
            -- session/project rows cover every PHASE8 §4b classification
            -- case (rollup vs imported vs hand-written vs project).
            INSERT INTO summaries (text, level, tags, session_id, model_id,
                                   unigram_count, created_at, updated_at,
                                   embedding_version, embedding) VALUES
                ('episode one',  'episode', '',          1, NULL, 7, 1700000100, 1700000300, NULL, NULL),
                ('episode two',  'episode', '',          2, NULL, 5, 1700000150, 1700000250, NULL, NULL),
                ('rollup one',   'session', 'a,b',       1, 1,    3, 1700000200, 1700000200, 1, X'01'),
                ('imported one', 'session', 'imported',  NULL, 1,  2, 1700000210, 1700000210, NULL, NULL),
                ('hand one',     'session', '',          NULL, NULL, 1, 1700000220, 1700000220, NULL, NULL),
                ('proj in',      'project', 'imported',  NULL, NULL, 1, 1700000230, 1700000230, NULL, NULL),
                ('legacy proj',  'project', '',          1, NULL, 4, 1700000200, 1700000200, NULL, NULL);
            CREATE TABLE turn_summaries (
                turn_summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                turn_id INTEGER REFERENCES turns(turn_id) ON DELETE SET NULL,
                session_id INTEGER REFERENCES sessions(session_id),
                turn_model_id INTEGER, summary_model_id INTEGER,
                turn_datetime INTEGER NOT NULL,
                summarized_on INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            INSERT INTO turn_summaries (text, session_id, turn_datetime) VALUES
                ('in span A',      1, 1700000200),  -- -> episode one
                ('in span B',      2, 1700000200),  -- -> episode two (collision)
                ('after spans',    1, 1700000900),  -- NULL (outside)
                ('',               1, 1700000200),  -- NULL (marker)
                (NULL,             1, 1700000200),  -- NULL (null text)
                ('no session',  NULL, 1700000200);  -- NULL (no session)
            CREATE TABLE document_sources (
                document_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, path TEXT, year INTEGER,
                tags TEXT NOT NULL DEFAULT '',
                imported_at INTEGER NOT NULL);
            INSERT INTO document_sources (title, path, year, tags, imported_at)
                VALUES ('Doc', '~/doc.md', 2026, 'd', 1760000000),
                       ('Pathless', NULL, NULL, '', 1760000001),
                       ('Empty',    '',   NULL, '', 1760000002);
        )";
        char* fixture_err = nullptr;
        int fixture_rc = sqlite3_exec(raw, ddl, nullptr, nullptr, &fixture_err);
        if (fixture_rc != SQLITE_OK)
            std::println("    fixture DDL failed: {}", fixture_err ? fixture_err : "?");
        sqlite3_free(fixture_err);
        CHECK(fixture_rc == SQLITE_OK);
        // Seed the AUTOINCREMENT watermark ABOVE MAX(id): rows were deleted
        // in the fixture's "past life"; the rebuild must not restart below it.
        CHECK(sqlite3_exec(raw,
            "UPDATE sqlite_sequence SET seq = 42 WHERE name = 'turn_summaries'",
            nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
    }
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);   // runs the migration
        db.close();
    }
    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, "SELECT value FROM settings WHERE key='db_version'") == "0.17");
    // Canonical (fresh-install) column order: no project_id anywhere.
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('knowledge') WHERE cid = 5") == "trigger");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('knowledge') WHERE cid = 11") == "created_at");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('knowledge') WHERE cid = 1") == "kind");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('knowledge') WHERE name = 'project_id'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('knowledge') WHERE name = 'level'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'project_id'") == 0);
    // sessions gains the derived span columns (PHASE4).
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('sessions') WHERE cid = 4") == "started_on");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('sessions') WHERE cid = 5") == "ended_on");
    // Span derived from turns MIN/MAX; a session without turns is NULL/NULL.
    CHECK(p1_scalar_text(raw, "SELECT started_on || '-' || ended_on FROM sessions WHERE guid = 'guid-a'") == "1700000100-1700000500");
    CHECK(p1_scalar_text(raw, "SELECT ifnull(started_on,'NULL') || '-' || ifnull(ended_on,'NULL') FROM sessions WHERE guid = 'guid-b'") == "NULL-NULL");
    // The AUTOINCREMENT counter survives the rebuild.
    CHECK(p1_scalar_int(raw, "SELECT seq FROM sqlite_sequence WHERE name = 'sessions'") == 2);
    // PHASE7: the projects table AND every pre-unification decisions/insights
    // object are GONE; the unified knowledge tables + view exist instead.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'projects'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('decisions','decisions_terms','decision_evidence','insights','insight_evidence','insights_terms')") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name IN ('decisions_view','insights_view')") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='knowledge_view'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('knowledge_episode_evidence','knowledge_document_evidence')") == 2);
    // knowledge_document_evidence FKs: document_id SET NULL, knowledge_id CASCADE.
    CHECK(p1_scalar_text(raw, "SELECT on_delete FROM pragma_foreign_key_list('knowledge_document_evidence') WHERE \"table\" = 'documents'") == "SET NULL");
    CHECK(p1_scalar_text(raw, "SELECT on_delete FROM pragma_foreign_key_list('knowledge_document_evidence') WHERE \"table\" = 'knowledge'") == "CASCADE");
    // PHASE8: summaries is GONE — dropped with its junction and view.
    // Everything moved: episodes to `episodes` (PHASE5), session/project
    // rows to knowledge notes (§4b).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('summaries','summaries_terms','summaries_view')") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 2);
    CHECK(p1_scalar_text(raw, "SELECT episode_id || '|' || span_start || '-' || span_end || '|' || unigram_count "
                              "FROM episodes WHERE text = 'episode one'") == "1|1700000100-1700000300|2");  // count recomputed by the leg's final reindex
    CHECK(p1_scalar_text(raw, "SELECT episode_id || '|' || span_start || '-' || span_end || '|' || unigram_count "
                              "FROM episodes WHERE text = 'episode two'") == "2|1700000150-1700000250|2");
    // episode_id backfill: in-span rows get their OWN session's episode even
    // when the other session's span covers the same second; marker, NULL-text,
    // after-span, and no-session rows stay NULL.
    CHECK(p1_scalar_text(raw, "SELECT ifnull(episode_id,'NULL') FROM turn_summaries WHERE text = 'in span A'") == "1");
    CHECK(p1_scalar_text(raw, "SELECT ifnull(episode_id,'NULL') FROM turn_summaries WHERE text = 'in span B'") == "2");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE episode_id IS NOT NULL") == 2);
    // turn_summaries rebuilt in canonical column order with episode_id.
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('turn_summaries') WHERE cid = 4") == "episode_id");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('turn_summaries')") == 13);
    // session_id FK gained ON DELETE SET NULL.
    CHECK(p1_scalar_text(raw, "SELECT on_delete FROM pragma_foreign_key_list('turn_summaries') WHERE \"table\" = 'sessions'") == "SET NULL");
    // The seeded AUTOINCREMENT watermark (42 > MAX(id)=6) survived the
    // rebuild — exactly one row (INSERT into sqlite_sequence would append a
    // duplicate; the restore must UPDATE).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_sequence WHERE name = 'turn_summaries'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT seq FROM sqlite_sequence WHERE name = 'turn_summaries'") == 42);
    // Watermark settings rows deleted with the rollup machinery, and the two
    // dead keys (phon_weight, ini_migrated) removed by the same leg (S40);
    // surviving settings (db_version) untouched.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM settings WHERE key LIKE '%boundary_watermark_turn_id'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM settings WHERE key IN ('phon_weight', 'ini_migrated')") == 0);
    CHECK(p1_scalar_text(raw, "SELECT value FROM settings WHERE key = 'db_version'") == "0.17");
    // All-table reindex ran: junctions consistent, episodes indexed.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries_terms WHERE turn_summary_id NOT IN (SELECT turn_summary_id FROM turn_summaries)") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes_terms") > 0);
    // No stray rebuild tables / repointed REFERENCES (legacy_alter_table trap).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE sql LIKE '%_rebuild%'") == 0);
    // document_sources.path is unique: pathless/empty sources got their own
    // document_source_id as a placeholder; real paths are untouched.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_document_sources_path'") == 1);
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Doc'") == "~/doc.md");
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Pathless'") == "2");
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Empty'") == "3");
    // §4a: ids preserved across the gap, all kind='decision'. Unknown status
    // words (fact/lesson) moved into tags with status 'current'; roadmap and
    // deprecated are vocabulary and stay. New columns start NULL.
    CHECK(p1_scalar_text(raw, "SELECT group_concat(knowledge_id) FROM knowledge WHERE kind='decision'") == "1,3,5,7,9");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE kind = 'decision'") == 5);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE kind='decision' AND status NOT IN ('current','roadmap','deprecated')") == 0);
    CHECK(p1_scalar_text(raw, "SELECT tags FROM knowledge WHERE text = 'a fact'") == "fact");
    CHECK(p1_scalar_text(raw, "SELECT tags FROM knowledge WHERE text = 'a lesson'") == "x,lesson");
    CHECK(p1_scalar_text(raw, "SELECT status FROM knowledge WHERE text = 'roadmap ruling'") == "roadmap");
    CHECK(p1_scalar_text(raw, "SELECT status FROM knowledge WHERE text = 'deprecated ruling'") == "deprecated");
    CHECK(p1_scalar_int(raw, "SELECT unigram_count FROM knowledge WHERE text = 'first ruling'") == 2);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE kind='decision' AND trigger IS NULL AND rationale IS NULL AND scope IS NULL AND supersedes IS NULL AND session_id IS NULL AND model_id IS NULL") == 5);
    // sqlite_sequence: knowledge seq >= max migrated id (AUTOINCREMENT floor).
    CHECK(p1_scalar_int(raw, "SELECT seq >= 9 FROM sqlite_sequence WHERE name = 'knowledge'") == 1);
    // §4b (PHASE8): every non-episode summaries row became a knowledge
    // note. Ids continue above the highest decision id (9), ascending in
    // old summary_id order: rollup one=10, imported one=11, hand one=12,
    // proj in=13, legacy proj=14.
    CHECK(p1_scalar_text(raw, "SELECT group_concat(knowledge_id) FROM knowledge WHERE kind='note'")
          == "10,11,12,13,14");
    CHECK(p1_scalar_text(raw, "SELECT status || '|' || tags FROM knowledge WHERE text='rollup one'")
          == "retired|a,b,legacy_session");
    CHECK(p1_scalar_text(raw, "SELECT status || '|' || tags FROM knowledge WHERE text='imported one'")
          == "current|imported,legacy_note");
    CHECK(p1_scalar_text(raw, "SELECT status || '|' || tags FROM knowledge WHERE text='hand one'")
          == "current|legacy_note");
    CHECK(p1_scalar_text(raw, "SELECT status || '|' || tags FROM knowledge WHERE text='proj in'")
          == "current|imported,legacy_project");
    CHECK(p1_scalar_text(raw, "SELECT status || '|' || tags FROM knowledge WHERE text='legacy proj'")
          == "retired|legacy_project");
    // session_id / model_id / created_at carried across (unigram_count is
    // recomputed by the leg's final reindex, like episodes above).
    CHECK(p1_scalar_text(raw, "SELECT ifnull(session_id,'N') || '-' || ifnull(model_id,'N') || '-' "
                              "|| created_at || '-' || unigram_count FROM knowledge WHERE text='rollup one'")
          == "1-1-1700000200-2");
    CHECK(p1_scalar_text(raw, "SELECT embedding_version || '-' || hex(embedding) FROM knowledge WHERE text='rollup one'")
          == "1-01");
    // knowledge row count = decisions + non-episode summaries rows (5 + 5).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge") == 10);
    // The leg's final reindex indexed knowledge: every row is covered.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_terms") > 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE knowledge_id NOT IN "
                             "(SELECT knowledge_id FROM knowledge_terms)") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'protect_orphan_turn_summaries'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_foreign_key_check") == 0);
    sqlite3_close(raw);

    // PHASE9: PERMANENT equivalence check — the migrated DB and a fresh DB
    // built by this binary must agree on every schema object: per-table
    // column shape, FK actions, index/trigger SQL (whitespace-normalized,
    // sorted), and the view set. Zero differences allowed; the first
    // differing object prints on failure. (This is what Hal verified by
    // hand against the standalone SQL file; now it can never silently rot.)
    {
        const std::string fresh_db = TEMP_DB + "_fresh";
        fs::remove(fresh_db);
        fs::remove(fresh_db + "-wal");
        fs::remove(fresh_db + "-shm");
        { ragger::sqlite::Backend fresh(emb, fresh_db); }   // fresh 0.17 install

        auto norm_ws = [](std::string s) {
            std::string out;
            bool space = false;
            for (char c : s) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { space = true; continue; }
                if (space && !out.empty()) out += ' ';
                space = false;
                out += c;
            }
            return out;
        };
        sqlite3* mig = nullptr;
        sqlite3* fre = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &mig) == SQLITE_OK);
        CHECK(sqlite3_open(fresh_db.c_str(), &fre) == SQLITE_OK);
        auto rows = [](sqlite3* d, const std::string& sql) {
            std::vector<std::string> out;
            sqlite3_stmt* st = nullptr;
            if (sqlite3_prepare_v2(d, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) {
                std::println("    equivalence prepare failed: {} ({})", sql, sqlite3_errmsg(d));
                CHECK(false);
            } else {
                while (sqlite3_step(st) == SQLITE_ROW) {
                    std::string r;
                    for (int i = 0; i < sqlite3_column_count(st); ++i) {
                        const unsigned char* t = sqlite3_column_text(st, i);
                        r += (t ? reinterpret_cast<const char*>(t) : "<null>");
                        r += '|';
                    }
                    out.push_back(r);
                }
                sqlite3_finalize(st);
            }
            return out;
        };
        auto compare = [&](const std::string& label, std::vector<std::string> a,
                           std::vector<std::string> b) {
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a != b) {
                std::println("    EQUIVALENCE DIFF in {}:", label);
                for (size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
                    const std::string& x = i < a.size() ? a[i] : "<missing in migrated>";
                    const std::string& y = i < b.size() ? b[i] : "<missing in fresh>";
                    if (x != y) {
                        std::println("      migrated: {}", x);
                        std::println("      fresh:    {}", y);
                        break;
                    }
                }
            }
            CHECK(a == b);
        };

        const std::string tables_sql =
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name";
        auto t_mig = rows(mig, tables_sql);
        compare("table set", t_mig, rows(fre, tables_sql));
        CHECK(t_mig.size() >= 15);

        for (const auto& trow : t_mig) {
            std::string t = trow;
            t.pop_back();  // trailing '|' from rows()
            compare("columns of " + t,
                    rows(mig, "SELECT * FROM pragma_table_info('" + t + "')"),
                    rows(fre, "SELECT * FROM pragma_table_info('" + t + "')"));
            compare("foreign keys of " + t,
                    rows(mig, "SELECT \"from\",\"table\",\"to\",on_delete FROM pragma_foreign_key_list('" + t + "')"),
                    rows(fre, "SELECT \"from\",\"table\",\"to\",on_delete FROM pragma_foreign_key_list('" + t + "')"));
        }

        {
            auto idx  = rows(mig, "SELECT type||'|'||name||'|'||COALESCE(sql,'<none>') "
                                  "FROM sqlite_master WHERE type IN ('index','trigger')");
            auto idxf = rows(fre, "SELECT type||'|'||name||'|'||COALESCE(sql,'<none>') "
                                  "FROM sqlite_master WHERE type IN ('index','trigger')");
            for (auto& s : idx)  s = norm_ws(s);
            for (auto& s : idxf) s = norm_ws(s);
            compare("indexes/triggers", idx, idxf);
        }
        compare("views",
                rows(mig, "SELECT name FROM sqlite_master WHERE type = 'view'"),
                rows(fre, "SELECT name FROM sqlite_master WHERE type = 'view'"));

        sqlite3_close(mig);
        sqlite3_close(fre);
        fs::remove(fresh_db);
        fs::remove(fresh_db + "-wal");
        fs::remove(fresh_db + "-shm");
    }

    // The pre-migration backup archive lands next to the DB; remove it.
    for (const auto& e : fs::directory_iterator(fs::path(TEMP_DB).parent_path()))
        if (e.path().filename().string().starts_with("ragger_test_backend_BACKUP_"))
            fs::remove(e.path());
    cleanup();
}

// PHASE9: the leg's pre-checks run BEFORE any change and refuse duplicates
// with a pointed message; the transaction rollback must leave the DB
// exactly as it was (db_version still 0.16, placeholder paths not applied).
void test_migrate_0_16_to_0_17_rejects(ragger::Embedder& emb) {
    auto make_fixture = [&](bool dup_chunks, bool dup_episodes) {
        cleanup();
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        std::string ddl = R"(
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO settings VALUES ('db_version', '0.16');
            CREATE TABLE turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT NOT NULL,
                assistant_text TEXT, model_id INTEGER, session_id INTEGER,
                created_at INTEGER NOT NULL,
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            CREATE TABLE document_sources (
                document_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, path TEXT, year INTEGER,
                tags TEXT NOT NULL DEFAULT '',
                imported_at INTEGER NOT NULL);
            INSERT INTO document_sources (title, path, year, tags, imported_at)
                VALUES ('Doc', '/dup.md', 2026, '', 1760000000),
                       ('Pathless', NULL, NULL, '', 1760000001);
            CREATE TABLE documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER,
                document_source_id INTEGER REFERENCES document_sources(document_source_id),
                modified_on INTEGER,
                embedding_version INTEGER, embedding BLOB);
            CREATE TABLE summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, level TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                session_id INTEGER, model_id INTEGER,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
        )";
        if (dup_chunks)
            ddl += "INSERT INTO documents (text, chunk_index, document_source_id, modified_on) "
                   "VALUES ('dup a', 1, 1, 100), ('dup b', 1, 1, 101);";
        else
            ddl += "INSERT INTO documents (text, chunk_index, document_source_id, modified_on) "
                   "VALUES ('ok', 1, 1, 100);";
        if (dup_episodes)
            ddl += "INSERT INTO summaries (text, level, session_id, created_at, updated_at) "
                   "VALUES ('e1','episode',1,500,600), ('e2','episode',1,500,700);";
        else
            ddl += "INSERT INTO summaries (text, level, session_id, created_at, updated_at) "
                   "VALUES ('e1','episode',1,500,600);";
        CHECK(sqlite3_exec(raw, ddl.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
    };

    auto expect_refused = [&](const char* needle) {
        bool threw = false;
        try {
            ragger::sqlite::Backend db(emb, TEMP_DB);
            db.close();
        } catch (const std::runtime_error& e) {
            threw = true;
            const std::string m = e.what();
            CHECK(m.find(needle) != std::string::npos);
            CHECK(m.find("nothing was changed") != std::string::npos);
        }
        CHECK(threw);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT value FROM settings WHERE key = 'db_version'") == "0.16");
        // The placeholder-path UPDATE that runs ahead of the checks must be
        // rolled back with everything else.
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM document_sources WHERE path IS NULL") == 1);
        sqlite3_close(raw);
        for (const auto& e : fs::directory_iterator(fs::path(TEMP_DB).parent_path()))
            if (e.path().filename().string().starts_with("ragger_test_backend_BACKUP_"))
                fs::remove(e.path());
        cleanup();
    };

    make_fixture(/*dup_chunks=*/true, /*dup_episodes=*/false);
    expect_refused("2 chunks share (source 1, chunk_index 1)");

    make_fixture(/*dup_chunks=*/false, /*dup_episodes=*/true);
    expect_refused("2 episodes share (session 1, created_at 500)");
}

// PHASE9: final schema assembly — the kLegs predicate drives backup/VACUUM
// gating; every table has a <table>_view except an explicit allow-list (a
// future table without a view fails here); users_view never exposes hashes.
void test_p9_schema_assembly(ragger::Embedder& emb) {
    using ragger::sqlite::Backend;

    // kLegs origins are pending; the target, fresh, and unknown are not.
    // PHASE9A: 0.16 is the only supported origin; pre-0.16 versions are not
    // pending (create_schema() refuses them before touching the file).
    CHECK(!Backend::migration_pending("0.12"));
    CHECK(!Backend::migration_pending("0.15"));
    CHECK(Backend::migration_pending("0.16"));
    CHECK(!Backend::migration_pending("0.17"));
    CHECK(!Backend::migration_pending(""));
    CHECK(!Backend::migration_pending("0.11"));
    CHECK(!Backend::migration_pending("0.99"));

    cleanup();
    Backend db(emb, TEMP_DB);

    sqlite3* raw = p1_open_raw();
    // Views exist only to render date integers as date strings, so a table
    // with no date column (evidence links, the terms index, the <t>_terms
    // junctions, settings) has none.
    std::set<std::string> allowed_no_view = {
        "sqlite_sequence", "semqlite_config", "settings", "terms",
        "knowledge_episode_evidence", "knowledge_document_evidence"};
    for (const auto& tt : ragger::sqlite::kTextTables)
        allowed_no_view.insert(tt.junction);
    std::vector<std::string> tables;
    sqlite3_stmt* st = nullptr;
    CHECK(sqlite3_prepare_v2(raw,
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
        -1, &st, nullptr) == SQLITE_OK);
    while (sqlite3_step(st) == SQLITE_ROW)
        tables.push_back(reinterpret_cast<const char*>(sqlite3_column_text(st, 0)));
    sqlite3_finalize(st);
    CHECK(tables.size() >= 15);  // sanity: the full 0.17 table set
    for (const auto& t : tables) {
        if (allowed_no_view.count(t)) continue;
        char* q = sqlite3_mprintf(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'view' AND name = '%q_view'",
            t.c_str());
        bool has = false;
        if (sqlite3_prepare_v2(raw, q, -1, &st, nullptr) == SQLITE_OK) {
            if (sqlite3_step(st) == SQLITE_ROW) has = sqlite3_column_int(st, 0) == 1;
            sqlite3_finalize(st);
        }
        sqlite3_free(q);
        if (!has) std::println("    MISSING view for table {}", t);
        CHECK(has);
    }

    // Every remaining view converts a date integer; no evidence or term-junction
    // view exists.
    {
        sqlite3_stmt* vs = nullptr;
        CHECK(sqlite3_prepare_v2(raw,
            "SELECT name, sql FROM sqlite_master WHERE type = 'view'",
            -1, &vs, nullptr) == SQLITE_OK);
        int nviews = 0;
        while (sqlite3_step(vs) == SQLITE_ROW) {
            ++nviews;
            const std::string vname = reinterpret_cast<const char*>(sqlite3_column_text(vs, 0));
            const std::string vsql  = reinterpret_cast<const char*>(sqlite3_column_text(vs, 1));
            const bool retired = vname.ends_with("_terms_view") || vname.ends_with("_evidence_view");
            if (retired || vsql.find("datetime(") == std::string::npos)
                std::println("    unexpected view {}", vname);
            CHECK(!retired);
            CHECK(vsql.find("datetime(") != std::string::npos);
        }
        sqlite3_finalize(vs);
        CHECK(nviews == 9);
    }

    // users_view must not expose hashes (inspection views are readable by
    // anyone who can open the DB).
    CHECK(sqlite3_prepare_v2(raw, "PRAGMA table_info(users_view)", -1, &st, nullptr) == SQLITE_OK);
    int users_cols = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
        std::string col = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        if (col.find("hash") != std::string::npos)
            std::println("    users_view LEAKS column {}", col);
        CHECK(col.find("hash") == std::string::npos);
        ++users_cols;
    }
    sqlite3_finalize(st);
    CHECK(users_cols == 4);  // id, username, created_at, updated_at

    sqlite3_close(raw);
    db.close();
    cleanup();
}

// Documents are identified by path (0.17): one source per path regardless of
// title; no path -> invalid_argument; find_document_source reports it.
void test_document_path_unique(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    CHECK(db.find_document_source("/tmp/u.md").is_null());

    ragger::DocumentChunk a;
    a.text = "First chunk."; a.title = "Original Title"; a.path = "/tmp/u.md"; a.chunk_index = 1;
    db.store_document(a);
    ragger::DocumentChunk b = a;
    b.text = "Second chunk."; b.title = "A Different Title"; b.chunk_index = 2;
    db.store_document(b);                       // same path -> same source

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM document_sources") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents") == 2);
    CHECK(p1_scalar_text(raw, "SELECT title FROM document_sources") == "Original Title");
    // The schema itself refuses a second source with the same path.
    CHECK(sqlite3_exec(raw, "INSERT INTO document_sources (title, path, imported_at) "
                            "VALUES ('x', '/tmp/u.md', 1)", nullptr, nullptr, nullptr)
          == SQLITE_CONSTRAINT);
    sqlite3_close(raw);

    auto found = db.find_document_source("/tmp/u.md");
    CHECK(!found.is_null());
    CHECK(found["title"] == "Original Title");

    ragger::DocumentChunk c;
    c.text = "No path."; c.title = "Pathless";
    bool threw = false;
    try { db.store_document(c); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // PHASE9: (source, chunk_index) is the chunk key — a second chunk at
    // position 1 of the same document is a caller error with a pointed
    // message.
    ragger::DocumentChunk dup = a;
    dup.text = "Different text, same slot.";
    threw = false;
    try { db.store_document(dup); }
    catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("document chunk already exists: /tmp/u.md chunk 1")
              != std::string::npos);
    }
    CHECK(threw);

    db.close();
    cleanup();
}

// list_document_sources / delete_document_source: ids are the primary key
// (gaps after a delete are kept), delete removes chunks + term rows + the
// source in one go, and the path can then be imported again.
void test_document_list_and_delete(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    auto add = [&](const char* path, const char* title, const char* text, int chunk) {
        ragger::DocumentChunk d;
        d.text = text; d.title = title; d.path = path; d.chunk_index = chunk; d.year = 2026;
        return db.store_document(d);
    };
    add("/tmp/a.md", "Doc A", "Alpine zebra crossings are unusual.", 1);
    add("/tmp/a.md", "Doc A", "Zebra stripes confuse flies.", 2);
    add("/tmp/b.md", "Doc B", "Barometric pressure and weather.", 1);
    add("/tmp/c.md", "Doc C", "Cobblestone streets of old towns.", 1);

    auto docs = db.list_document_sources();
    CHECK(docs.size() == 3);
    CHECK(docs[0]["chunks"] == 2);
    CHECK(docs[0]["path"] == "/tmp/a.md");
    const int a_id = docs[0]["id"].get<int>();
    const int b_id = docs[1]["id"].get<int>();
    const int c_id = docs[2]["id"].get<int>();

    auto info = db.delete_document_source(b_id);       // middle one -> a gap
    CHECK(info["chunks"] == 1);
    CHECK(info["title"] == "Doc B");

    docs = db.list_document_sources();
    CHECK(docs.size() == 2);
    CHECK(docs[0]["id"].get<int>() == a_id);
    CHECK(docs[1]["id"].get<int>() == c_id);           // key not renumbered

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM documents WHERE document_source_id = {}", b_id)) == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents") == 3);
    // No orphaned term rows for the deleted chunk (CASCADE).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms "
                             "WHERE document_id NOT IN (SELECT document_id FROM documents)") == 0);
    sqlite3_close(raw);

    CHECK(db.find_document_source("/tmp/b.md").is_null());
    add("/tmp/b.md", "Doc B again", "Barometric pressure revisited.", 1);   // re-import is now allowed

    bool threw = false;
    try { db.delete_document_source(9999); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    db.close();
    cleanup();
}

// invalidate_all_caches(): a backend with warm caches does not see rows
// written by ANOTHER connection (CLI/hand edit) until its caches are dropped.
void test_invalidate_all_caches(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend daemon(emb, TEMP_DB);
    {
        ragger::DocumentChunk d;
        d.text = "Original zebra chunk."; d.title = "T"; d.path = "/tmp/z.md"; d.chunk_index = 1;
        daemon.store_document(d);
    }
    auto has = [&](const char* q, const char* needle) {
        // LR S76: documents are probed through their own verb now.
        ragger::DocumentSearchQuery dq;
        dq.query = q; dq.limit = 10;
        for (auto& h : daemon.agent_document_search(dq))
            if (h.snippet.find(needle) != std::string::npos) return true;
        return false;
    };
    CHECK(has("zebra", "zebra"));                      // warms the document cache
    {
        ragger::sqlite::Backend cli(emb, TEMP_DB);     // second connection = the CLI
        ragger::DocumentChunk d;
        d.text = "Newly imported quokka chunk."; d.title = "Q"; d.path = "/tmp/q.md"; d.chunk_index = 1;
        cli.store_document(d);
        cli.close();
    }
    CHECK(!has("quokka", "quokka"));                   // stale: cache predates the write
    daemon.invalidate_all_caches();
    CHECK(has("quokka", "quokka"));                    // rebuilt from the DB
    daemon.close();
    cleanup();
}

// ---- Phase 5 steps 2+3: insights write path + search integration --------
static bool p5_throws_invalid_insight(ragger::sqlite::Backend& db,
                                      const ragger::KnowledgeInput& in) {
    try { db.store_knowledge(in); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

// Happy path: trigger optional, evidence required and linked, row embedded
// and indexed, search surfaces it with source="insight".
void test_p5_insight_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ep1 = p1_make_episode(db, "insight evidence one");
    int ep2 = p1_make_episode(db, "insight evidence two");

    ragger::KnowledgeInput in;
    in.kind = "insight";
    in.text      = "The embed-table lists must stay in sync by hand.";
    in.rationale = "Nothing enforces kAllTables vs count_embeddable_rows.";
    in.tags      = "maintenance";
    in.evidence  = {ep1, ep2, ep1};   // duplicate is ignored
    int id = db.store_knowledge(in).id;
    CHECK(id > 0);

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM knowledge WHERE knowledge_id = {}", id);
    CHECK(p1_scalar_text(raw, "SELECT text" + w) == "The embed-table lists must stay in sync by hand.");
    CHECK(p1_scalar_text(raw, "SELECT trigger" + w) == "<NULL>");   // optional -> NULL
    CHECK(p1_scalar_text(raw, "SELECT status" + w) == "current");
    CHECK(p1_scalar_text(raw, "SELECT tags" + w) == "maintenance");
    CHECK(p1_scalar_int(raw, "SELECT embedding IS NOT NULL" + w) == 1);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count > 0" + w) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM knowledge_episode_evidence WHERE knowledge_id = {}", id)) == 2);
    sqlite3_close(raw);

    // Search finds it tagged as an insight.
    auto resp = db.search("embed table lists sync by hand", 10, 0.0f);
    bool found = false;
    for (const auto& h : resp.results)
        if (h.metadata.value("source", "") == "knowledge" && h.id == id) found = true;
    CHECK(found);

    // RESTRICT: evidence episodes can't vanish from under the insight.
    raw = p1_open_raw();
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "DELETE FROM episodes WHERE episode_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, ep1);
    CHECK(sqlite3_step(st) == SQLITE_CONSTRAINT);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// Validation: evidence REQUIRED (empty or unknown ids throw), text required,
// status allow-list, supersedes flips the old row and hides it from search.
void test_p5_insight_validation(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts = p1_make_episode(db, "validation evidence");

    ragger::KnowledgeInput ok;
    ok.kind = "insight";
    ok.text = "Insights always cite their sources.";
    ok.evidence = {ts};

    auto bad = ok; bad.text = "   ";
    CHECK(p5_throws_invalid_insight(db, bad));
    bad = ok; bad.evidence = {};               // a note, not an insight
    CHECK(p5_throws_invalid_insight(db, bad));
    bad = ok; bad.evidence = {9999};
    CHECK(p5_throws_invalid_insight(db, bad));
    bad = ok; bad.status = "superseded";
    CHECK(p5_throws_invalid_insight(db, bad));
    bad = ok; bad.supersedes = 9999;
    CHECK(p5_throws_invalid_insight(db, bad));

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE kind = 'insight'") == 0);   // nothing written
    sqlite3_close(raw);

    // supersedes: old row flips to 'superseded', keeps its text (honest
    // history), and disappears from search.
    int a_id = db.store_knowledge(ok).id;
    ragger::KnowledgeInput b = ok;
    b.text = "Insights cite turn summaries, never raw turns.";
    b.supersedes = a_id;
    auto rb = db.store_knowledge(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT status FROM knowledge WHERE knowledge_id = {}", a_id)) == "superseded");
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT supersedes FROM knowledge WHERE knowledge_id = {}", rb.id)) == a_id);
    sqlite3_close(raw);

    auto resp = db.search("Insights always cite their sources", 10, 0.0f);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "knowledge") continue;
        if (h.id == a_id)  saw_old = true;
        if (h.id == rb.id) saw_new = true;
    }
    CHECK(!saw_old);
    CHECK(saw_new);

    // A superseded insight can't be superseded again.
    b.supersedes = a_id;
    CHECK(p5_throws_invalid_insight(db, b));

    db.close();
    cleanup();
}

// Phase 5 Step 2: the C++ embed/index string for insights (trigger+"\\n"+text,
// or bare text when trigger is NULL) MUST equal kDecisionEmbedTextSQL, or a
// housekeeping rebuild silently changes vectors/counts. Mirrors
// test_structured_decision_rebuild_is_stable; covers both trigger branches.
void test_p5_insight_rebuild_is_stable(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts = p1_make_episode(db, "rebuild evidence");

    ragger::KnowledgeInput a;
    a.kind = "insight";
    a.text = "Supersede rather than rewrite when the old wording mattered.";
    a.trigger = "Use when retiring an insight";
    a.evidence = {ts};
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "insight";
    b.text = "Migration legs are defensive CREATEs; no rebuild needed.";
    b.evidence = {ts};                       // no trigger
    int b_id = db.store_knowledge(b).id;

    sqlite3* raw = p1_open_raw();
    const std::string wa = std::format(" FROM knowledge WHERE knowledge_id = {}", a_id);
    const std::string wb = std::format(" FROM knowledge WHERE knowledge_id = {}", b_id);
    const std::string blob_a = p1_scalar_text(raw, "SELECT hex(embedding)" + wa);
    const std::string blob_b = p1_scalar_text(raw, "SELECT hex(embedding)" + wb);
    const long long uni_a = p1_scalar_int(raw, "SELECT unigram_count" + wa);
    const long long bi_a  = p1_scalar_int(raw, "SELECT bigram_count" + wa);
    const long long uni_b = p1_scalar_int(raw, "SELECT unigram_count" + wb);
    const long long bi_b  = p1_scalar_int(raw, "SELECT bigram_count" + wb);
    sqlite3_close(raw);

    db.rebuild_embeddings(emb, {}, "knowledge");
    db.reindex_table("knowledge", {});

    raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, "SELECT hex(embedding)" + wa) == blob_a);
    CHECK(p1_scalar_text(raw, "SELECT hex(embedding)" + wb) == blob_b);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count" + wa) == uni_a);
    CHECK(p1_scalar_int(raw, "SELECT bigram_count" + wa) == bi_a);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count" + wb) == uni_b);
    CHECK(p1_scalar_int(raw, "SELECT bigram_count" + wb) == bi_b);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// Phase 5 Step 4: the insight layer in agent_search (handles, status,
// trigger), agent_get expand=sources/chain, agent_update transition table,
// agent_review queue.
void test_p5_insight_layer(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int ts = p1_make_episode(db, "layer evidence");

    ragger::KnowledgeInput a;
    a.kind = "insight";
    a.text     = "P5gizmo flange insight: casting follows machining.";
    a.trigger  = "Use when inspecting gizmo flanges";
    a.evidence = {ts};
    int a_id = db.store_knowledge(a).id;

    // ---- agent_search with layers=["insight"]
    ragger::AgentSearchQuery q;
    q.query = "gizmo flange casting";
    q.layers = {"insight"};
    bool found = false;
    for (const auto& h : db.agent_search(q)) {
        CHECK(h.layer == "knowledge");
        if (h.id == "knowledge:" + std::to_string(a_id)) {
            found = true;
            CHECK(h.extra.value("status", "") == "current");
            CHECK(h.extra.value("trigger", "") == a.trigger);
        }
    }
    CHECK(found);

    // ---- agent_get expand=sources -> the evidence episode (LR S80)
    ragger::AgentGetQuery g;
    g.ids = {"knowledge:" + std::to_string(a_id)};
    g.expand = "sources";
    auto got = db.agent_get(g);
    CHECK(!got["items"][0]["expanded"].empty());
    CHECK(got["items"][0]["expanded"][0]["layer"] == "episode");

    // ---- agent_get expand=chain -> supersede lineage oldest -> newest
    ragger::KnowledgeInput b = a;
    b.text = "P5gizmo flange insight two: machining follows casting.";
    b.supersedes = a_id;
    int b_id = db.store_knowledge(b).id;
    g.ids = {"knowledge:" + std::to_string(b_id)};
    g.expand = "chain";
    got = db.agent_get(g);
    const auto& exp = got["items"][0]["expanded"];
    CHECK(exp.size() == 2);
    CHECK(exp[0]["id"] == "knowledge:" + std::to_string(a_id));
    CHECK(exp[1]["id"] == "knowledge:" + std::to_string(b_id));

    // ---- agent_update: tags free; illegal transition rejected
    ragger::UpdateRequest u;
    u.id = "knowledge:" + std::to_string(b_id);
    u.tags = "flange";
    auto row = db.agent_update(u);
    CHECK(row["tags"] == "flange");
    u.tags.reset();
    u.status = "pending";   // current -> pending is not a listed transition
    bool threw = false;
    try { db.agent_update(u); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // ---- review: pending insights listed, approvable through the handle
    ragger::KnowledgeInput p;
    p.kind = "insight";
    p.text = "P5 pending insight: torque to spec.";
    p.evidence = {ts};
    p.status = "pending";
    int p_id = db.store_knowledge(p).id;
    auto listed = db.agent_review("list", {});
    bool saw_ins = false;
    for (const auto& item : listed["pending"])
        if (item["id"] == "knowledge:" + std::to_string(p_id)) saw_ins = true;
    CHECK(saw_ins);
    auto done = db.agent_review("approve", {"knowledge:" + std::to_string(p_id)});
    CHECK(done["done"].size() == 1);
    CHECK(done["failed"].empty());

    // ---- reject through an insight handle (review 7): pending -> retired,
    // and the queue no longer lists it.
    ragger::KnowledgeInput rj;
    rj.kind = "insight";
    rj.text = "P5 pending insight to reject.";
    rj.evidence = {ts};
    rj.status = "pending";
    int rj_id = db.store_knowledge(rj).id;
    auto rej = db.agent_review("reject", {"knowledge:" + std::to_string(rj_id)});
    CHECK(rej["done"].size() == 1);
    CHECK(rej["failed"].empty());
    {
        ragger::AgentGetQuery g;
        g.ids = {"knowledge:" + std::to_string(rj_id)};
        auto got = db.agent_get(g);
        CHECK(got["items"][0]["status"] == "retired");
    }
    bool still_listed = false;
    for (const auto& item : db.agent_review("list", {})["pending"])
        if (item["id"] == "knowledge:" + std::to_string(rj_id)) still_listed = true;
    CHECK(!still_listed);

    (void)a_id; (void)b_id;
    db.close();
    cleanup();
}

// ---- PHASE7: unified knowledge table ----------------------------------
// kind rules: decision requires a trigger; insight requires >=1 evidence;
// note requires neither. kind is reported by agent_get.
void test_p7_knowledge_kinds(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    auto throws_invalid = [&db](ragger::KnowledgeInput in) {
        try { db.store_knowledge(in); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };

    // note: no trigger, no evidence -> fine.
    ragger::KnowledgeInput n;
    n.kind = "note";
    n.text = "P7 note: the rehearsal room key lives under the mat.";
    int n_id = db.store_knowledge(n).id;
    CHECK(n_id > 0);

    // decision without trigger -> rejected; with trigger -> stored.
    ragger::KnowledgeInput d;
    d.kind = "decision";
    d.text = "P7 decision: always rehearse bars 40-52 slowly.";
    CHECK(throws_invalid(d));
    d.trigger = "Use when scheduling rehearsal time";
    int d_id = db.store_knowledge(d).id;
    CHECK(d_id > 0);

    // insight without evidence -> rejected; with evidence -> stored.
    int ts = p1_make_episode(db, "P7 insight evidence");
    ragger::KnowledgeInput i;
    i.kind = "insight";
    i.text = "P7 insight: slow practice transfers to tempo.";
    CHECK(throws_invalid(i));
    i.evidence = {ts};
    int i_id = db.store_knowledge(i).id;
    CHECK(i_id > 0);

    // Unknown kind -> rejected (vocabulary may grow, but only via code).
    ragger::KnowledgeInput u;
    u.kind = "wisdom";
    u.text = "P7 nonsense kind row.";
    CHECK(throws_invalid(u));

    // agent_get reports kind.
    ragger::AgentGetQuery g;
    g.ids = {"knowledge:" + std::to_string(n_id),
             "knowledge:" + std::to_string(d_id),
             "knowledge:" + std::to_string(i_id)};
    auto got = db.agent_get(g);
    CHECK(got["items"][0]["kind"] == "note");
    CHECK(got["items"][1]["kind"] == "decision");
    CHECK(got["items"][2]["kind"] == "insight");

    db.close();
    cleanup();
}

// supersedes crosses kinds: a decision may retire the insight it promotes.
void test_p7_cross_kind_supersedes(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts = p1_make_episode(db, "P7 promotion evidence");

    ragger::KnowledgeInput i;
    i.kind = "insight";
    i.text = "P7 insight before promotion: damp the reed first.";
    i.evidence = {ts};
    int i_id = db.store_knowledge(i).id;

    ragger::KnowledgeInput d;
    d.kind = "decision";
    d.text = "P7 promoted decision: damp the reed before every reed scrape.";
    d.trigger = "Use when scraping a reed";
    d.supersedes = i_id;
    auto r = db.store_knowledge(d);
    CHECK(r.superseded.has_value() && *r.superseded == i_id);

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT status FROM knowledge WHERE knowledge_id = {}", i_id)) == "superseded");
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT kind FROM knowledge WHERE knowledge_id = {}", r.id)) == "decision");
    sqlite3_close(raw);

    // Chain expansion crosses kinds, oldest first.
    ragger::AgentGetQuery g;
    g.ids = {"knowledge:" + std::to_string(r.id)};
    g.expand = "chain";
    const auto exp = db.agent_get(g)["items"][0]["expanded"];
    CHECK(exp.size() == 2);
    CHECK(exp[0]["id"] == "knowledge:" + std::to_string(i_id));
    CHECK(exp[0]["kind"] == "insight");
    CHECK(exp[1]["kind"] == "decision");

    db.close();
    cleanup();
}

// agent_search `kinds` narrows knowledge hits; the legacy layer aliases
// 'decision'/'insight' mean the same thing (covered in insight_layer test).
void test_p7_kinds_filter(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput d;
    d.kind = "decision";
    d.text = "P7 filter decision: oboe reeds bend northward.";
    d.trigger = "Use when filtering P7 knowledge by kind";
    int d_id = db.store_knowledge(d).id;

    ragger::KnowledgeInput n;
    n.kind = "note";
    n.text = "P7 filter note: oboe reeds bend northward, allegedly.";
    int n_id = db.store_knowledge(n).id;

    ragger::AgentSearchQuery q;
    q.query = "oboe reeds bend northward";
    q.kinds = {"decision"};
    bool saw_d = false, saw_n = false;
    for (const auto& h : db.agent_search(q)) {
        CHECK(h.layer == "knowledge");
        if (h.id == "knowledge:" + std::to_string(d_id)) saw_d = true;
        if (h.id == "knowledge:" + std::to_string(n_id)) saw_n = true;
    }
    CHECK(saw_d);
    CHECK(!saw_n);

    db.close();
    cleanup();
}

// Legacy typed handles are rejected with a pointer, not silently mapped.
void test_p7_legacy_handles_rejected(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput n;
    n.kind = "note";
    n.text = "P7 legacy handle target.";
    int n_id = db.store_knowledge(n).id;

    ragger::AgentGetQuery g;
    g.ids = {"decision:" + std::to_string(n_id)};
    bool threw = false;
    try { db.agent_get(g); }
    catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()) == "use knowledge:<id>");
    }
    CHECK(threw);

    ragger::UpdateRequest u;
    u.id = "insight:" + std::to_string(n_id);
    u.tags = "x";
    threw = false;
    try { db.agent_update(u); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Looser junk stays a per-item 'not found', not a throw.
    ragger::AgentGetQuery m;
    m.ids = {"decision:", "insight:-2"};
    const auto got = db.agent_get(m);
    CHECK(got["items"].size() == 2);
    CHECK(got["items"][0].contains("error"));
    CHECK(got["items"][1].contains("error"));

    db.close();
    cleanup();
}

// PHASE8 D.6: notes are found via layers ["knowledge"] + kinds ["note"];
// layers ["summary"] is rejected with a message pointing at knowledge;
// agent_get on a `summary:<id>` handle is rejected.
void test_p8_summary_layer_gone(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const int note_id = std::stoi(db.store("The bakery sells sourdough loaves daily."));

    ragger::AgentSearchQuery q;
    q.query = "sourdough loaves";
    q.layers = {"knowledge"};
    q.kinds = {"note"};
    bool found = false;
    for (const auto& h : db.agent_search(q))
        if (h.id == "knowledge:" + std::to_string(note_id)) found = true;
    CHECK(found);

    q.layers = {"summary"};
    bool threw = false;
    try { db.agent_search(q); }
    catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("knowledge") != std::string::npos);
    }
    CHECK(threw);

    ragger::AgentGetQuery g;
    g.ids = {"summary:" + std::to_string(note_id)};
    threw = false;
    try { db.agent_get(g); }
    catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("knowledge") != std::string::npos);
    }
    CHECK(threw);

    db.close();
    cleanup();
}

// Phase 2 Step 1: degraded (text-only) search must apply the same decision
// status allow-list as ensure_dec_cache() — a superseded row stays hidden.
void test_p2_text_only_hides_superseded(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Prefer WAL checkpointing before archiving the database.";
    a.trigger = "Use when archiving a live SQLite database";
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Checkpoint in TRUNCATE mode without closing the connection.";
    b.trigger = "Use when archiving a live SQLite database";
    b.supersedes = a_id;
    auto rb = db.store_knowledge(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    // Text-only path with A's exact words: A must not appear, B must.
    auto resp = db.search_text_only("Prefer WAL checkpointing before archiving the database.", 10);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "knowledge") continue;
        if (h.id == a_id)  saw_old = true;
        if (h.id == rb.id) saw_new = true;
    }
    CHECK(!saw_old);
    CHECK(saw_new);

    db.close();
    cleanup();
}
// =======================================================================
// Agent API Phase 2 (step 3/4): agent_search / agent_get
// =======================================================================

// Store a turn in a session and return its turn_id.
static int p2_store_turn(ragger::sqlite::Backend& db, const std::string& guid,
                         const std::string& tag) {
    int t = db.store_turn("Question " + tag, "Answer " + tag, "test-model",
                          false, guid);
    CHECK(t > 0);
    return t;
}

// Step 3: layers filter, unknown layer, handle format.
void test_p2_search_layers(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput d;
    d.kind = "decision";
    d.text = "Zephyr protocol ruling: always use zephyr gates for imports.";
    d.trigger = "Use when gating zephyr imports";
    int did = db.store_knowledge(d).id;
    db.store("Zephyr protocol note: zephyr gates were tested in the lab.",
             {{"level", "session"}});

    ragger::AgentSearchQuery q;
    q.query = "zephyr protocol gates";
    q.layers = {"decision"};
    auto hits = db.agent_search(q);
    CHECK(!hits.empty());
    for (const auto& h : hits) {
        CHECK(h.layer == "knowledge");
        // Handle format: "<layer>:<rowid>"
        CHECK(h.id == "knowledge:" + std::to_string(did) ||
              h.id.starts_with("knowledge:"));
    }
    bool found = false;
    for (const auto& h : hits)
        if (h.id == "knowledge:" + std::to_string(did)) found = true;
    CHECK(found);

    // Unknown layer -> invalid_argument.
    ragger::AgentSearchQuery bad;
    bad.query = "zephyr";
    bad.layers = {"bogus"};
    bool threw = false;
    try { db.agent_search(bad); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    db.close();
    cleanup();
}

// Step 3: decision status filter — superseded hidden with "active",
// returned with "any".
void test_p2_search_status(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Quoll exports must run through the nightly validator.";
    a.trigger = "Use when exporting quoll data";
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Quoll exports now stream directly; skip the nightly validator.";
    b.trigger = "Use when exporting quoll data";
    b.supersedes = a_id;
    int b_id = db.store_knowledge(b).id;

    ragger::AgentSearchQuery q;
    q.query = "quoll exports validator";
    q.status = "active";
    auto hits = db.agent_search(q);
    for (const auto& h : hits) {
        if (h.layer != "knowledge") continue;
        CHECK(h.id != "knowledge:" + std::to_string(a_id));
    }

    q.status = "any";
    auto any_hits = db.agent_search(q);
    bool saw_old = false, saw_new = false;
    for (const auto& h : any_hits) {
        if (h.id == "knowledge:" + std::to_string(a_id)) { saw_old = true; CHECK(h.extra["status"] == "superseded"); }
        if (h.id == "knowledge:" + std::to_string(b_id)) saw_new = true;
    }
    CHECK(saw_old);
    CHECK(saw_new);

    db.close();
    cleanup();
}

// Step 3: since/until parsing — future since -> 0 hits; "30d" -> hits;
// malformed -> throws.
void test_p2_search_since(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("Kestrel habitat notes: kestrels nest in old towers.");

    ragger::AgentSearchQuery q;
    q.query = "kestrel habitat nest";
    q.since = "2099-01-01";
    CHECK(db.agent_search(q).empty());

    q.since = "30d";
    CHECK(!db.agent_search(q).empty());

    q.since = "2026-13-45";
    bool threw = false;
    try { db.agent_search(q); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    db.close();
    cleanup();
}

// Step 3: snippet cutting — 1000-char text -> <=243 bytes ending in "…".
void test_p2_search_snippet(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    std::string long_text = "Wombat warren survey. ";
    while (long_text.size() < 1000) long_text += "The wombat surveyed the warren thoroughly. ";
    db.store(long_text);

    ragger::AgentSearchQuery q;
    q.query = "wombat warren survey";
    auto hits = db.agent_search(q);
    CHECK(!hits.empty());
    bool saw_long = false;
    for (const auto& h : hits) {
        CHECK(h.snippet.size() <= 243);
        if (h.snippet.size() > 100) {   // the long row (others are short)
            saw_long = true;
            CHECK(h.snippet.ends_with("…"));
        }
    }
    CHECK(saw_long);

    db.close();
    cleanup();
}

// Step 3: max_tokens budget — 3 matches, budget 1 -> exactly 1 hit.
void test_p2_search_max_tokens(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("Xenon lamp calibration record one.");
    db.store("Xenon lamp calibration record two.");
    db.store("Xenon lamp calibration record three.");

    ragger::AgentSearchQuery q;
    q.query = "xenon lamp calibration record";
    q.max_tokens = 1;
    auto hits = db.agent_search(q);
    CHECK(hits.size() == 1);

    db.close();
    cleanup();
}

// Step 4: mode 1 — decision handle returns full row with trigger + evidence.
void test_p2_get_decision(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int ts1 = p1_make_episode(db, "alpha");
    int ts2 = p1_make_episode(db, "beta");

    ragger::KnowledgeInput d;
    d.kind = "decision";
    d.text = "Always cite the failing test when filing a bug.";
    d.trigger = "Use when filing a bug report";
    d.rationale = "Reproducible failures need receipts.";
    d.scope = "Not needed for feature requests.";
    d.evidence = {ts1, ts2};
    int did = db.store_knowledge(d).id;

    ragger::AgentGetQuery q;
    q.ids = {"knowledge:" + std::to_string(did)};
    auto got = db.agent_get(q);
    CHECK(got["items"].size() == 1);
    const auto& item = got["items"][0];
    CHECK(!item.contains("error"));
    CHECK(item["layer"] == "knowledge");
    CHECK(item["trigger"] == "Use when filing a bug report");
    CHECK(item["rationale"] == "Reproducible failures need receipts.");
    CHECK(item["evidence"].size() == 2);
    CHECK(item["evidence"][0] == ts1);
    CHECK(item["evidence"][1] == ts2);

    // Unknown id -> per-item error, not a throw.
    ragger::AgentGetQuery bad;
    bad.ids = {"knowledge:999999"};
    auto bad_got = db.agent_get(bad);
    CHECK(bad_got["items"].size() == 1);
    CHECK(bad_got["items"][0].contains("error"));
    CHECK(bad_got["items"][0]["error"] == "not found");

    // Malformed handle -> same per-item error.
    ragger::AgentGetQuery malformed;
    malformed.ids = {"nonsense:1", "decision:-3", "decision:"};
    auto m_got = db.agent_get(malformed);
    CHECK(m_got["items"].size() == 3);
    for (const auto& it : m_got["items"]) CHECK(it.contains("error"));

    db.close();
    cleanup();
}

// Step 4: expand=chain — A <- B <- C, get(B) -> expanded [A, B, C].
void test_p2_get_chain(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Chain link one: use plan alpha for the widget.";
    a.trigger = "Use when planning the widget";
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Chain link two: use plan beta for the widget.";
    b.trigger = "Use when planning the widget";
    b.supersedes = a_id;
    int b_id = db.store_knowledge(b).id;

    ragger::KnowledgeInput c;
    c.kind = "decision";
    c.text = "Chain link three: use plan gamma for the widget.";
    c.trigger = "Use when planning the widget";
    c.supersedes = b_id;
    int c_id = db.store_knowledge(c).id;

    ragger::AgentGetQuery q;
    q.ids = {"knowledge:" + std::to_string(b_id)};
    q.expand = "chain";
    auto got = db.agent_get(q);
    const auto& exp = got["items"][0]["expanded"];
    CHECK(exp.size() == 3);
    CHECK(exp[0]["id"] == "knowledge:" + std::to_string(a_id));
    CHECK(exp[1]["id"] == "knowledge:" + std::to_string(b_id));
    CHECK(exp[2]["id"] == "knowledge:" + std::to_string(c_id));

    db.close();
    cleanup();
}

// Step 4: expand=context — 6 turns in one session, get(turn 3) -> 1,2,4,5.
void test_p2_get_context(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "ctx-session-guid";
    std::vector<int> ids;
    for (int i = 0; i < 6; ++i)
        ids.push_back(p2_store_turn(db, guid, "ctx" + std::to_string(i)));
    CHECK(ids.size() == 6);

    ragger::AgentGetQuery q;
    q.ids = {"turn:" + std::to_string(ids[2])};
    q.expand = "context";
    auto got = db.agent_get(q);
    const auto& exp = got["items"][0]["expanded"];
    CHECK(exp.size() == 4);
    CHECK(exp[0]["id"] == "turn:" + std::to_string(ids[0]));
    CHECK(exp[1]["id"] == "turn:" + std::to_string(ids[1]));
    CHECK(exp[2]["id"] == "turn:" + std::to_string(ids[3]));
    CHECK(exp[3]["id"] == "turn:" + std::to_string(ids[4]));

    db.close();
    cleanup();
}

// Step 4: mode 2 — history paging: 25 turns, limit 10 -> 10/10/5,
// next_after null on the last page.
void test_p2_get_history(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "hist-session-guid";
    for (int i = 0; i < 25; ++i)
        p2_store_turn(db, guid, "h" + std::to_string(i));

    ragger::AgentGetQuery q;
    q.session = guid;
    q.limit = 10;

    auto p1 = db.agent_get(q);
    CHECK(p1["session"] == guid);
    CHECK(p1["turns"].size() == 10);
    CHECK(!p1["next_after"].is_null());
    int after = p1["next_after"].get<int>();

    q.after_turn = after;
    auto p2 = db.agent_get(q);
    CHECK(p2["turns"].size() == 10);
    CHECK(!p2["next_after"].is_null());
    after = p2["next_after"].get<int>();

    q.after_turn = after;
    auto p3 = db.agent_get(q);
    CHECK(p3["turns"].size() == 5);
    CHECK(p3["next_after"].is_null());

    // Ascending turn_id order across pages.
    CHECK(p1["turns"][0]["id"] == "turn:1");

    db.close();
    cleanup();
}

// Step 4: exactly one mode — both or neither -> invalid_argument.
void test_p2_get_mode_exclusive(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    bool threw_both = false, threw_neither = false;
    ragger::AgentGetQuery both;
    both.ids = {"knowledge:1"};
    both.session = "some-guid";
    try { db.agent_get(both); }
    catch (const std::invalid_argument&) { threw_both = true; }

    ragger::AgentGetQuery neither;
    try { db.agent_get(neither); }
    catch (const std::invalid_argument&) { threw_neither = true; }

    CHECK(threw_both);
    CHECK(threw_neither);

    db.close();
    cleanup();
}

// Step 5: possible_supersedes hint — a near-duplicate decision flags the
// original; an unrelated one returns an empty list. No row is ever changed.
void test_p2_possible_supersedes(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Always run the full test suite before tagging a release.";
    a.trigger = "Use when preparing a release tag";
    int a_id = db.store_knowledge(a).id;

    // Near-duplicate: same ruling reworded, same trigger.
    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Run the complete test suite before every release tag.";
    b.trigger = "Use when preparing a release tag";
    auto rb = db.store_knowledge(b);
    bool flags_a = false;
    for (const auto& [id, score] : rb.possible_supersedes) {
        CHECK(score >= 0.80f);
        if (id == a_id) flags_a = true;
    }
    CHECK(flags_a);
    // Flag only: A must still be 'current'.
    CHECK(!rb.superseded.has_value());

    // Unrelated text -> empty list.
    ragger::KnowledgeInput c;
    c.kind = "decision";
    c.text = "The harpsichord was tuned in meantone temperament in 1742.";
    c.trigger = "Use when tuning baroque keyboard instruments";
    auto rc = db.store_knowledge(c);
    CHECK(rc.possible_supersedes.empty());

    db.close();
    cleanup();
}
// Step 1: three sessions with episodes at known created_at -> newest first;
// paging limit 2 over 5 rows -> 2, 2, 1 with next_cursor null at the end.
void test_p4_timeline_order_and_paging(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Five episodes, one per "session", spaced a day apart (PHASE5: episodes
    // live in their own table; close each via store_episode_close around a
    // single seeded member).
    const std::vector<std::string> days = {
        "2026-04-01 10:00:00", "2026-04-02 10:00:00", "2026-04-03 10:00:00",
        "2026-04-04 10:00:00", "2026-04-05 10:00:00"};
    for (size_t i = 0; i < days.size(); ++i) {
        const std::string guid = "p4tl-" + std::to_string(i);
        const int t = db.store_turn("p4tl q" + std::to_string(i),
                                    "p4tl answer " + std::string(60, 'a') + std::to_string(i),
                                    "test-model", false, guid, days[i]);
        CHECK(db.finalize_turn_summary(t, "p4tl summary " + std::to_string(i), "summarizer-model"));
        sqlite3* raw = p1_open_raw();
        const int tsid = static_cast<int>(p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
        sqlite3_close(raw);
        ragger::EpisodeClose ec;
        ec.text         = "Timeline episode " + std::to_string(i);
        ec.model_name   = "summarizer-model";
        ec.session_guid = guid;
        ec.span_start   = p5_epoch(days[i]);
        ec.span_end     = p5_epoch(days[i]);
        ec.member_turn_summary_ids = {tsid};
        CHECK(db.store_episode_close(ec) > 0);
    }

    ragger::TimelineQuery q;
    q.granularity = "episode";
    q.limit = 100;
    auto page = db.agent_timeline(q);
    CHECK(page["items"].size() == 5);
    // Newest first: created strings strictly descending.
    for (size_t i = 1; i < page["items"].size(); ++i)
        CHECK(page["items"][i - 1]["created"].get<std::string>() >
              page["items"][i]["created"].get<std::string>());
    CHECK(page["items"][0]["text"].get<std::string>() == "Timeline episode 4");
    CHECK(page["next_cursor"].is_null());

    // Paging: 2, 2, 1; cursor null only on the last page.
    ragger::TimelineQuery p;
    p.granularity = "episode";
    p.limit = 2;
    int seen = 0;
    for (int guard = 0; guard < 5; ++guard) {
        auto pg = db.agent_timeline(p);
        const size_t want = (guard == 0 || guard == 1) ? 2 : 1;
        CHECK(pg["items"].size() == want);
        seen += static_cast<int>(pg["items"].size());
        if (pg["next_cursor"].is_null()) break;
        p.cursor = pg["next_cursor"].get<std::string>();
        if (guard == 0) CHECK(!pg["next_cursor"].is_null());
    }
    CHECK(seen == 5);

    // since/until window: only the middle three days.
    ragger::TimelineQuery w;
    w.granularity = "episode";
    w.since = "2026-04-02";
    w.until = "2026-04-04";
    auto win = db.agent_timeline(w);
    CHECK(win["items"].size() == 3);

    // Bad granularity / bad cursor -> invalid_argument.
    bool threw = false;
    try {
        ragger::TimelineQuery bad; bad.granularity = "week"; db.agent_timeline(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {
        ragger::TimelineQuery bad; bad.cursor = "not-a-cursor"; db.agent_timeline(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Negative cursor forms (review 7): every malformed shape must reject as
    // invalid_argument -- including overflowing digit runs, which must not
    // escape as std::out_of_range (HTTP 500 instead of 400).
    for (const char* bad_cursor : {
            "123",                        // no colon
            "123:",                       // empty rowid part
            ":5",                         // empty epoch part
            "a:b",                        // non-digits
            "-1:5",                       // negative epoch ('-' is not a digit)
            "1:-5",                       // negative rowid
            "1:2:3",                      // extra colon
            " 1:5",                       // leading space
            "99999999999999999999:1",     // epoch overflow (>18 digits)
            "1:99999999999999999999",     // rowid overflow
            "99999999999999999999999999999999:1" /* absurd length */}) {
        bool rejected = false;
        try {
            ragger::TimelineQuery bad; bad.cursor = bad_cursor; db.agent_timeline(bad);
        } catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }

    db.close();
    cleanup();
}

// PHASE5 C.3: granularity=session returns ONE ROW PER SESSION in range —
// {session_id, guid, name, started_on, ended_on, episodes, snippet} — and
// never reads summaries. Sessions with no turns still appear (NULL bounds).
void test_p4_timeline_session_fallback(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string ga = "p4sf-a";
    const std::string gb = "p4sf-b";
    const std::string gc = "p4sf-c";

    auto seed = [&](const std::string& guid, const std::string& day, int idx) {
        const std::string stamp = day + " 08:00:00";
        const int t = db.store_turn("p4sf q" + guid + std::to_string(idx),
                                    "p4sf answer " + std::string(60, 'b') + guid + std::to_string(idx),
                                    "test-model", false, guid, stamp);
        CHECK(db.finalize_turn_summary(t, "p4sf summary " + guid + std::to_string(idx), "summarizer-model"));
        sqlite3* raw = p1_open_raw();
        const int tsid = static_cast<int>(p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
        sqlite3_close(raw);
        ragger::EpisodeClose ec;
        ec.text         = guid + " episode " + std::to_string(idx);
        ec.model_name   = "summarizer-model";
        ec.session_guid = guid;
        ec.span_start   = p5_epoch(stamp);
        ec.span_end     = p5_epoch(stamp);
        ec.member_turn_summary_ids = {tsid};
        CHECK(db.store_episode_close(ec) > 0);
    };

    // Session A: two episodes (04-01, 04-02). Session B: two (04-03, 04-04).
    seed(ga, "2026-04-01", 0);
    seed(ga, "2026-04-02", 1);
    seed(gb, "2026-04-03", 0);
    seed(gb, "2026-04-04", 1);
    // Session C: no turns at all -> still one row, NULL bounds, 0 episodes.
    {
        sqlite3* raw = p1_open_raw();
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(raw, "INSERT INTO sessions (guid) VALUES ('p4sf-c')",
                                 -1, &st, nullptr) == SQLITE_OK);
        CHECK(sqlite3_step(st) == SQLITE_DONE);
        sqlite3_finalize(st);
        sqlite3_close(raw);
    }

    ragger::TimelineQuery q;
    q.granularity = "session";
    auto page = db.agent_timeline(q);
    CHECK(page["items"].size() == 3);   // one row per session, always
    // Newest first by started_on: B (04-03) then A (04-01) then C (no span).
    CHECK(page["items"][0]["guid"].get<std::string>() == gb);
    CHECK(page["items"][0]["episodes"].get<int>() == 2);
    CHECK(page["items"][0]["snippet"].get<std::string>() == "p4sf-b episode 1");
    CHECK(page["items"][1]["guid"].get<std::string>() == ga);
    CHECK(page["items"][1]["episodes"].get<int>() == 2);
    CHECK(page["items"][1]["snippet"].get<std::string>() == "p4sf-a episode 1");
    CHECK(page["items"][2]["guid"].get<std::string>() == gc);
    CHECK(page["items"][2]["episodes"].get<int>() == 0);
    CHECK(page["items"][2]["snippet"].get<std::string>() == "");
    CHECK(page["items"][2]["started_on"].is_null());
    CHECK(page["items"][2]["ended_on"].is_null());
    // started_on/ended_on rendered for sessions with turns.
    CHECK(!page["items"][0]["started_on"].is_null());

    // Date filter on sessions matches span overlap: since=04-02 keeps B
    // (span 04-03..04-04) but drops A (span 04-01..04-02 08:00 < since).
    ragger::TimelineQuery w;
    w.granularity = "session";
    w.since = "2026-04-02 12:00:00";
    auto wp = db.agent_timeline(w);
    CHECK(wp["items"].size() == 1);
    CHECK(wp["items"][0]["guid"].get<std::string>() == gb);

    db.close();
    cleanup();
}

// Step 1: turn granularity — summarized turns show the summary, an
// unsummarized turn falls back to raw user text.
void test_p4_timeline_turn_fallback(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "p4tf-1";
    int t1 = db.store_turn("Raw question one", "Answer one", "test-model", false, guid);
    int t2 = db.store_turn("Raw question two", "Answer two", "test-model", false, guid);
    CHECK(db.finalize_turn_summary(t1, "Summarized turn one", "summarizer-model"));

    ragger::TimelineQuery q;
    q.granularity = "turn";
    auto page = db.agent_timeline(q);
    CHECK(page["items"].size() == 2);
    // Handles are turn:<id> (resolvable by agent_get either way).
    bool saw_summarized = false, saw_raw = false;
    for (const auto& item : page["items"]) {
        if (item["id"].get<std::string>() == "turn:" + std::to_string(t1)) {
            CHECK(item["text"].get<std::string>() == "Summarized turn one");
            saw_summarized = true;
        }
        if (item["id"].get<std::string>() == "turn:" + std::to_string(t2)) {
            CHECK(item["text"].get<std::string>() == "Raw question two");
            saw_raw = true;
        }
    }
    CHECK(saw_summarized);
    CHECK(saw_raw);

    db.close();
    cleanup();
}
// Step 2: update rules — text rejected, non-editable layers rejected,
// notes editable (tags/project), rollups rejected, transitions enforced.
void test_p4_update_rules(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    auto throws_invalid = [&db](const ragger::UpdateRequest& req) {
        try { db.agent_update(req); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };

    // Rule 1: text present -> rejected with the exact message.
    {
        ragger::UpdateRequest r;
        r.id = "knowledge:1";
        r.text = "anything";
        bool threw = false;
        try { db.agent_update(r); }
        catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()) ==
                  "text cannot be edited in place; store a new decision/note with supersedes");
        }
        CHECK(threw);
    }

    // Rule 2: turn / turn_summary handles rejected.
    int ts_id = p1_make_turn_summary(db, "p4u");   // returns the turn_summary_id
    {
        ragger::UpdateRequest r; r.id = "turn_summary:" + std::to_string(ts_id); r.tags = "x";
        CHECK(throws_invalid(r));
    }

    // Rule 3 (PHASE8): notes are knowledge rows. The legacy `summary:<id>`
    // handle is rejected with a pointed message; `knowledge:<id>` edits work.
    const std::string note_id = db.store("A note to curate.", {{"level", "session"}});
    {
        ragger::UpdateRequest r;
        r.id = "summary:" + note_id;
        r.tags = "curated,keep2";
        CHECK(throws_invalid(r));
    }
    {
        ragger::UpdateRequest r;
        r.id = "knowledge:" + note_id;
        r.tags = "curated,keep2";
        auto row = db.agent_update(r);
        CHECK(row["tags"].get<std::string>() == "curated,keep2");
    }
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT tags FROM knowledge WHERE knowledge_id = " + note_id)
              == "curated,keep2");
        sqlite3_close(raw);
    }

    // Rule 4: transition table.
    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text = "P4 transition target ruling.";
    in.trigger = "Use when testing transitions";
    in.status = "pending";
    int pending_id = db.store_knowledge(in).id;
    auto dec_status = [&db](int id) {
        sqlite3* raw = p1_open_raw();
        std::string s = p1_scalar_text(raw, "SELECT status FROM knowledge WHERE knowledge_id = "
                                  + std::to_string(id));
        sqlite3_close(raw);
        return s;
    };
    {   // pending -> superseded rejected
        ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(pending_id);
        r.status = "superseded";
        CHECK(throws_invalid(r));
    }
    {   // pending -> current allowed
        ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(pending_id);
        r.status = "current";
        auto row = db.agent_update(r);
        CHECK(row["status"].get<std::string>() == "current");
        CHECK(dec_status(pending_id) == "current");
    }
    {   // current -> pending NOT allowed (table says no)
        ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(pending_id);
        r.status = "pending";
        CHECK(throws_invalid(r));
    }
    {   // current -> current not a listed transition either
        ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(pending_id);
        r.status = "current";
        CHECK(throws_invalid(r));
    }
    {   // current -> provisional -> retired allowed; retired -> anything rejected
        ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(pending_id);
        r.status = "provisional";
        db.agent_update(r);
        r.status = "retired";
        db.agent_update(r);
        CHECK(dec_status(pending_id) == "retired");
        r.status = "current";
        CHECK(throws_invalid(r));
    }

    db.close();
    cleanup();
}

// Step 2/5: a decision change is visible to normal search immediately
// (decision cache invalidated).
void test_p4_update_cache_invalidation(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text = "Kestrel migrations run only on Wednesdays.";
    in.trigger = "Use when scheduling kestrel migrations";
    in.status = "pending";
    int id = db.store_knowledge(in).id;

    // Pending is hidden from active search.
    ragger::AgentSearchQuery q;
    q.query = "kestrel migrations Wednesdays";
    q.layers = {"decision"};
    auto hits = db.agent_search(q);
    for (const auto& h : hits) CHECK(h.id != "knowledge:" + std::to_string(id));

    ragger::UpdateRequest r; r.id = "knowledge:" + std::to_string(id); r.status = "current";
    db.agent_update(r);

    hits = db.agent_search(q);
    bool found = false;
    for (const auto& h : hits)
        if (h.id == "knowledge:" + std::to_string(id)) found = true;
    CHECK(found);

    db.close();
    cleanup();
}

// Step 3: review queue — list, approve, reject, per-id failure.
void test_p4_review(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput a;
    a.kind = "decision";
    a.text = "Review candidate alpha: wagons ship at dawn.";
    a.trigger = "Use when shipping wagons";
    a.status = "pending";
    int a_id = db.store_knowledge(a).id;

    ragger::KnowledgeInput b;
    b.kind = "decision";
    b.text = "Review candidate beta: kayaks ship at dusk.";
    b.trigger = "Use when shipping kayaks";
    b.status = "pending";
    int b_id = db.store_knowledge(b).id;

    auto listed = db.agent_review("list", {});
    CHECK(listed["pending"].size() == 2);
    CHECK(listed["pending"][0]["id"].get<std::string>() == "knowledge:" + std::to_string(a_id));
    CHECK(!listed["pending"][0]["trigger"].get<std::string>().empty());

    // Approve alpha (bare integer accepted too).
    auto done = db.agent_review("approve", {std::to_string(a_id)});
    CHECK(done["done"].size() == 1);
    CHECK(done["failed"].empty());
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT status FROM knowledge WHERE knowledge_id = "
                                  + std::to_string(a_id)) == "current");
        sqlite3_close(raw);
    }
    // Approved decision is now visible in normal search.
    ragger::AgentSearchQuery q;
    q.query = "wagons ship dawn";
    q.layers = {"decision"};
    bool found = false;
    for (const auto& h : db.agent_search(q))
        if (h.id == "knowledge:" + std::to_string(a_id)) found = true;
    CHECK(found);

    // Approving an already-current decision fails THAT id only.
    auto again = db.agent_review("approve", {
        "knowledge:" + std::to_string(a_id),
        "knowledge:" + std::to_string(b_id)});
    CHECK(again["done"].size() == 1);
    CHECK(again["failed"].size() == 1);
    CHECK(again["failed"][0]["id"].get<std::string>() == "knowledge:" + std::to_string(a_id));

    // Reject beta -> retired, invisible in search, row still present.
    auto rej = db.agent_review("reject", {"knowledge:" + std::to_string(b_id)});
    CHECK(rej["done"].size() == 1);
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT status FROM knowledge WHERE knowledge_id = "
                                  + std::to_string(b_id)) == "retired");
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge WHERE knowledge_id = "
                                  + std::to_string(b_id)) == 1);
        sqlite3_close(raw);
    }
    q.query = "kayaks ship dusk";
    for (const auto& h : db.agent_search(q))
        CHECK(h.id != "knowledge:" + std::to_string(b_id));

    // Queue is empty now.
    CHECK(db.agent_review("list", {})["pending"].empty());

    // Bad action -> invalid_argument.
    bool threw = false;
    try { db.agent_review("zap", {}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    db.close();
    cleanup();
}

// =======================================================================
// PHASE1 Step 3a: Transaction RAII (raw sqlite; own scratch table).
// =======================================================================
static void p1r_exec_raw(sqlite3* raw, const char* sql) {
    CHECK(sqlite3_exec(raw, sql, nullptr, nullptr, nullptr) == SQLITE_OK);
}

void p1r_test_transaction() {
    std::println("  p1r_test_transaction...");
    sqlite3* raw = p1_open_raw();
    p1r_exec_raw(raw, "CREATE TABLE IF NOT EXISTS p1r_t (v INTEGER)");
    p1r_exec_raw(raw, "DELETE FROM p1r_t");

    // (i) commit persists.
    {
        ragger::Transaction tx(raw);
        p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (1)");
        tx.commit();
    }
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM p1r_t") == 1);

    // (ii) scope exit without commit rolls back.
    {
        ragger::Transaction tx(raw);
        p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (2)");
    }
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM p1r_t") == 1);

    // (iii) a Transaction constructed inside another neither commits nor
    // rolls back the outer: outer commit persists both rows...
    {
        ragger::Transaction outer(raw);
        p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (3)");
        {
            ragger::Transaction inner(raw);   // joins the outer transaction
            p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (4)");
            inner.commit();                   // must be a no-op for the outer
        }
        // Still inside the outer transaction (same connection sees its own
        // uncommitted rows; the open transaction is what proves inner did
        // not commit).
        CHECK(sqlite3_get_autocommit(raw) == 0);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM p1r_t WHERE v IN (3,4)") == 2);
        outer.commit();
    }
    CHECK(sqlite3_get_autocommit(raw) == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM p1r_t") == 3);

    // ...and outer rollback removes both (inner commit did not commit them).
    {
        ragger::Transaction outer(raw);
        p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (5)");
        {
            ragger::Transaction inner(raw);
            p1r_exec_raw(raw, "INSERT INTO p1r_t VALUES (6)");
            inner.commit();
        }
    }  // outer dtor rolls back both rows
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM p1r_t") == 3);

    p1r_exec_raw(raw, "DROP TABLE p1r_t");
    sqlite3_close(raw);
    std::println(" OK");
}

// =======================================================================
// PHASE1 Step 3d: turn_summaries index keyed by turn_summary_id.
// =======================================================================
void p1_test_turn_summary_index_keying(ragger::Embedder& emb) {
    std::println("  p1_test_turn_summary_index_keying...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int a = db.store_turn("Question alpha one", "Answer alpha", "test-model");
    int b = db.store_turn("Question beta two three", "Answer beta", "test-model");
    int c = db.store_turn("Question gamma", "Answer gamma", "test-model");
    CHECK(a > 0 && b > 0 && c > 0);

    // Finalize in reverse order (C, B, A) so turn_summary_id and turn_id
    // diverge — the order is the whole point of the test.
    const std::string text_c = "Short gamma summary.";
    const std::string text_b = "Medium beta summary with more words here.";
    const std::string text_a = "Alpha summary is the longest one of the three summaries by far.";
    CHECK(db.finalize_turn_summary(c, text_c, "summarizer-model"));
    CHECK(db.finalize_turn_summary(b, text_b, "summarizer-model"));
    CHECK(db.finalize_turn_summary(a, text_a, "summarizer-model"));

    sqlite3* raw = p1_open_raw();
    // Non-vacuity: if all ids coincide the test proves nothing — stop here.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_summary_id != turn_id") >= 1);
    // Every junction row joins a real summary row.
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM turn_summaries_terms j LEFT JOIN turn_summaries t "
        "ON t.turn_summary_id = j.turn_summary_id WHERE t.turn_summary_id IS NULL") == 0);
    // Counts landed on the summary rows (not lost to a mis-keyed UPDATE).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE unigram_count = 0 AND text != ''") == 0);
    // The longest summary text has the largest unigram_count.
    CHECK(p1_scalar_text(raw,
        "SELECT text FROM turn_summaries WHERE text != '' ORDER BY unigram_count DESC LIMIT 1") == text_a);
    sqlite3_close(raw);

    // mark_turn_summarized: marker row exists, indexes zero terms for it.
    int d = db.store_turn("Question delta", "Answer delta", "test-model");
    CHECK(d > 0);
    CHECK(db.mark_turn_summarized(d, "trivial-model"));
    raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = "
                             + std::to_string(d) + " AND text = ''") == 1);
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM turn_summaries_terms j JOIN turn_summaries t "
        "ON t.turn_summary_id = j.turn_summary_id WHERE t.turn_id = " + std::to_string(d)) == 0);
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// =======================================================================
// PHASE1 Step 4: import-idempotency probes compare INTEGER epochs.
// Each probe gets a POSITIVE case — these were never tested.
// =======================================================================
void p1t_test_import_probes(ragger::Embedder& emb) {
    std::println("  p1t_test_import_probes...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // note (PHASE8): store with explicit source_timestamp, probe must find it.
    ragger::KnowledgeInput nin;
    nin.kind = "note";
    nin.text = "Probe note text.";
    nin.source_timestamp = "2026-04-15 09:30:00";
    int nid = db.store_knowledge(nin).id;
    CHECK(nid > 0);
    CHECK(db.note_exists_exact("Probe note text.", "2026-04-15 09:30:00"));
    CHECK(!db.note_exists_exact("Probe note text.", "2026-04-15 09:30:01"));

    // decision: same contract.
    int did = db.store_decision("Probe decision text.", "current", "", "2026-04-15 09:30:00");
    CHECK(did > 0);
    CHECK(db.decision_exists_exact("Probe decision text.", "2026-04-15 09:30:00"));
    CHECK(!db.decision_exists_exact("Probe decision text.", "2026-04-15 09:30:01"));

    // turn fuzzy window: ±5s around the probe instant.
    int t = db.store_turn("Fuzzy probe question", "Fuzzy probe answer", "test-model",
                          false, "", "2026-04-15 09:30:00");
    CHECK(t > 0);
    CHECK(db.turn_exists_fuzzy("Fuzzy probe question", "2026-04-15 09:30:03", 5));
    CHECK(!db.turn_exists_fuzzy("Fuzzy probe question", "2026-04-15 09:30:10", 5));
    CHECK(!db.turn_exists_fuzzy("Different question text", "2026-04-15 09:30:00", 5));

    // update_turn_meta must write an INTEGER epoch, agreeing with a store
    // that used the same source_timestamp.
    int t2 = db.store_turn("Meta probe question", "Meta probe answer", "test-model");
    CHECK(t2 > 0);
    CHECK(db.update_turn_meta(t2, "2026-04-16 10:00:00", ""));
    int t3 = db.store_turn("Meta probe question two", "Meta probe answer two", "test-model",
                           false, "", "2026-04-16 10:00:00");
    CHECK(t3 > 0);
    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, "SELECT typeof(created_at) FROM turns WHERE turn_id = "
                              + std::to_string(t2)) == "integer");
    CHECK(p1_scalar_int(raw, "SELECT created_at FROM turns WHERE turn_id = " + std::to_string(t2))
          == p1_scalar_int(raw, "SELECT created_at FROM turns WHERE turn_id = " + std::to_string(t3)));

    // search_by_metadata: after/before bracket the stored row and the
    // returned created_at renders datetime-shaped (intentional change from
    // the raw epoch string).
    auto results = db.search_by_metadata({{"kind", "note"}}, 0,
                                         "2026-04-15 09:00:00", "2026-04-15 10:00:00");
    CHECK(results.size() == 1);
    CHECK(results[0].text == "Probe note text.");
    const std::string& ts = results[0].timestamp;
    std::regex dt_re(R"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)");
    CHECK(std::regex_match(ts, dt_re));
    // Outside the bracket → empty.
    CHECK(db.search_by_metadata({{"kind", "note"}}, 0,
                                "2026-04-15 09:31:00", "2026-04-15 10:00:00").empty());
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// =======================================================================
// S51 (a) (B §6 item 3): search_by_metadata temporal boundaries.
// The existing probe brackets ONE row; this pins the boundary behaviour with
// three rows a day apart.
//
// Semantics AS IMPLEMENTED (src/sqlite/maintenance.cpp search_by_metadata):
//   after  -> created_at >= ?   INCLUSIVE lower bound
//   before -> created_at <  ?   EXCLUSIVE upper bound
// i.e. a half-open interval [after, before). Note the asymmetry: it is not
// "both inclusive", so after==before is the EMPTY set, not a single row.
// created_at is an INTEGER epoch and both bounds resolve through
// resolve_epoch() (local-time '%F %T'), so raw-string comparison never
// applies here.
// =======================================================================
void p11_search_by_metadata_boundaries(ragger::Embedder& emb) {
    std::println("  p11_search_by_metadata_boundaries...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string t0 = "2026-05-01 12:00:00";
    const std::string t1 = "2026-05-02 12:00:00";   // t0 + 1d
    const std::string t2 = "2026-05-03 12:00:00";   // t0 + 2d
    for (const auto& [ts, txt] : std::vector<std::pair<std::string, std::string>>{
             {t0, "boundary row alpha"}, {t1, "boundary row beta"}, {t2, "boundary row gamma"}}) {
        ragger::KnowledgeInput in;
        in.kind = "note";
        in.text = txt;
        in.tags = "boundary";
        in.source_timestamp = ts;
        CHECK(db.store_knowledge(in).id > 0);
    }

    auto texts = [](std::vector<ragger::SearchResult> v) {
        std::sort(v.begin(), v.end(), [](const ragger::SearchResult& a, const ragger::SearchResult& b) {
            return a.timestamp > b.timestamp;
        });
        std::string s;
        for (const auto& r : v) { s += r.text; s += "|"; }
        return s;
    };

    // INCLUSIVE lower bound: the row exactly ON the bound is included.
    CHECK(texts(db.search_by_metadata({{"tags", "boundary"}}, 0, t1, "")) ==
          "boundary row gamma|boundary row beta|");
    // EXCLUSIVE upper bound: the row exactly ON the bound is NOT included.
    CHECK(texts(db.search_by_metadata({{"tags", "boundary"}}, 0, "", t1)) ==
          "boundary row alpha|");
    // Half-open: [t1, t1) is empty (this is where "both inclusive" would give 1).
    CHECK(db.search_by_metadata({{"tags", "boundary"}}, 0, t1, t1).empty());
    // [t0, t2) -> the two older rows; ORDER BY created_at DESC.
    CHECK(texts(db.search_by_metadata({{"tags", "boundary"}}, 0, t0, t2)) ==
          "boundary row beta|boundary row alpha|");
    // Open lower bound: [.., t2) excludes the row ON t2 (upper bound is <).
    CHECK(texts(db.search_by_metadata({{"tags", "boundary"}}, 0, "", t2)) ==
          "boundary row beta|boundary row alpha|");
    CHECK(texts(db.search_by_metadata({{"tags", "boundary"}}, 0, t2, "")) ==
          "boundary row gamma|");
    // A lower bound one second past the newest row matches nothing (>= on a
    // value no row has).
    CHECK(db.search_by_metadata({{"tags", "boundary"}}, 0, "2026-05-03 12:00:01").empty());
    // Timestamps render as local datetime strings, never bare epochs.
    for (const auto& r : db.search_by_metadata({{"tags", "boundary"}}))
        CHECK(r.timestamp.size() == 19 && r.timestamp[4] == '-' && r.timestamp[13] == ':');

    db.close();
    cleanup();
    std::println(" OK");
}


// =======================================================================
// S51 (b) (B §6 item 11-ish, C §7): agent_search on the DEGRADED path
// (no usable embedder -> search_text_only) with a document hit and a time
// LR S76 MIGRATION: this window test used the documents corpus (the only
// degraded-path fixture at the time); documents left agent_search's layers,
// so the same degraded-window contract is now pinned on turn_summaries.
// Window semantics AS IMPLEMENTED (src/sqlite/agent_read.cpp): the bounds are
// resolved by resolve_agent_time_bound() and then compared as STRINGS against
// the rendered local datetime of each hit — `since` inclusive, `until`
// inclusive, with a bare "YYYY-MM-DD" until widened to 23:59:59 (end_of_day).
// That is [since, until] — deliberately different from search_by_metadata's
// half-open [after, before).
//
// Runs without an ONNX model: the DB-only constructor leaves the embedder
// null, which is exactly the "unready" condition.
// =======================================================================
void p11_agent_search_unready_window(ragger::Embedder& emb) {
    std::println("  p11_agent_search_unready_window...");
    cleanup();

    // Writes need an embedder (require_embedder() throws without one), so the
    // fixture is written with the model and the DB is then REOPENED through the
    // DB-only constructor: that is the unready condition, reached the way a
    // server without a model reaches it.
    {
        ragger::sqlite::Backend w(emb, TEMP_DB);
        const std::string gs = "p11-window";
        const int t1 = w.store_turn("early q", "otter slide rule specifications",
                                    "test-model", true, gs, "2026-06-01 10:00:00");
        CHECK(w.finalize_turn_summary(t1, "otter slide rule specifications", "summarizer-model"));
        const int t2 = w.store_turn("late q", "otter slide rule calibration",
                                    "test-model", true, gs, "2026-06-05 10:00:00");
        CHECK(w.finalize_turn_summary(t2, "otter slide rule calibration", "summarizer-model"));
    }

    ragger::sqlite::Backend db(TEMP_DB);   // DB-only: embedder == nullptr

    // Handles sorted so score ordering can't make the assertion flaky.
    auto ids = [](std::vector<ragger::AgentHit> hits) {
        std::vector<std::string> v;
        for (const auto& h : hits) v.push_back(h.id);
        std::sort(v.begin(), v.end());
        std::string s;
        for (const auto& x : v) { s += x; s += "|"; }
        return s;
    };

    ragger::AgentSearchQuery q;
    q.query = "otter slide rule";
    q.limit = 10;

    // Degraded path finds both summaries. NOTE: matched_by is deliberately NOT
    // asserted here -- see p11_agent_search_matched_by_no_vector_signal below
    // (FINDING, parked unregistered).
    auto all = db.agent_search(q);
    CHECK(all.size() == 2);
    for (const auto& h : all) {
        CHECK(h.layer == "turn_summary");
        // Rendered local datetime, never a bare epoch.
        CHECK(h.created.size() == 19);
        CHECK(h.created[4] == '-' && h.created[7] == '-' &&
              h.created[10] == ' ' && h.created[13] == ':' && h.created[16] == ':');
    }

    // since (inclusive lower bound, date widened to 00:00:00).
    q.since = "2026-06-03";
    CHECK(ids(db.agent_search(q)) == "turn_summary:2|");
    // until (inclusive upper bound, date widened to 23:59:59).
    q.since = "";
    q.until = "2026-06-03";
    CHECK(ids(db.agent_search(q)) == "turn_summary:1|");
    // A bare `since` date is widened to 00:00:00, so the row on that day is in
    // (and so is every later row: this bound alone has no upper edge).
    q.until = "";
    q.since = "2026-06-01";
    CHECK(ids(db.agent_search(q)) == "turn_summary:1|turn_summary:2|");
    // Empty window.
    q.since = "2026-06-02";
    q.until = "2026-06-04";
    CHECK(db.agent_search(q).empty());

    db.close();
    cleanup();
    std::println(" OK");
}


// =======================================================================
// S51 (c) (C §7): agent_timeline at TURN granularity across TWO interleaved
// sessions. Existing coverage: turn granularity for ONE session
// (test_p4_timeline_turn_fallback) and session/episode granularity for two.
// Nothing pinned that a session-filtered turn timeline excludes the other
// session's rows while the two are interleaved in time, nor that keyset
// paging over a filtered timeline neither repeats nor skips.
// =======================================================================
void p11_timeline_turn_interleaved_sessions(ragger::Embedder& emb) {
    std::println("  p11_timeline_turn_interleaved_sessions...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string ga = "p11-interleave-a", gb = "p11-interleave-b";
    // A1, B1, A2, B2 — interleaved, ascending in time.
    int a1 = db.store_turn("Alpha first question",  "a1", "test-model", true, ga, "2026-07-01 10:00:00");
    int b1 = db.store_turn("Bravo first question",  "b1", "test-model", true, gb, "2026-07-01 10:01:00");
    int a2 = db.store_turn("Alpha second question", "a2", "test-model", true, ga, "2026-07-01 10:02:00");
    int b2 = db.store_turn("Bravo second question", "b2", "test-model", true, gb, "2026-07-01 10:03:00");
    CHECK(a1 > 0 && b1 > 0 && a2 > 0 && b2 > 0);

    auto collect = [&](const std::string& session, int limit) {
        std::vector<std::string> ids;
        ragger::TimelineQuery q;
        q.granularity = "turn";
        q.session = session;
        q.limit = limit;
        for (int guard = 0; guard < 10; ++guard) {
            auto page = db.agent_timeline(q);
            for (const auto& it : page["items"]) ids.push_back(it["id"].get<std::string>());
            if (page["next_cursor"].is_null()) return ids;
            q.cursor = page["next_cursor"].get<std::string>();
        }
        return ids;   // guard exhausted; the CHECK below fails loudly
    };
    auto turn = [](int id) { return "turn:" + std::to_string(id); };

    // (locals, not literals inside CHECK: the macro splits on top-level commas)
    const std::vector<std::string> expect_a{turn(a2), turn(a1)};
    const std::vector<std::string> expect_b{turn(b2), turn(b1)};

    // Session-filtered: only that session's turns, newest first.
    CHECK(collect(ga, 20) == expect_a);
    CHECK(collect(gb, 20) == expect_b);

    // Paging one at a time must yield exactly the same set, in the same order
    // (no repeat across pages, no skip).
    CHECK(collect(ga, 1) == expect_a);
    CHECK(collect(gb, 1) == expect_b);

    // Unfiltered: all four, strictly descending in time (the interleave shows).
    ragger::TimelineQuery all;
    all.granularity = "turn";
    all.limit = 20;
    auto page = db.agent_timeline(all);
    CHECK(page["items"].size() == 4);
    for (size_t i = 1; i < page["items"].size(); ++i)
        CHECK(page["items"][i - 1]["created"].get<std::string>() >
              page["items"][i]["created"].get<std::string>());
    CHECK(page["items"][0]["id"].get<std::string>() == turn(b2));
    CHECK(page["items"][3]["id"].get<std::string>() == turn(a1));

    db.close();
    cleanup();
    std::println(" OK");
}


// =======================================================================
// S51 (d): time-bound formats, ONE CASE PER FORMAT.
//
// SPEC DEVIATION, stated rather than papered over: the step names `recipe.cpp`,
// but recipe.cpp parses NO timestamps at all (parse_recipe reads name /
// description / max_tokens / chars_per_token / layers[kind,limit|count]; see
// include/recipe.h) and no recipe consumer formats a timestamp either — grepped
// src/recipe*.cpp and src/memory.cpp for time/date/strftime/localtime. The only
// multi-format timestamp PARSER in the query path is resolve_agent_time_bound()
// (file-static in src/sqlite/agent_read.cpp), so its accepted shapes are pinned
// here through the public surface that reaches it. Formats, read from the code:
//   "<N>d" / "<N>h"   N days/hours AGO (rejected above 100000000)
//   "YYYY-MM-DD"      since -> +00:00:00, until -> +23:59:59 (end_of_day)
//   "YYYY-MM-DD HH:MM:SS"  used verbatim
//   anything else     std::invalid_argument
// =======================================================================
void p11_time_bound_formats(ragger::Embedder& emb) {
    std::println("  p11_time_bound_formats...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "p11-tbf";
    int d1 = db.store_turn("July ten",  "x", "test-model", true, guid, "2026-07-10 08:00:00");
    int d2 = db.store_turn("July eleven", "x", "test-model", true, guid, "2026-07-11 08:00:00");
    // Relative rows, anchored to now so the "<N>d"/"<N>h" formats have meaning.
    const auto wall = std::time(nullptr);
    auto stamp = [](std::time_t t) {
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[20];
        std::strftime(buf, sizeof buf, "%F %T", &tm);
        return std::string(buf);
    };
    int r3 = db.store_turn("Three hours ago", "x", "test-model", true, guid, stamp(wall - 3 * 3600));
    int r4 = db.store_turn("Forty hours ago", "x", "test-model", true, guid, stamp(wall - 40 * 3600));
    CHECK(d1 > 0 && d2 > 0 && r3 > 0 && r4 > 0);

    auto count = [&](const std::string& since, const std::string& until) {
        ragger::TimelineQuery q;
        q.granularity = "turn";
        q.session = guid;
        q.since = since;
        q.until = until;
        q.limit = 100;
        return db.agent_timeline(q)["items"].size();
    };
    auto throws_bad = [&](const std::string& bound) {
        ragger::TimelineQuery q;
        q.granularity = "turn";
        q.session = guid;
        q.since = bound;
        try { db.agent_timeline(q); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };

    // "YYYY-MM-DD": widened to the whole local day on BOTH sides, so a
    // since==until date still selects that day's rows.
    CHECK(count("2026-07-10", "2026-07-10") == 1);
    CHECK(count("2026-07-11", "2026-07-11") == 1);
    CHECK(count("2026-07-10", "2026-07-11") == 2);
    // "YYYY-MM-DD HH:MM:SS": verbatim, inclusive at the exact instant.
    CHECK(count("2026-07-10 08:00:00", "2026-07-10 08:00:00") == 1);
    CHECK(count("2026-07-10 08:00:01", "2026-07-11 23:59:59") == 1);
    // "<N>h" / "<N>d": ago-relative against the two relative rows.
    CHECK(count("12h", "") == 1);     // only the 3h-old row
    CHECK(count("2d", "") == 2);      // 3h and 40h old; July rows are older
    CHECK(count("", "12h") == 3);     // everything OLDER than 12h ago
    // Rejections (each a distinct branch of the parser).
    CHECK(throws_bad("2026-13-45"));            // month out of range
    CHECK(throws_bad("2026-07-10 25:00:00"));   // hour out of range
    CHECK(throws_bad("100000001d"));            // relative cap
    CHECK(throws_bad("abc"));                   // no shape matches
    CHECK(throws_bad("1d2"));                   // digit suffix, not a shape

    db.close();
    cleanup();
    std::println(" OK");
}


// =======================================================================
// FINDING (S51 b) — PARKED, NOT REGISTERED IN main(). Do not "fix" this by
// weakening it; it asserts what matched_by SHOULD be.
//
// On the degraded path (no embedder) every candidate carries vec_score = -1.0f
// as the "no vector signal" sentinel (search_text_only), and agent_search then
// labels hits with:
//     vec>0 && bm25>0 -> "both";  bm25>0 -> "lexical";  else -> "vector"
// The final `else` therefore reports "vector" for a hit that provably has NO
// vector signal, whenever its text score is exactly 0.0.
//
// That is reachable, not theoretical: idf = log(N/df) is 0 when the query term
// occurs in EVERY row of the corpus, so a 2-document fixture whose two rows
// both contain the query scores 0.0 on both. Observed (2026-11-03):
//     id=document:1 layer=document matched=vector score=0
//     id=document:2 layer=document matched=vector score=0
// The agent API surfaces matched_by to callers, so "vector" here is a lie about
// provenance. Suggested shape of a fix (Reid's call -- it is an API contract):
// branch on the two signals explicitly and give the no-positive-signal case its
// own label ("neither"), or treat vec_score <= 0 as "no vector" so the degraded
// path can never claim "vector".
// =======================================================================
void p11_agent_search_matched_by_no_vector_signal(ragger::Embedder& emb) {
    std::println("  p11_agent_search_matched_by_no_vector_signal...");
    cleanup();
    {
        ragger::sqlite::Backend w(emb, TEMP_DB);
        ragger::DocumentChunk c;
        c.path = "/docs/matched-by.md";
        c.title = "matched by";
        c.text = "walrus barometer reading";
        c.imported_at = "2026-06-01 10:00:00";
        CHECK(w.store_document(c, /*defer_embedding=*/true) > 0);
    }
    ragger::sqlite::Backend db(TEMP_DB);   // unready embedder
    ragger::AgentSearchQuery q;
    q.query = "walrus barometer";
    auto hits = db.agent_search(q);
    CHECK(hits.size() == 1);
    // No embedder exists, so no hit can have matched by vector.
    CHECK(hits[0].matched_by != "vector");
    db.close();
    cleanup();
    std::println(" OK");
}


// =======================================================================
// PHASE1 Step 5: embedder / vector-type swap must not dangle.
// =======================================================================
void p1_test_embedder_swap(ragger::Embedder& emb, const std::string& model_dir) {
    std::println("  p1_test_embedder_swap...");
    (void)emb;
    cleanup();

    // (a) dtype swap: blobs written after set_vector_type use the NEW dtype.
    {
        std::optional<ragger::Embedder> e(model_dir);
        ragger::sqlite::Backend db(*e, TEMP_DB);
        const int dims = ragger::config().embedding_dimensions;
        db.store("F16 width probe.");
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT length(embedding) FROM knowledge "
                                  "WHERE text = 'F16 width probe.'") == 2 * dims);
        sqlite3_close(raw);

        db.set_vector_type(ragger::vector_codec::VectorType::F32);
        db.store("F32 width probe.");
        raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT length(embedding) FROM knowledge "
                                  "WHERE text = 'F32 width probe.'") == 4 * dims);
        sqlite3_close(raw);
        db.close();
    }
    cleanup();

    // (b) embedder swap: destroy the original object after repointing;
    // store/search must keep working (before the fix this was a UAF).
    {
        std::optional<ragger::Embedder> first(model_dir);
        ragger::sqlite::Backend db(*first, TEMP_DB);
        std::optional<ragger::Embedder> second(model_dir);
        db.set_embedder(*second);
        first.reset();  // frees the embedder the backend originally pointed at

        db.store("After swap probe.");
        auto resp = db.search("After swap probe", 5, 0.0f);
        bool found = false;
        for (const auto& r : resp.results)
            if (r.text == "After swap probe.") found = true;
        CHECK(found);
        db.close();
    }
    cleanup();
    std::println(" OK");
}

// =======================================================================
// PHASE2 Step 1: close() really closes; row+postings atomic on the
// stable store paths.
// =======================================================================
void p2_test_close_really_closes(ragger::Embedder& emb) {
    std::println("  p2_test_close_really_closes...");
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        int t = db.store_turn("close probe user", "close probe assistant text");
        CHECK(t > 0);
        // WAL is live while the connection is open.
        CHECK(fs::exists(TEMP_DB + "-wal"));
        db.close();
        // SQLite deletes the WAL only when the LAST connection closes
        // cleanly — before the fix sqlite3_close returned SQLITE_BUSY
        // (TextIndex still held statements) and the file stayed.
        CHECK(!fs::exists(TEMP_DB + "-wal"));
        CHECK(!fs::exists(TEMP_DB + "-shm"));
        db.close();  // second close is harmless
    }
    cleanup();
    std::println(" OK");
}

void p2_test_write_paths_postings(ragger::Embedder& emb) {
    std::println("  p2_test_write_paths_postings...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int t = db.store_turn("alpha beta gamma", "", "m");  // partial turn
    CHECK(t > 0);
    CHECK(db.finalize_turn(t, "delta epsilon zeta final answer"));

    ragger::DocumentChunk chunk;
    chunk.path = "/docs/p2probe.md";
    chunk.text = "theta iota kappa document body text";
    chunk.title = "p2 probe";
    int doc_id = db.store_document(chunk);
    CHECK(doc_id > 0);

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns WHERE turn_id = " + std::to_string(t)) == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns_terms WHERE turn_id = " + std::to_string(t)) > 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms WHERE document_id = " + std::to_string(doc_id)) > 0);
    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" OK");
}

void p2_test_store_turn_under_outer_begin(ragger::Embedder& emb) {
    std::println("  p2_test_store_turn_under_outer_begin...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Outer raw BEGIN on a second connection, then store_turn, then
    // COMMIT: the deferred outer BEGIN holds no write lock, so the
    // store's BEGIN IMMEDIATE runs and commits normally. (Same-connection
    // join semantics are covered by p1r_test_transaction; Backend does not
    // expose its own sqlite3* to open an outer transaction on it.)
    sqlite3* raw = p1_open_raw();
    p1r_exec_raw(raw, "BEGIN");
    int t = db.store_turn("outer tx user", "outer tx assistant reply text");
    CHECK(t > 0);
    p1r_exec_raw(raw, "COMMIT");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns WHERE turn_id = " + std::to_string(t)) == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns_terms WHERE turn_id = " + std::to_string(t)) > 0);
    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" OK");
}

// =======================================================================
// PHASE2 Step 2: backfill/rebuild must drop ALL warm caches;
// has_embeddings must see turn_summaries.
// =======================================================================
void p2_test_has_embeddings_turn_summaries(ragger::Embedder& emb) {
    std::println("  p2_test_has_embeddings_turn_summaries...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int t = db.store_turn("haseemb user", "haseemb assistant", "m",
                          /*defer_embedding=*/true);
    CHECK(t > 0);
    CHECK(!db.has_embeddings());  // nothing embedded yet
    CHECK(db.finalize_turn_summary(t, "A finalized summary sentence.", "sum-model"));
    // The ONLY embedded row is a turn_summaries row — before the fix the
    // UNION omitted that table and this returned false.
    CHECK(db.has_embeddings());
    db.close();
    cleanup();
    std::println(" OK");
}

// S38: cleanup_old_conversations deletes only OLD turns that HAVE a
// turn_summaries row (marker rows count); unsummarized turns survive.
// Failures throw; no open transaction may survive them.
void p10_cleanup_old_conversations(ragger::Embedder& emb) {
    std::println("  p10_cleanup_old_conversations...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    const std::string old_ts = "2026-01-01 00:00:00";  // far past a 1h cutoff
    const int a = db.store_turn("old a", "answer a", "", false, "", old_ts);
    const int b = db.store_turn("old b", "answer b", "", false, "", old_ts);
    const int c = db.store_turn("old c", "answer c", "", false, "", old_ts);
    const int d = db.store_turn("recent d", "answer d");
    CHECK(a > 0 && b > 0 && c > 0 && d > 0);
    CHECK(db.finalize_turn_summary(a, "summary a", "summarizer-model"));
    CHECK(db.mark_turn_summarized(b, "trivial-skip-model"));  // marker row
    // c deliberately gets NO summary row.
    CHECK(db.finalize_turn_summary(d, "summary d", "summarizer-model"));

    CHECK(db.cleanup_old_conversations(1.0f) == 2);  // A and B only

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM turns WHERE turn_id IN ({},{})", a, b)) == 0);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM turns WHERE turn_id IN ({},{})", c, d)) == 2);
    // The FK (turn_summaries.turn_id ON DELETE SET NULL) keeps the summaries.
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM turn_summaries WHERE turn_id IS NULL AND text = 'summary a'") == 1);
    // foreign_key_check: no violation rows.
    {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(raw, "PRAGMA foreign_key_check", -1, &st, nullptr) == SQLITE_OK);
        CHECK(sqlite3_step(st) == SQLITE_DONE);
        sqlite3_finalize(st);
    }

    // Injection (8.0.8): a failing DELETE throws and deletes nothing.
    // C must match the DELETE first, so mark it summarized (the first pass
    // correctly skipped it).
    CHECK(db.mark_turn_summarized(c, "trivial-skip-model"));
    char* err = nullptr;
    CHECK(sqlite3_exec(raw,
        "CREATE TRIGGER p10_boom_del BEFORE DELETE ON turns "
        "BEGIN SELECT RAISE(ABORT,'p10 injected'); END;",
        nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);
    bool threw = false;
    try {
        db.cleanup_old_conversations(1.0f);
    } catch (const std::runtime_error& e) {
        threw = true;
        CHECK(std::string(e.what()).find("p10 injected") != std::string::npos);
    }
    CHECK(threw);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM turns WHERE turn_id = {}", c)) == 1);
    // No open transaction survives the throw (second-connection write works).
    err = nullptr;
    CHECK(sqlite3_exec(raw, "INSERT INTO models (name) VALUES ('p10_cleanup_after_throw')",
                       nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);
    err = nullptr;
    CHECK(sqlite3_exec(raw, "DROP TRIGGER p10_boom_del", nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);
    // After the trigger is gone the marked turn ages out normally.
    CHECK(db.cleanup_old_conversations(1.0f) == 1);
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// S37: finalize_turn_summary/mark_turn_summarized distinguish UNIQUE (false,
// "already summarized") from any other INSERT failure (throw), and an empty
// model name on mark_turn_summarized is invalid_argument, not a silent false.
void p10_summary_insert_failures(ragger::Embedder& emb) {
    std::println("  p10_summary_insert_failures...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    const int t1 = db.store_turn("q one", "a one");

    // (a) Duplicate finalize: false, row count unchanged.
    CHECK(db.finalize_turn_summary(t1, "first summary text.", "summarizer-model"));
    CHECK(!db.finalize_turn_summary(t1, "second summary text.", "summarizer-model"));

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = {}", t1)) == 1);

    // (b) Injected failure (8.0.8 recipe): throws with the trigger's message.
    char* err = nullptr;
    CHECK(sqlite3_exec(raw,
        "CREATE TRIGGER p10_boom BEFORE INSERT ON turn_summaries "
        "BEGIN SELECT RAISE(ABORT,'p10 injected'); END;",
        nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);
    const int t2 = db.store_turn("q two", "a two");
    bool threw = false;
    try {
        db.finalize_turn_summary(t2, "boom text", "summarizer-model");
    } catch (const std::runtime_error& e) {
        threw = true;
        CHECK(std::string(e.what()).find("p10 injected") != std::string::npos);
    }
    CHECK(threw);
    try {
        db.mark_turn_summarized(t2, "some-model");
        threw = false;
    } catch (const std::runtime_error& e) {
        threw = true;
        CHECK(std::string(e.what()).find("p10 injected") != std::string::npos);
    }
    CHECK(threw);
    // No open transaction may survive the throw: the backend's handle is not
    // reachable from here, so the proxy is a write on this second connection —
    // it would hit SQLITE_BUSY while the backend still held the write lock.
    err = nullptr;
    const int rc = sqlite3_exec(raw,
        "INSERT INTO models (name) VALUES ('p10_after_throw')", nullptr, nullptr, &err);
    if (rc != SQLITE_OK)
        std::println("  second-connection write failed: {} ({})", rc, err ? err : "?");
    CHECK(rc == SQLITE_OK);
    sqlite3_free(err);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = {}", t2)) == 0);
    err = nullptr;
    CHECK(sqlite3_exec(raw, "DROP TRIGGER p10_boom", nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);

    // (c) Empty model name: invalid_argument, and nothing was marked.
    threw = false;
    try {
        db.mark_turn_summarized(t2, "");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(!db.turn_summary_exists(t2));
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// S36: deferred writes leave embedding AND embedding_version NULL through
// the single bind_embedding*/empty-vector choke point (the explicit
// defer_embedding bind branches are redundant). Passes before and after the
// refactor; the non-deferred twins assert the embedder path stays non-NULL.
void p10_defer_binds_null(ragger::Embedder& emb) {
    std::println("  p10_defer_binds_null...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const int kid  = std::stoi(db.store("deferred note", {}, /*defer_embedding=*/true));
    const int kid2 = std::stoi(db.store("embedded note", {}, /*defer_embedding=*/false));
    const int did  = db.store_decision("deferred decision", "current", "", "", /*defer_embedding=*/true);
    ragger::DocumentChunk c1; c1.text = "deferred chunk"; c1.path = "/p/d1.md";
    c1.chunk_index = 1;
    ragger::DocumentChunk c2; c2.text = "embedded chunk"; c2.path = "/p/d2.md";
    c2.chunk_index = 1;
    const int doc1 = db.store_document(c1, /*defer_embedding=*/true);
    const int doc2 = db.store_document(c2, /*defer_embedding=*/false);

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NULL AND embedding_version IS NULL FROM knowledge WHERE knowledge_id = {}", kid)) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NOT NULL AND embedding_version IS NOT NULL FROM knowledge WHERE knowledge_id = {}", kid2)) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NULL AND embedding_version IS NULL FROM knowledge WHERE knowledge_id = {}", did)) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NULL AND embedding_version IS NULL FROM documents WHERE document_id = {}", doc1)) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NOT NULL AND embedding_version IS NOT NULL FROM documents WHERE document_id = {}", doc2)) == 1);
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// S35(1): fetch_knowledge_row's COALESCE subqueries always yield '' — the
// session/session_name/model keys must be present even when the row has no
// session or model (pin for the dead is_null-branch removal: JSON identical
// before and after).
void p10_knowledge_get_absent_keys(ragger::Embedder& emb) {
    std::println("  p10_knowledge_get_absent_keys...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::KnowledgeInput a;
    a.kind = "note";
    a.text = "p10 note without session or model";
    int a_id = db.store_knowledge(a).id;
    ragger::AgentGetQuery g;
    g.ids = {"knowledge:" + std::to_string(a_id)};
    auto item = db.agent_get(g)["items"][0];
    CHECK(item.contains("session"));
    CHECK(item["session"] == "");
    CHECK(item.contains("model"));
    CHECK(item["model"] == "");
    CHECK(item.contains("trigger"));
    CHECK(item["trigger"].is_null());
    db.close();
    cleanup();
    std::println(" OK");
}

// S45 pin: search_text_only must score documents through the SAME lambda as
// the other corpora (was a hand-expanded copy). Records exact ids/sources/
// scores from the unfixed code; must be identical after the refactor.
void p10_text_only_one_path(ragger::Embedder& emb) {
    std::println("  p10_text_only_one_path...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int kn = db.store("zephyr quoll knowledge note").empty() ? -1 : 1;
    CHECK(kn > 0);
    int t = p2_store_turn(db, "s45-guid", "zephyr quoll question");
    CHECK(db.finalize_turn_summary(t, "zephyr quoll summary gist", "s45-model"));
    ragger::DocumentChunk dc;
    dc.text = "zephyr quoll document chunk";
    dc.path = "s45-src";
    int doc = db.store_document(dc, false);
    CHECK(doc > 0);

    auto resp = db.search_text_only("zephyr quoll", 10);
    std::vector<std::string> got;
    for (const auto& r : resp.results) {
        std::println("    hit id={} source={} score={:.8f} ts='{}'", r.id,
                     r.metadata.value("source", "?"), r.bm25_score, r.timestamp);
        got.push_back(r.metadata.value("source", "?"));
    }
    // Recorded from the UNFIXED code (hand-expanded documents branch):
    const std::vector<std::string> want{"document", "knowledge", "turn_summary"};
    CHECK(got == want);
    for (const auto& r : resp.results) {
        CHECK(r.bm25_score == 0.0f);   // idf tie: term in all 3 docs
        // Rendered local datetime, never a bare epoch (9B step 2).
        CHECK(r.timestamp.size() == 19 && r.timestamp[4] == '-' && r.timestamp[13] == ':');
    }

    db.close();
    cleanup();
    std::println(" OK");
}

// S44 pin: turn_meta + insert_turn_summary refactor. A marker row (via
// mark_turn_summarized) and a real row (via finalize_turn_summary) must get
// the same turn_datetime and session_id as their source turn, before and
// after the refactor (pure: no behavior change).
void p10_turn_meta_columns(ragger::Embedder& emb) {
    std::println("  p10_turn_meta_columns...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    const std::string guid = "s44-guid";
    int t1 = p2_store_turn(db, guid, "s44 alpha text");
    int t2 = p2_store_turn(db, guid, "s44 beta text");

    CHECK(db.finalize_turn_summary(t1, "s44 summary one", "s44-model"));
    CHECK(db.mark_turn_summarized(t2, "s44-model"));

    // Read back directly: turn_datetime == source turn's created_at;
    // session_id == the session row for guid.
    auto q = [&](const char* sql, int turn) {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(raw, sql, -1, &st, nullptr);
        sqlite3_bind_int(st, 1, turn);
        int64_t dt = -1; int sid = -1;
        if (sqlite3_step(st) == SQLITE_ROW) { dt = sqlite3_column_int64(st, 0); sid = sqlite3_column_int(st, 1); }
        sqlite3_finalize(st); sqlite3_close(raw);
        return std::pair<int64_t,int>{dt, sid};
    };
    auto src = [&](int turn) {
        return q("SELECT created_at, session_id FROM turns WHERE turn_id = ?", turn);
    };
    auto sum = [&](int turn) {
        return q("SELECT turn_datetime, session_id FROM turn_summaries WHERE turn_id = ?", turn);
    };
    auto s1 = src(t1), r1 = sum(t1);
    auto s2 = src(t2), r2 = sum(t2);
    CHECK(r1.first  == s1.first);   // turn_datetime matches
    CHECK(r1.second == s1.second);  // session_id matches
    CHECK(r2.first  == s2.first);
    CHECK(r2.second == s2.second);

    db.close();
    cleanup();
    std::println(" OK");
}

void p10_unknown_session_guid(ragger::Embedder& emb) {
    std::println("  p10_unknown_session_guid...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    // One known turn so the DB has a session; the queries use an unknown GUID.
    p2_store_turn(db, "s43-known-guid", "s43 turn");

    // agent_get mode 2: unknown GUID -> empty page, no error.
    ragger::AgentGetQuery g;
    g.session = "s43-no-such-guid";
    auto hist = db.agent_get(g);
    CHECK(hist["turns"].is_array());
    CHECK(hist["turns"].empty());
    CHECK(hist["next_after"].is_null());

    // agent_timeline: unknown GUID -> no items, no error (turn granularity).
    ragger::TimelineQuery t;
    t.granularity = "turn";
    t.session = "s43-no-such-guid";
    auto tl = db.agent_timeline(t);
    CHECK(tl["items"].is_array());
    CHECK(tl["items"].empty());
    CHECK(tl["next_cursor"].is_null());

    // agent_search session filter with an unknown GUID matches nothing.
    ragger::AgentSearchQuery aq;
    aq.query = "s43 turn";
    aq.session = "s43-no-such-guid";
    auto hits = db.agent_search(aq);
    CHECK(hits.empty());

    // Ruling 2026-11-05 (Reid): "matches nothing" is the contract. Same for
    // the knowledge granularity (LR S79); an unknown exclude_session still
    // excludes nothing (results unchanged).
    ragger::TimelineQuery kt;
    kt.granularity = "knowledge";
    kt.session = "s43-no-such-guid";
    CHECK(db.agent_timeline(kt)["items"].empty());

    ragger::KnowledgeInput k;
    k.kind = "note";
    k.text = "s43 knowledge note about kites";
    db.store_knowledge(k);
    ragger::AgentSearchQuery eq;
    eq.query = "s43 knowledge note kites";
    eq.limit = 20;
    const size_t base = db.agent_search(eq).size();
    CHECK(base > 0);
    eq.exclude_session = "s43-no-such-guid";
    CHECK(db.agent_search(eq).size() == base);

    db.close();
    cleanup();
    std::println(" OK");
}

void p10_embeddable_counts(ragger::Embedder& emb) {
    std::println("  p10_embeddable_counts...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // turn_summaries fixture: 2 rows with text, 1 marker ('' = skipped on
    // purpose), 1 legacy NULL-text row. embed_tables() skips the last two;
    // count_embeddable_rows() must agree (B m7).
    int t1 = db.store_turn("q one", "a one");
    int t2 = db.store_turn("q two", "a two");
    int t3 = db.store_turn("q three", "a three");
    int t4 = db.store_turn("q four", "a four");
    CHECK(db.finalize_turn_summary(t1, "summary one", "test-model"));
    CHECK(db.finalize_turn_summary(t2, "summary two", "test-model"));
    CHECK(db.mark_turn_summarized(t3, "test-model"));
    CHECK(db.finalize_turn_summary(t4, "summary four", "test-model"));
    {   // Reshape one summary into the legacy NULL-text form.
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        char* err = nullptr;
        const std::string sql =
            "UPDATE turn_summaries SET text = NULL WHERE text = 'summary four'";
        CHECK(sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, &err) == SQLITE_OK);
        sqlite3_free(err);
        sqlite3_close(raw);
    }
    CHECK(db.count_embeddable_rows("turn_summaries") == 2);

    CHECK(!db.store("a knowledge note").empty());
    CHECK(db.count_embeddable_rows("knowledge") == 1);

    ragger::DocumentChunk chunk;
    chunk.text = "a document chunk";
    chunk.path = "/p/doc.md";
    chunk.title = "Doc";
    CHECK(db.store_document(chunk) > 0);
    CHECK(db.count_embeddable_rows("documents") == 1);

    // A full rebuild touches exactly the embeddable rows, and afterwards the
    // counts still agree (rebuild does not change text).
    const int rebuilt = db.rebuild_embeddings(emb);
    CHECK(rebuilt == db.count_embeddable_rows("all"));

    db.close();
    cleanup();
    std::println(" OK");
}

void p2_test_backfill_invalidates_dec_cache(ragger::Embedder& emb) {
    std::println("  p2_test_backfill_invalidates_dec_cache...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    const std::string stored = "The cat sat on the mat.";
    const std::string query  = "A feline rested upon a rug";  // no shared stems
    int d = db.store_decision(stored, "current", "", "", /*defer_embedding=*/true);
    CHECK(d > 0);

    // Warm the decision cache while the row still has no vector.
    auto before = db.search(query, 5, 0.0f);
    bool found_before = false;
    for (const auto& r : before.results)
        if (r.text == stored && r.score > 0.2f) found_before = true;
    CHECK(!found_before);

    int n = db.backfill_embeddings(emb);
    CHECK(n >= 1);

    // Same connection, same caches: only a cache drop makes the freshly
    // embedded decision findable semantically.
    auto after = db.search(query, 5, 0.0f);
    bool found_after = false;
    for (const auto& r : after.results)
        if (r.text == stored && r.score > 0.2f) found_after = true;
    CHECK(found_after);
    // Insights: skipped — the public store_insight() requires evidence and
    // exposes no defer_embedding knob, so the deferral scenario cannot be
    // built through the public API (noted in REPORT-P2).
    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE2 Step 6: the db-version gate must throw, not std::exit (B §1.11).
void p2_test_db_version_gate(ragger::Embedder& emb) {
    std::println("  p2_test_db_version_gate...");
    const std::string path = "/tmp/ragger_test_dbversion_gate.db";
    std::filesystem::remove(path);
    {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(path.c_str(), &raw) == SQLITE_OK);
        char* err = nullptr;
        const char* sql =
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "INSERT INTO settings (key, value) VALUES ('db_version', '0.99');";
        CHECK(sqlite3_exec(raw, sql, nullptr, nullptr, &err) == SQLITE_OK);
        sqlite3_free(err);
        sqlite3_close(raw);
    }
    bool threw = false;
    try {
        ragger::sqlite::Backend db(emb, path);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string msg = e.what();
        CHECK(msg.find("schema version mismatch") != std::string::npos);
        CHECK(msg.find("0.99") != std::string::npos);
        // The old text named a script that does not exist; the new text
        // must not.
        CHECK(msg.find("migrate_to_db") == std::string::npos);
    }
    CHECK(threw);  // construction must reject, not exit the process
    std::filesystem::remove(path);
    std::println(" ok");
}

// PHASE3 Step 3: the agent-layer set derived from the registry must remain
// exactly the layers agent_search validates today (PHASE5 adds `episode`;
// PHASE8 removes `summary` with the summaries table).
void p3_test_known_layers() {
    std::println("  p3_test_known_layers...");
    const std::set<std::string> expected = {
        "turn", "turn_summary", "document", "knowledge", "episode", "session"};
    std::set<std::string> got;
    for (const auto& t : ragger::sqlite::kTextTables)
        if (t.layer && *t.layer) got.insert(t.layer);
    CHECK(got == expected);
    std::println(" ok");
}

// PHASE2 Step 6: DB-only backend (null embedder) throws on embedder use
// instead of dereferencing null (B §1.17).
void p2_test_require_embedder(ragger::Embedder& emb) {
    std::println("  p2_test_require_embedder...");
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        db.store("hello there general kenobi");
        db.store_turn("hello there", "general kenobi", "test-model");
        db.close();
    }
    {
        ragger::sqlite::Backend db_only(TEMP_DB);
        // Write path: null embedder -> require_embedder() throws.
        bool threw = false;
        try {
            db_only.store_turn("another", "turn", "test-model");
        } catch (const std::runtime_error& e) {
            threw = true;
            CHECK(std::string(e.what()) == "embedder not available");
        }
        CHECK(threw);
        // Read path: corpus is non-empty (one embedded turn above), so
        // search reaches the embedder gate and throws the readiness
        // message rather than feeding Eigen an empty vector.
        threw = false;
        try {
            db_only.search("hello", 5, 0.0f);
        } catch (const std::runtime_error& e) {
            threw = true;
            CHECK(std::string(e.what()) ==
                  "semantic search unavailable: embedder not ready");
        }
        CHECK(threw);
        db_only.close();
    }
    cleanup();
    std::println(" ok");
}

// PHASE3 Step 4: one status vocabulary — lists pinned to their pre-refactor
// membership; transition_ok pinned to the agent_update table (moved
// verbatim), including the pairs the spec calls out.
void p3s_test_status_vocabulary() {
    std::println("  p3s_test_status_vocabulary...");
    namespace st = ragger::sqlite::status;
    CHECK(st::in_list_sql(st::kSearchActive) == "'current','provisional'");
    CHECK(st::in_list_sql(st::kCacheLoaded) == "'current','provisional','roadmap'");
    CHECK(st::in_list_sql(st::kCreatable) == "'current','provisional','pending'");
    CHECK(st::in_list_sql(st::kMigrationKnown) ==
          "'current','provisional','pending','superseded','retired','roadmap','deprecated'");
    CHECK(st::in_list_sql(st::kAgentSettable) ==
          "'current','provisional','pending','retired'");

    // transition_ok — verbatim table semantics.
    CHECK(st::transition_ok("pending", "current"));
    CHECK(st::transition_ok("pending", "provisional"));
    CHECK(st::transition_ok("pending", "retired"));
    CHECK(st::transition_ok("provisional", "current"));
    CHECK(st::transition_ok("provisional", "retired"));
    CHECK(!st::transition_ok("provisional", "pending"));
    CHECK(st::transition_ok("current", "provisional"));
    CHECK(st::transition_ok("current", "retired"));
    CHECK(!st::transition_ok("current", "superseded"));   // superseded only via supersedes
    CHECK(!st::transition_ok("current", "current"));     // strict: same-status not listed
    CHECK(st::transition_ok("roadmap", "current"));
    CHECK(st::transition_ok("roadmap", "provisional"));
    CHECK(!st::transition_ok("superseded", "current"));  // terminal
    CHECK(!st::transition_ok("retired", "current"));     // terminal
    CHECK(!st::transition_ok("deprecated", "current"));  // not in table
    CHECK(!st::transition_ok("nonsense", "current"));
    std::println(" ok");
}

// S10 (v0.17 CR, B m1 / Q8 / round-2 ruling "roadmap dropped"): every
// offered status must be reachable by an allowed transition, and every
// legacy-known non-terminal status must be able to leave. NOTE: 'pending'
// is creation-only (kCreatable) — no transition edge targets it; the
// review's "every kAgentSettable value is the target of some allowed edge"
// claim is false for 'pending' under the verbatim table (flagged as a
// review error; asserted honestly here).
void p10_test_agent_settable_targets() {
    std::println("  p10_test_agent_settable_targets...");
    namespace st = ragger::sqlite::status;
    for (const auto* to : {"current", "provisional", "retired"}) {
        bool any = false;
        for (const auto& from : st::kMigrationKnown)
            if (st::transition_ok(from, to)) { any = true; break; }
        CHECK(any);
    }
    CHECK(!st::transition_ok("pending", "pending"));   // creation-only
    // Every legacy-known value that is not terminal has >= 1 allowed target.
    for (const auto* from : {"current", "provisional", "pending", "roadmap"}) {
        bool any = false;
        for (const auto& to : st::kMigrationKnown)
            if (st::transition_ok(from, to)) { any = true; break; }
        CHECK(any);
    }
    // Terminal: retired and superseded have no outgoing edge; deprecated is
    // not in the table at all (also terminal).
    for (const auto& to : st::kMigrationKnown) {
        CHECK(!st::transition_ok("retired", to));
        CHECK(!st::transition_ok("superseded", to));
        CHECK(!st::transition_ok("deprecated", to));
    }
    std::println(" ok");
}

// LR S60: the CLI's decision status path is agent_update on knowledge:<id>
// (the parallel CLI-only setter chain is gone). Same discipline:
// unknown word -> invalid_argument; superseded -> invalid_argument;
// same-status -> invalid_argument (not a listed edge); unknown id ->
// invalid_argument (not false).
void lr60_test_decision_status_via_agent_update(ragger::Embedder& emb) {
    std::println("  lr60_test_decision_status_via_agent_update...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int id = db.store_decision("lr60 vocabulary decision", "current", "", "", true);
    auto update = [&](const char* to) {
        ragger::UpdateRequest req;
        req.id = "knowledge:" + std::to_string(id);
        req.status = to;
        db.agent_update(req);
    };
    auto throws_iav = [&](const char* to) {
        try { update(to); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(throws_iav("roadmap"));      // legacy word: not offered
    CHECK(throws_iav("nonsense"));     // unknown word
    CHECK(throws_iav("superseded"));   // never reachable here
    CHECK(throws_iav("current"));      // same-status is not a listed edge
    update("provisional");
    {
        sqlite3* raw = p1_open_raw();
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(raw, "SELECT status FROM knowledge WHERE knowledge_id = ?", -1, &s, nullptr);
        sqlite3_bind_int(s, 1, id);
        std::string got = "<?>";
        if (sqlite3_step(s) == SQLITE_ROW) got = reinterpret_cast<const char*>(sqlite3_column_text(s, 0));
        sqlite3_finalize(s);
        sqlite3_close(raw);
        CHECK(got == "provisional");
    }
    update("retired");                            // legal edge
    CHECK(throws_iav("current"));                 // retired is terminal
    {
        ragger::UpdateRequest req;
        req.id = "knowledge:999999";              // no such id -> throws
        req.status = "current";
        bool threw = false;
        try { db.agent_update(req); }
        catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()).find("not found") != std::string::npos);
        }
        CHECK(threw);
    }
    db.close(); cleanup();
    std::println(" ok");
}

// PHASE3 Step 5 pin: one row per cached table -> search must surface each
// with the SAME metadata keys it carried pre-refactor (captured from the
// five hand-written loaders). The single load_cache() must not change any
// key, value, or the set of rows that participate in search.
void p3_cache_pin(ragger::Embedder& emb) {
    std::println("  p3_cache_pin...");
    try {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("cachepin summary sentence");                       // summaries
    ragger::DocumentChunk chunk;
    chunk.text  = "cachepin document chunk text";
    chunk.title = "PinDocTitle";
    chunk.path  = "/cachepin/doc.txt";
    db.store_document(chunk, /*defer_embedding=*/false);          // documents
    db.store_decision("cachepin decision ruling", "current", "", "",
                      /*defer_embedding=*/false);                 // decisions
    int t = db.store_turn("cachepin user", "cachepin assistant", "m",
                          /*defer_embedding=*/false, "cachepin-session-guid");
    CHECK(t > 0);
    CHECK(db.finalize_turn_summary(t, "cachepin turn summary", "sum-model"));

    ragger::KnowledgeInput in;
    in.kind = "insight";
    in.text     = "cachepin insight ruling";
    {   // LR S80: episode evidence over the cachepin turn (no extra rows).
        sqlite3* raw = p1_open_raw();
        const int ts = static_cast<int>(p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
        const long long when = p1_scalar_int(raw, std::format(
            "SELECT created_at FROM turns WHERE turn_id = {}", t));
        sqlite3_close(raw);
        ragger::EpisodeClose ec;
        ec.text = "cachepin episode";
        ec.model_name = "sum-model";
        ec.session_guid = "cachepin-session-guid";
        ec.span_start = when;
        ec.span_end   = when;
        ec.member_turn_summary_ids = {ts};
        in.evidence = {db.store_episode_close(ec)};
    }
    db.store_knowledge(in);                                         // insights

    auto resp = db.search("cachepin", 50, 0.0f);
    std::map<std::string, std::set<std::string>> keys_by_source;
    std::map<std::string, int> count_by_source;
    for (const auto& h : resp.results) {
        const std::string src = h.metadata.value("source", "?");
        count_by_source[src]++;
        for (auto& [k, v] : h.metadata.items()) keys_by_source[src].insert(k);
    }
    for (const auto& [src, keys] : keys_by_source) {
        std::string joined;
        for (const auto& k : keys) { if (!joined.empty()) joined += ","; joined += k; }
        std::println("    {} x{} [{}]", src, count_by_source[src], joined);
    }

    // Pinned (PHASE8: the note rides the knowledge corpus — one row each of
    // note, decision, insight land under "knowledge").
    CHECK(count_by_source["document"] == 1);
    CHECK(count_by_source["knowledge"] == 3);
    CHECK(count_by_source["turn_summary"] == 1);
    CHECK(keys_by_source["document"] == std::set<std::string>({"source", "title"}));
    CHECK(keys_by_source["knowledge"] == std::set<std::string>({"source", "kind", "status"}));
    CHECK(keys_by_source["turn_summary"] ==
          std::set<std::string>({"source", "turn_id", "session_id", "datetime"}));
    db.close();
    cleanup();
    std::println(" OK");
    } catch (const std::exception& e) {
        std::println("  p3_cache_pin EXCEPTION: {}", e.what());
        throw;
    }
}

// PHASE3 Step 6: the shared write-back updates the right table/pk and
// invalidates the matching cache; a missing id returns false.
void p3_test_update_embedding(ragger::Embedder& emb) {
    std::println("  p3_test_update_embedding...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int d = db.store_decision("updatemb deferred ruling", "current", "", "",
                              /*defer_embedding=*/true);
    CHECK(d > 0);
    std::vector<float> v(static_cast<size_t>(ragger::config().embedding_dimensions), 0.0f);
    v[0] = 1.0f;
    CHECK(db.update_decision_embedding(d, v));
    CHECK(!db.update_decision_embedding(d + 999, v));
    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT embedding IS NOT NULL FROM knowledge WHERE knowledge_id = {}", d)) == 1);
    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" OK");
}


// PHASE4 Step 4: sessions.started_on/ended_on are the derived turn span.
static std::string p4_span(sqlite3* raw, const std::string& guid) {
    std::string sql = "SELECT ifnull(started_on,'NULL') || '-' || "
                      "ifnull(ended_on,'NULL') FROM sessions WHERE guid = '" + guid + "'";
    return p1_scalar_text(raw, sql);
}

void p4_session_span(ragger::Embedder& emb) {
    std::println("  p4_session_span...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // (a) Out-of-order source timestamps widen to MIN/MAX, not insert order.
    db.store_turn("span b", "", "", false, "span-s", "2026-04-02 10:00:00");
    db.store_turn("span a", "", "", false, "span-s", "2026-04-01 09:00:00");
    db.store_turn("span c", "", "", false, "span-s", "2026-04-03 11:00:00");

    // (b) A turn with no session: no error, no session row created.
    db.store_turn("orphan turn", "", "", false, "", "");

    // (c) A session with no turns is NULL/NULL (raw insert: nothing else
    // creates a session without turns on the live path).
    {
        sqlite3* r2 = p1_open_raw();
        CHECK(sqlite3_exec(r2, "INSERT INTO sessions (guid) VALUES ('span-none')",
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(r2);
    }

    sqlite3* raw = p1_open_raw();
    CHECK(p4_span(raw, "span-s") ==
          p1_scalar_text(raw, "SELECT (SELECT MIN(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s')) || '-' || "
                              "(SELECT MAX(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s'))"));
    CHECK(p4_span(raw, "span-none") == "NULL-NULL");

    // Dedup path: re-storing 'span b' with a later timestamp updates the
    // row's created_at and widens ended_on (touch on the dedup branch).
    db.store_turn("span b", "", "", false, "span-s", "2026-04-05 08:00:00");
    CHECK(p4_span(raw, "span-s") ==
          p1_scalar_text(raw, "SELECT (SELECT MIN(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s')) || '-' || "
                              "(SELECT MAX(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s'))"));
    CHECK(p1_scalar_text(raw, "SELECT ended_on FROM sessions WHERE guid='span-s'") ==
          p1_scalar_text(raw, "SELECT MAX(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s')"));
    sqlite3_close(raw);

    // (d) update_turn_meta moving the ONLY 'span-none'-adjacent turn... use
    // span-s: move one turn to a new session; both spans are recomputed.
    {
        sqlite3* r2 = p1_open_raw();
        const std::string moved = p1_scalar_text(r2,
            "SELECT turn_id FROM turns WHERE user_text = 'span a'");
        sqlite3_close(r2);
        CHECK(db.update_turn_meta(std::stoi(moved), "2026-04-10 12:00:00", "span-t"));
    }
    raw = p1_open_raw();
    CHECK(p4_span(raw, "span-t") ==
          p1_scalar_text(raw, "SELECT created_at || '-' || created_at FROM turns "
                              "WHERE session_id = (SELECT session_id FROM sessions WHERE guid='span-t')"));
    CHECK(p4_span(raw, "span-s") ==
          p1_scalar_text(raw, "SELECT (SELECT MIN(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s')) || '-' || "
                              "(SELECT MAX(created_at) FROM turns WHERE session_id = "
                              "(SELECT session_id FROM sessions WHERE guid='span-s'))"));
    // The moved turn left span-s at a later timestamp than before.
    CHECK(p1_scalar_text(raw, "SELECT started_on FROM sessions WHERE guid='span-s'") > "1774000000");
    sqlite3_close(raw);

    // Move the last remaining... empty the old session completely:
    // span-none never had turns; verify a session emptied of turns goes
    // NULL/NULL by moving every span-s turn out is overkill — instead move
    // the sole turn of a two-turn session out and check shrink.
    {
        db.store_turn("solo turn", "", "", false, "span-u", "2026-04-06 06:00:00");
        sqlite3* r2 = p1_open_raw();
        const std::string tid = p1_scalar_text(r2,
            "SELECT turn_id FROM turns WHERE user_text = 'solo turn'");
        sqlite3_close(r2);
        CHECK(db.update_turn_meta(std::stoi(tid), "2026-04-07 07:00:00", "span-s"));
        raw = p1_open_raw();
        CHECK(p4_span(raw, "span-u") == "NULL-NULL");   // emptied -> NULL
        sqlite3_close(raw);
    }

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE5 E.1/E.2: store_episode_close contract + episode reads.
static std::vector<int> p5_seed_members(ragger::sqlite::Backend& db,
                                        const std::string& guid,
                                        const std::vector<std::string>& stamps) {
    std::vector<int> ids;
    for (size_t i = 0; i < stamps.size(); ++i) {
        const int t = db.store_turn("p5 q" + guid + std::to_string(i),
                                    "p5 answer " + std::string(60, 'c') + guid + std::to_string(i),
                                    "test-model", false, guid, stamps[i]);
        CHECK(db.finalize_turn_summary(t, "p5 summary " + guid + std::to_string(i),
                                       "summarizer-model"));
        sqlite3* raw = p1_open_raw();
        ids.push_back(static_cast<int>(p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t))));
        sqlite3_close(raw);
    }
    return ids;
}

void p5_test_episode_close(ragger::Embedder& emb) {
    std::println("  p5_test_episode_close...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string ga = "p5ec-a", gb = "p5ec-b";
    auto m = p5_seed_members(db, ga, {"2026-05-01 10:00:00", "2026-05-01 10:05:00",
                                      "2026-05-01 10:10:00"});
    CHECK(m.size() == 3);

    // (a) happy path: episode row + membership + terms + last_episode_end.
    ragger::EpisodeClose ec;
    ec.text = "p5ec first episode";
    ec.model_name = "summarizer-model";
    ec.session_guid = ga;
    ec.span_start = p5_epoch("2026-05-01 10:00:00");
    ec.span_end   = p5_epoch("2026-05-01 10:10:00");
    ec.member_turn_summary_ids = m;
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 1);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE episode_id = "
                                 + std::to_string(ep)) == 3);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes_terms WHERE episode_id = "
                                 + std::to_string(ep)) > 0);
        CHECK(p1_scalar_text(raw, "SELECT span_start || '-' || span_end FROM episodes "
                                  "WHERE episode_id = " + std::to_string(ep)) ==
              std::to_string(ec.span_start) + "-" + std::to_string(ec.span_end));
        sqlite3_close(raw);
    }
    CHECK(db.last_episode_end(ga) == "2026-05-01 10:10:00");

    // (S41.4) Episode cache metadata: `layer` was dead (no consumer anywhere);
    // `level` stays (stats_logger's collection_of fallback chain reads it).
    {
        auto resp = db.search("p5ec first episode", 10, 0.0f);
        bool saw_episode = false;
        for (const auto& r : resp.results) {
            if (r.metadata.value("source", "") != "episode") continue;
            saw_episode = true;
            CHECK(r.metadata.value("level", "") == "episode");
            CHECK(!r.metadata.contains("layer"));
        }
        CHECK(saw_episode);
    }

    // (c) member already in an episode -> throws, nothing written.
    {
        ragger::EpisodeClose ec2 = ec;
        ec2.text = "p5ec overlap attempt";
        ec2.span_start = p5_epoch("2026-05-01 09:00:00");   // different key, same members
        ec2.span_end   = p5_epoch("2026-05-01 10:10:00");
        bool threw = false;
        try { db.store_episode_close(ec2); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 1);   // rolled back
        sqlite3_close(raw);
    }

    // (b) member from ANOTHER session -> throws; no episode row, no member touched.
    auto n = p5_seed_members(db, gb, {"2026-05-01 10:07:00"});
    {
        ragger::EpisodeClose ec3;
        ec3.text = "p5ec foreign member";
        ec3.model_name = "summarizer-model";
        ec3.session_guid = ga;
        ec3.span_start = p5_epoch("2026-05-01 11:00:00");
        ec3.span_end   = p5_epoch("2026-05-01 11:00:00");
        ec3.member_turn_summary_ids = {m[0], n[0]};   // m[0] already assigned too
        bool threw = false;
        try { db.store_episode_close(ec3); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 1);
        CHECK(p1_scalar_text(raw, "SELECT ifnull(episode_id,'NULL') FROM turn_summaries "
                                  "WHERE turn_summary_id = " + std::to_string(n[0])) == "NULL");
        sqlite3_close(raw);
    }

    // (d) marker row (text = '') as member -> throws.
    {
        const int t = db.store_turn("p5ec marker q", "p5ec marker answer " + std::string(60, 'd'),
                                    "test-model", false, ga, "2026-05-01 12:00:00");
        CHECK(db.mark_turn_summarized(t, "bad"));
        sqlite3* raw = p1_open_raw();
        const int marker = static_cast<int>(p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
        sqlite3_close(raw);
        ragger::EpisodeClose ec4;
        ec4.text = "p5ec marker member";
        ec4.model_name = "summarizer-model";
        ec4.session_guid = ga;
        ec4.span_start = p5_epoch("2026-05-01 12:00:00");
        ec4.span_end   = p5_epoch("2026-05-01 12:00:00");
        ec4.member_turn_summary_ids = {marker};
        bool threw = false;
        try { db.store_episode_close(ec4); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        sqlite3* r2 = p1_open_raw();
        CHECK(p1_scalar_int(r2, "SELECT COUNT(*) FROM episodes") == 1);
        sqlite3_close(r2);
    }

    // (e) second close with the SAME (session, span_start) -> returns 0,
    // one episode row (unique key, "already closed", not an error).
    {
        ragger::EpisodeClose ec5;
        ec5.text = "p5ec duplicate span";
        ec5.model_name = "summarizer-model";
        ec5.session_guid = ga;
        ec5.span_start = ec.span_start;
        ec5.span_end   = ec.span_end;
        ec5.member_turn_summary_ids = {n[0]};
        CHECK(db.store_episode_close(ec5) == 0);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 1);
        // Unassigned: n[0] (the duplicate attempt rolled back) + the (d) marker row.
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE episode_id IS NULL") == 2);
        sqlite3_close(raw);
    }

    // Validation rejects: empty text / empty members / inverted span.
    {
        ragger::EpisodeClose bad;
        bad.session_guid = ga; bad.span_start = 1; bad.span_end = 2;
        bad.member_turn_summary_ids = {m[1]};
        bool threw = false;
        try { db.store_episode_close(bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);   // empty text
        bad.text = "x"; bad.member_turn_summary_ids.clear();
        threw = false;
        try { db.store_episode_close(bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);   // no members
        bad.member_turn_summary_ids = {m[1]}; bad.span_start = 5; bad.span_end = 4;
        threw = false;
        try { db.store_episode_close(bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);   // inverted span
    }

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE9B step 1: store_episode_close must never create a session.
void p9b_test_episode_close_unknown_session(ragger::Embedder& emb) {
    std::println("  p9b_test_episode_close_unknown_session...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    sqlite3* raw = p1_open_raw();
    const int sessions_before = static_cast<int>(p1_scalar_int(raw, "SELECT COUNT(*) FROM sessions"));
    sqlite3_close(raw);

    ragger::EpisodeClose ec;
    ec.text = "p9bec unknown session";
    ec.model_name = "summarizer-model";
    ec.session_guid = "p9bec-never-seen";
    ec.span_start = p5_epoch("2026-05-01 10:00:00");
    ec.span_end   = p5_epoch("2026-05-01 10:00:00");
    ec.member_turn_summary_ids = {1};
    bool threw = false;
    std::string what;
    try { db.store_episode_close(ec); }
    catch (const std::exception& e) { threw = true; what = e.what(); }
    CHECK(threw);
    CHECK(what.find("unknown session") != std::string::npos);

    // Old behavior (pinned before the fix): the unknown GUID silently CREATED
    // a sessions row; the call only failed later on member assignment, and the
    // created session survived. The fix must reject before touching anything.
    raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sessions") == sessions_before);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM episodes") == 0);
    sqlite3_close(raw);

    // A known GUID still works (seed one turn + summary, close it).
    const int t = db.store_turn("p9bec q", "p9bec answer", "test-model", false,
                                "p9bec-known", "2026-05-01 10:00:00");
    CHECK(db.finalize_turn_summary(t, "p9bec summary", "summarizer-model"));
    raw = p1_open_raw();
    const int ts_id = static_cast<int>(p1_scalar_int(raw, std::format(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", t)));
    sqlite3_close(raw);
    ragger::EpisodeClose ok;
    ok.text = "p9bec known session episode";
    ok.model_name = "summarizer-model";
    ok.session_guid = "p9bec-known";
    ok.span_start = p5_epoch("2026-05-01 10:00:00");
    ok.span_end   = p5_epoch("2026-05-01 10:00:00");
    ok.member_turn_summary_ids = {ts_id};
    CHECK(db.store_episode_close(ok) > 0);

    db.close();
    cleanup();
    std::println(" OK");
}

// LR S56 (time rule): an episode's created_at is the time of its LAST member
// turn (= span_end), not the moment it was closed. Pin first: the old INSERT
// omits created_at, so it falls to DEFAULT unixepoch() (close time, which is
// "now" in a test run — far from a seeded 2026-05-01 span).
void lr56_test_episode_created_at_is_span_end(ragger::Embedder& emb) {
    std::println("  lr56_test_episode_created_at_is_span_end...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "lr56ec-a";
    auto m = p5_seed_members(db, guid, {"2026-05-01 10:00:00", "2026-05-01 10:10:00"});
    CHECK(m.size() == 2);

    ragger::EpisodeClose ec;
    ec.text = "lr56ec episode";
    ec.model_name = "summarizer-model";
    ec.session_guid = guid;
    ec.span_start = p5_epoch("2026-05-01 10:00:00");
    ec.span_end   = p5_epoch("2026-05-01 10:10:00");
    ec.member_turn_summary_ids = m;
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, "SELECT created_at FROM episodes WHERE episode_id = "
                             + std::to_string(ep)) == ec.span_end);
    // The close must not move a member's turn_datetime (rule 18).
    CHECK(p1_scalar_int(raw, "SELECT turn_datetime FROM turn_summaries WHERE turn_summary_id = "
                             + std::to_string(m[0])) == p5_epoch("2026-05-01 10:00:00"));
    CHECK(p1_scalar_int(raw, "SELECT turn_datetime FROM turn_summaries WHERE turn_summary_id = "
                             + std::to_string(m[1])) == p5_epoch("2026-05-01 10:10:00"));
    sqlite3_close(raw);

    db.close();
    cleanup();
    std::println(" OK");
}

// LR S57: finalize_turn applies the shared merge rule (merge_assistant_text)
// instead of overwriting assistant_text. Pin first: the old code UPDATEs
// unconditionally, so a differing resend replaced the stored text.
void lr57_test_finalize_turn_merges(ragger::Embedder& emb) {
    std::println("  lr57_test_finalize_turn_merges...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // The step's expected helper values.
    CHECK(ragger::sqlite::merge_assistant_text("", "A") == std::optional<std::string>("A"));
    CHECK(ragger::sqlite::merge_assistant_text("A", "A") == std::nullopt);
    CHECK(ragger::sqlite::merge_assistant_text("AB", "A") == std::nullopt);
    CHECK(ragger::sqlite::merge_assistant_text("A", "AB") == std::optional<std::string>("AB"));
    CHECK(ragger::sqlite::merge_assistant_text("A", "B") == std::optional<std::string>("A\n\nB"));

    const auto stored = [&](int turn_id) {
        sqlite3* raw = p1_open_raw();
        std::string v = p1_scalar_text(raw, "SELECT assistant_text FROM turns WHERE turn_id = "
                                 + std::to_string(turn_id));
        sqlite3_close(raw);
        return v;
    };

    // Partial turn -> fill; repeat -> no change; superset -> replace;
    // contained -> no change (old code: OVERWRITE); differing -> append.
    const int t = db.store_turn("lr57 q", "", "test-model", false, "lr57-a");
    CHECK(t > 0);
    CHECK(db.finalize_turn(t, "A", "m1"));
    CHECK(stored(t) == "A");
    CHECK(db.finalize_turn(t, "A", "m1"));
    CHECK(stored(t) == "A");
    CHECK(db.finalize_turn(t, "AB", "m1"));
    CHECK(stored(t) == "AB");
    CHECK(db.finalize_turn(t, "A", "m1"));
    CHECK(stored(t) == "AB");
    CHECK(db.finalize_turn(t, "C", "m1"));
    CHECK(stored(t) == "AB\n\nC");

    // Unknown turn still returns false.
    CHECK(!db.finalize_turn(99999, "x", "m1"));

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE9B step 2: reads return local wall-time strings, never raw epochs.
void p9b_test_timestamp_formats(ragger::Embedder& emb) {
    std::println("  p9b_test_timestamp_formats...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // (a) search_text_only on a document chunk: timestamp is the source's
    // imported_at as local "YYYY-MM-DD HH:MM:SS" (old code: raw epoch int).
    ragger::DocumentChunk dc;
    dc.text = "p9bts harpsichord concerto movement";
    dc.path = "/tmp/p9bts.md";
    dc.title = "p9bts";
    dc.chunk_index = 1;
    dc.imported_at = "2026-05-01 09:30:00";
    db.store_document(dc);
    auto resp = db.search_text_only("harpsichord", 10);
    CHECK(!resp.results.empty());
    bool found_doc = false;
    for (const auto& r : resp.results) {
        if (r.metadata.value("source", std::string()) == "document") {
            found_doc = true;
            CHECK(r.timestamp == "2026-05-01 09:30:00");
        }
    }
    CHECK(found_doc);

    // (b) no source row -> empty string, not 1970-01-01. (imported_at itself
    // is NOT NULL in the schema; the LEFT JOIN's NULL case is a chunk whose
    // document_source_id is NULL.)
    sqlite3* raw = p1_open_raw();
    CHECK(sqlite3_exec(raw, "UPDATE documents SET document_source_id = NULL",
                       nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);
    resp = db.search_text_only("harpsichord", 10);
    found_doc = false;
    for (const auto& r : resp.results) {
        if (r.metadata.value("source", std::string()) == "document") {
            found_doc = true;
            CHECK(r.timestamp.empty());
        }
    }
    CHECK(found_doc);

    // (c) find_turn_by_text returns the local format too.
    const int t = db.store_turn("p9bts probe question", "p9bts answer",
                                "test-model", false, "p9bts-g", "2026-05-01 08:15:00");
    CHECK(t > 0);
    auto rec = db.find_turn_by_text("p9bts probe question");
    CHECK(rec.has_value());
    CHECK(rec->timestamp == "2026-05-01 08:15:00");

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE9B step 4: failure injection (review recipe). A BEFORE trigger that
// RAISE(ABORT) makes exactly one write path fail deterministically; the
// error is constraint-class (SQLITE_CONSTRAINT_TRIGGER) and sqlite3_errmsg
// contains "p9b injected".
static void p9b_trigger(const std::string& sql) {
    sqlite3* raw = p1_open_raw();
    CHECK(sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);
}

void p9b_test_checked_writes(ragger::Embedder& emb) {
    std::println("  p9b_test_checked_writes...");

    // --- create_user: duplicate returns -1 (never a stale rowid) ----------
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        ragger::UserStore um(TEMP_DB);
        const int id1 = um.create_user("p9b-alice", "h1");
        CHECK(id1 > 0);
        const int dup = um.create_user("p9b-alice", "h2");
        CHECK(dup == -1);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM users WHERE username='p9b-alice'") == 1);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    // --- create_user under injected failure: -1, no row. (Old code read
    // sqlite3_changes() after a failed INSERT — the PREVIOUS statement's
    // count — so this could return a stale rowid. Note what it returned.)
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        ragger::UserStore um(TEMP_DB);
        CHECK(um.create_user("p9b-carol", "h1") > 0);
        CHECK(um.create_user("p9b-dave", "h2") > 0);
        // The previous statement on this connection (dave's INSERT) changed
        // 1 row: the old code read sqlite3_changes() after the failed INSERT
        // and returned that stale count as if it were the new rowid.
        p9b_trigger("CREATE TRIGGER p9b_boom_users BEFORE INSERT ON users "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        const int r = um.create_user("p9b-bob", "h3");
        std::println("    create_user under trigger returned {}", r);
        CHECK(r == -1);
        p9b_trigger("DROP TRIGGER p9b_boom_users");
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM users WHERE username='p9b-bob'") == 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    // --- set_setting failure must throw, not vanish -------------------------
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        p9b_trigger("CREATE TRIGGER p9b_boom_settings BEFORE INSERT ON settings "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        bool threw = false;
        std::string what;
        try { db.set_setting("p9b_k", "v"); }
        catch (const std::runtime_error& e) { threw = true; what = e.what(); }
        CHECK(threw);
        CHECK(what.find("p9b injected") != std::string::npos);
        p9b_trigger("DROP TRIGGER p9b_boom_settings");
        db.set_setting("p9b_k", "v");   // works again
        CHECK(db.get_setting("p9b_k") == std::optional<std::string>("v"));
        db.close(); cleanup();
    }

    // --- get_or_create_model: a failed INSERT must not return a stale rowid
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        const int t1 = db.store_turn("q1", "a1", "p9b-model-a", false);
        CHECK(t1 > 0);
        p9b_trigger("CREATE TRIGGER p9b_boom_models BEFORE INSERT ON models "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        bool threw = false;
        try { db.store_turn("q2", "a2", "p9b-model-new", false); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        p9b_trigger("DROP TRIGGER p9b_boom_models");
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns") == 1);
        // The old code returned sqlite3_last_insert_rowid() after the failed
        // INSERT — a rowid of an unrelated models row (or 0) — so the turn
        // was stored with a model_id that names nothing.
        CHECK(p1_scalar_int(raw,
            "SELECT COUNT(*) FROM turns t LEFT JOIN models m ON t.model_id = m.model_id "
            "WHERE t.model_id IS NOT NULL AND m.model_id IS NULL") == 0);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM models WHERE name='p9b-model-new'") == 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    // --- get_or_create_session: same -------------------------------------------------
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        const int t1 = db.store_turn("q1", "a1", "p9b-model-a", false, "p9b-sess-a");
        CHECK(t1 > 0);
        p9b_trigger("CREATE TRIGGER p9b_boom_sessions BEFORE INSERT ON sessions "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        bool threw = false;
        try { db.store_turn("q2", "a2", "p9b-model-a", false, "p9b-sess-new"); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        p9b_trigger("DROP TRIGGER p9b_boom_sessions");
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turns") == 1);
        CHECK(p1_scalar_int(raw,
            "SELECT COUNT(*) FROM turns t LEFT JOIN sessions s ON t.session_id = s.session_id "
            "WHERE t.session_id IS NOT NULL AND s.session_id IS NULL") == 0);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sessions WHERE guid='p9b-sess-new'") == 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    // --- delete_document_source: atomic (old hand-rolled BEGIN/COMMIT) -----
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        ragger::DocumentChunk c1;
        c1.text = "p9b atomic chunk one"; c1.path = "/tmp/p9b-atomic.md"; c1.title = "t"; c1.chunk_index = 1;
        ragger::DocumentChunk c2 = c1;
        c2.text = "p9b atomic chunk two"; c2.chunk_index = 2;
        db.store_document(c1);
        db.store_document(c2);
        auto src = db.find_document_source("/tmp/p9b-atomic.md");
        const int src_id = src.at("id").get<int>();
        p9b_trigger("CREATE TRIGGER p9b_boom_ds BEFORE DELETE ON document_sources "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        bool threw = false;
        try { db.delete_document_source(src_id); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents") == 2);   // rolled back
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM document_sources") == 1);
        sqlite3_close(raw);
        p9b_trigger("DROP TRIGGER p9b_boom_ds");
        db.delete_document_source(src_id);
        raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents") == 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    std::println(" OK");
}

// PHASE9B step 5: reindex_all is atomic; reindex_table commits per batch.
void p9b_test_reindex_atomic(ragger::Embedder& emb) {
    std::println("  p9b_test_reindex_atomic...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    for (int i = 0; i < 3; ++i)
        db.store(std::format("p9bri harpsichord note {}", i));
    ragger::DocumentChunk c;
    c.text = "p9bri harpsichord document one"; c.path = "/tmp/p9bri-1.md"; c.title = "d1"; c.chunk_index = 1;
    db.store_document(c);
    c.text = "p9bri harpsichord document two"; c.path = "/tmp/p9bri-2.md"; c.chunk_index = 1;
    db.store_document(c);

    auto counts = [&]() {
        sqlite3* raw = p1_open_raw();
        ragger::json j;
        j["terms"] = p1_scalar_int(raw, "SELECT COUNT(*) FROM terms");
        for (const char* t : {"turns_terms","knowledge_terms","documents_terms",
                              "turn_summaries_terms","episodes_terms"})
            j[t] = p1_scalar_int(raw, std::string("SELECT COUNT(*) FROM ") + t);
        sqlite3_close(raw);
        return j;
    };
    const ragger::json before = counts();
    CHECK(before["terms"].get<int>() > 0);
    CHECK(before["knowledge_terms"].get<int>() > 0);
    CHECK(before["documents_terms"].get<int>() > 0);

    // Inject a failure in the THIRD reindexed table (registry order: turns,
    // knowledge, documents, ...). BEFORE UPDATE on the base table (the counts
    // write-back), not on the junction: reset_terms_table() DROPs the
    // junctions inside reindex_all, and a trigger dies with its table.
    p9b_trigger("CREATE TRIGGER p9b_boom_docs BEFORE UPDATE ON documents "
                "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
    bool threw = false;
    try { db.reindex_all({}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    const ragger::json after_fail = counts();
    for (auto& [k, v] : after_fail.items())
        CHECK(v.get<int>() == before[k].get<int>());

    // Happy path: same totals as the per-table loop, search still works.
    p9b_trigger("DROP TRIGGER p9b_boom_docs");
    auto done = db.reindex_all({});
    CHECK(done.size() == 6);   // LR S81: sessions joins the registry (lexical-only)
    int total = 0;
    for (auto& [table, n] : done) total += n;
    CHECK(total == 3 + 2);   // 3 notes + 2 document chunks (turns empty here)
    const ragger::json after_ok = counts();
    CHECK(after_ok["terms"].get<int>() == before["terms"].get<int>());
    auto hits = db.search_text_only("harpsichord", 20);
    CHECK(!hits.results.empty());

    // Single-table reindex: failure rolls back the batch, success restores.
    p9b_trigger("CREATE TRIGGER p9b_boom_kn BEFORE UPDATE ON knowledge "
                "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
    threw = false;
    try { db.reindex_table("knowledge", {}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    p9b_trigger("DROP TRIGGER p9b_boom_kn");
    const int n = db.reindex_table("knowledge", {});
    CHECK(n == 3);
    CHECK(counts()["knowledge_terms"].get<int>() == before["knowledge_terms"].get<int>());

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE9B step 6: store() is the single implementation of the store verb;
// category=decision writes kind='decision' (NULL trigger) with store()'s
// validation and transaction.
void p9b_test_store_one_function(ragger::Embedder& emb) {
    std::println("  p9b_test_store_one_function...");

    // store_kind_for: the single rule.
    CHECK(ragger::store_kind_for(ragger::json{{"category", "decision"}}) == "decision");
    CHECK(ragger::store_kind_for(ragger::json::object()) == "note");
    CHECK(ragger::store_kind_for(ragger::json{{"category", "fact"}}) == "note");
    CHECK(ragger::store_kind_for(ragger::json{{"category", ""}}) == "note");
    CHECK(ragger::store_kind_for(ragger::json()) == "note");

    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        const std::string did = db.store("p9bs of by hal use the semqlite shell",
                                         ragger::json{{"category", "decision"}});
        const std::string nid = db.store("p9bs a plain note about harpsichords",
                                         ragger::json::object());
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT kind FROM knowledge WHERE knowledge_id = " + did) == "decision");
        CHECK(p1_scalar_text(raw, "SELECT ifnull(trigger,'NULL') FROM knowledge WHERE knowledge_id = " + did) == "NULL");
        CHECK(p1_scalar_text(raw, "SELECT kind FROM knowledge WHERE knowledge_id = " + nid) == "note");
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_terms WHERE knowledge_id = " + did) > 0);
        sqlite3_close(raw);
        // indexed: a keyword search finds it
        auto hits = db.search_text_only("semqlite", 10);
        bool found = false;
        for (const auto& r : hits.results)
            if (r.text.find("semqlite") != std::string::npos) found = true;
        CHECK(found);
        db.close(); cleanup();
    }

    // Invalid status is rejected on BOTH paths and writes nothing.
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        bool threw = false;
        std::string what;
        try { db.store("p9bs roadmap note", ragger::json{{"category", "decision"}, {"status", "roadmap"}}); }
        catch (const std::invalid_argument& e) { threw = true; what = e.what(); }
        CHECK(threw);
        CHECK(what == "status must be one of: current, provisional, pending");
        threw = false;
        try { db.store_decision("p9bs roadmap decision", "roadmap", "", "", true); }
        catch (const std::invalid_argument& e) { threw = true; what = e.what(); }
        CHECK(threw);
        CHECK(what == "status must be one of: current, provisional, pending");
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge") == 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    // Atomicity: an indexing failure must not leave a committed, unindexed
    // row (the old store_decision had no transaction).
    {
        cleanup();
        ragger::sqlite::Backend db(emb, TEMP_DB);
        p9b_trigger("CREATE TRIGGER p9b_boom_kn BEFORE INSERT ON knowledge_terms "
                    "BEGIN SELECT RAISE(ABORT, 'p9b injected'); END");
        bool threw = false;
        try { db.store_decision("p9bs atomic decision", "current", "", "", true); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge") == 0);
        sqlite3_close(raw);
        p9b_trigger("DROP TRIGGER p9b_boom_kn");
        const int did = db.store_decision("p9bs atomic decision", "current", "", "", true);
        CHECK(did > 0);
        raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_terms WHERE knowledge_id = " + std::to_string(did)) > 0);
        sqlite3_close(raw);
        db.close(); cleanup();
    }

    std::println(" OK");
}

void p5_test_episodes_needing_close(ragger::Embedder& emb) {
    std::println("  p5_test_episodes_needing_close...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Old (idle) turns everywhere: 2020 timestamps vs idle_minutes=60.
    const std::string gq = "p5nc-queued", gm = "p5nc-marker", gc = "p5nc-closed";

    // gq: content summary after (no) episode -> queued.
    auto q = p5_seed_members(db, gq, {"2020-01-01 10:00:00"});
    // gm: ONLY a marker row after -> NOT queued (the verified bug).
    {
        const int t = db.store_turn("p5nc marker q", "p5nc marker answer " + std::string(60, 'e'),
                                    "test-model", false, gm, "2020-01-01 10:00:00");
        CHECK(db.mark_turn_summarized(t, "bad"));
    }
    auto closes = db.episodes_needing_close(60);
    CHECK(std::find(closes.begin(), closes.end(), gq) != closes.end());
    CHECK(std::find(closes.begin(), closes.end(), gm) == closes.end());

    // gc: summary covered by a closed episode -> NOT queued.
    auto c = p5_seed_members(db, gc, {"2020-01-01 10:00:00"});
    {
        ragger::EpisodeClose ec;
        ec.text = "p5nc closed"; ec.model_name = "summarizer-model";
        ec.session_guid = gc;
        ec.span_start = p5_epoch("2020-01-01 10:00:00");
        ec.span_end   = p5_epoch("2020-01-01 10:00:00");
        ec.member_turn_summary_ids = c;
        CHECK(db.store_episode_close(ec) > 0);
    }
    closes = db.episodes_needing_close(60);
    CHECK(std::find(closes.begin(), closes.end(), gc) == closes.end());
    // l2_summaries_since: nothing after the episode end for gc; content-only
    // rows (no markers) for gq since ''.
    CHECK(db.l2_summaries_since(gc, db.last_episode_end(gc)).empty());
    auto since_all = db.l2_summaries_since(gq, "");
    CHECK(since_all.size() == 1);
    CHECK(since_all[0].text == "p5 summary " + gq + "0");

    db.close();
    cleanup();
    std::println(" OK");
}

void p5_test_episode_reads(ragger::Embedder& emb) {
    std::println("  p5_test_episode_reads...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Session A: episode spanning 09:00..11:00 with two members. Session B:
    // a turn summary at 10:05 — INSIDE A's span (the same-second /
    // cross-session collision the old BETWEEN heuristic mis-assigned).
    const std::string ga = "p5er-a", gb = "p5er-b";
    auto a = p5_seed_members(db, ga, {"2026-06-01 09:00:00", "2026-06-01 11:00:00"});
    auto b = p5_seed_members(db, gb, {"2026-06-01 10:05:00"});
    ragger::EpisodeClose ec;
    ec.text = "p5er episode A"; ec.model_name = "summarizer-model";
    ec.session_guid = ga;
    ec.span_start = p5_epoch("2026-06-01 09:00:00");
    ec.span_end   = p5_epoch("2026-06-01 11:00:00");
    ec.member_turn_summary_ids = a;
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);

    // (f) expand=sources returns EXACTLY the recorded members.
    {
        ragger::AgentGetQuery g;
        g.ids = {"episode:" + std::to_string(ep)};
        g.expand = "sources";
        auto items = db.agent_get(g);
        CHECK(items["items"].size() == 1);
        const auto& item = items["items"][0];
        CHECK(item.contains("expanded"));
        CHECK(item["expanded"].size() == 2);
        for (const auto& e : item["expanded"]) {
            const int id = e["id"].get<std::string>() == "turn_summary:" + std::to_string(a[0])
                               ? a[0] : a[1];
            CHECK(id == a[0] || id == a[1]);
        }
    }

    // (i) episode:N fetch shape + update rejection.
    {
        ragger::AgentGetQuery g;
        g.ids = {"episode:" + std::to_string(ep)};
        auto items = db.agent_get(g);
        const auto& row = items["items"][0];
        CHECK(row["layer"] == "episode");
        CHECK(row["text"] == "p5er episode A");
        CHECK(row["session"] == ga);
        CHECK(row["span_start"].get<std::string>() == "2026-06-01 09:00:00");
        CHECK(row["span_end"].get<std::string>() == "2026-06-01 11:00:00");

        ragger::UpdateRequest r;
        r.id = "episode:" + std::to_string(ep);
        r.tags = "x";
        bool threw = false;
        try { db.agent_update(r); }
        catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()) == "episodes are immutable");
        }
        CHECK(threw);
    }

    // (h) span-overlap date filter: since=10:30 (inside the span, after
    // span_start) still returns the episode.
    {
        ragger::TimelineQuery q;
        q.granularity = "episode";
        q.since = "2026-06-01 10:30:00";
        auto page = db.agent_timeline(q);
        CHECK(page["items"].size() == 1);
        CHECK(page["items"][0]["id"].get<std::string>() == "episode:" + std::to_string(ep));
    }

    // Episode corpus in search: layer=episode hits.
    {
        ragger::AgentSearchQuery q;
        q.query = "p5er episode";
        auto hits = db.agent_search(q);
        bool found = false;
        for (const auto& h : hits)
            if (h.layer == "episode" && h.id == "episode:" + std::to_string(ep)) found = true;
        CHECK(found);
    }

    db.close();
    cleanup();
    std::println(" OK");
}


// PHASE6 6: the single "pending content L2" predicate — episodes_needing_close
// and episode_candidate_turns agree; unsummarized turns block a close.
void p6_test_pending_predicate(ragger::Embedder& emb) {
    std::println("  p6_test_pending_predicate...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    auto queued = [&](const std::string& g) {
        auto v = db.episodes_needing_close(60);
        return std::find(v.begin(), v.end(), g) != v.end();
    };

    // (a) post-episode content L2 without an embedding -> NOT queued; with
    //     one -> queued (the old disagreement, fixed).
    const std::string ga = "p6pp-a";
    auto a = p5_seed_members(db, ga, {"2020-02-01 10:00:00"});
    CHECK(queued(ga));                       // baseline: everything embedded
    {
        sqlite3* raw = p1_open_raw();
        CHECK(sqlite3_exec(raw, ("UPDATE turn_summaries SET embedding = NULL WHERE turn_summary_id = "
                                 + std::to_string(a[0])).c_str(),
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        CHECK(!queued(ga));                  // un-embeddable row must not queue
    }
    {
        sqlite3* raw = p1_open_raw();
        const auto vec = emb.encode("p5 summary " + ga + "0");
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(raw, "UPDATE turn_summaries SET embedding = ? WHERE turn_summary_id = ?",
                                 -1, &st, nullptr) == SQLITE_OK);
        sqlite3_bind_blob(st, 1, vec.data(), static_cast<int>(vec.size() * sizeof(float)), SQLITE_TRANSIENT);
        sqlite3_bind_int(st, 2, a[0]);
        CHECK(sqlite3_step(st) == SQLITE_DONE);
        sqlite3_finalize(st);
        sqlite3_close(raw);
        CHECK(queued(ga));                   // backfill restores queueing
    }

    // (b) an unsummarized turn in the session blocks the close; marking it
    //     (even as a bad/skip marker) unblocks.
    const std::string gb = "p6pp-b";
    auto b = p5_seed_members(db, gb, {"2020-02-01 10:00:00"});
    CHECK(queued(gb));
    const int t2 = db.store_turn("p6pp extra q",
                                 "p6pp extra answer " + std::string(60, 'u'),
                                 "test-model", false, gb, "2020-02-01 11:00:00");
    CHECK(!queued(gb));                      // late L2 could still land in-span
    CHECK(db.mark_turn_summarized(t2, "bad"));
    CHECK(queued(gb));

    // (c) rows already carrying an episode_id are never candidates again,
    //     even with since='' (a late L2 inside a closed span must not be
    //     re-episoded into an overlapping episode).
    {
        ragger::EpisodeClose ec;
        ec.text = "p6pp close"; ec.model_name = "summarizer-model";
        ec.session_guid = ga;
        ec.span_start = p5_epoch("2020-02-01 10:00:00");
        ec.span_end   = p5_epoch("2020-02-01 10:00:00");
        ec.member_turn_summary_ids = {a[0]};
        CHECK(db.store_episode_close(ec) > 0);
        CHECK(db.episode_candidate_turns(ga, "").empty());
    }

    db.close();
    cleanup();
    std::println(" OK");
}


// semqlite_config (the facts-about-the-file table SemanticSQLite reads): the
// embedding identity stamps live there under SemanticSQLite's key names, the
// legacy `settings` rows move over once, and Ragger's change-management keys
// stay in `settings`. Model-free: uses the DB-only constructor.
static void sq_test_semqlite_config() {
    std::print("  sq_test_semqlite_config...\n");
    cleanup();
    {
        ragger::sqlite::Backend db(TEMP_DB);
        CHECK(!db.get_semqlite_config("embedding_model").has_value());

        // The three identity stamps redirect to semqlite_config, renamed.
        db.set_setting("embedding_model", "org/model");
        db.set_setting("vector_type", "f32");
        db.set_setting("dimensions", "384");
        CHECK(db.get_semqlite_config("embedding_model").value_or("") == "org/model");
        CHECK(db.get_semqlite_config("embedding_vector_type").value_or("") == "f32");
        CHECK(db.get_semqlite_config("embedding_dims").value_or("") == "384");
        // ...and read back through the old names.
        CHECK(db.get_setting("vector_type").value_or("") == "f32");
        CHECK(db.get_setting("dimensions").value_or("") == "384");

        // Change-management state stays in `settings`.
        db.set_setting("desired_embedding_model", "x/y");
        CHECK(!db.get_semqlite_config("desired_embedding_model").has_value());

        // Direct semqlite_config access (publish_semqlite_identity uses it).
        db.set_semqlite_config("embedding_offset", "0");
        CHECK(db.get_semqlite_config("embedding_offset").value_or("") == "0");
        db.close();
    }

    sqlite3* raw = p1_open_raw();
    // Nothing duplicated in `settings`.
    CHECK(p1_scalar_int(raw,
        "SELECT count(*) FROM settings WHERE key IN "
        "('embedding_model','vector_type','dimensions')") == 0);
    CHECK(p1_scalar_int(raw,
        "SELECT count(*) FROM settings WHERE key = 'desired_embedding_model'") == 1);

    // Legacy move: a file from before this change keeps its stamps in
    // `settings`. One of them ALSO already exists in semqlite_config, which
    // must win (INSERT OR IGNORE).
    CHECK(sqlite3_exec(raw,
        "DELETE FROM semqlite_config WHERE key IN ('embedding_model','embedding_dims');"
        "UPDATE semqlite_config SET value = 'bf16' WHERE key = 'embedding_vector_type';"
        "INSERT INTO settings (key, value) VALUES "
        "('embedding_model','old/model'),('vector_type','int8'),('dimensions','128')",
        nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(raw);

    {
        ragger::sqlite::Backend db(TEMP_DB);   // reopening runs the move
        CHECK(db.get_semqlite_config("embedding_model").value_or("") == "old/model");
        CHECK(db.get_semqlite_config("embedding_dims").value_or("") == "128");
        CHECK(db.get_semqlite_config("embedding_vector_type").value_or("") == "bf16");
        db.close();
    }
    raw = p1_open_raw();
    CHECK(p1_scalar_int(raw,
        "SELECT count(*) FROM settings WHERE key IN "
        "('embedding_model','vector_type','dimensions')") == 0);
    sqlite3_close(raw);
    cleanup();
    std::println(" ok");
}

// ---------------------------------------------------------------------------
// LR S66/S67 — back-fill passes (see src/backfill.cpp). These run against a
// stub inference endpoint (bf_stub: same shape as test_summarizer's StubChat)
// on the shared TEMP_DB fixture. Reid runs the real passes (rule 17); the
// executor's job is that the code exists and is pinned here: counts, written
// values, in-place column rules, dry-run byte-identity, rerun-is-no-op,
// failure injection, and the trigger re-embed invariant.
// ---------------------------------------------------------------------------

namespace {

struct bf_stub {
    int ls = -1;
    int port = 0;
    std::atomic<bool> running{true};
    std::atomic<int> calls{0};
    std::string reply;
    std::thread srv;
    explicit bf_stub(std::string reply_text) : reply(std::move(reply_text)) {
        ls = ::socket(AF_INET, SOCK_STREAM, 0);
        CHECK(ls >= 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        CHECK(::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
        CHECK(::listen(ls, 8) == 0);
        socklen_t len = sizeof a;
        ::getsockname(ls, reinterpret_cast<sockaddr*>(&a), &len);
        port = ntohs(a.sin_port);
        srv = std::thread([this] {
            while (running.load()) {
                int c = ::accept(ls, nullptr, nullptr);
                if (c < 0) break;
                std::string req;
                char buf[8192];
                size_t hdr_end = std::string::npos;
                long long cl = -1;
                while (true) {
                    ssize_t n = ::recv(c, buf, sizeof buf, 0);
                    if (n <= 0) break;
                    req.append(buf, static_cast<size_t>(n));
                    if (hdr_end == std::string::npos) {
                        hdr_end = req.find("\r\n\r\n");
                        if (hdr_end != std::string::npos) {
                            std::string head = req.substr(0, hdr_end);
                            for (auto& ch : head) ch = static_cast<char>(::tolower(ch));
                            auto p = head.find("content-length:");
                            if (p != std::string::npos)
                                cl = std::stoll(head.substr(p + 15));
                        }
                    }
                    if (cl >= 0 && req.size() >= hdr_end + 4 + static_cast<size_t>(cl))
                        break;
                }
                if (cl >= 0) {
                    const std::string body = req.substr(hdr_end + 4, static_cast<size_t>(cl));
                    if (body.find("\"messages\"") != std::string::npos)
                        ++calls;
                }
                const std::string out =
                    R"({"choices":[{"message":{"content":")" + reply + R"("}}]})";
                const std::string resp =
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(out.size()) +
                    "\r\nConnection: close\r\n\r\n" + out;
                ::send(c, resp.data(), resp.size(), 0);
                ::close(c);
            }
        });
    }
    ~bf_stub() {
        running.store(false);
        ::close(ls);
        if (srv.joinable()) srv.join();
    }
    ragger::InferenceClient client() const {
        ragger::InferenceClient c(
            {ragger::Endpoint("stub", "http://127.0.0.1:" + std::to_string(port) + "/v1")},
            "stub-model", 64);
        c.memory_model = "stub-model";
        c.timeout_sec = 5;
        return c;
    }
};

// Full-content dump of the tables the back-fill may touch — a dry-run pass
// must leave this string byte-identical.
std::string bf_dump(sqlite3* raw) {
    static const char* tables[] = {"turn_summaries", "knowledge", "sessions", "episodes",
                                   "turn_summaries_terms", "knowledge_terms", "models"};
    std::string out;
    for (const char* t : tables) {
        char* err = nullptr;
        char** rows = nullptr;
        int nrows = 0, ncols = 0;
        const std::string sql = std::string("SELECT * FROM ") + t + " ORDER BY 1";
        CHECK(sqlite3_get_table(raw, sql.c_str(), &rows, &nrows, &ncols, &err) == SQLITE_OK);
        out += std::format("{}:{}x{}\n", t, nrows, ncols);
        for (int i = 0; i <= nrows; ++i)
            for (int j = 0; j < ncols; ++j) {
                const char* v = rows[i * ncols + j];
                out += (v ? v : "<NULL>");
                out += '\x1f';
            }
        sqlite3_free_table(rows);
    }
    return out;
}

int bf_ts_id(sqlite3* raw, int turn_id) {
    return static_cast<int>(p1_scalar_int(
        raw, "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(turn_id)));
}

} // namespace

// S66 kinds l2-null / l2-empty / l2-regenerate: counts, written values, the
// in-place column rules (turn_datetime/session_id/episode_id/summarized_on
// never change), floor copy without a model call, guard behavior, rerun
// no-op, dry-run byte-identity, failure injection, unfiltered-regenerate
// refusal, and the not-configured refusal.
void lr66_test_backfill_l2(ragger::Embedder& emb) {
    std::println("  lr66_test_backfill_l2...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    // (1) NULL-text row on a LONG turn -> model path; stub's 2-word reply
    // is strictly shorter -> summary wins, model = stub-model.
    const std::string long_u = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima ";
    const std::string long_a = "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu ";
    const int t_null = db.store_turn(long_u, long_a, "turn-model", false, "bf-l2-a");
    {
        ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on) "
                    "VALUES (?, NULL, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000000)");
        s.bind(1, t_null).bind(2, t_null);
        CHECK(s.exec());
    }
    const int ts_null = bf_ts_id(raw, t_null);
    CHECK(p1_scalar_int(raw, std::format("SELECT summarized_on FROM turn_summaries WHERE turn_summary_id = {}", ts_null)) == 1700000000);

    // (2) '' row (legacy skip, model not 'bad') on a long turn -> candidate.
    const int t_empty = db.store_turn(long_u + " extra", long_a + " more", "turn-model", false, "bf-l2-a");
    CHECK(db.mark_turn_summarized(t_empty, "old-model"));
    const int ts_empty = bf_ts_id(raw, t_empty);

    // (3) poison-abandon '' row -> never an l2-empty candidate.
    const int t_bad = db.store_turn(long_u, long_a, "turn-model", false, "bf-l2-a");
    CHECK(db.mark_turn_summarized(t_bad, "bad"));

    // (4) truly-empty turn ('' row, turn content blanked) -> stays ''.
    const int t_blank = db.store_turn("short q", "short a", "turn-model", false, "bf-l2-a");
    CHECK(db.mark_turn_summarized(t_blank, "old-model"));
    {
        ragger::Stmt s(raw, "UPDATE turns SET user_text = '', assistant_text = '' WHERE turn_id = ?");
        s.bind(1, t_blank);
        CHECK(s.exec());
    }

    {
        bf_stub stub("Brief summary.");
        auto client = stub.client();
        // Dry-run first: byte-identical DB.
        const std::string before = bf_dump(raw);
        auto dry = ragger::backfill_l2(db, client, "null", /*dry_run=*/true);
        CHECK(dry.candidates == 1);
        CHECK(dry.applied == 0);
        CHECK(bf_dump(raw) == before);

        auto res = ragger::backfill_l2(db, client, "null", false);
        CHECK(res.candidates == 1);
        CHECK(res.applied == 1);
        CHECK(stub.calls == 1);   // long source: exactly one model call
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_null)) == "Brief summary.");
        CHECK(p1_scalar_int(raw, "SELECT summary_model_id FROM turn_summaries ts JOIN models m ON m.model_id = ts.summary_model_id WHERE ts.turn_summary_id = " + std::to_string(ts_null)) > 0);
        CHECK(p1_scalar_text(raw, "SELECT m.name FROM turn_summaries ts JOIN models m ON m.model_id = ts.summary_model_id WHERE ts.turn_summary_id = " + std::to_string(ts_null)) == "stub-model");
        CHECK(p1_scalar_int(raw, "SELECT embedding IS NOT NULL FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_null)) == 1);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries_terms WHERE turn_summary_id = " + std::to_string(ts_null)) > 0);
        // In-place rules: summarized_on untouched (D-8), turn_datetime kept.
        CHECK(p1_scalar_int(raw, "SELECT summarized_on FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_null)) == 1700000000);
        CHECK(p1_scalar_int(raw, "SELECT ts.turn_datetime == t.created_at FROM turn_summaries ts JOIN turns t ON t.turn_id = ts.turn_id WHERE ts.turn_summary_id = " + std::to_string(ts_null)) == 1);

        // Rerun: nothing left to fix.
        auto again = ragger::backfill_l2(db, client, "null", false);
        CHECK(again.candidates == 0);
    }

    // l2-empty: only the legacy '' row (not 'bad', not the blanked turn).
    {
        bf_stub stub("Concise recap.");
        auto client = stub.client();
        auto res = ragger::backfill_l2(db, client, "empty", false);
        CHECK(res.candidates == 1);
        CHECK(res.applied == 1);
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_empty)) == "Concise recap.");
        // The blanked turn's '' row and the poison row are untouched.
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_id = " + std::to_string(t_blank)).empty());
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_id = " + std::to_string(t_bad)).empty());
        auto again = ragger::backfill_l2(db, client, "empty", false);
        CHECK(again.candidates == 0);
    }

    // Floor copy: NULL-text row at/below summarizer_min_words words -> copied
    // verbatim with the TURN's model, ZERO model calls. The source is kept
    // OVER summarize_transcript's 120-char trivial shortcut on purpose: with
    // the floor disabled the model WOULD be called, so the stub.calls == 0
    // CHECK is a real pin (a shorter source is answered by the shortcut and
    // hides the difference — found by negative control).
    {
        const std::string fu = "configuration authentication rotation configuration authentication rotation configuration authentication rotation";
        const std::string fa = "please revisit";
        const int t_short = db.store_turn(fu, fa, "turn-model", false, "bf-l2-b");
        {
            ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on) "
                        "VALUES (?, NULL, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000001)");
            s.bind(1, t_short).bind(2, t_short);
            CHECK(s.exec());
        }
        const int ts_short = bf_ts_id(raw, t_short);
        bf_stub stub("NEVER USED");
        auto client = stub.client();
        auto res = ragger::backfill_l2(db, client, "null", false);
        CHECK(res.candidates == 1);
        CHECK(res.applied == 1);
        CHECK(stub.calls == 0);   // floor: no model call at all
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_short)) == fu + " | " + fa);
        CHECK(p1_scalar_text(raw, "SELECT m.name FROM turn_summaries ts JOIN models m ON m.model_id = ts.summary_model_id WHERE ts.turn_summary_id = " + std::to_string(ts_short)) == "turn-model");
    }

    // l2-regenerate: refuses with no filter; --ids and --model select.
    {
        bf_stub stub("Regenerated text.");
        auto client = stub.client();
        bool threw = false;
        try { ragger::backfill_l2(db, client, "regenerate", false); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        auto res = ragger::backfill_l2(db, client, "regenerate", false, "", {ts_null});
        CHECK(res.candidates == 1);
        CHECK(res.applied == 1);
        CHECK(p1_scalar_text(raw, "SELECT text FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_null)) == "Regenerated text.");
        // By model: every row whose summary_model is 'stub-model'.
        auto by_model = ragger::backfill_l2(db, client, "regenerate", false, "stub-model");
        CHECK(by_model.candidates >= 2);
        CHECK(by_model.applied == by_model.candidates);
    }

    // Failure injection: an ABORT trigger on one row -> that row unchanged,
    // the pass reports it and continues with the rest.
    {
        const int t_x = db.store_turn(long_u, long_a, "turn-model", false, "bf-l2-c");
        {
            ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on) "
                        "VALUES (?, NULL, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000002)");
            s.bind(1, t_x).bind(2, t_x);
            CHECK(s.exec());
        }
        const int ts_x = bf_ts_id(raw, t_x);
        CHECK(sqlite3_exec(raw, std::format(
            "CREATE TRIGGER bf_fail BEFORE UPDATE ON turn_summaries "
            "WHEN old.turn_summary_id = {} BEGIN SELECT RAISE(ABORT, 'bf injected'); END",
            ts_x).c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        bf_stub stub("Injected-fail summary.");
        auto client = stub.client();
        auto res = ragger::backfill_l2(db, client, "null", false);
        CHECK(res.candidates == 1);
        CHECK(res.applied == 0);
        CHECK(res.failed == 1);
        CHECK(p1_scalar_int(raw, "SELECT text IS NULL FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_x)) == 1);
        CHECK(sqlite3_exec(raw, "DROP TRIGGER bf_fail", nullptr, nullptr, nullptr) == SQLITE_OK);
        // With the trigger gone the same pass fixes it (resumable).
        auto res2 = ragger::backfill_l2(db, client, "null", false);
        CHECK(res2.candidates == 1);
        CHECK(res2.applied == 1);
    }

    // Not configured: refusal, nothing written.
    {
        ragger::InferenceClient none({}, "", 64);
        const std::string before = bf_dump(raw);
        bool threw = false;
        try { ragger::backfill_l2(db, none, "empty", false); }
        catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()) == "inference not configured");
        }
        CHECK(threw);
        CHECK(bf_dump(raw) == before);
    }

    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" ok");
}

// S66 kind triggers: decision/insight rows with no trigger get one drafted
// into the NULL column ONLY, and the row is re-embedded + re-indexed to the
// trigger+text formula (kKnowledgeEmbedTextSQL). The vector invariant is
// pinned by rebuild-stability: rebuild_embeddings() re-encodes from that SQL
// expression, so an identical blob proves the stored vector was computed
// from the same text (same pattern as the rebuild-stability pins above).
void lr66_test_backfill_triggers(ragger::Embedder& emb) {
    std::println("  lr66_test_backfill_triggers...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    const std::string dtext =
        "we chose SQLite FTS5 custom terms index over FTS because stemming and phonetics are ours";
    const int d1 = db.store_decision(dtext, "current", "tech", "");
    // A decision that already has a trigger: never touched.
    ragger::KnowledgeInput in;
    in.kind = "decision";
    in.text = "pin-first tests catch regressions the suite cannot see";
    in.trigger = "Use when choosing a test order";
    const auto d2 = db.store_knowledge(in);
    // An insight without a trigger: also a candidate.
    {
        ragger::Stmt s(raw, "INSERT INTO knowledge (kind, text, status, tags, created_at) "
                    "VALUES ('insight', 'episode spans must keep both embedding signals', 'current', '', unixepoch())");
        CHECK(s.exec());
    }
    // A note: never a trigger candidate.
    db.store("plain note about the garden", ragger::json{{"tags", {"x"}}});

    // Dry-run: counts only.
    {
        bf_stub stub("whentriggerdraft short");
        auto client = stub.client();
        const std::string before = bf_dump(raw);
        auto dry = ragger::backfill_triggers(db, client, true);
        CHECK(dry.candidates == 2);   // d1 + the insight (not d2, not the note)
        CHECK(bf_dump(raw) == before);
        CHECK(stub.calls == 0);
    }

    {
        bf_stub stub("whentriggerdraft short");
        auto client = stub.client();
        auto res = ragger::backfill_triggers(db, client, false);
        CHECK(res.candidates == 2);
        CHECK(res.applied == 2);
        CHECK(stub.calls == 2);
        CHECK(p1_scalar_text(raw, "SELECT trigger FROM knowledge WHERE knowledge_id = " + std::to_string(d1)) == "whentriggerdraft short");
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_terms kt JOIN terms t ON t.term_id = kt.term_id WHERE kt.knowledge_id = " + std::to_string(d1) + " AND t.term = 'whentriggerdraft'") >= 1);
        // d2's trigger untouched; text/status never change anywhere.
        CHECK(p1_scalar_text(raw, "SELECT trigger FROM knowledge WHERE knowledge_id = " + std::to_string(d2.id)) == "Use when choosing a test order");
        // Vector invariant: stored embedding == encode(trigger || '\n' || text).
        // Pinned via rebuild-stability (see header comment).
        const std::string hex_before = p1_scalar_text(raw, "SELECT hex(embedding) FROM knowledge WHERE knowledge_id = " + std::to_string(d1));
        CHECK(!hex_before.empty());
        db.rebuild_embeddings(emb, {}, "knowledge");
        CHECK(p1_scalar_text(raw, "SELECT hex(embedding) FROM knowledge WHERE knowledge_id = " + std::to_string(d1)) == hex_before);
        // Rerun: nothing left.
        auto again = ragger::backfill_triggers(db, client, false);
        CHECK(again.candidates == 0);
    }

    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" ok");
}

// S67 kind stranded: membership-only attach of L2 rows inside a closed span
// of the SAME session; the greatest-span_start episode wins on overlap (ties
// counted); the open tail is counted and left alone; rerun is a no-op.
void lr67_test_backfill_stranded(ragger::Embedder& emb) {
    std::println("  lr67_test_backfill_stranded...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    const std::string g = "bf-strand-a";
    auto m = p5_seed_members(db, g, {"2026-05-01 10:00:00", "2026-05-01 10:05:00",
                                     "2026-05-01 10:10:00"});
    ragger::EpisodeClose ec;
    ec.text = "bf-strand ep1";
    ec.model_name = "summarizer-model";
    ec.session_guid = g;
    ec.span_start = p5_epoch("2026-05-01 10:00:00");
    ec.span_end   = p5_epoch("2026-05-01 10:05:00");
    ec.member_turn_summary_ids = {m[0], m[1]};
    const int ep1 = db.store_episode_close(ec);
    CHECK(ep1 > 0);

    // A late L2 inside ep1's span (stranded), and one after the last span
    // (open tail).
    auto late = p5_seed_members(db, g, {"2026-05-01 10:02:00", "2026-05-01 10:20:00"});

    // A second, overlapping episode (earlier span_start) for the tie case.
    auto m2 = p5_seed_members(db, g, {"2026-05-01 09:55:00"});
    ragger::EpisodeClose ec2;
    ec2.text = "bf-strand ep2";
    ec2.model_name = "summarizer-model";
    ec2.session_guid = g;
    ec2.span_start = p5_epoch("2026-05-01 09:55:00");
    ec2.span_end   = p5_epoch("2026-05-01 10:04:00");
    ec2.member_turn_summary_ids = {m2[0]};
    const int ep2 = db.store_episode_close(ec2);
    CHECK(ep2 > 0);

    // Another session covering the same time: never attached across sessions.
    const std::string g2 = "bf-strand-b";
    auto other = p5_seed_members(db, g2, {"2026-05-01 10:02:00"});

    // Dry-run: counts, nothing written.
    {
        const std::string before = bf_dump(raw);
        auto dry = ragger::backfill_stranded(db, true);
        CHECK(dry.candidates == 1);            // late[0] (inside ep1/ep2)
        CHECK(dry.ties == 1);                  // covered by both spans
        CHECK(dry.open_tail >= 1);             // late[1] after the last span (+ session b's row)
        CHECK(bf_dump(raw) == before);
    }

    auto res = ragger::backfill_stranded(db, false);
    CHECK(res.candidates == 1);
    CHECK(res.applied == 1);
    // Greatest span_start <= t wins: ep1 (10:00) over ep2 (09:55).
    CHECK(p1_scalar_int(raw, "SELECT episode_id FROM turn_summaries WHERE turn_summary_id = " + std::to_string(late[0])) == ep1);
    // Open tail untouched; foreign session untouched.
    CHECK(p1_scalar_int(raw, "SELECT episode_id FROM turn_summaries WHERE turn_summary_id = " + std::to_string(late[1])) == 0 /* NULL reads as 0 */);
    CHECK(p1_scalar_int(raw, "SELECT episode_id IS NULL FROM turn_summaries WHERE turn_summary_id = " + std::to_string(other[0])) == 1);
    // The episode itself is unchanged (membership-only).
    CHECK(p1_scalar_text(raw, "SELECT text FROM episodes WHERE episode_id = " + std::to_string(ep1)) == "bf-strand ep1");

    auto again = ragger::backfill_stranded(db, false);
    CHECK(again.candidates == 0);

    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" ok");
}

// S67 kind session-names: unnamed sessions with L2 content get a drafted
// title with name_source='summarizer'; a named session is never touched;
// rerun is a no-op.
void lr67_test_backfill_session_names(ragger::Embedder& emb) {
    std::println("  lr67_test_backfill_session_names...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    // Unnamed session with one L2 row.
    const int t1 = db.store_turn("bf-name question", "bf-name answer about the pipeline",
                                 "test-model", false, "bf-name-a");
    CHECK(db.finalize_turn_summary(t1, "bf-name summary", "summarizer-model"));
    // Named session with L2 rows: never touched.
    const int t2 = db.store_turn("bf-name2 q", "bf-name2 a", "test-model", false,
                                 "bf-name-b", "", "user-given-title", "hermes_title");
    CHECK(db.finalize_turn_summary(t2, "bf-name2 summary", "summarizer-model"));
    // Unnamed session with NO L2 rows: not a candidate.
    db.store_turn("bf-name3 q", "bf-name3 a", "test-model", false, "bf-name-c");

    {
        const std::string before = bf_dump(raw);
        ragger::InferenceClient none({}, "", 64);
        auto dry = ragger::backfill_session_names(db, none, true);   // dry-run: inference unused
        CHECK(dry.candidates == 1);
        CHECK(bf_dump(raw) == before);
    }
    {
        bf_stub stub("Pipeline discussion");
        auto client = stub.client();
        auto res = ragger::backfill_session_names(db, client, false);
        CHECK(res.candidates == 1);
        CHECK(res.applied == 1);
        CHECK(stub.calls == 1);
        CHECK(p1_scalar_text(raw, "SELECT name FROM sessions WHERE guid = 'bf-name-a'") == "Pipeline discussion");
        CHECK(p1_scalar_text(raw, "SELECT name_source FROM sessions WHERE guid = 'bf-name-a'") == "summarizer");
        CHECK(p1_scalar_text(raw, "SELECT name FROM sessions WHERE guid = 'bf-name-b'") == "user-given-title");
        CHECK(p1_scalar_int(raw, "SELECT name IS NULL FROM sessions WHERE guid = 'bf-name-c'") == 1);
        auto again = ragger::backfill_session_names(db, client, false);
        CHECK(again.candidates == 0);
        // Direct guard pin (selection alone cannot show it): the writer must
        // refuse a non-NULL name even when called with the id.
        const int sid_b = static_cast<int>(p1_scalar_int(
            raw, "SELECT session_id FROM sessions WHERE guid = 'bf-name-b'"));
        CHECK(!db.apply_session_name(sid_b, "clobbered", "summarizer"));
        CHECK(p1_scalar_text(raw, "SELECT name FROM sessions WHERE guid = 'bf-name-b'") == "user-given-title");
    }

    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" ok");
}

// S68 (rule 18): the time invariants as one scenario — L2 turn_datetime
// always equals its turn's created_at (before AND after a regenerate
// back-fill), episodes carry created_at = span_end, and copy rows record
// the S62 provenance model.
void lr68_test_time_invariants(ragger::Embedder& emb) {
    std::println("  lr68_test_time_invariants...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    const std::string g = "bf-time-a";
    auto m = p5_seed_members(db, g, {"2026-06-01 09:00:00", "2026-06-01 09:10:00"});
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM turn_summaries ts JOIN turns t ON t.turn_id = ts.turn_id "
        "WHERE ts.turn_datetime != t.created_at") == 0);

    ragger::EpisodeClose ec;
    ec.text = "bf-time episode";
    ec.model_name = "summarizer-model";
    ec.session_guid = g;
    ec.span_start = p5_epoch("2026-06-01 09:00:00");
    ec.span_end   = p5_epoch("2026-06-01 09:10:00");
    ec.member_turn_summary_ids = m;
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);
    CHECK(p1_scalar_int(raw, "SELECT created_at FROM episodes WHERE episode_id = " + std::to_string(ep))
          == p5_epoch("2026-06-01 09:10:00"));

    // Regenerate both rows: turn_datetime (and episode membership) unchanged.
    {
        bf_stub stub("Regenerated time text.");
        auto client = stub.client();
        auto res = ragger::backfill_l2(db, client, "regenerate", false, "", {m[0], m[1]});
        CHECK(res.candidates == 2);
        CHECK(res.applied == 2);
    }
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM turn_summaries ts JOIN turns t ON t.turn_id = ts.turn_id "
        "WHERE ts.turn_datetime != t.created_at") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM turn_summaries WHERE episode_id = " + std::to_string(ep)) == 2);

    // Copy provenance (S62 table): a floor copy records the TURN's model.
    const int t_short = db.store_turn("tiny", "yes", "turn-model-x", false, "bf-time-b");
    {
        ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on) "
                    "VALUES (?, NULL, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000003)");
        s.bind(1, t_short).bind(2, t_short);
        CHECK(s.exec());
    }
    {
        bf_stub stub("unused");
        auto client = stub.client();
        auto res = ragger::backfill_l2(db, client, "null", false);
        CHECK(res.applied == 1);
    }
    CHECK(p1_scalar_text(raw,
        "SELECT m.name FROM turn_summaries ts JOIN models m ON m.model_id = ts.summary_model_id "
        "WHERE ts.turn_id = " + std::to_string(t_short)) == "turn-model-x");
    CHECK(p1_scalar_int(raw,
        "SELECT ts.turn_datetime == t.created_at FROM turn_summaries ts JOIN turns t ON t.turn_id = ts.turn_id "
        "WHERE ts.turn_id = " + std::to_string(t_short)) == 1);

    sqlite3_close(raw);
    db.close();
    cleanup();
    std::println(" ok");
}

// LR S73 (R13, Q5, review #3): episodes satisfy the session/exclude_session
// filters, and episode hits carry extra = {session, span_start, span_end}
// (spans rendered like memory_timeline renders `created`). Before this step
// session_of was filled only for turn_summary/knowledge, so
// layers:["episode"] + session returned [] and extra was null.
void lr73_test_episode_session_filter_and_extra(ragger::Embedder& emb) {
    std::println("  lr73_test_episode_session_filter_and_extra...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string ga = "lr73-a", gb = "lr73-b";
    auto ma = p5_seed_members(db, ga, {"2026-07-01 09:00:00", "2026-07-01 09:10:00"});
    auto mb = p5_seed_members(db, gb, {"2026-07-02 09:00:00", "2026-07-02 09:10:00"});

    auto close_ep = [&](const std::string& g, std::vector<int>& m,
                        const std::string& text, const char* s, const char* e) {
        ragger::EpisodeClose ec;
        ec.text = text;
        ec.model_name = "summarizer-model";
        ec.session_guid = g;
        ec.span_start = p5_epoch(s);
        ec.span_end   = p5_epoch(e);
        ec.member_turn_summary_ids = m;
        const int ep = db.store_episode_close(ec);
        CHECK(ep > 0);
        return ep;
    };
    const int epa = close_ep(ga, ma, "lr73 alpha episode about kites",
                             "2026-07-01 09:00:00", "2026-07-01 09:10:00");
    const int epb = close_ep(gb, mb, "lr73 beta episode about kites",
                             "2026-07-02 09:00:00", "2026-07-02 09:10:00");

    // Session filter over the episode layer: exactly that session's episode.
    ragger::AgentSearchQuery q;
    q.query = "lr73 episode kites";
    q.layers = {"episode"};
    q.session = ga;
    auto hits = db.agent_search(q);
    CHECK(hits.size() == 1);
    CHECK(!hits.empty() && hits[0].id == "episode:" + std::to_string(epa));

    q.session = "";
    q.exclude_session = ga;
    hits = db.agent_search(q);
    CHECK(hits.size() == 1);
    CHECK(!hits.empty() && hits[0].id == "episode:" + std::to_string(epb));

    // Episode hits carry extra with the owning session GUID and the span
    // rendered as local "YYYY-MM-DD HH:MM:SS" (memory_timeline's format).
    q.exclude_session = "";
    hits = db.agent_search(q);
    CHECK(hits.size() == 2);
    for (const auto& h : hits) {
        CHECK(h.extra.is_object());
        const bool is_a = (h.id == "episode:" + std::to_string(epa));
        CHECK(h.extra.value("session", "") == (is_a ? ga : gb));
        CHECK(h.extra.value("span_start", std::string()) ==
              (is_a ? "2026-07-01 09:00:00" : "2026-07-02 09:00:00"));
        CHECK(h.extra.value("span_end", std::string()) ==
              (is_a ? "2026-07-01 09:10:00" : "2026-07-02 09:10:00"));
    }

    db.close();
    cleanup();
    std::println(" OK");
}

// LR S75: raw turns are an OPT-IN corpus. Default layers never return a
// turn:* hit; layers:["turn"] finds a phrase that exists only in the raw
// assistant text (never in its L2 summary); hits carry turn:<N>, created =
// the turn's time, extra.session; memory_get("turn:<N>") works; degraded
// (no embedder) mirrors the contract with text-only scoring per corpus.
void lr75_test_raw_turn_opt_in(ragger::Embedder& emb) {
    std::println("  lr75_test_raw_turn_opt_in...");
    cleanup();
    int t1 = 0, t2 = 0;
    const std::string s = "lr75-sess";
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        t1 = db.store_turn("How did the reptile review end?",
                           "The purple newt hypothesis was rejected in 1998.",
                           "test-model", false, s, "2026-08-03 11:22:33");
        CHECK(t1 > 0);
        CHECK(db.finalize_turn_summary(t1, "The reptile review concluded negatively.",
                                       "summarizer-model"));
        // A second turn with NO summary row at all — the raw corpus must find
        // it too (it is not gated on summarization).
        t2 = db.store_turn("And the salamander appendix?",
                           "The orange salamander appendix was withdrawn.",
                           "test-model", false, s, "2026-08-03 11:25:00");
        CHECK(t2 > 0);

        // S75 report: raw-turn cache cost for this fixture. Per row the warm
        // cache holds dim*4 float32 bytes + display text + a small meta json.
        {
            const long long text_bytes =
                std::string_view("How did the reptile review end?").size() +
                std::string_view("The purple newt hypothesis was rejected in 1998.").size() +
                std::string_view("And the salamander appendix?").size() +
                std::string_view("The orange salamander appendix was withdrawn.").size();
            std::println("    [s75] fixture turns=2 text_bytes={} "
                         "cache ≈ {} B (float32 emb) + text/meta",
                         text_bytes, 2 * 384 * 4);
        }

        ragger::AgentSearchQuery q;
        q.limit = 10;

        // Default layers: the raw-only phrase must not surface a turn hit.
        q.query = "purple newt hypothesis rejected";
        q.layers = {};
        for (const auto& h : db.agent_search(q))
            CHECK(!h.id.starts_with("turn:"));

        // Opt-in: finds the turn by raw text its summary never contained.
        q.layers = {"turn"};
        bool found = false;
        for (const auto& h : db.agent_search(q)) {
            CHECK(h.layer == "turn");
            if (h.id != "turn:" + std::to_string(t1)) continue;
            found = true;
            CHECK(h.created == "2026-08-03 11:22:33");
            CHECK(h.extra.contains("session"));
            CHECK(h.extra["session"].get<std::string>() == s);
        }
        CHECK(found);

        // The unsummarized turn is findable as well.
        q.query = "orange salamander appendix";
        found = false;
        for (const auto& h : db.agent_search(q))
            if (h.id == "turn:" + std::to_string(t2)) found = true;
        CHECK(found);

        // Session filter works on the raw corpus too. NOTE: an UNKNOWN guid
        // is the parked FINDING (filter silently ignored) — not asserted here;
        // use a session that exists but does not own t2.
        const int t3 = db.store_turn("Unrelated question",
                                     "Nothing about salamanders here.",
                                     "test-model", false, "lr75-other",
                                     "2026-08-03 12:00:00");
        CHECK(t3 > 0);
        q.query = "orange salamander appendix";
        q.session = "lr75-other";
        for (const auto& h : db.agent_search(q))
            CHECK(h.id != "turn:" + std::to_string(t2));
        q.session.clear();

        // memory_get by handle.
        ragger::AgentGetQuery g;
        g.ids = {"turn:" + std::to_string(t1)};
        const auto got = db.agent_get(g);
        CHECK(got["items"][0]["layer"] == "turn");
        CHECK(got["items"][0]["assistant_text"].get<std::string>()
                  .contains("purple newt"));
    }
    // Degraded (DB-only constructor = unready embedder): same contract via
    // text-only scoring; default layers still exclude turn.
    {
        ragger::sqlite::Backend db(TEMP_DB);
        ragger::AgentSearchQuery q;
        q.query = "purple newt hypothesis rejected";
        q.limit = 10;
        q.layers = {"turn"};
        bool found = false;
        for (const auto& h : db.agent_search(q))
            if (h.id == "turn:" + std::to_string(t1)) found = true;
        CHECK(found);
        q.layers = {};
        for (const auto& h : db.agent_search(q))
            CHECK(!h.id.starts_with("turn:"));
        db.close();
    }
    cleanup();
    std::println(" OK");
}

// LR S76 (D-7, Q-A): documents get their own verb. document_search finds a
// chunk with extra {source_path, chunk_index}; `source` filters by path;
// memory_search's default layers no longer include document, and naming the
// `document` layer is an invalid_argument pointing to document_search; the
// flat quick search still covers documents; degraded mirrors the contract.
void lr76_test_document_search(ragger::Embedder& emb) {
    std::println("  lr76_test_document_search...");
    cleanup();
    int doc_id = 0;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        ragger::DocumentChunk d;
        d.text = "The quarterly otter census was published in March.";
        d.title = "Otter census";
        d.path = "/docs/lr76-census.md";
        d.chunk_index = 2;
        doc_id = db.store_document(d);
        CHECK(doc_id > 0);

        ragger::DocumentSearchQuery dq;
        dq.query = "otter census published";
        auto hits = db.agent_document_search(dq);
        bool found = false;
        for (const auto& h : hits) {
            CHECK(h.layer == "document");
            if (h.id != "document:" + std::to_string(doc_id)) continue;
            found = true;
            CHECK(h.extra["source_path"] == "/docs/lr76-census.md");
            CHECK(h.extra["chunk_index"] == 2);
            CHECK(h.snippet.find("otter") != std::string::npos);
        }
        CHECK(found);

        // source filter (substring, case-insensitive).
        dq.source = "LR76";
        found = false;
        for (const auto& h : db.agent_document_search(dq))
            if (h.id == "document:" + std::to_string(doc_id)) found = true;
        CHECK(found);
        dq.source = "no-such-path";
        for (const auto& h : db.agent_document_search(dq))
            CHECK(h.id != "document:" + std::to_string(doc_id));
        dq.source.clear();

        // memory_search default layers: no document hits.
        ragger::AgentSearchQuery q;
        q.query = "otter census published";
        q.limit = 10;
        for (const auto& h : db.agent_search(q))
            CHECK(!h.id.starts_with("document:"));

        // Naming the document layer is rejected with the pointer.
        bool threw = false;
        try {
            q.layers = {"document"};
            db.agent_search(q);
        } catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()).find("document_search") != std::string::npos);
        }
        CHECK(threw);

        // The flat quick search still covers documents (Hermes keeps one call
        // for "anything").
        bool flat = false;
        for (const auto& r : db.search("otter census published", 10, 0.0f).results)
            if (r.metadata.value("source", "") == "document") flat = true;
        CHECK(flat);
    }
    // Degraded (unready embedder): document_search still finds via text-only.
    {
        ragger::sqlite::Backend db(TEMP_DB);
        ragger::DocumentSearchQuery dq;
        dq.query = "otter census published";
        bool found = false;
        for (const auto& h : db.agent_document_search(dq))
            if (h.id == "document:" + std::to_string(doc_id)) found = true;
        CHECK(found);
        db.close();
    }
    cleanup();
    std::println(" OK");
}

// LR S77 (Q-A): mode = hybrid | semantic | keyword. hybrid is the default
// (byte-identical to today, pinned by every existing search test); keyword
// is text-index only and works without an embedder; semantic is vector-only
// (ranks a paraphrase above an unrelated row) and needs an embedder; unknown
// mode is invalid_argument; matched_by reflects the mode.
void lr77_test_search_mode(ragger::Embedder& emb) {
    std::println("  lr77_test_search_mode...");
    cleanup();
    int a_id = 0, c_id = 0, doc_id = 0;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        const std::string gs = "lr77";
        const int t1 = db.store_turn("q1", "Freezing garlic bread keeps it fresh for months",
                                     "test-model", false, gs, "2026-08-04 09:00:00");
        CHECK(db.finalize_turn_summary(t1, "Freezing garlic bread keeps it fresh for months",
                                       "summarizer-model"));
        const int t2 = db.store_turn("q2", "The stock market closed lower on Friday",
                                     "test-model", false, gs, "2026-08-04 09:01:00");
        CHECK(db.finalize_turn_summary(t2, "The stock market closed lower on Friday",
                                       "summarizer-model"));
        const int t3 = db.store_turn("q3", "The xylophonically tuned bell rang",
                                     "test-model", false, gs, "2026-08-04 09:02:00");
        CHECK(db.finalize_turn_summary(t3, "The xylophonically tuned bell rang",
                                       "summarizer-model"));

        ragger::AgentSearchQuery q;
        q.limit = 10;

        // keyword: exact rare token, matched_by lexical.
        q.query = "xylophonically";
        q.mode = "keyword";
        bool found = false;
        for (const auto& h : db.agent_search(q))
            if (h.id == "turn_summary:" + std::to_string(t3)) {
                found = true;
                CHECK(h.matched_by == "lexical");
            }
        CHECK(found);
        c_id = t3;

        // semantic: paraphrase ranks above the unrelated row; matched_by vector.
        q.query = "how do I keep garlic bread fresh";
        q.mode = "semantic";
        int rank_a = -1, rank_b = -1, i = 0;
        for (const auto& h : db.agent_search(q)) {
            if (h.id == "turn_summary:" + std::to_string(t1)) { rank_a = i; CHECK(h.matched_by == "vector"); }
            if (h.id == "turn_summary:" + std::to_string(t2)) rank_b = i;
            ++i;
        }
        CHECK(rank_a >= 0);
        CHECK(rank_b == -1 || rank_a < rank_b);
        a_id = t1;

        // unknown mode -> invalid_argument.
        bool threw = false;
        try {
            q.mode = "fuzzy";
            db.agent_search(q);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        q.mode = "hybrid";

        // hybrid default finds the keyword row again (unchanged behavior).
        q.query = "xylophonically";
        found = false;
        for (const auto& h : db.agent_search(q))
            if (h.id == "turn_summary:" + std::to_string(c_id)) found = true;
        CHECK(found);

        // document_search takes the same mode. (Two docs: with a one-row
        // corpus the idf term weight is 0 and every text score is 0 — the
        // parked matched_by FINDING shape, not what this step pins.)
        ragger::DocumentChunk other;
        other.text = "Harbor lantern oil deliveries, weekly.";
        other.title = "Lantern oil";
        other.path = "/docs/lr77-oil.md";
        CHECK(db.store_document(other) > 0);
        ragger::DocumentChunk d;
        d.text = "Kelpie lighthouse maintenance log, xylophonically indexed.";
        d.title = "Kelpie log";
        d.path = "/docs/lr77-kelpie.md";
        doc_id = db.store_document(d);
        CHECK(doc_id > 0);
        ragger::DocumentSearchQuery dq;
        dq.query = "xylophonically";
        dq.mode = "keyword";
        found = false;
        for (const auto& h : db.agent_document_search(dq))
            if (h.id == "document:" + std::to_string(doc_id)) {
                found = true;
                CHECK(h.matched_by == "lexical");
            }
        CHECK(found);
        threw = false;
        try {
            dq.mode = "telepathy";
            db.agent_document_search(dq);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    // Degraded (no embedder): keyword works; semantic is a clean 400-class
    // invalid_argument, not a runtime_error.
    {
        ragger::sqlite::Backend db(TEMP_DB);
        ragger::AgentSearchQuery q;
        q.query = "xylophonically";
        q.limit = 10;
        q.mode = "keyword";
        bool found = false;
        for (const auto& h : db.agent_search(q))
            if (h.id == "turn_summary:" + std::to_string(c_id)) found = true;
        CHECK(found);
        bool threw = false;
        try {
            q.mode = "semantic";
            db.agent_search(q);
        } catch (const std::invalid_argument& e) {
            threw = true;
            CHECK(std::string(e.what()).find("embedder") != std::string::npos);
        }
        CHECK(threw);
        ragger::DocumentSearchQuery dq;
        dq.query = "xylophonically";
        dq.mode = "keyword";
        found = false;
        for (const auto& h : db.agent_document_search(dq))
            if (h.id == "document:" + std::to_string(doc_id)) found = true;
        CHECK(found);
        db.close();
    }
    cleanup();
    std::println(" OK");
}

// LR S78 (logic §2): memory_search `tags` filters knowledge hits to rows
// carrying ALL listed tags (trimmed, case-insensitive); an array or a comma
// string are both accepted; layers without tags are excluded when tags are
// given; memory_update accepts the same array/string forms.
void lr78_test_tags_filter(ragger::Embedder& emb) {
    std::println("  lr78_test_tags_filter...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::KnowledgeInput k1;
    k1.kind = "note"; k1.text = "quay lantern inventory alpha"; k1.tags = "a,b";
    const auto r1 = db.store_knowledge(k1);
    ragger::KnowledgeInput k2;
    k2.kind = "note"; k2.text = "quay lantern inventory beta"; k2.tags = "a";
    const auto r2 = db.store_knowledge(k2);
    // A turn_summary matching the same query must be excluded when tags filter.
    const int t = db.store_turn("quay lantern inventory?", "quay lantern inventory gamma",
                                "test-model", false, "lr78", "2026-08-05 09:00:00");
    CHECK(db.finalize_turn_summary(t, "quay lantern inventory gamma", "summarizer-model"));

    auto ids_with = [&](const ragger::json& tags) {
        ragger::AgentSearchQuery q;
        q.query = "quay lantern inventory";
        q.limit = 20;
        if (tags.is_array()) {
            for (const auto& x : tags) q.tags.push_back(x.get<std::string>());
        } else if (tags.is_string()) {
            const std::string s = tags.get<std::string>();
            std::string::size_type start = 0;
            while (start <= s.size()) {
                auto comma = s.find(',', start);
                const auto end = comma == std::string::npos ? s.size() : comma;
                if (end > start) q.tags.push_back(s.substr(start, end - start));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        std::set<std::string> got;
        for (const auto& h : db.agent_search(q)) got.insert(h.id);
        return got;
    };
    const std::string id1 = "knowledge:" + std::to_string(r1.id);
    const std::string id2 = "knowledge:" + std::to_string(r2.id);
    const std::string tid = "turn_summary:" + std::to_string(t);

    // No tags: both knowledge rows + the summary.
    auto all = ids_with(ragger::json::array());
    CHECK(all.count(id1) && all.count(id2) && all.count(tid));

    // AND rule: ["a","b"] -> only the first; ["a"] -> both knowledge rows,
    // summary excluded (layers without tags drop out).
    auto ab = ids_with(ragger::json{"a", "b"});
    CHECK((ab == std::set<std::string>{id1}));
    auto a = ids_with(ragger::json{"a"});
    CHECK((a == std::set<std::string>{id1, id2}));
    // Case-insensitive + comma string form.
    auto ac = ids_with(ragger::json{"A"});
    CHECK((ac == std::set<std::string>{id1, id2}));
    auto cs = ids_with(std::string("a, b"));
    CHECK((cs == std::set<std::string>{id1}));
    // A tag nobody has: nothing.
    CHECK(ids_with(ragger::json{"z"}).empty());

    // memory_update accepts arrays and strings alike.
    ragger::UpdateRequest u;
    u.id = id2;
    u.tags = std::string("x,y");
    db.agent_update(u);
    auto after = ids_with(ragger::json{"x", "y"});
    CHECK(after.count(id2));
    CHECK((ids_with(ragger::json{"a"}) == std::set<std::string>{id1}));  // id2 lost 'a'

    db.close();
    cleanup();
    std::println(" OK");
}

// LR S79 (logic §2, P10): history reads carry the hierarchy.
// (a) session paging returns each turn's turn_summary_id and episode_id
//     when they exist; (b) expand=context for a turn returns the 2 turns
//     before/after in SESSION ORDER (created_at, turn_id), not raw id
//     order; (c) memory_timeline granularity "knowledge" pages newest-first
//     with <epoch>:<rowid> cursors, no gaps or repeats.
void lr79_test_history_hierarchy(ragger::Embedder& emb) {
    std::println("  lr79_test_history_hierarchy...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string ga = "lr79-a", gb = "lr79-b";
    // Session A's id order deliberately differs from its time order:
    //   A1 10:00, B1 10:01 (other session), A2 10:05, A3 10:02,
    //   A4 10:06, A5 10:03, A6 10:07
    // A's time order: A1 A3 A5 A2 A4 A6.
    auto store = [&](const std::string& g, const std::string& n, const char* ts) {
        return db.store_turn("lr79 q " + n, "lr79 a " + n, "test-model", false, g, ts);
    };
    const int a1 = store(ga, "a1", "2026-08-06 10:00:00");
    const int b1 = store(gb, "b1", "2026-08-06 10:01:00");
    const int a2 = store(ga, "a2", "2026-08-06 10:05:00");
    const int a3 = store(ga, "a3", "2026-08-06 10:02:00");
    const int a4 = store(ga, "a4", "2026-08-06 10:06:00");
    const int a5 = store(ga, "a5", "2026-08-06 10:03:00");
    const int a6 = store(ga, "a6", "2026-08-06 10:07:00");
    CHECK(b1 > 0);

    // (b) context around A2: before = A3, A5; after = A4, A6 (session order).
    // The old id-window code returned A1, A3, A4 (ids a2-2..a2+2 within the
    // session): A1 is outside the time window and A5 was missing.
    ragger::AgentGetQuery gq;
    gq.ids = {"turn:" + std::to_string(a2)};
    gq.expand = "context";
    const auto got = db.agent_get(gq);
    std::vector<std::string> exp_ids;
    if (!got["items"].empty() && got["items"][0].contains("expanded"))
        for (const auto& e : got["items"][0]["expanded"])
            exp_ids.push_back(e["id"].get<std::string>());
    const std::vector<std::string> want = {
        "turn:" + std::to_string(a3), "turn:" + std::to_string(a5),
        "turn:" + std::to_string(a4), "turn:" + std::to_string(a6)};
    CHECK((exp_ids == want));

    // (a) summaries + an episode so the hierarchy fields exist.
    CHECK(db.finalize_turn_summary(a1, "lr79 summary a1", "summarizer-model"));
    CHECK(db.finalize_turn_summary(a3, "lr79 summary a3", "summarizer-model"));
    auto tsid = [&](int turn) {
        sqlite3* raw = p1_open_raw();
        const long long v = p1_scalar_int(raw, std::format(
            "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = {}", turn));
        sqlite3_close(raw);
        return static_cast<int>(v);
    };
    ragger::EpisodeClose ec;
    ec.text = "lr79 episode one";
    ec.model_name = "summarizer-model";
    ec.session_guid = ga;
    ec.span_start = p5_epoch("2026-08-06 10:00:00");
    ec.span_end   = p5_epoch("2026-08-06 10:02:00");
    ec.member_turn_summary_ids = {tsid(a1), tsid(a3)};
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);

    ragger::AgentGetQuery hq;
    hq.session = ga;
    hq.limit = 100;
    const auto hist = db.agent_get(hq);
    CHECK(hist["turns"].size() == 6);
    int seen_a1 = 0, seen_a2 = 0;
    for (const auto& t : hist["turns"]) {
        const int id = std::stoi(t["id"].get<std::string>().substr(5));
        if (id == a1) {
            CHECK(t["turn_summary_id"] == tsid(a1));
            CHECK(t["episode_id"] == ep);
            ++seen_a1;
        } else if (id == a2) {
            CHECK(t["turn_summary_id"].is_null());
            CHECK(t["episode_id"].is_null());
            ++seen_a2;
        }
    }
    CHECK(seen_a1 == 1 && seen_a2 == 1);

    // (c) knowledge timeline: 5 rows, limit 2 -> pages 2+2+1, no repeats.
    for (int i = 0; i < 5; ++i) {
        ragger::KnowledgeInput k;
        k.kind = "note";
        k.text = "lr79 knowledge note number " + std::to_string(i);
        db.store_knowledge(k);
    }
    ragger::TimelineQuery tq;
    tq.granularity = "knowledge";
    tq.limit = 2;
    std::set<std::string> seen_ids;
    int pages = 0;
    while (true) {
        const auto tl = db.agent_timeline(tq);
        for (const auto& it : tl["items"]) {
            CHECK(it["id"].get<std::string>().starts_with("knowledge:"));
            CHECK(it["kind"] == "note");
            CHECK(seen_ids.insert(it["id"].get<std::string>()).second);
        }
        ++pages;
        if (tl["next_cursor"].is_null()) break;
        tq.cursor = tl["next_cursor"].get<std::string>();
        CHECK(pages < 10);   // guard: paging must terminate
    }
    CHECK(pages == 3);
    CHECK(seen_ids.size() == 5);

    db.close();
    cleanup();
    std::println(" OK");
}

// LR S74 (review #2, Q-A): per-corpus candidate pools. One merged pool of
// max(limit*5,50) lets a big corpus starve a narrow filter (live probe:
// layers:[turn_summary] + session -> 0 hits at limit 3). Fixture: 3 target
// L2 rows in one session, outranked on the query by 80 denser L2 rows in
// another session and 60 knowledge rows. Per-corpus pools must return all
// 3 targets at limit 3.
void lr74_test_per_corpus_pools(ragger::Embedder& emb) {
    std::println("  lr74_test_per_corpus_pools...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string gt = "lr74-target", gf = "lr74-filler";
    for (int i = 0; i < 3; ++i) {
        const int t = db.store_turn("lr74 q " + std::to_string(i),
                                    "wombat target " + std::to_string(i),
                                    "test-model", false, gt,
                                    "2026-08-01 09:0" + std::to_string(i) + ":00");
        CHECK(db.finalize_turn_summary(t, "wombat target note " + std::to_string(i),
                                       "summarizer-model"));
    }
    for (int i = 0; i < 80; ++i) {
        const int t = db.store_turn("lr74 filler q " + std::to_string(i),
                                    "wombat wombat wombat wombat filler " + std::to_string(i),
                                    "test-model", false, gf,
                                    "2026-08-02 09:00:00");
        CHECK(db.finalize_turn_summary(t, "wombat wombat wombat wombat filler " + std::to_string(i),
                                       "summarizer-model"));
    }
    for (int i = 0; i < 60; ++i)
        CHECK(!db.store("wombat wombat wombat knowledge " + std::to_string(i)).empty());

    ragger::AgentSearchQuery q;
    q.query = "wombat";
    q.layers = {"turn_summary"};
    q.session = gt;
    q.limit = 3;
    auto hits = db.agent_search(q);
    std::println("    [pin] layers:[turn_summary] session:target limit:3 -> {} hits", hits.size());
    CHECK(hits.size() == 3);
    for (const auto& h : hits)
        CHECK(h.id.starts_with("turn_summary:"));

    db.close();
    cleanup();
    std::println(" OK");
}


// LR S80 (D2): evidence points at EPISODES, not turn summaries. Table
// knowledge_episode_evidence (CASCADE/RESTRICT); validation wording
// "evidence: episode N does not exist or is empty"; missing-evidence message
// names episode ids; fetch + expand=sources return episode handles.
void lr80_test_episode_evidence(ragger::Embedder& emb) {
    std::println("  lr80_test_episode_evidence...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string guid = "lr80-evidence";
    auto members = p5_seed_members(db, guid, {"2026-06-01 10:00:00", "2026-06-01 10:05:00"});
    ragger::EpisodeClose ec;
    ec.text = "lr80 episode text";
    ec.model_name = "summarizer-model";
    ec.session_guid = guid;
    ec.span_start = p5_epoch("2026-06-01 10:00:00");
    ec.span_end   = p5_epoch("2026-06-01 10:05:00");
    ec.member_turn_summary_ids = members;
    const int ep = db.store_episode_close(ec);
    CHECK(ep > 0);

    // Insight backed by an episode; a duplicated id is OR IGNORE (one row).
    ragger::KnowledgeInput in;
    in.kind = "insight";
    in.text = "lr80 insight backed by an episode";
    in.evidence = {ep, ep};
    const auto res = db.store_knowledge(in);
    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM knowledge_episode_evidence WHERE knowledge_id = {}",
        res.id)) == 1);
    CHECK(p1_scalar_int(raw,
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'knowledge_turn_evidence'") == 0);
    // RESTRICT: an episode with evidence cannot be deleted.
    CHECK(sqlite3_exec(raw, std::format(
        "DELETE FROM episodes WHERE episode_id = {}", ep).c_str(),
        nullptr, nullptr, nullptr) != SQLITE_OK);
    // CASCADE: deleting the knowledge row removes the link.
    CHECK(sqlite3_exec(raw, std::format(
        "DELETE FROM knowledge WHERE knowledge_id = {}", res.id).c_str(),
        nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM knowledge_episode_evidence") == 0);
    sqlite3_close(raw);

    // Unknown episode id -> the exact contract wording.
    ragger::KnowledgeInput bad;
    bad.kind = "insight";
    bad.text = "lr80 bad insight";
    bad.evidence = {9999};
    bool threw = false;
    try { db.store_knowledge(bad); }
    catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find(
            "evidence: episode 9999 does not exist or is empty") != std::string::npos;
    }
    CHECK(threw);

    // An ''-text episode is not evidence either.
    int empty_ep = 0;
    {
        sqlite3* r2 = p1_open_raw();
        CHECK(sqlite3_exec(r2, "INSERT INTO episodes (text, span_start, span_end)"
                               " VALUES ('', 1, 2)", nullptr, nullptr, nullptr) == SQLITE_OK);
        empty_ep = static_cast<int>(p1_scalar_int(r2, "SELECT last_insert_rowid()"));
        sqlite3_close(r2);
    }
    bad.evidence = {empty_ep};
    threw = false;
    try { db.store_knowledge(bad); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Missing evidence names episode ids (not turn_summary ids).
    bad.evidence.clear();
    threw = false;
    try { db.store_knowledge(bad); }
    catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("episode_id") != std::string::npos;
    }
    CHECK(threw);

    // fetch: evidence[] carries the episode id; expand=sources returns the
    // episode's full row.
    ragger::KnowledgeInput ok2;
    ok2.kind = "insight";
    ok2.text = "lr80 insight for fetch";
    ok2.evidence = {ep};
    const auto k2 = db.store_knowledge(ok2);
    ragger::AgentGetQuery g;
    g.ids = {std::format("knowledge:{}", k2.id)};
    auto got = db.agent_get(g);
    const auto& item = got["items"][0];
    CHECK(item["evidence"].size() == 1 && item["evidence"][0] == ep);
    g.expand = "sources";
    auto src = db.agent_get(g);
    const auto& ex = src["items"][0]["expanded"];
    CHECK(ex.size() == 1 && ex[0]["id"] == std::format("episode:{}", ep));
    CHECK(ex[0]["layer"] == "episode");

    // ---- LR S86: expand=sources follows the new hierarchy end to end:
    // knowledge -> its episodes (in order); episode -> member summaries in
    // time order with ''-marker members excluded; turn_summary -> its turn.
    auto members2 = p5_seed_members(db, guid, {"2026-06-01 11:00:00"});
    ragger::EpisodeClose ec2;
    ec2.text = "lr80 episode text two";
    ec2.model_name = "summarizer-model";
    ec2.session_guid = guid;
    ec2.span_start = p5_epoch("2026-06-01 11:00:00");
    ec2.span_end   = p5_epoch("2026-06-01 11:05:00");
    ec2.member_turn_summary_ids = members2;
    const int ep2 = db.store_episode_close(ec2);
    CHECK(ep2 > 0);

    ragger::KnowledgeInput d86;
    d86.kind = "decision";
    d86.text = "lr80 decision citing two episodes";
    d86.trigger = "when probing s86";
    d86.evidence = {ep, ep2};
    const auto dres = db.store_knowledge(d86);
    g.ids = {std::format("knowledge:{}", dres.id)};
    g.expand = "sources";
    const auto got86 = db.agent_get(g);
    const auto& ex86 = got86["items"][0]["expanded"];
    CHECK(ex86.size() == 2 && ex86[0]["id"] == std::format("episode:{}", ep)
                              && ex86[1]["id"] == std::format("episode:{}", ep2));

    // A ''-text marker member joins ep; expansion must exclude it.
    // (turn_summaries is UNIQUE(turn_id), so the marker rides its own turn.)
    auto marker_member = p5_seed_members(db, guid, {"2026-06-01 12:00:00"})[0];
    {
        sqlite3* r3 = p1_open_raw();
        char* errmsg = nullptr;
        const int rc3 = sqlite3_exec(r3, std::format(
            "UPDATE turn_summaries SET text = '', episode_id = {}"
            " WHERE turn_summary_id = {}", ep, marker_member).c_str(),
            nullptr, nullptr, &errmsg);
        if (rc3 != SQLITE_OK) std::println("  marker update: {}", errmsg ? errmsg : "?");
        CHECK(rc3 == SQLITE_OK);
        sqlite3_close(r3);
    }
    g.ids = {std::format("episode:{}", ep)};
    const auto ep_src = db.agent_get(g);
    const auto& ex_ep = ep_src["items"][0]["expanded"];
    CHECK(ex_ep.size() == members.size());   // the marker member is excluded
    for (size_t i = 0; i < members.size(); ++i)
        CHECK(ex_ep[i]["id"] == std::format("turn_summary:{}", members[i]));

    // turn_summary -> its turn (turn:N).
    g.ids = {std::format("turn_summary:{}", members[0])};
    const auto ts_src = db.agent_get(g);
    const auto& ex_ts = ts_src["items"][0]["expanded"];
    CHECK(ex_ts.size() == 1 && ex_ts[0]["layer"] == "turn");
}


// LR S81 (D5): sessions as a lexical-only title corpus. Search layer
// `session` (hit id session:<guid>, matched_by lexical, extra
// {started_on, ended_on}); semantic mode excludes it; the default corpus
// set never includes it; reindex rebuilds sessions_terms from name alone;
// embedding loops skip the table.
void lr81_test_session_corpus(ragger::Embedder& emb) {
    std::println("  lr81_test_session_corpus...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string g1 = "lr81-named";
    CHECK(db.store_turn("lr81 q", "lr81 a", "test-model", false, g1, "",
                        "Zephyr gardening plans", "test") > 0);
    const std::string g2 = "lr81-unnamed";
    CHECK(db.store_turn("lr81 q2", "lr81 a2", "test-model", false, g2) > 0);

    // A word in the title finds the session; unnamed stays absent.
    ragger::AgentSearchQuery q;
    q.query = "zephyr gardening";
    q.layers = {"session"};
    auto hits = db.agent_search(q);
    CHECK(hits.size() == 1);
    CHECK(hits[0].id == "session:" + g1);
    CHECK(hits[0].layer == "session");
    CHECK(hits[0].matched_by == "lexical");
    CHECK(hits[0].extra.contains("started_on"));
    CHECK(hits[0].extra.contains("ended_on"));

    // semantic mode excludes the lexical-only corpus; keyword finds it.
    q.mode = "semantic";
    CHECK(db.agent_search(q).empty());
    q.mode = "keyword";
    CHECK(db.agent_search(q).size() == 1);
    q.mode = "hybrid";

    // The default corpus set never includes sessions.
    ragger::AgentSearchQuery d;
    d.query = "zephyr gardening plans";
    for (const auto& h : db.agent_search(d))
        CHECK(h.layer != "session");

    // reindex rebuilds sessions_terms from name alone: a raw UPDATE bypasses
    // the write-path index, so the row is invisible until reindex_table.
    const std::string g3 = "lr81-rawname";
    CHECK(db.store_turn("lr81 q3", "lr81 a3", "test-model", false, g3) > 0);
    {
        sqlite3* raw = p1_open_raw();
        CHECK(sqlite3_exec(raw, "UPDATE sessions SET name = 'Wombat tax review'"
                                " WHERE guid = 'lr81-rawname'",
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
    }
    q.query = "wombat tax";
    CHECK(db.agent_search(q).empty());            // not indexed yet
    db.reindex_table("sessions");
    auto rh = db.agent_search(q);
    CHECK(rh.size() == 1 && rh[0].id == "session:" + g3);

    // Embedding loops skip the lexical-only table.
    CHECK(db.count_embeddable_rows("sessions") == 0);
    CHECK(db.rebuild_embeddings(emb, {}, "sessions") == 0);
    db.close();
}

// ---------------------------------------------------------------------------
// LR S90 — maintenance-verb contract: dry-run (SELECT-only, writes nothing),
// cooperative cancel (committed work stays, the in-flight transaction rolls
// back, a rerun finishes the rest), and failure injection (the failing row
// unchanged, earlier rows kept, the message names the row). Reid runs the
// real verbs (rule 17); this pins the code.
// ---------------------------------------------------------------------------
void lr90_test_maintenance_verbs(ragger::Embedder& emb) {
    std::println("  lr90_test_maintenance_verbs...");
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    sqlite3* raw = p1_open_raw();

    // ---- reindex: plan counts, cancel after the first batch, rerun finishes
    // Chunks hang off one source: kDocEmbedTextSQL is NULL for a source-less
    // chunk (three-valued logic), and such chunks are neither embeddable nor
    // indexable by design — store_document always attaches a source.
    CHECK(sqlite3_exec(raw,
        "INSERT INTO document_sources (title, imported_at) VALUES ('s90 doc', 100)",
        nullptr, nullptr, nullptr) == SQLITE_OK);
    for (int i = 1; i <= 205; ++i) {
        const std::string sql = std::format(
            "INSERT INTO documents (text, chunk_index, document_source_id, modified_on)"
            " VALUES ('alpha bravo charlie doc {}', {}, 1, 100)", i, i);
        CHECK(sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
    }
    {
        auto plan = db.reindex_plan("documents");
        CHECK(plan.size() == 1 && plan[0].first == "documents" && plan[0].second == 205);
        CHECK(db.reindex_plan("all").size() == ragger::sqlite::kTextTableCount);
    }
    {
        ragger::CancelFlag cancel{false};
        const ragger::ProgressFn prog = [&](const char*, int done, int) {
            if (done >= 200) cancel.store(true);
        };
        bool threw = false;
        try { db.reindex_table("documents", prog, &cancel); }
        catch (const ragger::cancelled_error&) { threw = true; }
        CHECK(threw);
        // Batch one committed; the rest untouched (consistent partial state).
        CHECK(p1_scalar_int(raw,
              "SELECT COUNT(DISTINCT document_id) FROM documents_terms") == 200);
        // Rerun without cancel finishes the rest.
        CHECK(db.reindex_table("documents") == 205);
        CHECK(p1_scalar_int(raw,
              "SELECT COUNT(DISTINCT document_id) FROM documents_terms") == 205);
    }
    {
        // reindex_all is ONE transaction: a cancel rolls back the terms reset
        // and every partial batch — the DB reads exactly as before the run.
        const int terms_before = p1_scalar_int(raw, "SELECT COUNT(*) FROM terms");
        const int jt_before = p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms");
        ragger::CancelFlag cancel{false};
        const ragger::ProgressFn prog = [&](const char*, int done, int) {
            if (done >= 1) cancel.store(true);
        };
        bool threw = false;
        try { db.reindex_all(prog, &cancel); }
        catch (const ragger::cancelled_error&) { threw = true; }
        CHECK(threw);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms") == jt_before);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM terms") == terms_before);
    }

    // ---- re-embed: cancel after the first row; backfill finishes the rest
    std::vector<int> ts_ids;
    for (int i = 1; i <= 3; ++i) {
        const int tt = db.store_turn(
            std::format("question number {} alpha bravo", i),
            std::format("answer number {} charlie delta", i),
            "turn-model", false, "s90-a");
        ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on)"
                            " VALUES (?, ?, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000000)");
        s.bind(1, tt).bind(2, std::format("summary text number {} echo foxtrot", i)).bind(3, tt);
        CHECK(s.exec());
        ts_ids.push_back(bf_ts_id(raw, tt));
    }
    {
        ragger::CancelFlag cancel{false};
        const ragger::ProgressFn prog = [&](const char*, int done, int) {
            if (done >= 1) cancel.store(true);
        };
        bool threw = false;
        try { db.rebuild_embeddings(emb, prog, "turn_summaries", &cancel); }
        catch (const ragger::cancelled_error&) { threw = true; }
        CHECK(threw);
        CHECK(p1_scalar_int(raw,
              "SELECT COUNT(*) FROM turn_summaries WHERE embedding IS NOT NULL") == 1);
        // Rerun (cheap stale/missing mode) completes the rest.
        CHECK(db.backfill_embeddings(emb, "turn_summaries") == 2);
        CHECK(p1_scalar_int(raw,
              "SELECT COUNT(*) FROM turn_summaries WHERE embedding IS NOT NULL") == 3);
    }

    // ---- backfill: cancel after the first row; rerun applies the rest
    {
        const std::string long_u = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima ";
        const std::string long_a = "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu ";
        for (int i = 0; i < 3; ++i) {
            const int tt = db.store_turn(long_u + std::to_string(i), long_a + std::to_string(i),
                                         "turn-model", false, "s90-b");
            ragger::Stmt s(raw, "INSERT INTO turn_summaries (turn_id, text, turn_datetime, summarized_on)"
                                " VALUES (?, NULL, (SELECT created_at FROM turns WHERE turn_id = ?), 1700000000)");
            s.bind(1, tt).bind(2, tt);
            CHECK(s.exec());
        }
        bf_stub stub("Brief recap.");
        auto client = stub.client();
        ragger::CancelFlag cancel{false};
        const ragger::ProgressFn prog = [&](const char*, int done, int) {
            if (done >= 1) cancel.store(true);
        };
        bool threw = false;
        try { ragger::backfill_l2(db, client, "null", false, "", {}, prog, &cancel); }
        catch (const ragger::cancelled_error&) { threw = true; }
        CHECK(threw);
        CHECK(p1_scalar_int(raw,
              "SELECT COUNT(*) FROM turn_summaries WHERE text = 'Brief recap.'") == 1);
        auto res = ragger::backfill_l2(db, client, "null", false);
        CHECK(res.candidates == 2 && res.applied == 2);
    }

    // ---- failure injection: failing row unchanged, earlier rows kept,
    //      the message names the row.
    {
        // re-embed: a BEFORE UPDATE trigger aborts the write for one row.
        CHECK(sqlite3_exec(raw, "UPDATE turn_summaries SET embedding_version = 999",
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        const int victim = ts_ids[1];
        const std::string trg = std::format(
            "CREATE TRIGGER s90_inj BEFORE UPDATE ON turn_summaries WHEN old.turn_summary_id = {}"
            " BEGIN SELECT RAISE(ABORT, 'injected'); END", victim);
        CHECK(sqlite3_exec(raw, trg.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        std::string msg;
        bool threw = false;
        try { db.rebuild_embeddings(emb, {}, "turn_summaries"); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        sqlite3_exec(raw, "DROP TRIGGER s90_inj", nullptr, nullptr, nullptr);
        CHECK(threw);
        CHECK(msg.find("turn_summaries") != std::string::npos);
        CHECK(msg.find(std::to_string(victim)) != std::string::npos);
        // The victim kept the sentinel version (its UPDATE never landed);
        // the earlier row was re-encoded.
        CHECK(p1_scalar_int(raw, "SELECT embedding_version FROM turn_summaries WHERE turn_summary_id = "
                                  + std::to_string(victim)) == 999);
        CHECK(p1_scalar_int(raw, "SELECT embedding_version != 999 FROM turn_summaries WHERE turn_summary_id = "
                                  + std::to_string(ts_ids[0])) == 1);
    }
    {
        // reindex: a BEFORE INSERT trigger aborts one record's junction write;
        // its whole batch rolls back (per-batch transaction).
        const int doc2 = static_cast<int>(p1_scalar_int(raw, "SELECT MIN(document_id) FROM documents"));
        const std::string trg = std::format(
            "CREATE TRIGGER s90_inj2 BEFORE INSERT ON documents_terms WHEN new.document_id = {}"
            " BEGIN SELECT RAISE(ABORT, 'injected'); END", doc2);
        CHECK(sqlite3_exec(raw, trg.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        const int jt_before = p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms");
        std::string msg;
        bool threw = false;
        try { db.reindex_table("documents"); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        sqlite3_exec(raw, "DROP TRIGGER s90_inj2", nullptr, nullptr, nullptr);
        CHECK(threw);
        CHECK(msg.find("documents") != std::string::npos);
        CHECK(msg.find(std::to_string(doc2)) != std::string::npos);
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM documents_terms") == jt_before);
    }

    // ---- dry-run paths: counts only, DB byte-identical.
    {
        db.store_turn("unsummarized question", "unsummarized answer",
                      "turn-model", false, "s90-c");
        // Normalize versions so the stale count is deterministic.
        CHECK(sqlite3_exec(raw, ("UPDATE turn_summaries SET embedding_version = "
                                 + std::to_string(db.embedding_version())
                                 + " WHERE embedding IS NOT NULL").c_str(),
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        const std::string before = bf_dump(raw);
        CHECK(db.count_embeddable_rows("all") > 0);
        (void)db.reindex_plan("all");
        const auto plan = db.housekeeping_plan(100000.0f);   // cutoff far in the past
        CHECK(plan.purge_turns == 0);
        CHECK(plan.unsummarized == 1);
        CHECK(plan.embed_missing == 205);   // the raw-inserted document chunks
        CHECK(plan.embed_stale == 0);
        CHECK(bf_dump(raw) == before);
    }
    sqlite3_close(raw);
    db.close();
}

int main() {
    TestEnv env("sqlite_backend");  // must be first: isolates from ~/.ragger
    // Need config for model dir
    auto model_dir = env.model_dir();

    // Model-free unit tests run even when the embedding model is absent.
    p1r_test_transaction();
    p3_test_known_layers();
    p3s_test_status_vocabulary();
    p10_test_agent_settable_targets();
    sq_test_semqlite_config();

    if (!fs::exists(model_dir + "/model.onnx") && !fs::exists(model_dir + "/onnx/model.onnx")) {
        std::cerr << "Skipping backend tests: model not found at " << model_dir << "\n";
        std::println("test_sqlite_backend: SKIPPED (no model)");
        return 0;
    }

    ragger::Embedder emb(model_dir);

    test_store_and_count(emb);
    test_store_with_metadata(emb);
    test_search_basic(emb);
    test_search_fts_special_chars(emb);
    test_search_min_score(emb);
    test_search_min_score_before_topk(emb);
    test_search_limit(emb);
    test_load_all(emb);
    test_rebuild_embeddings(emb);
    test_rebuild_embeddings_empty_db(emb);
    test_rebuild_embeddings_count_matches(emb);
    test_search_text_only(emb);
    test_search_timing(emb);
    test_search_by_metadata(emb);
    test_timestamp_format(emb);
    test_v2_summaries_backing(emb);
    test_store_document(emb);
    test_search_merges_three_corpora(emb);
    test_turn_summary_search_metadata(emb);
    test_store_turn(emb);
    test_store_turn_dedup(emb);
    test_store_turn_dedup_same_session(emb);
    test_store_turn_creates_no_turn_summaries_row(emb);
    test_store_turn_partial_turn_creates_no_turn_summaries_row(emb);
    test_finalize_turn_summary_rejects_duplicate(emb);
    test_finalize_turn_summary_missing_turn_returns_false(emb);
    test_turn_summary_survives_turn_deletion(emb);
    test_unsummarized_turns_excludes_summarized(emb);
    test_mark_turn_summarized_trivial_turn(emb);
    test_reset_abandoned_turn_summaries(emb);
    p9a_test_reset_null_text_rows(emb);
    p9a_test_rejects_pre_016_databases(emb);
    test_note_primitives(emb);
    test_path_normalization(emb);
    test_structured_decision_basic(emb);
    test_structured_decision_rebuild_is_stable(emb);
    test_structured_decision_validation(emb);
    test_structured_decision_supersedes(emb);
    test_structured_decision_evidence(emb);
    test_orphan_turn_summary_protected(emb);
    test_migrate_0_16_to_0_17(emb);
    test_migrate_0_16_to_0_17_rejects(emb);
    test_p9_schema_assembly(emb);
    test_document_path_unique(emb);
    test_document_list_and_delete(emb);
    test_invalidate_all_caches(emb);
    test_p5_insight_basic(emb);
    test_p5_insight_validation(emb);
    test_p5_insight_rebuild_is_stable(emb);
    test_p5_insight_layer(emb);
    test_p7_knowledge_kinds(emb);
    test_p7_cross_kind_supersedes(emb);
    test_p7_kinds_filter(emb);
    test_p7_legacy_handles_rejected(emb);
    test_p8_summary_layer_gone(emb);
    test_p4_timeline_order_and_paging(emb);
    test_p4_timeline_session_fallback(emb);
    test_p4_timeline_turn_fallback(emb);
    test_p4_update_rules(emb);
    test_p4_update_cache_invalidation(emb);
    test_p4_review(emb);
    test_p2_text_only_hides_superseded(emb);
    test_p2_search_layers(emb);
    test_p2_search_status(emb);
    test_p2_search_since(emb);
    test_p2_search_snippet(emb);
    test_p2_search_max_tokens(emb);
    test_p2_get_decision(emb);
    test_p2_get_chain(emb);
    test_p2_get_context(emb);
    test_p2_get_history(emb);
    test_p2_get_mode_exclusive(emb);
    test_p2_possible_supersedes(emb);

    p1_test_turn_summary_index_keying(emb);
    p1t_test_import_probes(emb);
    p11_search_by_metadata_boundaries(emb);
    p11_agent_search_unready_window(emb);
    p11_timeline_turn_interleaved_sessions(emb);
    p11_time_bound_formats(emb);
    p1_test_embedder_swap(emb, model_dir);
    p2_test_close_really_closes(emb);
    p2_test_write_paths_postings(emb);
    p2_test_store_turn_under_outer_begin(emb);
    p2_test_has_embeddings_turn_summaries(emb);
    p10_cleanup_old_conversations(emb);
    p10_summary_insert_failures(emb);
    p10_defer_binds_null(emb);
    p10_knowledge_get_absent_keys(emb);
    p10_text_only_one_path(emb);
    p10_turn_meta_columns(emb);
    // p10_unknown_session_guid: FINDING 2026-11-03 parked; RULED 2026-11-05
    // ("matches nothing" is the contract) — registered with the fix.
    p10_unknown_session_guid(emb);
    p10_embeddable_counts(emb);
    p2_test_backfill_invalidates_dec_cache(emb);
    p2_test_db_version_gate(emb);
    p2_test_require_embedder(emb);
    p3_cache_pin(emb);
    p3_test_update_embedding(emb);
    p4_session_span(emb);
    p5_test_episode_close(emb);
    p9b_test_episode_close_unknown_session(emb);
    lr56_test_episode_created_at_is_span_end(emb);
    lr57_test_finalize_turn_merges(emb);
    p9b_test_timestamp_formats(emb);
    p9b_test_checked_writes(emb);
    p9b_test_reindex_atomic(emb);
    p9b_test_store_one_function(emb);
    lr60_test_decision_status_via_agent_update(emb);
    p5_test_episodes_needing_close(emb);
    p5_test_episode_reads(emb);
    p6_test_pending_predicate(emb);
    lr66_test_backfill_l2(emb);
    lr66_test_backfill_triggers(emb);
    lr67_test_backfill_stranded(emb);
    lr67_test_backfill_session_names(emb);
    lr68_test_time_invariants(emb);
    lr73_test_episode_session_filter_and_extra(emb);
    lr74_test_per_corpus_pools(emb);
    lr75_test_raw_turn_opt_in(emb);
    lr76_test_document_search(emb);
    lr77_test_search_mode(emb);
    lr78_test_tags_filter(emb);
    lr79_test_history_hierarchy(emb);
    lr80_test_episode_evidence(emb);
    lr81_test_session_corpus(emb);
    lr90_test_maintenance_verbs(emb);

    std::println("test_sqlite_backend: all passed");
    return 0;
}
