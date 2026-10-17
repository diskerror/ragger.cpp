/**
 * SQLite backend integration tests
 *
 * Requires ONNX model files at the configured model_dir.
 * Uses a temp DB that's deleted after the test.
 */
#include "config.h"
#include "embedder.h"
#include "sqlite/backend.h"
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

    // Lean v2 summaries: tags (comma-joined) + level round-trip.
    // status was removed from summaries in v0.12.0; any status key in
    // metadata is silently dropped.
    ragger::json meta = {
        {"level", "session"},
        {"tags", {"test", "unit"}}
    };
    std::string id = db.store("Metadata round-trip test.", meta);
    CHECK(!id.empty());

    auto all = db.load_all();
    CHECK(all.size() == 1);
    CHECK(all[0].metadata["level"] == "session");
    // Tags stored as comma-separated string in the tags column.
    CHECK(all[0].metadata["tags"] == "test,unit");

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

void test_delete_memory(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Store a memory
    std::string id1 = db.store("Memory to delete.");
    std::string id2 = db.store("Memory to keep.");
    CHECK(db.count() == 2);

    // Delete the first memory
    bool deleted = db.delete_memory(std::stoi(id1));
    CHECK(deleted);
    CHECK(db.count() == 1);

    // Verify the correct one was deleted
    auto all = db.load_all();
    CHECK(all.size() == 1);
    CHECK(all[0].text == "Memory to keep.");

    // Try deleting non-existent ID
    deleted = db.delete_memory(99999);
    CHECK(!deleted);
    CHECK(db.count() == 1);

    db.close();
    cleanup();
}

void test_delete_batch(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Store 3 memories
    std::string id1 = db.store("First memory.");
    std::string id2 = db.store("Second memory.");
    std::string id3 = db.store("Third memory.");
    CHECK(db.count() == 3);

    // Delete 2 by ID
    std::vector<int> to_delete = {std::stoi(id1), std::stoi(id3)};
    int deleted_count = db.delete_batch(to_delete);
    CHECK(deleted_count == 2);
    CHECK(db.count() == 1);

    // Verify the correct one remains
    auto all = db.load_all();
    CHECK(all.size() == 1);
    CHECK(all[0].text == "Second memory.");

    // Empty vector → returns 0
    deleted_count = db.delete_batch({});
    CHECK(deleted_count == 0);

    // Delete with non-existent IDs
    deleted_count = db.delete_batch({99999, 88888});
    CHECK(deleted_count == 0);

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

    // Filter by level
    auto results = db.search_by_metadata({{"level", "turn"}});
    CHECK(results.size() == 2);
    for (auto& r : results) CHECK(r.metadata["level"] == "turn");

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

void test_delete_respects_keep(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int initial_count = db.count();
    
    // Store a memory with {"keep": true} in metadata
    ragger::json meta = {{"keep", true}, {"collection", "memory"}};
    auto id = db.store("protected memory", meta);
    
    // Try to delete — should return false
    CHECK(!db.delete_memory(std::stoi(id)));
    
    // Memory should still exist
    CHECK(db.count() == initial_count + 1);
    
    // Store without keep
    auto id2 = db.store("deletable memory", {{"collection", "memory"}});
    
    // Delete should work
    CHECK(db.delete_memory(std::stoi(id2)));
    
    // Only the protected one should remain
    CHECK(db.count() == initial_count + 1);
    
    db.close();
    cleanup();
}

void test_delete_batch_respects_keep(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Store mix of keep and non-keep
    ragger::json keep_meta = {{"keep", true}, {"collection", "memory"}};
    ragger::json normal_meta = {{"collection", "memory"}};
    
    auto id1 = db.store("keep me", keep_meta);
    auto id2 = db.store("delete me", normal_meta);
    auto id3 = db.store("delete me too", normal_meta);
    
    int initial_count = db.count();
    
    // Batch delete all three
    int deleted = db.delete_batch({std::stoi(id1), std::stoi(id2), std::stoi(id3)});
    
    // Only 2 should be deleted (not the keep one)
    CHECK(deleted == 2);
    CHECK(db.count() == initial_count - 2);
    
    // Verify the protected one remains
    auto all = db.load_all();
    bool found_protected = false;
    for (const auto& mem : all) {
        if (mem.text == "keep me") {
            found_protected = true;
            break;
        }
    }
    CHECK(found_protected);
    
    db.close();
    cleanup();
}

void test_timestamp_format(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    db.store("Timestamp format test.");
    auto all = db.load_all();
    CHECK(all.size() == 1);

    // `summaries.created_at` is now an INTEGER epoch (v0.12.0 migration),
    // not the old "YYYY-MM-DD HH:MM:SS" TEXT format. load_all() reads the
    // column via column_text() on a raw INTEGER, which SQLite renders back
    // as plain base-10 digits (no separators) -- confirm that shape and
    // that it's a plausible "recent" epoch value, rather than asserting the
    // old fixed-19-char formatted-string length.
    auto& ts = all[0].timestamp;
    CHECK(!ts.empty());
    CHECK(ts.find_first_not_of("0123456789") == std::string::npos);
    int64_t epoch_val = std::stoll(ts);
    int64_t now_epoch = static_cast<int64_t>(std::time(nullptr));
    CHECK(epoch_val > 0);
    CHECK(epoch_val <= now_epoch + 5);       // not in the future
    CHECK(now_epoch - epoch_val < 300);      // stored "just now"

    db.close();
    cleanup();
}

// Guards the v2 read-path wiring: the generic store/read API operates on the
// lean `summaries` table (issue #33 split), not the dropped pre-v2 `memories`
// table. level/tags round-trip; there is no metadata blob.
// status was removed from summaries in v0.12.0.
void test_v2_summaries_backing(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);
        db.store("v2 summaries backing check.",
                 {{"level", "session"}, {"tags", {"note"}}});

        auto all = db.load_all();
        CHECK(all.size() == 1);
        CHECK(all[0].metadata["level"] == "session");
        CHECK(all[0].metadata["tags"] == "note");
        db.close();
    }

    // Inspect the raw DB: the row must be in `summaries`, and the legacy
    // `memories` table must not exist in a fresh v2 database.
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM summaries", -1, &st, nullptr);
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

// Retry/regeneration dedup: when the agent re-answers the SAME user prompt
// without an intervening new prompt, store_turn must keep-latest (update the
// prior row's assistant_text in place) instead of inserting a duplicate. The
// match is cross-session by design — a session change between two identical
// prompts is itself a breakage signal (TUI crash-restart minting fresh GUIDs).
void test_store_turn_dedup(ragger::Embedder& emb) {
    cleanup();
    int t1 = 0;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        // First answer to a prompt, in session A.
        t1 = db.store_turn("Explain the general_search recipe.",
                               "It searches summaries.", "model-a",
                               /*defer_embedding=*/false, "sessionA");
        CHECK(t1 > 0);

        // Sleep across a second boundary: local_timestamp() has second-level
        // resolution, so without a real gap here old_ts would trivially equal
        // new_ts and this test would not exercise the drift this guards
        // against (a dedup that lands in the same second as the original).
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));

        // Same prompt re-answered — DIFFERENT session (sessionB), different
        // assistant text. Must update t1 in place, not insert a new row.
        int t2 = db.store_turn("Explain the general_search recipe.",
                               "It searches summaries, documents, and decisions.",
                               "model-b", /*defer_embedding=*/false, "sessionB");
        CHECK(t2 == t1);  // same row id returned — keep-latest

        // Cross a second boundary before the next insert: local_timestamp()
        // has second-level resolution, and turn3 landing in the same second
        // as t1's dedup-updated created_at is itself a separate known
        // same-second collision (harmless for the summarizer's matching, but
        // it would otherwise make this assertion block conflate two issues).
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));

        // A genuinely different prompt still inserts a fresh row.
        int t3 = db.store_turn("What about decisions?", "They are first-class.",
                               "model-b", /*defer_embedding=*/false, "sessionB");
        CHECK(t3 != t1);

        db.close();
    }

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    // Exactly two turn rows: the deduped pair collapsed to one, plus the
    // distinct third prompt.
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turns", -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);

    // The surviving deduped row carries the LATEST assistant_text and is not
    // duplicated in the FTS index (one hit for the prompt's distinctive term).
    sqlite3_prepare_v2(raw,
        "SELECT assistant_text FROM turns "
        "WHERE user_text = 'Explain the general_search recipe.'",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* a = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        CHECK(a && std::string(a).find("decisions") != std::string::npos);
    }
    // Only one such row (no duplicate).
    CHECK(sqlite3_step(st) != SQLITE_ROW);
    sqlite3_finalize(st);

    // Custom terms index agrees: the prompt's distinctive stemmed bigram term
    // resolves to exactly one turn_id (dedup didn't leave a stale/duplicate
    // turns_terms row behind). Snowball's English algorithm leaves "general"
    // unstemmed (unlike the old Porter pipeline, which reduced it to
    // "genera"), so the stored literal bigram token is "general_search".
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM turns_terms tt "
        "JOIN terms t ON t.term_id = tt.term_id "
        "WHERE t.term = 'general_search'",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);

    // Regression: a dedup that moves the turn's created_at must not leave a
    // stray turn_summaries row for this turn. Since placeholders no longer
    // exist (per Reid's design decision — no placeholder mechanism), the
    // deduped turn should have ZERO turn_summaries rows until explicitly
    // finalized.
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM turn_summaries ts WHERE ts.turn_id = ?",
        -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_int(st, 0) == 0);
    sqlite3_finalize(st);

    sqlite3_close(raw);
    cleanup();
}

// PHASE-REVIEW-0.17 step 2: same-session regeneration dedup. The existing
// test_store_turn_dedup covers the cross-session path (crash-restart GUID
// churn); this pins the ordinary healthy-agent case: the re-answer arrives
// in the SAME session as the original prompt.
void test_store_turn_dedup_same_session(ragger::Embedder& emb) {
    cleanup();
    const std::string S = "dedup-same-session-guid";
    int t1 = 0;
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);

        t1 = db.store_turn("U1", "A1", "test-model",
                           /*defer_embedding=*/false, S);
        CHECK(t1 > 0);

        // Same prompt, same session, new answer, inside the 5-minute window
        // (no sleep needed: same-second is well within it). Must keep-latest.
        int t2 = db.store_turn("U1", "A2", "test-model",
                               /*defer_embedding=*/false, S);
        CHECK(t2 == t1);  // same row id returned — dedup branch reached

        db.close();
    }

    {
        // Exactly one row for that session after the deduped pair, and the
        // surviving row carries the LATEST assistant_text.
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(raw,
            "SELECT COUNT(*) FROM turns t "
            "JOIN sessions s ON s.session_id = t.session_id WHERE s.guid = ?",
            -1, &st, nullptr);
        sqlite3_bind_text(st, 1, S.c_str(), -1, SQLITE_STATIC);
        CHECK(sqlite3_step(st) == SQLITE_ROW);
        CHECK(sqlite3_column_int(st, 0) == 1);
        sqlite3_finalize(st);

        sqlite3_prepare_v2(raw,
            "SELECT assistant_text FROM turns WHERE turn_id = ?",
            -1, &st, nullptr);
        sqlite3_bind_int(st, 1, t1);
        CHECK(sqlite3_step(st) == SQLITE_ROW);
        const char* a = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        CHECK(a && std::string(a) == "A2");
        sqlite3_finalize(st);
        sqlite3_close(raw);
    }

    {
        // A genuinely different prompt in the same session inserts a fresh row.
        ragger::sqlite::Backend db(emb, TEMP_DB);
        int t3 = db.store_turn("U2", "A2b", "test-model",
                               /*defer_embedding=*/false, S);
        CHECK(t3 > 0 && t3 != t1);
        db.close();
    }

    {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(raw,
            "SELECT COUNT(*) FROM turns t "
            "JOIN sessions s ON s.session_id = t.session_id WHERE s.guid = ?",
            -1, &st, nullptr);
        sqlite3_bind_text(st, 1, S.c_str(), -1, SQLITE_STATIC);
        CHECK(sqlite3_step(st) == SQLITE_ROW);
        CHECK(sqlite3_column_int(st, 0) == 2);
        sqlite3_finalize(st);
        sqlite3_close(raw);
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


// Summary primitives (issue #22): store_summary, update_summary_text —
// the deterministic backbone the summarization pipeline drives.
// current_session_summary and set_summary_status were removed when
// summaries.status was dropped (v0.12.0 boundary-detection rework).
void test_summary_primitives(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    int l2 = db.store_summary("User asked about France; capital is Paris.",
                              "turn", "memo-model");
    CHECK(l2 > 0);

    int l3 = db.store_summary("Discussed European capitals.",
                              "session", "memo-model");
    CHECK(l3 > 0);

    CHECK(db.update_summary_text(l3, "Discussed European capitals, focus France.",
                                  "memo-model"));

    int l3b = db.store_summary("Switched to cooking techniques.",
                               "session", "memo-model");
    CHECK(l3b > 0);

    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM summaries WHERE level='turn'",
                       -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='session'",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries s JOIN models m ON m.model_id=s.model_id "
        "WHERE m.name='memo-model'", -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 3);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    CHECK(!db.update_summary_text(99999, "x", "memo-model"));

    // Recipe ingredients (issue #23): recency-based fetch.
    auto turns = db.recent_summaries("turn", 5);
    CHECK(turns.size() == 1);
    auto sessions = db.recent_summaries("session", 5);
    CHECK(sessions.size() == 2);
    CHECK(db.recent_summaries("project", 5).empty());
    CHECK(db.recent_summaries("turn", 0).empty());
    CHECK(db.current_decisions(5).empty());      // none created

    db.close();
    cleanup();
}

// Phase 2 (EPISODE_PLAN): episode layer + boundary-triggered session/project
// rollups. Deterministic backend-level verification of the bloat-bug fix — no
// inference needed. Reproduces the shape of the original 1956–1977 defect
// (repeated rollups after a "close") and asserts exactly one session row.
void test_episode_rollup_no_dup(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string G = "phase2-session";

    // No episodes yet.
    CHECK(db.last_episode_end(G).empty());
    CHECK(db.episode_texts(G).empty());

    // Two L2 turn summaries land (distinct timestamps → distinct rows).
    // Turn-level summaries now live in turn_summaries, keyed by a real
    // turn_id -- create the underlying turns first, then finalize each
    // one's summary, rather than writing level='turn' rows into
    // `summaries` directly (that path no longer exists post-db_version
    // 0.12; see l2_summaries_since's redirect to turn_summaries).
    int t1 = db.store_turn("Q1", "A1", "memo", false, G, "2026-01-01 10:00:00");
    int t2 = db.store_turn("Q2", "A2", "memo", false, G, "2026-01-01 10:00:02");
    CHECK(db.finalize_turn_summary(t1, "Turn one: discussed knowledge graphs.", "memo"));
    CHECK(db.finalize_turn_summary(t2, "Turn two: phonetic pun edges.", "memo"));

    // Before any episode, l2_summaries_since(all) returns both.
    CHECK(db.l2_summaries_since(G, "").size() == 2);

    // Close episode 1 spanning both turns.
    int ep1 = db.store_episode("Episode 1: KG design + phonetic edges.",
                               "memo", G, "2026-01-01 10:00:00",
                               "2026-01-01 10:00:02");
    CHECK(ep1 > 0);
    CHECK(db.last_episode_end(G) == "2026-01-01 10:00:02");
    CHECK(db.episode_texts(G).size() == 1);
    // No L2 past the episode end now.
    CHECK(db.l2_summaries_since(G, db.last_episode_end(G)).empty());

    // Second episode.
    int t3 = db.store_turn("Q3", "A3", "memo", false, G, "2026-01-01 11:00:00");
    CHECK(db.finalize_turn_summary(t3, "Turn three: curated fiction corpus.", "memo"));
    db.store_episode("Episode 2: curated corpus bet.", "memo", G,
                     "2026-01-01 11:00:00", "2026-01-01 11:00:00");
    CHECK(db.episode_texts(G).size() == 2);

    // Episode rows are immutable: timestamp=first, updated_at=last (span).
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='episode' "
        "AND session_id=(SELECT session_id FROM sessions WHERE guid='phase2-session')",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    // --- Boundary-detection session/project rollup (new design) ---------
    // A run closes on session_id edge (session) or a >= gap_days time gap
    // (project). Both are immutable-insert-per-closed-run, no upsert. We
    // exercise the raw store_session_summary/store_project_summary +
    // bounded_*_rollup_texts primitives directly here (the housekeeping
    // scan itself -- sessions_needing_close_boundary/watermark advance --
    // is exercised functionally in the summarizer smoke test, not here).

    // bounded_session_rollup_texts pulls in episodes + trailing turn
    // summaries whose span falls entirely within [first_ts, last_ts] for
    // this session_guid -- both episodes above qualify for a run spanning
    // the whole session.
    auto first_ep = ragger::parse_db_timestamp("2026-01-01 10:00:00");
    auto last_ep  = ragger::parse_db_timestamp("2026-01-01 11:00:00");
    CHECK(first_ep.has_value() && last_ep.has_value());
    auto sess_texts = db.bounded_session_rollup_texts(
        G, static_cast<int64_t>(*first_ep), static_cast<int64_t>(*last_ep));
    CHECK(sess_texts.size() == 2);  // both episode texts, oldest-first
    CHECK(sess_texts[0] == "Episode 1: KG design + phonetic edges.");
    CHECK(sess_texts[1] == "Episode 2: curated corpus bet.");

    // A LATER run of the SAME session_guid must not leak into an EARLIER
    // run's bounded query -- add a third episode outside [first_ep,last_ep]
    // and confirm it's excluded.
    int t4 = db.store_turn("Q4", "A4", "memo", false, G, "2026-01-02 09:00:00");
    CHECK(db.finalize_turn_summary(t4, "Turn four: unrelated later topic.", "memo"));
    db.store_episode("Episode 3: later unrelated run.", "memo", G,
                     "2026-01-02 09:00:00", "2026-01-02 09:00:00");
    auto sess_texts2 = db.bounded_session_rollup_texts(
        G, static_cast<int64_t>(*first_ep), static_cast<int64_t>(*last_ep));
    CHECK(sess_texts2.size() == 2);  // unchanged -- later episode excluded

    // store_session_summary: one immutable insert per closed run. Two
    // separate closes of the SAME session_guid produce TWO rows (unlike
    // the old running-upsert model, which kept exactly one).
    int sess1 = db.store_session_summary("Session run 1 rollup.", "memo", G,
                                         static_cast<int64_t>(*first_ep),
                                         static_cast<int64_t>(*last_ep));
    CHECK(sess1 > 0);
    auto later_ep = ragger::parse_db_timestamp("2026-01-02 09:00:00");
    CHECK(later_ep.has_value());
    int sess2 = db.store_session_summary("Session run 2 rollup.", "memo", G,
                                         static_cast<int64_t>(*later_ep),
                                         static_cast<int64_t>(*later_ep));
    CHECK(sess2 > 0);
    CHECK(sess1 != sess2);

    raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='session' "
        "AND session_id=(SELECT session_id FROM sessions WHERE guid='phase2-session')",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    // bounded_project_rollup_texts: gathers level='session' rows (produced
    // above) whose span falls entirely within [first_ts, last_ts],
    // session-unscoped.
    auto proj_texts = db.bounded_project_rollup_texts(
        static_cast<int64_t>(*first_ep), static_cast<int64_t>(*later_ep));
    CHECK(proj_texts.size() == 2);
    CHECK(proj_texts[0] == "Session run 1 rollup.");
    CHECK(proj_texts[1] == "Session run 2 rollup.");

    // store_project_summary: one immutable insert per closed project run.
    int proj1 = db.store_project_summary("Project run 1 rollup.", "memo",
                                         static_cast<int64_t>(*first_ep),
                                         static_cast<int64_t>(*later_ep));
    CHECK(proj1 > 0);

    raw = nullptr;
    CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*), session_id FROM summaries WHERE level='project'",
        -1, &st, nullptr);
    CHECK(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    CHECK(sqlite3_column_type(st, 1) == SQLITE_NULL);  // session-unscoped
    sqlite3_finalize(st);
    sqlite3_close(raw);

    // watermark advance is idempotent / monotonic bookkeeping -- exercised
    // via advance_session_boundary_watermark + a fresh
    // sessions_needing_close_boundary() call. With everything already
    // consumed above, calling it now with a turn_id shouldn't throw.
    db.advance_session_boundary_watermark(t4);
    db.advance_project_boundary_watermark(t4);

    db.close();
    cleanup();
}

// Regression: sessions_needing_close_boundary() must NOT re-close a run
// that already has a session summary covering it, even when the boundary
// watermark is zero/unset (e.g. right after a schema migration that never
// stamped the watermark key -- get_watermark() then defaults to 0).
//
// Reid's invariant: "there should not be an automatic lookback to do any
// session summary before the last session summary." The watermark is a
// belt-and-suspenders optimization; correctness must come structurally
// from the summaries that already exist, not from the watermark value.
//
// Pre-fix behavior (the bug): watermark=0 -> every historical closed run
// (last_turn_id > 0) re-qualifies -> the housekeeping scan re-summarizes
// all of session history, one duplicate level='session' row per tick.
void test_session_close_no_lookback_before_existing_summary(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    const std::string G1 = "already-summarized-session";
    const std::string G2 = "later-session";

    // A closed run for G1: two consecutive turns, then a turn in a DIFFERENT
    // session (G2) which edge-closes G1's run and becomes the open tail.
    int t1 = db.store_turn("Q1", "A1", "memo", false, G1, "2026-03-01 10:00:00");
    int t2 = db.store_turn("Q2", "A2", "memo", false, G1, "2026-03-01 10:00:05");
    CHECK(db.finalize_turn_summary(t1, "Turn one.", "memo"));
    CHECK(db.finalize_turn_summary(t2, "Turn two.", "memo"));
    (void)db.store_turn("Q3", "A3", "memo", false, G2, "2026-03-01 11:00:00");

    auto first_ts = ragger::parse_db_timestamp("2026-03-01 10:00:00");
    auto last_ts  = ragger::parse_db_timestamp("2026-03-01 10:00:05");
    CHECK(first_ts.has_value() && last_ts.has_value());

    // G1's run has already been summarized once (the normal, correct close).
    int s1 = db.store_session_summary("G1 run rollup.", "memo", G1,
                                      static_cast<int64_t>(*first_ts),
                                      static_cast<int64_t>(*last_ts));
    CHECK(s1 > 0);

    // Simulate the post-migration state: watermark is 0 (never stamped).
    db.advance_session_boundary_watermark(0);

    // The scan must NOT return G1's already-summarized run. (G2's run is the
    // open tail -- excluded by the MAX(grp) rule regardless.)
    auto runs = db.sessions_needing_close_boundary();
    for (const auto& r : runs) {
        CHECK(r.session_guid != G1 &&
               "already-summarized run re-closed with zeroed watermark (lookback bug)");
    }

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

        // (3) Decision inserted directly into the L6 table with NO embedding,
        // mirroring how the external summarizer writes decisions. backfill
        // then embeds it and must invalidate the decision cache.
        {
            sqlite3* raw = nullptr;
            CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
            const char* sql =
                "INSERT INTO decisions (text, status, tags, created_at) "
                "VALUES ('We will migrate the database to PostgreSQL next quarter.', "
                "'current', '', '2026-01-01T00:00:00Z')";
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

        CHECK(source_of("interstellar spacecraft golden record") == "summary");
        CHECK(source_of("how plants make energy from light")     == "document");
        CHECK(source_of("plan to switch to PostgreSQL database")  == "decision");

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
    CHECK(sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    long long v = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    return v;
}
static std::string p1_scalar_text(sqlite3* raw, const std::string& sql) {
    sqlite3_stmt* st = nullptr;
    CHECK(sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
    CHECK(sqlite3_step(st) == SQLITE_ROW);
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
static bool p1_throws_invalid(ragger::sqlite::Backend& db, const ragger::DecisionInput& in) {
    try { db.store_structured_decision(in); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_structured_decision_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::DecisionInput in;
    in.text      = "  Use a custom inverted index, not FTS5.  ";
    in.trigger   = "Use when choosing a full-text index strategy";
    in.rationale = "FTS5 MATCH is AND-only without query rewriting.";
    in.tags      = "search,sqlite";
    auto r = db.store_structured_decision(in);
    CHECK(r.id > 0);
    CHECK(!r.superseded.has_value());

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM decisions WHERE decision_id = {}", r.id);
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
        if (h.metadata.value("source", "") == "decision" &&
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
    ragger::DecisionInput in;
    in.text    = "Keep 'current' as the active decision status.";
    in.trigger = "Use when choosing decision status values";
    int id = db.store_structured_decision(in).id;

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM decisions WHERE decision_id = {}", id);
    const std::string blob_before = p1_scalar_text(raw, "SELECT hex(embedding)" + w);
    const long long   uni_before  = p1_scalar_int(raw, "SELECT unigram_count" + w);
    const long long   bi_before   = p1_scalar_int(raw, "SELECT bigram_count" + w);
    sqlite3_close(raw);

    db.rebuild_embeddings(emb, false, "decisions");
    db.reindex_table("decisions", false);

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
    ragger::DecisionInput ok;
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
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM decisions") == 0);   // nothing written
    sqlite3_close(raw);

    // provisional and pending are valid starting statuses.
    auto p = ok; p.status = "provisional";
    CHECK(db.store_structured_decision(p).id > 0);
    p.status = "pending";
    CHECK(db.store_structured_decision(p).id > 0);
    db.close();
    cleanup();
}

void test_structured_decision_supersedes(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    ragger::DecisionInput a;
    a.text = "Embed decisions as text only.";
    a.trigger = "Use when choosing what to embed for a decision";
    int a_id = db.store_structured_decision(a).id;

    ragger::DecisionInput b;
    b.text = "Embed decisions as trigger then text.";
    b.trigger = a.trigger;
    b.supersedes = a_id;
    auto rb = db.store_structured_decision(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT status FROM decisions WHERE decision_id = {}", a_id)) == "superseded");
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT supersedes FROM decisions WHERE decision_id = {}", rb.id)) == a_id);
    // Old row is kept, text untouched (honest history).
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT text FROM decisions WHERE decision_id = {}", a_id)) == "Embed decisions as text only.");
    sqlite3_close(raw);

    // Search no longer surfaces the superseded row; it does surface the new one.
    auto resp = db.search("Embed decisions as text only", 10, 0.0f);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "decision") continue;
        if (h.id == a_id)  saw_old = true;
        if (h.id == rb.id) saw_new = true;
    }
    CHECK(!saw_old);
    CHECK(saw_new);

    // A superseded row can't be superseded again.
    ragger::DecisionInput c = b;
    c.supersedes = a_id;
    CHECK(p1_throws_invalid(db, c));
    db.close();
    cleanup();
}

void test_structured_decision_evidence(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts1 = p1_make_turn_summary(db, "evidence one");
    int ts2 = p1_make_turn_summary(db, "evidence two");

    ragger::DecisionInput in;
    in.text = "Link decisions to turn summaries.";
    in.trigger = "Use when recording why a decision was made";
    in.evidence = {ts1, ts2, ts1};   // duplicate is ignored
    int id = db.store_structured_decision(in).id;

    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM decision_evidence WHERE decision_id = {}", id)) == 2);
    // RESTRICT: a turn summary that is evidence cannot be deleted.
    int rc = sqlite3_exec(raw, std::format(
        "DELETE FROM turn_summaries WHERE turn_summary_id = {}", ts1).c_str(),
        nullptr, nullptr, nullptr);
    CHECK(rc != SQLITE_OK);
    // CASCADE: deleting the decision removes its links.
    CHECK(sqlite3_exec(raw, std::format(
        "DELETE FROM decisions WHERE decision_id = {}", id).c_str(),
        nullptr, nullptr, nullptr) == SQLITE_OK);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM decision_evidence") == 0);
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
    CHECK(err && std::string(err).find("sole evidence") != std::string::npos);
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
// existing decisions_terms rows intact.
void test_migrate_0_16_to_0_17(ragger::Embedder& emb) {
    cleanup();
    {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
        const char* ddl = R"(
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO settings VALUES ('db_version', '0.16');
            CREATE TABLE turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT NOT NULL,
                assistant_text TEXT, model_id INTEGER, session_id INTEGER,
                created_at INTEGER NOT NULL,
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            CREATE TABLE decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, status TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER, embedding BLOB);
            INSERT INTO decisions (text, status, tags, unigram_count) VALUES
                ('first ruling', 'current', '', 2),
                ('a fact',       'fact',    '', 2),
                ('a lesson',     'lesson',  'x', 2);
            CREATE TABLE sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT NOT NULL UNIQUE, name TEXT, name_source TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()));
            INSERT INTO sessions (guid, name, name_source) VALUES
                ('guid-a', 'A', 'test'), ('guid-b', NULL, NULL);
            INSERT INTO turns (user_text, session_id, created_at) VALUES
                ('t1', 1, 1700000500), ('t2', 1, 1700000100), ('t3', 1, 1700000300);
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
            INSERT INTO summaries (text, level, tags, session_id, unigram_count) VALUES
                ('episode one', 'episode', '', 1, 7),
                ('note two',    'session', 'x', 2, 3);
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
        CHECK(sqlite3_exec(raw, ddl, nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
    }
    {
        ragger::sqlite::Backend db(emb, TEMP_DB);   // runs the migration
        db.close();
    }
    sqlite3* raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, "SELECT value FROM settings WHERE key='db_version'") == "0.17");
    // Canonical (fresh-install) column order: no project_id anywhere.
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('decisions') WHERE cid = 4") == "trigger");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('decisions') WHERE cid = 9") == "created_at");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('summaries') WHERE cid = 6") == "created_at");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('decisions') WHERE name = 'project_id'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('summaries') WHERE name = 'project_id'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'project_id'") == 0);
    // sessions gains the derived span columns (PHASE4).
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('sessions') WHERE cid = 4") == "started_on");
    CHECK(p1_scalar_text(raw, "SELECT name FROM pragma_table_info('sessions') WHERE cid = 5") == "ended_on");
    // Span derived from turns MIN/MAX; a session without turns is NULL/NULL.
    CHECK(p1_scalar_text(raw, "SELECT started_on || '-' || ended_on FROM sessions WHERE guid = 'guid-a'") == "1700000100-1700000500");
    CHECK(p1_scalar_text(raw, "SELECT ifnull(started_on,'NULL') || '-' || ifnull(ended_on,'NULL') FROM sessions WHERE guid = 'guid-b'") == "NULL-NULL");
    // The AUTOINCREMENT counter survives the rebuild.
    CHECK(p1_scalar_int(raw, "SELECT seq FROM sqlite_sequence WHERE name = 'sessions'") == 2);
    // Insights and their links exist; the projects table is GONE.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('insights','insight_evidence','insights_terms')") == 3);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'projects'") == 0);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='insights_view'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_insights_status'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM summaries") == 2);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count FROM summaries WHERE text = 'episode one'") == 7);
    // No stray rebuild tables / repointed REFERENCES (legacy_alter_table trap).
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE sql LIKE '%_rebuild%'") == 0);
    // document_sources.path is unique: pathless/empty sources got their own
    // document_source_id as a placeholder; real paths are untouched.
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_document_sources_path'") == 1);
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Doc'") == "~/doc.md");
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Pathless'") == "2");
    CHECK(p1_scalar_text(raw, "SELECT path FROM document_sources WHERE title = 'Empty'") == "3");
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM decisions") == 3);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM decisions WHERE status != 'current'") == 0);
    CHECK(p1_scalar_text(raw, "SELECT tags FROM decisions WHERE text = 'a fact'") == "fact");
    CHECK(p1_scalar_text(raw, "SELECT tags FROM decisions WHERE text = 'a lesson'") == "x,lesson");
    CHECK(p1_scalar_int(raw, "SELECT unigram_count FROM decisions WHERE text = 'first ruling'") == 2);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'decision_evidence'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'protect_orphan_turn_summaries'") == 1);
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM pragma_foreign_key_check") == 0);
    sqlite3_close(raw);

    // The pre-migration backup archive lands next to the DB; remove it.
    for (const auto& e : fs::directory_iterator(fs::path(TEMP_DB).parent_path()))
        if (e.path().filename().string().starts_with("ragger_test_backend_BACKUP_"))
            fs::remove(e.path());
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
        ragger::AgentSearchQuery aq;
        aq.query = q; aq.layers = {"document"}; aq.limit = 10;
        for (auto& h : daemon.agent_search(aq))
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
                                      const ragger::DecisionInput& in) {
    try { db.store_insight(in); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

// Happy path: trigger optional, evidence required and linked, row embedded
// and indexed, search surfaces it with source="insight".
void test_p5_insight_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);
    int ts1 = p1_make_turn_summary(db, "insight evidence one");
    int ts2 = p1_make_turn_summary(db, "insight evidence two");

    ragger::DecisionInput in;
    in.text      = "The embed-table lists must stay in sync by hand.";
    in.rationale = "Nothing enforces kAllTables vs count_embeddable_rows.";
    in.tags      = "maintenance";
    in.evidence  = {ts1, ts2, ts1};   // duplicate is ignored
    int id = db.store_insight(in).id;
    CHECK(id > 0);

    sqlite3* raw = p1_open_raw();
    const std::string w = std::format(" FROM insights WHERE insight_id = {}", id);
    CHECK(p1_scalar_text(raw, "SELECT text" + w) == "The embed-table lists must stay in sync by hand.");
    CHECK(p1_scalar_text(raw, "SELECT trigger" + w) == "<NULL>");   // optional -> NULL
    CHECK(p1_scalar_text(raw, "SELECT status" + w) == "current");
    CHECK(p1_scalar_text(raw, "SELECT tags" + w) == "maintenance");
    CHECK(p1_scalar_int(raw, "SELECT embedding IS NOT NULL" + w) == 1);
    CHECK(p1_scalar_int(raw, "SELECT unigram_count > 0" + w) == 1);
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT COUNT(*) FROM insight_evidence WHERE insight_id = {}", id)) == 2);
    sqlite3_close(raw);

    // Search finds it tagged as an insight.
    auto resp = db.search("embed table lists sync by hand", 10, 0.0f);
    bool found = false;
    for (const auto& h : resp.results)
        if (h.metadata.value("source", "") == "insight" && h.id == id) found = true;
    CHECK(found);

    // RESTRICT (same semantics as decision_evidence): evidence summaries
    // can't vanish from under the insight.
    raw = p1_open_raw();
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "DELETE FROM turn_summaries WHERE turn_summary_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, ts1);
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
    int ts = p1_make_turn_summary(db, "validation evidence");

    ragger::DecisionInput ok;
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
    CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM insights") == 0);   // nothing written
    sqlite3_close(raw);

    // supersedes: old row flips to 'superseded', keeps its text (honest
    // history), and disappears from search.
    int a_id = db.store_insight(ok).id;
    ragger::DecisionInput b = ok;
    b.text = "Insights cite turn summaries, never raw turns.";
    b.supersedes = a_id;
    auto rb = db.store_insight(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    raw = p1_open_raw();
    CHECK(p1_scalar_text(raw, std::format(
        "SELECT status FROM insights WHERE insight_id = {}", a_id)) == "superseded");
    CHECK(p1_scalar_int(raw, std::format(
        "SELECT supersedes FROM insights WHERE insight_id = {}", rb.id)) == a_id);
    sqlite3_close(raw);

    auto resp = db.search("Insights always cite their sources", 10, 0.0f);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "insight") continue;
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
    int ts = p1_make_turn_summary(db, "rebuild evidence");

    ragger::DecisionInput a;
    a.text = "Supersede rather than rewrite when the old wording mattered.";
    a.trigger = "Use when retiring an insight";
    a.evidence = {ts};
    int a_id = db.store_insight(a).id;

    ragger::DecisionInput b;
    b.text = "Migration legs are defensive CREATEs; no rebuild needed.";
    b.evidence = {ts};                       // no trigger
    int b_id = db.store_insight(b).id;

    sqlite3* raw = p1_open_raw();
    const std::string wa = std::format(" FROM insights WHERE insight_id = {}", a_id);
    const std::string wb = std::format(" FROM insights WHERE insight_id = {}", b_id);
    const std::string blob_a = p1_scalar_text(raw, "SELECT hex(embedding)" + wa);
    const std::string blob_b = p1_scalar_text(raw, "SELECT hex(embedding)" + wb);
    const long long uni_a = p1_scalar_int(raw, "SELECT unigram_count" + wa);
    const long long bi_a  = p1_scalar_int(raw, "SELECT bigram_count" + wa);
    const long long uni_b = p1_scalar_int(raw, "SELECT unigram_count" + wb);
    const long long bi_b  = p1_scalar_int(raw, "SELECT bigram_count" + wb);
    sqlite3_close(raw);

    db.rebuild_embeddings(emb, false, "insights");
    db.reindex_table("insights", false);

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

    int ts = p1_make_turn_summary(db, "layer evidence");

    ragger::DecisionInput a;
    a.text     = "P5gizmo flange insight: casting follows machining.";
    a.trigger  = "Use when inspecting gizmo flanges";
    a.evidence = {ts};
    int a_id = db.store_insight(a).id;

    // ---- agent_search with layers=["insight"]
    ragger::AgentSearchQuery q;
    q.query = "gizmo flange casting";
    q.layers = {"insight"};
    bool found = false;
    for (const auto& h : db.agent_search(q)) {
        CHECK(h.layer == "insight");
        if (h.id == "insight:" + std::to_string(a_id)) {
            found = true;
            CHECK(h.extra.value("status", "") == "current");
            CHECK(h.extra.value("trigger", "") == a.trigger);
        }
    }
    CHECK(found);

    // ---- agent_get expand=sources -> the evidence turn_summary
    ragger::AgentGetQuery g;
    g.ids = {"insight:" + std::to_string(a_id)};
    g.expand = "sources";
    auto got = db.agent_get(g);
    CHECK(!got["items"][0]["expanded"].empty());
    CHECK(got["items"][0]["expanded"][0]["layer"] == "turn_summary");

    // ---- agent_get expand=chain -> supersede lineage oldest -> newest
    ragger::DecisionInput b = a;
    b.text = "P5gizmo flange insight two: machining follows casting.";
    b.supersedes = a_id;
    int b_id = db.store_insight(b).id;
    g.ids = {"insight:" + std::to_string(b_id)};
    g.expand = "chain";
    got = db.agent_get(g);
    const auto& exp = got["items"][0]["expanded"];
    CHECK(exp.size() == 2);
    CHECK(exp[0]["id"] == "insight:" + std::to_string(a_id));
    CHECK(exp[1]["id"] == "insight:" + std::to_string(b_id));

    // ---- agent_update: tags free; illegal transition rejected
    ragger::UpdateRequest u;
    u.id = "insight:" + std::to_string(b_id);
    u.tags = "flange";
    auto row = db.agent_update(u);
    CHECK(row["tags"] == "flange");
    u.tags.reset();
    u.status = "pending";   // current -> pending is not a listed transition
    bool threw = false;
    try { db.agent_update(u); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // ---- review: pending insights listed, approvable through the handle
    ragger::DecisionInput p;
    p.text = "P5 pending insight: torque to spec.";
    p.evidence = {ts};
    p.status = "pending";
    int p_id = db.store_insight(p).id;
    auto listed = db.agent_review("list", {});
    bool saw_ins = false;
    for (const auto& item : listed["pending"])
        if (item["id"] == "insight:" + std::to_string(p_id)) saw_ins = true;
    CHECK(saw_ins);
    auto done = db.agent_review("approve", {"insight:" + std::to_string(p_id)});
    CHECK(done["done"].size() == 1);
    CHECK(done["failed"].empty());

    // ---- reject through an insight handle (review 7): pending -> retired,
    // and the queue no longer lists it.
    ragger::DecisionInput rj;
    rj.text = "P5 pending insight to reject.";
    rj.evidence = {ts};
    rj.status = "pending";
    int rj_id = db.store_insight(rj).id;
    auto rej = db.agent_review("reject", {"insight:" + std::to_string(rj_id)});
    CHECK(rej["done"].size() == 1);
    CHECK(rej["failed"].empty());
    {
        ragger::AgentGetQuery g;
        g.ids = {"insight:" + std::to_string(rj_id)};
        auto got = db.agent_get(g);
        CHECK(got["items"][0]["status"] == "retired");
    }
    bool still_listed = false;
    for (const auto& item : db.agent_review("list", {})["pending"])
        if (item["id"] == "insight:" + std::to_string(rj_id)) still_listed = true;
    CHECK(!still_listed);

    (void)a_id; (void)b_id;
    db.close();
    cleanup();
}

// Phase 2 Step 1: degraded (text-only) search must apply the same decision
// status allow-list as ensure_dec_cache() — a superseded row stays hidden.
void test_p2_text_only_hides_superseded(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::DecisionInput a;
    a.text = "Prefer WAL checkpointing before archiving the database.";
    a.trigger = "Use when archiving a live SQLite database";
    int a_id = db.store_structured_decision(a).id;

    ragger::DecisionInput b;
    b.text = "Checkpoint in TRUNCATE mode without closing the connection.";
    b.trigger = "Use when archiving a live SQLite database";
    b.supersedes = a_id;
    auto rb = db.store_structured_decision(b);
    CHECK(rb.superseded.has_value() && *rb.superseded == a_id);

    // Text-only path with A's exact words: A must not appear, B must.
    auto resp = db.search_text_only("Prefer WAL checkpointing before archiving the database.", 10);
    bool saw_old = false, saw_new = false;
    for (const auto& h : resp.results) {
        if (h.metadata.value("source", "") != "decision") continue;
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

    ragger::DecisionInput d;
    d.text = "Zephyr protocol ruling: always use zephyr gates for imports.";
    d.trigger = "Use when gating zephyr imports";
    int did = db.store_structured_decision(d).id;
    db.store("Zephyr protocol note: zephyr gates were tested in the lab.",
             {{"level", "session"}});

    ragger::AgentSearchQuery q;
    q.query = "zephyr protocol gates";
    q.layers = {"decision"};
    auto hits = db.agent_search(q);
    CHECK(!hits.empty());
    for (const auto& h : hits) {
        CHECK(h.layer == "decision");
        // Handle format: "<layer>:<rowid>"
        CHECK(h.id == "decision:" + std::to_string(did) ||
              h.id.starts_with("decision:"));
    }
    bool found = false;
    for (const auto& h : hits)
        if (h.id == "decision:" + std::to_string(did)) found = true;
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

    ragger::DecisionInput a;
    a.text = "Quoll exports must run through the nightly validator.";
    a.trigger = "Use when exporting quoll data";
    int a_id = db.store_structured_decision(a).id;

    ragger::DecisionInput b;
    b.text = "Quoll exports now stream directly; skip the nightly validator.";
    b.trigger = "Use when exporting quoll data";
    b.supersedes = a_id;
    int b_id = db.store_structured_decision(b).id;

    ragger::AgentSearchQuery q;
    q.query = "quoll exports validator";
    q.status = "active";
    auto hits = db.agent_search(q);
    for (const auto& h : hits) {
        if (h.layer != "decision") continue;
        CHECK(h.id != "decision:" + std::to_string(a_id));
    }

    q.status = "any";
    auto any_hits = db.agent_search(q);
    bool saw_old = false, saw_new = false;
    for (const auto& h : any_hits) {
        if (h.id == "decision:" + std::to_string(a_id)) { saw_old = true; CHECK(h.extra["status"] == "superseded"); }
        if (h.id == "decision:" + std::to_string(b_id)) saw_new = true;
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

    int ts1 = p1_make_turn_summary(db, "alpha");
    int ts2 = p1_make_turn_summary(db, "beta");

    ragger::DecisionInput d;
    d.text = "Always cite the failing test when filing a bug.";
    d.trigger = "Use when filing a bug report";
    d.rationale = "Reproducible failures need receipts.";
    d.scope = "Not needed for feature requests.";
    d.evidence = {ts1, ts2};
    int did = db.store_structured_decision(d).id;

    ragger::AgentGetQuery q;
    q.ids = {"decision:" + std::to_string(did)};
    auto got = db.agent_get(q);
    CHECK(got["items"].size() == 1);
    const auto& item = got["items"][0];
    CHECK(!item.contains("error"));
    CHECK(item["layer"] == "decision");
    CHECK(item["trigger"] == "Use when filing a bug report");
    CHECK(item["rationale"] == "Reproducible failures need receipts.");
    CHECK(item["evidence"].size() == 2);
    CHECK(item["evidence"][0] == ts1);
    CHECK(item["evidence"][1] == ts2);

    // Unknown id -> per-item error, not a throw.
    ragger::AgentGetQuery bad;
    bad.ids = {"decision:999999"};
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

    ragger::DecisionInput a;
    a.text = "Chain link one: use plan alpha for the widget.";
    a.trigger = "Use when planning the widget";
    int a_id = db.store_structured_decision(a).id;

    ragger::DecisionInput b;
    b.text = "Chain link two: use plan beta for the widget.";
    b.trigger = "Use when planning the widget";
    b.supersedes = a_id;
    int b_id = db.store_structured_decision(b).id;

    ragger::DecisionInput c;
    c.text = "Chain link three: use plan gamma for the widget.";
    c.trigger = "Use when planning the widget";
    c.supersedes = b_id;
    int c_id = db.store_structured_decision(c).id;

    ragger::AgentGetQuery q;
    q.ids = {"decision:" + std::to_string(b_id)};
    q.expand = "chain";
    auto got = db.agent_get(q);
    const auto& exp = got["items"][0]["expanded"];
    CHECK(exp.size() == 3);
    CHECK(exp[0]["id"] == "decision:" + std::to_string(a_id));
    CHECK(exp[1]["id"] == "decision:" + std::to_string(b_id));
    CHECK(exp[2]["id"] == "decision:" + std::to_string(c_id));

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
    both.ids = {"decision:1"};
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

    ragger::DecisionInput a;
    a.text = "Always run the full test suite before tagging a release.";
    a.trigger = "Use when preparing a release tag";
    int a_id = db.store_structured_decision(a).id;

    // Near-duplicate: same ruling reworded, same trigger.
    ragger::DecisionInput b;
    b.text = "Run the complete test suite before every release tag.";
    b.trigger = "Use when preparing a release tag";
    auto rb = db.store_structured_decision(b);
    bool flags_a = false;
    for (const auto& [id, score] : rb.possible_supersedes) {
        CHECK(score >= 0.80f);
        if (id == a_id) flags_a = true;
    }
    CHECK(flags_a);
    // Flag only: A must still be 'current'.
    CHECK(!rb.superseded.has_value());

    // Unrelated text -> empty list.
    ragger::DecisionInput c;
    c.text = "The harpsichord was tuned in meantone temperament in 1742.";
    c.trigger = "Use when tuning baroque keyboard instruments";
    auto rc = db.store_structured_decision(c);
    CHECK(rc.possible_supersedes.empty());

    db.close();
    cleanup();
}
// Step 1: three sessions with episodes at known created_at -> newest first;
// paging limit 2 over 5 rows -> 2, 2, 1 with next_cursor null at the end.
void test_p4_timeline_order_and_paging(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Five episodes, one per "session", spaced a day apart. store_summary's
    // source_timestamp override sets created_at exactly.
    const std::vector<std::string> days = {
        "2026-04-01 10:00:00", "2026-04-02 10:00:00", "2026-04-03 10:00:00",
        "2026-04-04 10:00:00", "2026-04-05 10:00:00"};
    std::vector<int> ids;
    for (size_t i = 0; i < days.size(); ++i) {
        const std::string guid = "p4tl-" + std::to_string(i);
        ids.push_back(db.store_summary("Timeline episode " + std::to_string(i),
                                       "episode", "summarizer-model", guid, days[i]));
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

// Step 1: granularity=session returns the session summary when present, and
// the newest episode when not (never requires session summaries).
void test_p4_timeline_session_fallback(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    // Session A: two episodes + a session summary.
    const std::string ga = "p4sf-a";
    db.store_summary("A episode one", "episode", "summarizer-model", ga, "2026-04-01 08:00:00");
    db.store_summary("A episode two", "episode", "summarizer-model", ga, "2026-04-02 08:00:00");
    db.store_summary("A session wrap", "session", "summarizer-model", ga, "2026-04-02 12:00:00");
    // Session B: episodes only.
    const std::string gb = "p4sf-b";
    db.store_summary("B episode one", "episode", "summarizer-model", gb, "2026-04-03 08:00:00");
    db.store_summary("B episode two", "episode", "summarizer-model", gb, "2026-04-04 08:00:00");
    // Session C: nothing at all -> contributes no entry.
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
    CHECK(page["items"].size() == 2);   // A and B only
    // Newest first: B (its newest episode, 04-04) then A (session summary 04-02).
    CHECK(page["items"][0]["text"].get<std::string>() == "B episode two");
    CHECK(page["items"][0]["session"].get<std::string>() == gb);
    CHECK(page["items"][1]["text"].get<std::string>() == "A session wrap");
    CHECK(page["items"][1]["session"].get<std::string>() == ga);

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
        r.id = "decision:1";
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

    // Rule 3: agent note (model_id NULL) editable; rollup rejected.
    const std::string note_id = db.store("A note to curate.", {{"level", "session"}});
    {
        ragger::UpdateRequest r;
        r.id = "summary:" + note_id;
        r.tags = "curated,keep2";
        auto row = db.agent_update(r);
        CHECK(row["tags"].get<std::string>() == "curated,keep2");
    }
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT tags FROM summaries WHERE summary_id = " + note_id)
              == "curated,keep2");
        sqlite3_close(raw);
    }
    {   // rollup (model_id set) rejected
        int rollup = db.store_summary("Rollup text", "episode", "summarizer-model",
                                      "p4u-roll", "2026-04-01 08:00:00");
        ragger::UpdateRequest r; r.id = "summary:" + std::to_string(rollup); r.tags = "nope";
        CHECK(throws_invalid(r));
    }
    {   // summaries have no status
        ragger::UpdateRequest r; r.id = "summary:" + note_id; r.status = "current";
        CHECK(throws_invalid(r));
    }

    // Rule 4: transition table.
    ragger::DecisionInput in;
    in.text = "P4 transition target ruling.";
    in.trigger = "Use when testing transitions";
    in.status = "pending";
    int pending_id = db.store_structured_decision(in).id;
    auto dec_status = [&db](int id) {
        sqlite3* raw = p1_open_raw();
        std::string s = p1_scalar_text(raw, "SELECT status FROM decisions WHERE decision_id = "
                                  + std::to_string(id));
        sqlite3_close(raw);
        return s;
    };
    {   // pending -> superseded rejected
        ragger::UpdateRequest r; r.id = "decision:" + std::to_string(pending_id);
        r.status = "superseded";
        CHECK(throws_invalid(r));
    }
    {   // pending -> current allowed
        ragger::UpdateRequest r; r.id = "decision:" + std::to_string(pending_id);
        r.status = "current";
        auto row = db.agent_update(r);
        CHECK(row["status"].get<std::string>() == "current");
        CHECK(dec_status(pending_id) == "current");
    }
    {   // current -> pending NOT allowed (table says no)
        ragger::UpdateRequest r; r.id = "decision:" + std::to_string(pending_id);
        r.status = "pending";
        CHECK(throws_invalid(r));
    }
    {   // current -> current not a listed transition either
        ragger::UpdateRequest r; r.id = "decision:" + std::to_string(pending_id);
        r.status = "current";
        CHECK(throws_invalid(r));
    }
    {   // current -> provisional -> retired allowed; retired -> anything rejected
        ragger::UpdateRequest r; r.id = "decision:" + std::to_string(pending_id);
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

    ragger::DecisionInput in;
    in.text = "Kestrel migrations run only on Wednesdays.";
    in.trigger = "Use when scheduling kestrel migrations";
    in.status = "pending";
    int id = db.store_structured_decision(in).id;

    // Pending is hidden from active search.
    ragger::AgentSearchQuery q;
    q.query = "kestrel migrations Wednesdays";
    q.layers = {"decision"};
    auto hits = db.agent_search(q);
    for (const auto& h : hits) CHECK(h.id != "decision:" + std::to_string(id));

    ragger::UpdateRequest r; r.id = "decision:" + std::to_string(id); r.status = "current";
    db.agent_update(r);

    hits = db.agent_search(q);
    bool found = false;
    for (const auto& h : hits)
        if (h.id == "decision:" + std::to_string(id)) found = true;
    CHECK(found);

    db.close();
    cleanup();
}

// Step 3: review queue — list, approve, reject, per-id failure.
void test_p4_review(ragger::Embedder& emb) {
    cleanup();
    ragger::sqlite::Backend db(emb, TEMP_DB);

    ragger::DecisionInput a;
    a.text = "Review candidate alpha: wagons ship at dawn.";
    a.trigger = "Use when shipping wagons";
    a.status = "pending";
    int a_id = db.store_structured_decision(a).id;

    ragger::DecisionInput b;
    b.text = "Review candidate beta: kayaks ship at dusk.";
    b.trigger = "Use when shipping kayaks";
    b.status = "pending";
    int b_id = db.store_structured_decision(b).id;

    auto listed = db.agent_review("list", {});
    CHECK(listed["pending"].size() == 2);
    CHECK(listed["pending"][0]["id"].get<std::string>() == "decision:" + std::to_string(a_id));
    CHECK(!listed["pending"][0]["trigger"].get<std::string>().empty());

    // Approve alpha (bare integer accepted too).
    auto done = db.agent_review("approve", {std::to_string(a_id)});
    CHECK(done["done"].size() == 1);
    CHECK(done["failed"].empty());
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT status FROM decisions WHERE decision_id = "
                                  + std::to_string(a_id)) == "current");
        sqlite3_close(raw);
    }
    // Approved decision is now visible in normal search.
    ragger::AgentSearchQuery q;
    q.query = "wagons ship dawn";
    q.layers = {"decision"};
    bool found = false;
    for (const auto& h : db.agent_search(q))
        if (h.id == "decision:" + std::to_string(a_id)) found = true;
    CHECK(found);

    // Approving an already-current decision fails THAT id only.
    auto again = db.agent_review("approve", {
        "decision:" + std::to_string(a_id),
        "decision:" + std::to_string(b_id)});
    CHECK(again["done"].size() == 1);
    CHECK(again["failed"].size() == 1);
    CHECK(again["failed"][0]["id"].get<std::string>() == "decision:" + std::to_string(a_id));

    // Reject beta -> retired, invisible in search, row still present.
    auto rej = db.agent_review("reject", {"decision:" + std::to_string(b_id)});
    CHECK(rej["done"].size() == 1);
    {
        sqlite3* raw = p1_open_raw();
        CHECK(p1_scalar_text(raw, "SELECT status FROM decisions WHERE decision_id = "
                                  + std::to_string(b_id)) == "retired");
        CHECK(p1_scalar_int(raw, "SELECT COUNT(*) FROM decisions WHERE decision_id = "
                                  + std::to_string(b_id)) == 1);
        sqlite3_close(raw);
    }
    q.query = "kayaks ship dusk";
    for (const auto& h : db.agent_search(q))
        CHECK(h.id != "decision:" + std::to_string(b_id));

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

    // summary: store with explicit source_timestamp, probe must find it.
    int sid = db.store_summary("Probe summary text.", "session", "model-x", "",
                               "2026-04-15 09:30:00");
    CHECK(sid > 0);
    CHECK(db.summary_exists_exact("Probe summary text.", "2026-04-15 09:30:00"));
    CHECK(!db.summary_exists_exact("Probe summary text.", "2026-04-15 09:30:01"));

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
    auto results = db.search_by_metadata({{"level", "session"}}, 0,
                                         "2026-04-15 09:00:00", "2026-04-15 10:00:00");
    CHECK(results.size() == 1);
    CHECK(results[0].text == "Probe summary text.");
    const std::string& ts = results[0].timestamp;
    std::regex dt_re(R"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)");
    CHECK(std::regex_match(ts, dt_re));
    // Outside the bracket → empty.
    CHECK(db.search_by_metadata({{"level", "session"}}, 0,
                                "2026-04-15 09:31:00", "2026-04-15 10:00:00").empty());
    sqlite3_close(raw);

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
        CHECK(p1_scalar_int(raw, "SELECT length(embedding) FROM summaries "
                                  "WHERE text = 'F16 width probe.'") == 2 * dims);
        sqlite3_close(raw);

        db.set_vector_type(ragger::vector_codec::VectorType::F32);
        db.store("F32 width probe.");
        raw = p1_open_raw();
        CHECK(p1_scalar_int(raw, "SELECT length(embedding) FROM summaries "
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
// exactly the six layers agent_search validates today.
void p3_test_known_layers() {
    std::println("  p3_test_known_layers...");
    const std::set<std::string> expected = {
        "turn", "turn_summary", "summary", "document", "decision", "insight"};
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
          "'current','provisional','pending','roadmap','retired'");

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

    ragger::DecisionInput in;
    in.text     = "cachepin insight ruling";
    { sqlite3* raw = p1_open_raw();
      in.evidence = {static_cast<int>(p1_scalar_int(raw, "SELECT MAX(turn_summary_id) FROM turn_summaries"))};
      sqlite3_close(raw); }
    db.store_insight(in);                                         // insights

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

    // Pinned pre-refactor (five hand-written loaders, one row each).
    CHECK(count_by_source["summary"] == 1);
    CHECK(count_by_source["document"] == 1);
    CHECK(count_by_source["decision"] == 1);
    CHECK(count_by_source["insight"] == 1);
    CHECK(count_by_source["turn_summary"] == 1);
    CHECK(keys_by_source["summary"] == std::set<std::string>({"source", "level"}));
    CHECK(keys_by_source["document"] == std::set<std::string>({"source", "title"}));
    CHECK(keys_by_source["decision"] == std::set<std::string>({"source", "status"}));
    CHECK(keys_by_source["insight"] == std::set<std::string>({"source", "status"}));
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
        "SELECT embedding IS NOT NULL FROM decisions WHERE decision_id = {}", d)) == 1);
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

int main() {
    TestEnv env("sqlite_backend");  // must be first: isolates from ~/.ragger
    // Need config for model dir
    auto model_dir = env.model_dir();

    // Model-free unit tests run even when the embedding model is absent.
    p1r_test_transaction();
    p3_test_known_layers();
    p3s_test_status_vocabulary();

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
    test_delete_memory(emb);
    test_delete_batch(emb);
    test_search_by_metadata(emb);
    test_delete_respects_keep(emb);
    test_delete_batch_respects_keep(emb);
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
    test_summary_primitives(emb);
    test_episode_rollup_no_dup(emb);
    test_session_close_no_lookback_before_existing_summary(emb);
    test_path_normalization(emb);
    test_structured_decision_basic(emb);
    test_structured_decision_rebuild_is_stable(emb);
    test_structured_decision_validation(emb);
    test_structured_decision_supersedes(emb);
    test_structured_decision_evidence(emb);
    test_orphan_turn_summary_protected(emb);
    test_migrate_0_16_to_0_17(emb);
    test_document_path_unique(emb);
    test_document_list_and_delete(emb);
    test_invalidate_all_caches(emb);
    test_p5_insight_basic(emb);
    test_p5_insight_validation(emb);
    test_p5_insight_rebuild_is_stable(emb);
    test_p5_insight_layer(emb);
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
    p1_test_embedder_swap(emb, model_dir);
    p2_test_close_really_closes(emb);
    p2_test_write_paths_postings(emb);
    p2_test_store_turn_under_outer_begin(emb);
    p2_test_has_embeddings_turn_summaries(emb);
    p2_test_backfill_invalidates_dec_cache(emb);
    p2_test_db_version_gate(emb);
    p2_test_require_embedder(emb);
    p3_cache_pin(emb);
    p3_test_update_embedding(emb);
    p4_session_span(emb);

    std::println("test_sqlite_backend: all passed");
    return 0;
}
