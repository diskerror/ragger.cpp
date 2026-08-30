/**
 * SQLite backend integration tests
 *
 * Requires ONNX model files at the configured model_dir.
 * Uses a temp DB that's deleted after the test.
 */
#include "config.h"
#include "embedder.h"
#include "sqlite_backend.h"
#include "user_store.h"
#include "auth.h"
#include "util/time.h"
#include <sqlite3.h>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>
#include <thread>
#include <chrono>
#include <ctime>

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
    ragger::SqliteBackend db(emb, TEMP_DB);

    assert(db.count() == 0);

    db.store("Hello world, this is a test memory.");
    assert(db.count() == 1);

    db.store("Another memory about something else.");
    assert(db.count() == 2);

    db.close();
    cleanup();
}

void test_store_with_metadata(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Lean v2 summaries: tags (comma-joined) + level round-trip.
    // status was removed from summaries in v0.12.0; any status key in
    // metadata is silently dropped.
    ragger::json meta = {
        {"level", "session"},
        {"tags", {"test", "unit"}}
    };
    std::string id = db.store("Metadata round-trip test.", meta);
    assert(!id.empty());

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].metadata["level"] == "session");
    // Tags stored as comma-separated string in the tags column.
    assert(all[0].metadata["tags"] == "test,unit");

    db.close();
    cleanup();
}

void test_search_basic(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("The capital of France is Paris.");
    db.store("SQLite is a lightweight database engine.");
    db.store("Machine learning uses neural networks for prediction.");

    auto resp = db.search("What is the capital of France?", 3, 0.0f);
    assert(!resp.results.empty());
    // The France/Paris doc should be the top result
    assert(resp.results[0].text.find("Paris") != std::string::npos);

    db.close();
    cleanup();
}

// FTS5 MATCH is built from arbitrary user text; punctuation/quotes/operators
// must not produce a syntax error. The keyword pass should degrade to "no
// keyword contribution" rather than throw, and vector search still returns.
void test_search_fts_special_chars(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

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
    assert(!resp.results.empty());
    assert(resp.results[0].text.find("SQLite") != std::string::npos);

    db.close();
    cleanup();
}

void test_search_min_score(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("The weather in Los Angeles is sunny.");
    db.store("Quantum physics studies subatomic particles.");

    // Very high min_score should return no results for a vague query
    auto resp = db.search("random unrelated gibberish xyzzy", 5, 0.99f);
    assert(resp.results.empty());

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
    ragger::SqliteBackend db(emb, TEMP_DB);

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
        assert(r.score >= thr);

    // (2) Count how many candidates actually clear the threshold by asking for
    // a huge limit; with the fix, a smaller limit returns exactly
    // min(limit, qualifying), never fewer due to top-k stealing.
    auto all = db.search("What is the capital of France?", 100, thr);
    size_t qualifying = all.results.size();
    size_t expected = std::min<size_t>(5, qualifying);
    assert(resp.results.size() == expected);

    db.close();
    cleanup();
}

void test_search_limit(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    for (int i = 0; i < 10; ++i) {
        db.store("Test document number " + std::to_string(i) + " with some content.");
    }

    auto resp = db.search("test document", 3, 0.0f);
    assert((int)resp.results.size() <= 3);

    db.close();
    cleanup();
}

void test_load_all(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("First memory.");
    db.store("Second memory.");
    db.store("Third memory.");

    auto all = db.load_all();
    assert(all.size() == 3);

    db.close();
    cleanup();
}

void test_rebuild_embeddings(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Store some test documents
    db.store("Embedding rebuild test document one.");
    db.store("Embedding rebuild test document two.");
    db.store("Something completely different here.");

    // Rebuild all embeddings
    int count = db.rebuild_embeddings(emb);
    assert(count == 3);

    // Search should still work after rebuild
    auto resp = db.search("embedding test", 5, 0.0f);
    assert(!resp.results.empty());

    db.close();
    cleanup();
}

void test_rebuild_embeddings_empty_db(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Rebuild on empty DB should return 0
    int count = db.rebuild_embeddings(emb);
    assert(count == 0);

    db.close();
    cleanup();
}

void test_rebuild_embeddings_count_matches(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Store various numbers of documents
    for (int i = 0; i < 7; ++i) {
        db.store("Test document number " + std::to_string(i));
    }

    int total_count = db.count();
    int rebuild_count = db.rebuild_embeddings(emb);
    
    // Rebuild count should match total count
    assert(rebuild_count == total_count);
    assert(rebuild_count == 7);

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
    ragger::SqliteBackend db(emb, TEMP_DB);

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
    assert(db.turn_summary_exists(turn_id));

    // Must not throw, and must actually find the keyword across corpora.
    auto resp = db.search_text_only("harpsichord", 10);
    assert(resp.timing.contains("text_only"));
    assert(!resp.results.empty());

    // Every hit carries a source label and no vector signal.
    bool saw_turn_summary = false;
    for (const auto& r : resp.results) {
        assert(r.metadata.contains("source"));
        assert(r.vec_score < 0.0f);
        if (r.metadata["source"] == "turn_summary") {
            saw_turn_summary = true;
            assert(!r.timestamp.empty());   // resolved from turn_datetime
        }
    }
    assert(saw_turn_summary);

    // A query that matches nothing returns cleanly rather than throwing.
    auto none = db.search_text_only("zzzzznotawordzzzzz", 5);
    assert(none.results.empty());

    db.close();
    cleanup();
}

void test_search_timing(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("Timing test document.");
    auto resp = db.search("timing", 5, 0.0f);
    // Timing JSON should have keys
    assert(resp.timing.contains("embedding_ms") || resp.timing.contains("total_ms") ||
           !resp.timing.empty());

    db.close();
    cleanup();
}

void test_delete_memory(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Store a memory
    std::string id1 = db.store("Memory to delete.");
    std::string id2 = db.store("Memory to keep.");
    assert(db.count() == 2);

    // Delete the first memory
    bool deleted = db.delete_memory(std::stoi(id1));
    assert(deleted);
    assert(db.count() == 1);

    // Verify the correct one was deleted
    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == "Memory to keep.");

    // Try deleting non-existent ID
    deleted = db.delete_memory(99999);
    assert(!deleted);
    assert(db.count() == 1);

    db.close();
    cleanup();
}

void test_delete_batch(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Store 3 memories
    std::string id1 = db.store("First memory.");
    std::string id2 = db.store("Second memory.");
    std::string id3 = db.store("Third memory.");
    assert(db.count() == 3);

    // Delete 2 by ID
    std::vector<int> to_delete = {std::stoi(id1), std::stoi(id3)};
    int deleted_count = db.delete_batch(to_delete);
    assert(deleted_count == 2);
    assert(db.count() == 1);

    // Verify the correct one remains
    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == "Second memory.");

    // Empty vector → returns 0
    deleted_count = db.delete_batch({});
    assert(deleted_count == 0);

    // Delete with non-existent IDs
    deleted_count = db.delete_batch({99999, 88888});
    assert(deleted_count == 0);

    db.close();
    cleanup();
}

void test_search_by_metadata(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Lean v2: filterable columns are level / status / tags (+ time window).
    db.store("Apple note.",  {{"level", "turn"},    {"tags", {"fruit"}}});
    db.store("Banana note.", {{"level", "turn"},    {"tags", {"fruit", "tropical"}}});
    db.store("Car note.",    {{"level", "session"}, {"tags", {"vehicle"}}});

    // Filter by level
    auto results = db.search_by_metadata({{"level", "turn"}});
    assert(results.size() == 2);
    for (auto& r : results) assert(r.metadata["level"] == "turn");

    // Filter by tag (LIKE on the tags column)
    results = db.search_by_metadata({{"tags", "vehicle"}});
    assert(results.size() == 1);
    assert(results[0].text == "Car note.");

    // AND across columns
    results = db.search_by_metadata({{"level", "turn"}, {"tags", "tropical"}});
    assert(results.size() == 1);
    assert(results[0].text == "Banana note.");

    // Limit respected
    results = db.search_by_metadata({{"level", "turn"}}, 1);
    assert(results.size() == 1);

    // No matches → empty
    results = db.search_by_metadata({{"level", "project"}});
    assert(results.empty());

    // Unsupported key under lean schema → empty
    results = db.search_by_metadata({{"color", "red"}});
    assert(results.empty());

    db.close();
    cleanup();
}

void test_user_management(ragger::Embedder& emb) {
    cleanup();
    // User management now uses SqliteBackend (separate from storage backend)
    ragger::UserStore umgr(TEMP_DB);

    // create_user → returns valid ID
    int user_id = umgr.create_user("testuser", "abc123hash");
    assert(user_id > 0);

    int admin_id = umgr.create_user("adminuser", "def456hash");
    assert(admin_id > 0);
    assert(admin_id != user_id);

    // get_user_by_token_hash → finds created user
    auto user_opt = umgr.get_user_by_token_hash("abc123hash");
    assert(user_opt.has_value());
    assert(user_opt->username == "testuser");
    assert(user_opt->token_hash == "abc123hash");

    // get_user_by_username → finds created user
    user_opt = umgr.get_user_by_username("adminuser");
    assert(user_opt.has_value());
    assert(user_opt->username == "adminuser");
    assert(user_opt->token_hash == "def456hash");

    // get_user_by_token_hash with wrong hash → nullopt
    user_opt = umgr.get_user_by_token_hash("wronghash");
    assert(!user_opt.has_value());

    // get_user_by_username with wrong name → nullopt
    user_opt = umgr.get_user_by_username("nonexistent");
    assert(!user_opt.has_value());

    cleanup();
}

void test_delete_respects_keep(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    int initial_count = db.count();
    
    // Store a memory with {"keep": true} in metadata
    ragger::json meta = {{"keep", true}, {"collection", "memory"}};
    auto id = db.store("protected memory", meta);
    
    // Try to delete — should return false
    assert(!db.delete_memory(std::stoi(id)));
    
    // Memory should still exist
    assert(db.count() == initial_count + 1);
    
    // Store without keep
    auto id2 = db.store("deletable memory", {{"collection", "memory"}});
    
    // Delete should work
    assert(db.delete_memory(std::stoi(id2)));
    
    // Only the protected one should remain
    assert(db.count() == initial_count + 1);
    
    db.close();
    cleanup();
}

void test_delete_batch_respects_keep(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

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
    assert(deleted == 2);
    assert(db.count() == initial_count - 2);
    
    // Verify the protected one remains
    auto all = db.load_all();
    bool found_protected = false;
    for (const auto& mem : all) {
        if (mem.text == "keep me") {
            found_protected = true;
            break;
        }
    }
    assert(found_protected);
    
    db.close();
    cleanup();
}

void test_timestamp_format(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("Timestamp format test.");
    auto all = db.load_all();
    assert(all.size() == 1);

    // `summaries.created_at` is now an INTEGER epoch (v0.12.0 migration),
    // not the old "YYYY-MM-DD HH:MM:SS" TEXT format. load_all() reads the
    // column via column_text() on a raw INTEGER, which SQLite renders back
    // as plain base-10 digits (no separators) -- confirm that shape and
    // that it's a plausible "recent" epoch value, rather than asserting the
    // old fixed-19-char formatted-string length.
    auto& ts = all[0].timestamp;
    assert(!ts.empty());
    assert(ts.find_first_not_of("0123456789") == std::string::npos);
    int64_t epoch_val = std::stoll(ts);
    int64_t now_epoch = static_cast<int64_t>(std::time(nullptr));
    assert(epoch_val > 0);
    assert(epoch_val <= now_epoch + 5);       // not in the future
    assert(now_epoch - epoch_val < 300);      // stored "just now"

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
        ragger::SqliteBackend db(emb, TEMP_DB);
        db.store("v2 summaries backing check.",
                 {{"level", "session"}, {"tags", {"note"}}});

        auto all = db.load_all();
        assert(all.size() == 1);
        assert(all[0].metadata["level"] == "session");
        assert(all[0].metadata["tags"] == "note");
        db.close();
    }

    // Inspect the raw DB: the row must be in `summaries`, and the legacy
    // `memories` table must not exist in a fresh v2 database.
    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM summaries", -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);

    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='memories'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 0);   // no legacy table in fresh DB
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
        ragger::SqliteBackend db(emb, TEMP_DB);

        ragger::DocumentChunk doc;
        doc.text        = "The mitochondria is the powerhouse of the cell.";
        doc.title       = "Cell Biology Primer";
        doc.tags        = "biology,cells";
        doc.year        = 2021;
        doc.path        = "/tmp/primer.md";
        doc.chunk_index = 1;

        doc_id = db.store_document(doc);
        assert(doc_id > 0);
        db.close();
    }

    // Verify the dedicated columns round-trip via the raw documents table.
    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT text, title, tags, year, path, chunk_index "
        "FROM documents WHERE document_id = ?",
        -1, &st, nullptr);
    sqlite3_bind_int(st, 1, doc_id);
    assert(sqlite3_step(st) == SQLITE_ROW);

    auto col = [&](int i) {
        const char* s = reinterpret_cast<const char*>(sqlite3_column_text(st, i));
        return std::string(s ? s : "");
    };
    std::string text = col(0);
    assert(col(1) == "Cell Biology Primer");
    assert(col(2) == "biology,cells");
    assert(sqlite3_column_int(st, 3) == 2021);
    assert(col(4) == "/tmp/primer.md");
    assert(sqlite3_column_int(st, 5) == 1);

    // Grouping metadata must NOT be embedded into the body text.
    assert(text == "The mitochondria is the powerhouse of the cell.");
    assert(text.find("Cell Biology Primer") == std::string::npos);

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
        ragger::SqliteBackend db(emb, TEMP_DB);

        // Complete turn in one call — embedded immediately.
        int t1 = db.store_turn("What is the capital of France?",
                               "The capital of France is Paris.", "test-model");
        assert(t1 > 0);

        // Partial turn (no assistant yet) → embedding NULL → finalize fills it.
        int t2 = db.store_turn("Tell me about photosynthesis.", "", "test-model",
                               /*defer_embedding=*/true);
        assert(t2 > 0 && t2 != t1);
        assert(db.finalize_turn(t2, "Plants convert sunlight into energy.",
                                "test-model"));
        db.close();
    }

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

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
        assert(u && a && a[0] != '\0');          // assistant filled in
        // Embedded as f16 in the version-tagged format:
        // 1-byte version + 384 dims * 2 bytes/dim.
        assert(emb_bytes == 1 + 384 * 2);
        assert(model && std::string(model) == "test-model");
        ++rows;
    }
    sqlite3_finalize(st);
    assert(rows == 2);

    // The model row was created exactly once, not duplicated.
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM models WHERE name='test-model'",
                       -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);

    // finalize_turn on a non-existent id returns false.
    {
        ragger::SqliteBackend db2(emb, TEMP_DB);
        assert(!db2.finalize_turn(99999, "x", "test-model"));
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
        ragger::SqliteBackend db(emb, TEMP_DB);

        // First answer to a prompt, in session A.
        t1 = db.store_turn("Explain the general_search recipe.",
                               "It searches summaries.", "model-a",
                               /*defer_embedding=*/false, "sessionA");
        assert(t1 > 0);

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
        assert(t2 == t1);  // same row id returned — keep-latest

        // Cross a second boundary before the next insert: local_timestamp()
        // has second-level resolution, and turn3 landing in the same second
        // as t1's dedup-updated created_at is itself a separate known
        // same-second collision (harmless for the summarizer's matching, but
        // it would otherwise make this assertion block conflate two issues).
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));

        // A genuinely different prompt still inserts a fresh row.
        int t3 = db.store_turn("What about decisions?", "They are first-class.",
                               "model-b", /*defer_embedding=*/false, "sessionB");
        assert(t3 != t1);

        db.close();
    }

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    // Exactly two turn rows: the deduped pair collapsed to one, plus the
    // distinct third prompt.
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turns", -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);

    // The surviving deduped row carries the LATEST assistant_text and is not
    // duplicated in the FTS index (one hit for the prompt's distinctive term).
    sqlite3_prepare_v2(raw,
        "SELECT assistant_text FROM turns "
        "WHERE user_text = 'Explain the general_search recipe.'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* a = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        assert(a && std::string(a).find("decisions") != std::string::npos);
    }
    // Only one such row (no duplicate).
    assert(sqlite3_step(st) != SQLITE_ROW);
    sqlite3_finalize(st);

    // FTS index agrees: the prompt term resolves to exactly one rowid.
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM turns_fts WHERE turns_fts MATCH 'general_search'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 1);
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
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 0);
    sqlite3_finalize(st);

    sqlite3_close(raw);
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
    ragger::SqliteBackend db(emb, TEMP_DB);

    int t1 = db.store_turn("What is the capital of Spain?",
                           "The capital of Spain is Madrid.", "test-model");
    assert(t1 > 0);
    assert(!db.turn_summary_exists(t1));

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 0);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// A partial turn (no assistant_text yet) also gets no turn_summaries row —
// same as a complete turn, since store_turn() never creates one at all now.
void test_store_turn_partial_turn_creates_no_turn_summaries_row(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    int t1 = db.store_turn("Tell me about volcanoes.", "", "test-model",
                          /*defer_embedding=*/true);
    assert(t1 > 0);
    assert(!db.turn_summary_exists(t1));

    db.close();
    cleanup();
}

// finalize_turn_summary() is a plain INSERT now (no upsert): calling it
// twice on the same turn_id must fail on the second call (unique partial
// index on turn_id rejects the duplicate), leaving exactly ONE row with the
// FIRST call's text in place — not the second.
void test_finalize_turn_summary_rejects_duplicate(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    int t1 = db.store_turn("Summarize this exchange.",
                           "First finalize target.", "test-model");
    assert(t1 > 0);

    assert(db.finalize_turn_summary(t1, "First summary text.", "summarizer-model"));
    assert(!db.finalize_turn_summary(t1, "Second summary text.", "summarizer-model"));

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 1);   // no duplicate row
    sqlite3_finalize(st);

    sqlite3_prepare_v2(raw, "SELECT text FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        assert(txt && std::string(txt) == "First summary text.");
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
    ragger::SqliteBackend db(emb, TEMP_DB);

    const int missing_turn_id = 999999;
    assert(!db.finalize_turn_summary(missing_turn_id, "orphan text", "test-model"));

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, missing_turn_id);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_int(st, 0) == 0);
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
    ragger::SqliteBackend db(emb, TEMP_DB);

    int t1 = db.store_turn("A turn destined for deletion.",
                           "Its summary should survive.", "test-model");
    assert(t1 > 0);
    assert(db.finalize_turn_summary(t1, "Survivor summary text.", "summarizer-model"));

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);

    int64_t turn_datetime_before = 0;
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT turn_datetime FROM turn_summaries WHERE turn_id = ?",
                       -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_ROW);
    turn_datetime_before = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);

    // Raw delete of the parent turns row -- exercises the FK trigger, not a
    // backend method (there's no public "delete_turn" primitive to call).
    sqlite3_prepare_v2(raw, "PRAGMA foreign_keys = ON", -1, &st, nullptr);
    sqlite3_step(st);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw, "DELETE FROM turns WHERE turn_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_DONE);
    sqlite3_finalize(st);

    // The turn_summaries row still exists, turn_id is now NULL, and the
    // text/turn_datetime are unchanged from before the deletion.
    sqlite3_prepare_v2(raw,
        "SELECT text, turn_id, turn_datetime FROM turn_summaries "
        "WHERE turn_datetime = ?", -1, &st, nullptr);
    sqlite3_bind_int64(st, 1, turn_datetime_before);
    assert(sqlite3_step(st) == SQLITE_ROW);
    {
        const char* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        assert(txt && std::string(txt) == "Survivor summary text.");
    }
    assert(sqlite3_column_type(st, 1) == SQLITE_NULL);   // turn_id now NULL
    assert(sqlite3_column_int64(st, 2) == turn_datetime_before);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// unsummarized_turns() must exclude a turn that already has a finalized
// turn_summaries row and include one that doesn't.
void test_unsummarized_turns_excludes_summarized(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    int t_summarized = db.store_turn("This turn will be summarized.",
                                     "Summarized turn's reply.", "test-model");
    assert(t_summarized > 0);
    int t_pending = db.store_turn("This turn stays unsummarized.",
                                  "Pending turn's reply.", "test-model");
    assert(t_pending > 0);

    assert(db.finalize_turn_summary(t_summarized, "Already done.", "summarizer-model"));

    auto pending = db.unsummarized_turns(0);
    bool found_pending = false, found_summarized = false;
    for (auto& t : pending) {
        if (t.turn_id == t_pending) found_pending = true;
        if (t.turn_id == t_summarized) found_summarized = true;
    }
    assert(found_pending);
    assert(!found_summarized);

    db.close();
    cleanup();
}

// mark_turn_summarized() marks a trivial/skip-worthy turn as done WITHOUT
// real summary text -- text must remain NULL while summary_model_id gets
// resolved to the given model name.
void test_mark_turn_summarized_trivial_turn(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // store_turn() no longer creates any turn_summaries row automatically.
    int t1 = db.store_turn("ok", "ok", "test-model");
    assert(t1 > 0);
    assert(!db.turn_summary_exists(t1));

    assert(db.mark_turn_summarized(t1, "trivial-skip-model"));

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT ts.text, m.name FROM turn_summaries ts "
        "JOIN models m ON m.model_id = ts.summary_model_id "
        "WHERE ts.turn_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t1);
    assert(sqlite3_step(st) == SQLITE_ROW);
    assert(sqlite3_column_type(st, 0) == SQLITE_NULL);   // text stays NULL
    {
        const char* model = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        assert(model && std::string(model) == "trivial-skip-model");
    }
    sqlite3_finalize(st);
    sqlite3_close(raw);

    db.close();
    cleanup();
}

// Summary primitives (issue #22): store_summary, update_summary_text —
// the deterministic backbone the summarization pipeline drives.
// current_session_summary and set_summary_status were removed when
// summaries.status was dropped (v0.12.0 boundary-detection rework).
void test_summary_primitives(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    int l2 = db.store_summary("User asked about France; capital is Paris.",
                              "turn", "memo-model");
    assert(l2 > 0);

    int l3 = db.store_summary("Discussed European capitals.",
                              "session", "memo-model");
    assert(l3 > 0);

    assert(db.update_summary_text(l3, "Discussed European capitals, focus France.",
                                  "memo-model"));

    int l3b = db.store_summary("Switched to cooking techniques.",
                               "session", "memo-model");
    assert(l3b > 0);

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM summaries WHERE level='turn'",
                       -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='session'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries s JOIN models m ON m.model_id=s.model_id "
        "WHERE m.name='memo-model'", -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 3);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    assert(!db.update_summary_text(99999, "x", "memo-model"));

    // Recipe ingredients (issue #23): recency-based fetch.
    auto turns = db.recent_summaries("turn", 5);
    assert(turns.size() == 1);
    auto sessions = db.recent_summaries("session", 5);
    assert(sessions.size() == 2);
    assert(db.recent_summaries("project", 5).empty());
    assert(db.recent_summaries("turn", 0).empty());
    assert(db.current_decisions(5).empty());      // none created

    db.close();
    cleanup();
}

// Phase 2 (EPISODE_PLAN): episode layer + boundary-triggered session/project
// rollups. Deterministic backend-level verification of the bloat-bug fix — no
// inference needed. Reproduces the shape of the original 1956–1977 defect
// (repeated rollups after a "close") and asserts exactly one session row.
void test_episode_rollup_no_dup(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    const std::string G = "phase2-session";

    // No episodes yet.
    assert(db.last_episode_end(G).empty());
    assert(db.episode_texts(G).empty());

    // Two L2 turn summaries land (distinct timestamps → distinct rows).
    // Turn-level summaries now live in turn_summaries, keyed by a real
    // turn_id -- create the underlying turns first, then finalize each
    // one's summary, rather than writing level='turn' rows into
    // `summaries` directly (that path no longer exists post-db_version
    // 0.12; see l2_summaries_since's redirect to turn_summaries).
    int t1 = db.store_turn("Q1", "A1", "memo", false, G, "2026-01-01 10:00:00");
    int t2 = db.store_turn("Q2", "A2", "memo", false, G, "2026-01-01 10:00:02");
    assert(db.finalize_turn_summary(t1, "Turn one: discussed knowledge graphs.", "memo"));
    assert(db.finalize_turn_summary(t2, "Turn two: phonetic pun edges.", "memo"));

    // Before any episode, l2_summaries_since(all) returns both.
    assert(db.l2_summaries_since(G, "").size() == 2);

    // Close episode 1 spanning both turns.
    int ep1 = db.store_episode("Episode 1: KG design + phonetic edges.",
                               "memo", G, "2026-01-01 10:00:00",
                               "2026-01-01 10:00:02");
    assert(ep1 > 0);
    assert(db.last_episode_end(G) == "2026-01-01 10:00:02");
    assert(db.episode_texts(G).size() == 1);
    // No L2 past the episode end now.
    assert(db.l2_summaries_since(G, db.last_episode_end(G)).empty());

    // Second episode.
    int t3 = db.store_turn("Q3", "A3", "memo", false, G, "2026-01-01 11:00:00");
    assert(db.finalize_turn_summary(t3, "Turn three: curated fiction corpus.", "memo"));
    db.store_episode("Episode 2: curated corpus bet.", "memo", G,
                     "2026-01-01 11:00:00", "2026-01-01 11:00:00");
    assert(db.episode_texts(G).size() == 2);

    // Episode rows are immutable: timestamp=first, updated_at=last (span).
    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='episode' "
        "AND session_id=(SELECT session_id FROM sessions WHERE guid='phase2-session')",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
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
    assert(first_ep.has_value() && last_ep.has_value());
    auto sess_texts = db.bounded_session_rollup_texts(
        G, static_cast<int64_t>(*first_ep), static_cast<int64_t>(*last_ep));
    assert(sess_texts.size() == 2);  // both episode texts, oldest-first
    assert(sess_texts[0] == "Episode 1: KG design + phonetic edges.");
    assert(sess_texts[1] == "Episode 2: curated corpus bet.");

    // A LATER run of the SAME session_guid must not leak into an EARLIER
    // run's bounded query -- add a third episode outside [first_ep,last_ep]
    // and confirm it's excluded.
    int t4 = db.store_turn("Q4", "A4", "memo", false, G, "2026-01-02 09:00:00");
    assert(db.finalize_turn_summary(t4, "Turn four: unrelated later topic.", "memo"));
    db.store_episode("Episode 3: later unrelated run.", "memo", G,
                     "2026-01-02 09:00:00", "2026-01-02 09:00:00");
    auto sess_texts2 = db.bounded_session_rollup_texts(
        G, static_cast<int64_t>(*first_ep), static_cast<int64_t>(*last_ep));
    assert(sess_texts2.size() == 2);  // unchanged -- later episode excluded

    // store_session_summary: one immutable insert per closed run. Two
    // separate closes of the SAME session_guid produce TWO rows (unlike
    // the old running-upsert model, which kept exactly one).
    int sess1 = db.store_session_summary("Session run 1 rollup.", "memo", G,
                                         static_cast<int64_t>(*first_ep),
                                         static_cast<int64_t>(*last_ep));
    assert(sess1 > 0);
    auto later_ep = ragger::parse_db_timestamp("2026-01-02 09:00:00");
    assert(later_ep.has_value());
    int sess2 = db.store_session_summary("Session run 2 rollup.", "memo", G,
                                         static_cast<int64_t>(*later_ep),
                                         static_cast<int64_t>(*later_ep));
    assert(sess2 > 0);
    assert(sess1 != sess2);

    raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='session' "
        "AND session_id=(SELECT session_id FROM sessions WHERE guid='phase2-session')",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 2);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    // bounded_project_rollup_texts: gathers level='session' rows (produced
    // above) whose span falls entirely within [first_ts, last_ts],
    // session-unscoped.
    auto proj_texts = db.bounded_project_rollup_texts(
        static_cast<int64_t>(*first_ep), static_cast<int64_t>(*later_ep));
    assert(proj_texts.size() == 2);
    assert(proj_texts[0] == "Session run 1 rollup.");
    assert(proj_texts[1] == "Session run 2 rollup.");

    // store_project_summary: one immutable insert per closed project run.
    int proj1 = db.store_project_summary("Project run 1 rollup.", "memo",
                                         static_cast<int64_t>(*first_ep),
                                         static_cast<int64_t>(*later_ep));
    assert(proj1 > 0);

    raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*), session_id FROM summaries WHERE level='project'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    assert(sqlite3_column_type(st, 1) == SQLITE_NULL);  // session-unscoped
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
    ragger::SqliteBackend db(emb, TEMP_DB);

    const std::string G1 = "already-summarized-session";
    const std::string G2 = "later-session";

    // A closed run for G1: two consecutive turns, then a turn in a DIFFERENT
    // session (G2) which edge-closes G1's run and becomes the open tail.
    int t1 = db.store_turn("Q1", "A1", "memo", false, G1, "2026-03-01 10:00:00");
    int t2 = db.store_turn("Q2", "A2", "memo", false, G1, "2026-03-01 10:00:05");
    assert(db.finalize_turn_summary(t1, "Turn one.", "memo"));
    assert(db.finalize_turn_summary(t2, "Turn two.", "memo"));
    (void)db.store_turn("Q3", "A3", "memo", false, G2, "2026-03-01 11:00:00");

    auto first_ts = ragger::parse_db_timestamp("2026-03-01 10:00:00");
    auto last_ts  = ragger::parse_db_timestamp("2026-03-01 10:00:05");
    assert(first_ts.has_value() && last_ts.has_value());

    // G1's run has already been summarized once (the normal, correct close).
    int s1 = db.store_session_summary("G1 run rollup.", "memo", G1,
                                      static_cast<int64_t>(*first_ts),
                                      static_cast<int64_t>(*last_ts));
    assert(s1 > 0);

    // Simulate the post-migration state: watermark is 0 (never stamped).
    db.advance_session_boundary_watermark(0);

    // The scan must NOT return G1's already-summarized run. (G2's run is the
    // open tail -- excluded by the MAX(grp) rule regardless.)
    auto runs = db.sessions_needing_close_boundary();
    for (const auto& r : runs) {
        assert(r.session_guid != G1 &&
               "already-summarized run re-closed with zeroed watermark (lookback bug)");
    }

    db.close();
    cleanup();
}

void test_path_normalization(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    const char* home = std::getenv("HOME");
    assert(home != nullptr);
    std::string text = std::string("File at ") + home + "/Documents/test.txt is important.";
    db.store(text);

    auto all = db.load_all();
    assert(all.size() == 1);
    // Should be normalized to ~/
    assert(all[0].text.find("~/Documents/test.txt") != std::string::npos);
    assert(all[0].text.find(home) == std::string::npos);

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
        ragger::SqliteBackend db(emb, TEMP_DB);

        // (1) Summary via the public store API (embedded immediately).
        db.store("The Voyager probes left the solar system carrying golden records.");

        // (2) Document via store_document (embedded immediately).
        ragger::DocumentChunk doc;
        doc.text  = "Photosynthesis converts sunlight into chemical energy in plants.";
        doc.title = "Botany Notes";
        db.store_document(doc);

        // (3) Decision inserted directly into the L6 table with NO embedding,
        // mirroring how the external summarizer writes decisions. backfill
        // then embeds it and must invalidate the decision cache.
        {
            sqlite3* raw = nullptr;
            assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
            const char* sql =
                "INSERT INTO decisions (text, status, tags, created_at) "
                "VALUES ('We will migrate the database to PostgreSQL next quarter.', "
                "'current', '', '2026-01-01T00:00:00Z')";
            char* err = nullptr;
            assert(sqlite3_exec(raw, sql, nullptr, nullptr, &err) == SQLITE_OK);
            sqlite3_close(raw);
        }

        // Embed the unembedded decision row (and any other deferred rows).
        int filled = db.backfill_embeddings(emb);
        assert(filled >= 1);

        // Each corpus should surface for its own topical query, tagged by source.
        auto source_of = [&](const std::string& query) -> std::string {
            auto resp = db.search(query, 5, 0.0f, {});
            assert(!resp.results.empty());
            return resp.results[0].metadata.value("source", std::string());
        };

        assert(source_of("interstellar spacecraft golden record") == "summary");
        assert(source_of("how plants make energy from light")     == "document");
        assert(source_of("plan to switch to PostgreSQL database")  == "decision");

        // corpus_size in the timing payload reflects all three tables.
        auto resp = db.search("database", 5, 0.0f, {});
        assert(resp.timing.value("corpus_size", 0) == 3);

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
        ragger::SqliteBackend db(emb, TEMP_DB);

        const std::string G = "meta-shape-session";
        const std::string when = "2026-04-15 09:30:00";
        int t1 = db.store_turn("Q1", "A1", "memo", false, G, when);
        assert(t1 > 0);
        assert(db.finalize_turn_summary(
            t1, "Discussed the migration to a columnar store.", "memo"));

        auto resp = db.search("columnar store migration", 5, 0.0f, {});
        assert(!resp.results.empty());

        // Find the turn_summary hit among results.
        const ragger::SearchResult* hit = nullptr;
        for (const auto& r : resp.results) {
            if (r.metadata.value("source", std::string()) == "turn_summary") {
                hit = &r;
                break;
            }
        }
        assert(hit && "expected a turn_summary result");

        // datetime present and mirrors the stored turn timestamp.
        assert(hit->metadata.contains("datetime"));
        assert(hit->metadata["datetime"] == when);
        // The top-level result timestamp is also populated (not blanked).
        assert(hit->timestamp == when);
        // session_id and turn_id both present.
        assert(hit->metadata.contains("session_id"));
        assert(hit->metadata.contains("turn_id"));
        assert(hit->metadata["turn_id"] == t1);

        db.close();
    }
    cleanup();
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main() {
    // Need config for model dir
    ragger::init_config("");
    auto model_dir = ragger::config().resolved_model_dir();

    if (!fs::exists(model_dir + "/model.onnx")) {
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
    test_store_turn_creates_no_turn_summaries_row(emb);
    test_store_turn_partial_turn_creates_no_turn_summaries_row(emb);
    test_finalize_turn_summary_rejects_duplicate(emb);
    test_finalize_turn_summary_missing_turn_returns_false(emb);
    test_turn_summary_survives_turn_deletion(emb);
    test_unsummarized_turns_excludes_summarized(emb);
    test_mark_turn_summarized_trivial_turn(emb);
    test_summary_primitives(emb);
    test_episode_rollup_no_dup(emb);
    test_session_close_no_lookback_before_existing_summary(emb);
    test_path_normalization(emb);

    std::println("test_sqlite_backend: all passed");
    return 0;
}
