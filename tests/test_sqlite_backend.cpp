/**
 * SQLite backend integration tests
 *
 * Requires ONNX model files at the configured model_dir.
 * Uses a temp DB that's deleted after the test.
 */
#include "ragger/config.h"
#include "ragger/embedder.h"
#include "ragger/sqlite_backend.h"
#include "ragger/user_store.h"
#include "ragger/auth.h"
#include <sqlite3.h>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>

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

    // Lean v2 summaries: tags (comma-joined) + level/status round-trip.
    ragger::json meta = {
        {"level", "session"},
        {"status", "complete"},
        {"tags", {"test", "unit"}}
    };
    std::string id = db.store("Metadata round-trip test.", meta);
    assert(!id.empty());

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].metadata["level"] == "session");
    assert(all[0].metadata["status"] == "complete");
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
    db.store("Apple note.",  {{"level", "turn"},    {"status", "complete"}, {"tags", {"fruit"}}});
    db.store("Banana note.", {{"level", "turn"},    {"status", "current"},  {"tags", {"fruit"}}});
    db.store("Car note.",    {{"level", "session"}, {"status", "complete"}, {"tags", {"vehicle"}}});

    // Filter by level
    auto results = db.search_by_metadata({{"level", "turn"}});
    assert(results.size() == 2);
    for (auto& r : results) assert(r.metadata["level"] == "turn");

    // Filter by tag (LIKE on the tags column)
    results = db.search_by_metadata({{"tags", "vehicle"}});
    assert(results.size() == 1);
    assert(results[0].text == "Car note.");

    // AND across columns
    results = db.search_by_metadata({{"level", "turn"}, {"status", "current"}});
    assert(results.size() == 1);
    assert(results[0].text == "Banana note.");

    // Limit respected
    results = db.search_by_metadata({{"status", "complete"}}, 1);
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

    // Verify local-time "YYYY-MM-DD HH:MM:SS" pattern
    auto& ts = all[0].timestamp;
    assert(ts.length() == 19);
    assert(ts[4] == '-');
    assert(ts[7] == '-');
    assert(ts[10] == ' ');
    assert(ts[13] == ':');
    assert(ts[16] == ':');

    db.close();
    cleanup();
}

// Guards the v2 read-path wiring: the generic store/read API operates on the
// lean `summaries` table (issue #33 split), not the dropped pre-v2 `memories`
// table. level/status/tags round-trip; there is no metadata blob.
void test_v2_summaries_backing(ragger::Embedder& emb) {
    cleanup();
    {
        ragger::SqliteBackend db(emb, TEMP_DB);
        db.store("v2 summaries backing check.",
                 {{"level", "session"}, {"status", "complete"}, {"tags", {"note"}}});

        auto all = db.load_all();
        assert(all.size() == 1);
        assert(all[0].metadata["level"] == "session");
        assert(all[0].metadata["status"] == "complete");
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
        assert(emb_bytes == 384 * 2);            // embedded as f16 (2 bytes/value)
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
    {
        ragger::SqliteBackend db(emb, TEMP_DB);

        // First answer to a prompt, in session A.
        int t1 = db.store_turn("Explain the general_search recipe.",
                               "It searches summaries.", "model-a",
                               /*defer_embedding=*/false, "sessionA");
        assert(t1 > 0);

        // Same prompt re-answered — DIFFERENT session (sessionB), different
        // assistant text. Must update t1 in place, not insert a new row.
        int t2 = db.store_turn("Explain the general_search recipe.",
                               "It searches summaries, documents, and decisions.",
                               "model-b", /*defer_embedding=*/false, "sessionB");
        assert(t2 == t1);  // same row id returned — keep-latest

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

    sqlite3_close(raw);
    cleanup();
}
// L2/L3 summary primitives (issue #22): store_summary, current_session_summary,
// update_summary_text, set_summary_status — the deterministic backbone the
// forked summarization pipeline drives.
void test_summary_primitives(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    assert(!db.current_session_summary().has_value());

    int l2 = db.store_summary("User asked about France; capital is Paris.",
                              "turn", "complete", "memo-model");
    assert(l2 > 0);

    int l3 = db.store_summary("Discussed European capitals.",
                              "session", "current", "memo-model");
    assert(l3 > 0);
    auto cur = db.current_session_summary();
    assert(cur.has_value());
    assert(cur->first == l3);
    assert(cur->second == "Discussed European capitals.");

    assert(db.update_summary_text(l3, "Discussed European capitals, focus France.",
                                  "memo-model"));
    cur = db.current_session_summary();
    assert(cur->second == "Discussed European capitals, focus France.");

    // Topic shift: complete the old, start a new current.
    assert(db.set_summary_status(l3, "complete"));
    assert(!db.current_session_summary().has_value());
    int l3b = db.store_summary("Switched to cooking techniques.",
                               "session", "current", "memo-model");
    auto cur2 = db.current_session_summary();
    assert(cur2->first == l3b);

    sqlite3* raw = nullptr;
    assert(sqlite3_open(TEMP_DB.c_str(), &raw) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(raw, "SELECT COUNT(*) FROM summaries WHERE level='turn'",
                       -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries WHERE level='session' AND status='complete'",
        -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 1);
    sqlite3_finalize(st);
    sqlite3_prepare_v2(raw,
        "SELECT COUNT(*) FROM summaries s JOIN models m ON m.model_id=s.model_id "
        "WHERE m.name='memo-model'", -1, &st, nullptr);
    assert(sqlite3_step(st) == SQLITE_ROW && sqlite3_column_int(st, 0) == 3);
    sqlite3_finalize(st);
    sqlite3_close(raw);

    assert(!db.update_summary_text(99999, "x", "memo-model"));
    assert(!db.set_summary_status(99999, "complete"));

    // Recipe ingredients (issue #23): recency-based fetch.
    auto turns = db.recent_summaries("turn", 5);
    assert(turns.size() == 1);
    auto sessions = db.recent_summaries("session", 5);
    assert(sessions.size() == 2);                 // l3 (complete) + l3b (current)
    assert(db.recent_summaries("project", 5).empty());
    assert(db.recent_summaries("turn", 0).empty());
    assert(db.current_decisions(5).empty());      // none created

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
                "INSERT INTO decisions (text, status, tags, timestamp) "
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
    test_search_limit(emb);
    test_load_all(emb);
    test_rebuild_embeddings(emb);
    test_rebuild_embeddings_empty_db(emb);
    test_rebuild_embeddings_count_matches(emb);
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
    test_store_turn(emb);
    test_store_turn_dedup(emb);
    test_summary_primitives(emb);
    test_path_normalization(emb);

    std::println("test_sqlite_backend: all passed");
    return 0;
}
