/**
 * SQLite backend integration tests
 *
 * Requires ONNX model files at the configured model_dir.
 * Uses a temp DB that's deleted after the test.
 */
#include "ragger/config.h"
#include "ragger/embedder.h"
#include "ragger/sqlite_backend.h"
#include <cassert>
#include <filesystem>
#include <iostream>

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

    ragger::json meta = {
        {"category", "fact"},
        {"collection", "memory"},
        {"tags", {"test", "unit"}}
    };
    std::string id = db.store("Metadata round-trip test.", meta);
    assert(!id.empty());

    // Retrieve via load_all and check metadata survived
    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].metadata["category"] == "fact");
    assert(all[0].metadata["collection"] == "memory");
    assert(all[0].metadata["tags"].size() == 2);

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

void test_search_collection_filter(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("Apple is a fruit.", {{"collection", "food"}});
    db.store("Apple makes computers.", {{"collection", "tech"}});
    db.store("Bananas are yellow.", {{"collection", "food"}});

    // Search only "food" collection
    auto resp = db.search("apple", 5, 0.0f, {"food"});
    for (auto& r : resp.results) {
        assert(r.metadata["collection"] == "food");
    }

    // Search only "tech" collection
    resp = db.search("apple", 5, 0.0f, {"tech"});
    for (auto& r : resp.results) {
        assert(r.metadata["collection"] == "tech");
    }

    // Search all collections (empty filter)
    resp = db.search("apple", 5, 0.0f);
    assert(resp.results.size() >= 2);

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

void test_collections_list(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("Doc one.", {{"collection", "alpha"}});
    db.store("Doc two.", {{"collection", "beta"}});
    db.store("Doc three.", {{"collection", "alpha"}});

    auto colls = db.collections();
    assert(colls.size() == 2);
    bool has_alpha = false, has_beta = false;
    for (auto& c : colls) {
        if (c == "alpha") has_alpha = true;
        if (c == "beta") has_beta = true;
    }
    assert(has_alpha && has_beta);

    db.close();
    cleanup();
}

void test_load_all(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("First memory.", {{"collection", "test"}});
    db.store("Second memory.", {{"collection", "test"}});
    db.store("Third memory.", {{"collection", "other"}});

    // Load all
    auto all = db.load_all();
    assert(all.size() == 3);

    // Load by collection
    auto test_only = db.load_all("test");
    assert(test_only.size() == 2);

    db.close();
    cleanup();
}

void test_rebuild_bm25(ragger::Embedder& emb) {
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("BM25 index rebuild test document one.");
    db.store("BM25 index rebuild test document two.");
    db.store("Something completely different here.");

    int count = db.rebuild_bm25();
    assert(count == 3);

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

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    // Need config for model dir
    ragger::init_config("");
    auto model_dir = ragger::config().resolved_model_dir();

    if (!fs::exists(model_dir + "/model.onnx")) {
        std::cerr << "Skipping backend tests: model not found at " << model_dir << "\n";
        std::cout << "test_sqlite_backend: SKIPPED (no model)\n";
        return 0;
    }

    ragger::Embedder emb(model_dir);

    test_store_and_count(emb);
    test_store_with_metadata(emb);
    test_search_basic(emb);
    test_search_collection_filter(emb);
    test_search_min_score(emb);
    test_search_limit(emb);
    test_collections_list(emb);
    test_load_all(emb);
    test_rebuild_bm25(emb);
    test_search_timing(emb);

    std::cout << "test_sqlite_backend: all passed\n";
    return 0;
}
