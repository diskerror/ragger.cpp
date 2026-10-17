/**
 * Path normalization tests
 *
 * Tests that $HOME/ is replaced with ~/ in stored text.
 * Uses store + load_all to verify normalization through the public API
 * (normalize_path is a private static method in sqlite::Backend).
 */
#include "config.h"
#include "embedder.h"
#include "sqlite/backend.h"
#include "test_env.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <print>

namespace fs = std::filesystem;

static const std::string TEMP_DB = "/tmp/ragger_test_pathnorm.db";

static void cleanup() {
    fs::remove(TEMP_DB);
    fs::remove(TEMP_DB + "-wal");
    fs::remove(TEMP_DB + "-shm");
}

void test_normalize_home_in_stored_text(ragger::Embedder& emb) {
    std::println("  test_normalize_home_in_stored_text...");
    cleanup();

    ragger::sqlite::Backend db(emb, TEMP_DB);
    const char* home = std::getenv("HOME");
    assert(home != nullptr);

    std::string input = std::string(home) + "/Documents/test.md";
    db.store(input);

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == "~/Documents/test.md");

    db.close();
    cleanup();
    std::println(" OK");
}

void test_normalize_preserves_non_home_paths(ragger::Embedder& emb) {
    std::println("  test_normalize_preserves_non_home_paths...");
    cleanup();

    ragger::sqlite::Backend db(emb, TEMP_DB);
    std::string input = "/usr/local/bin/python";
    db.store(input);

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == "/usr/local/bin/python");

    db.close();
    cleanup();
    std::println(" OK");
}

void test_normalize_multiple_occurrences(ragger::Embedder& emb) {
    std::println("  test_normalize_multiple_occurrences...");
    cleanup();

    ragger::sqlite::Backend db(emb, TEMP_DB);
    const char* home = std::getenv("HOME");
    assert(home != nullptr);

    std::string input = "Source: " + std::string(home) + "/a.md and " +
                        std::string(home) + "/b.md";
    db.store(input);

    auto all = db.load_all();
    assert(all.size() == 1);
    // Both occurrences should be replaced
    assert(all[0].text == "Source: ~/a.md and ~/b.md");
    // Original home path should not appear
    assert(all[0].text.find(home) == std::string::npos);

    db.close();
    cleanup();
    std::println(" OK");
}

void test_normalize_partial_match_not_replaced(ragger::Embedder& emb) {
    std::println("  test_normalize_partial_match_not_replaced...");
    cleanup();

    ragger::sqlite::Backend db(emb, TEMP_DB);
    const char* home = std::getenv("HOME");
    assert(home != nullptr);

    // Home path without trailing component — should NOT be replaced
    // (normalize_path looks for HOME + "/", not HOME alone)
    std::string input = "Path is " + std::string(home);
    db.store(input);

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == input);  // unchanged

    db.close();
    cleanup();
    std::println(" OK");
}

void test_normalize_embedded_in_longer_text(ragger::Embedder& emb) {
    std::println("  test_normalize_embedded_in_longer_text...");
    cleanup();

    ragger::sqlite::Backend db(emb, TEMP_DB);
    const char* home = std::getenv("HOME");
    assert(home != nullptr);

    std::string input = "I stored notes at " + std::string(home) +
                        "/Projects/notes.md for future reference.";
    db.store(input);

    auto all = db.load_all();
    assert(all.size() == 1);
    assert(all[0].text == "I stored notes at ~/Projects/notes.md for future reference.");

    db.close();
    cleanup();
    std::println(" OK");
}

// PHASE2 Step 5: is_safe_model_repo (C §1.6).
void test_is_safe_model_repo() {
    std::println("  test_is_safe_model_repo...");
    using ragger::is_safe_model_repo;
    for (const char* ok : {"sentence-transformers/all-MiniLM-L6-v2", "a/b",
                           "a.b.c/d_e-f.g", "A1/B2"})
        CHECK(is_safe_model_repo(ok));
    for (const char* bad : {"x/../../../tmp/evil", "../a", "a/..", "a//b",
                            "a/b/c", "/a", "a/", "a\\b/c", "", "a/b c",
                            ".", "./a", "a/.", "..", "a/.."})
        CHECK(!is_safe_model_repo(bad));
    // Embedded NUL (built with an explicit length so the NUL is inside).
    CHECK(!is_safe_model_repo(std::string("a/b\0c", 5)));
    // Over-long.
    CHECK(!is_safe_model_repo(std::string(150, 'a') + "/" + std::string(60, 'b')));
    CHECK(is_safe_model_repo(std::string(100, 'a') + "/" + std::string(99, 'b')));  // 200 total
    std::println(" OK");
}

int main() {
    TestEnv env("path_normalization");  // must be first: isolates from ~/.ragger
    std::println("Running path normalization tests:");

    test_is_safe_model_repo();  // pure function — before the model gate

    auto cfg = ragger::config();
    assert(cfg.normalize_home_path == true);

    // Embedder needed for sqlite::Backend store()
    auto model_dir = env.model_dir();
    if (!std::filesystem::exists(model_dir + "/model.onnx") && !std::filesystem::exists(model_dir + "/onnx/model.onnx")) {
        std::cerr << "Skipping path normalization tests: model not found at " << model_dir << "\n";
        return 0;
    }
    ragger::Embedder emb(model_dir);

    test_normalize_home_in_stored_text(emb);
    test_normalize_preserves_non_home_paths(emb);
    test_normalize_multiple_occurrences(emb);
    test_normalize_partial_match_not_replaced(emb);
    test_normalize_embedded_in_longer_text(emb);

    std::println("test_path_normalization: all passed");
    return 0;
}
