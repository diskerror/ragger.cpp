/**
 * Path normalization tests
 *
 * Tests that $HOME/ is replaced with ~/ in stored text.
 * Uses store + load_all to verify normalization through the public API
 * (normalize_path is a private static method in SqliteBackend).
 */
#include "ragger/config.h"
#include "ragger/embedder.h"
#include "ragger/sqlite_backend.h"
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

    ragger::SqliteBackend db(emb, TEMP_DB);
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

    ragger::SqliteBackend db(emb, TEMP_DB);
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

    ragger::SqliteBackend db(emb, TEMP_DB);
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

    ragger::SqliteBackend db(emb, TEMP_DB);
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

    ragger::SqliteBackend db(emb, TEMP_DB);
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

int main() {
    std::println("Running path normalization tests:");

    ragger::init_config("", true);  // quiet
    auto cfg = ragger::config();
    assert(cfg.normalize_home_path == true);

    // Embedder needed for SqliteBackend store()
    ragger::Embedder emb(cfg.resolved_model_dir());

    test_normalize_home_in_stored_text(emb);
    test_normalize_preserves_non_home_paths(emb);
    test_normalize_multiple_occurrences(emb);
    test_normalize_partial_match_not_replaced(emb);
    test_normalize_embedded_in_longer_text(emb);

    std::println("test_path_normalization: all passed");
    return 0;
}
