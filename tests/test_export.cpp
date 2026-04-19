/**
 * Export functionality tests
 *
 * Tests the load_all pathway used by export, and verifies
 * data integrity for export scenarios.
 * Requires ONNX model files.
 */
#include "ragger/config.h"
#include "ragger/embedder.h"
#include "ragger/sqlite_backend.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

static const std::string TEMP_DB = "/tmp/ragger_test_export.db";

static void cleanup() {
    fs::remove(TEMP_DB);
    fs::remove(TEMP_DB + "-wal");
    fs::remove(TEMP_DB + "-shm");
}

void test_export_basic(ragger::Embedder& emb) {
    std::print("  test_export_basic..."); std::cout.flush();
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("# Title\n\nFirst paragraph about testing.",
             {{"collection", "reference"}, {"source", "doc.md"}});
    db.store("# Title\n\nSecond paragraph continues.",
             {{"collection", "reference"}, {"source", "doc.md"}});

    auto all = db.load_all("reference");
    assert(all.size() == 2);
    // Both should have text and metadata
    for (auto& r : all) {
        assert(!r.text.empty());
        assert(r.metadata["collection"] == "reference");
    }

    db.close();
    cleanup();
    std::cout << " OK\n";
}

void test_export_heading_deduplication(ragger::Embedder& emb) {
    std::print("  test_export_heading_deduplication..."); std::cout.flush();
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    // Store chunks with overlapping heading chains
    db.store("# Main Title\n\n## Section A\n\nContent A.",
             {{"collection", "reference"}, {"source", "test.md"}});
    db.store("# Main Title\n\n## Section A\n\nMore content A.",
             {{"collection", "reference"}, {"source", "test.md"}});
    db.store("# Main Title\n\n## Section B\n\nContent B.",
             {{"collection", "reference"}, {"source", "test.md"}});

    auto all = db.load_all("reference");
    assert(all.size() == 3);

    // Simulate heading deduplication (same logic as do_export_docs in main.cpp)
    std::set<std::string> seen_headings;
    int heading_count = 0;
    for (auto& r : all) {
        std::istringstream ss(r.text);
        std::string line;
        while (std::getline(ss, line)) {
            if (!line.empty() && line[0] == '#') {
                if (seen_headings.find(line) == seen_headings.end()) {
                    seen_headings.insert(line);
                    heading_count++;
                }
            }
        }
    }
    // Should have: "# Main Title", "## Section A", "## Section B" = 3 unique
    assert(heading_count == 3);

    db.close();
    cleanup();
    std::cout << " OK\n";
}

void test_export_by_collection(ragger::Embedder& emb) {
    std::print("  test_export_by_collection..."); std::cout.flush();
    cleanup();
    ragger::SqliteBackend db(emb, TEMP_DB);

    db.store("Alpha doc.", {{"collection", "alpha"}});
    db.store("Beta doc.", {{"collection", "beta"}});
    db.store("Alpha doc 2.", {{"collection", "alpha"}});

    auto alpha_only = db.load_all("alpha");
    assert(alpha_only.size() == 2);
    for (auto& r : alpha_only) {
        assert(r.metadata["collection"] == "alpha");
    }

    auto beta_only = db.load_all("beta");
    assert(beta_only.size() == 1);
    assert(beta_only[0].metadata["collection"] == "beta");

    db.close();
    cleanup();
    std::cout << " OK\n";
}

int main() {
    ragger::init_config("");
    auto model_dir = ragger::config().resolved_model_dir();

    if (!fs::exists(model_dir + "/model.onnx")) {
        std::cerr << "Skipping export tests: model not found at " << model_dir << "\n";
        std::println("test_export: SKIPPED (no model)");
        return 0;
    }

    ragger::Embedder emb(model_dir);

    std::println("Running export tests:");

    test_export_basic(emb);
    test_export_heading_deduplication(emb);
    test_export_by_collection(emb);

    std::println("test_export: all passed");
    return 0;
}
