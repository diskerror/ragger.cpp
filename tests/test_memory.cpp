/**
 * RaggerMemory facade tests
 *
 * Requires ONNX model files at the configured model_dir.
 */
#include "config.h"
#include "memory.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>
#include <stdexcept>

namespace fs = std::filesystem;

static const std::string TEMP_DB1 = "/tmp/ragger_test_mem1.db";
static const std::string TEMP_DB2 = "/tmp/ragger_test_mem2.db";

static void cleanup(const std::string& path) {
    fs::remove(path);
    fs::remove(path + "-wal");
    fs::remove(path + "-shm");
}

static void cleanup_all() {
    cleanup(TEMP_DB1);
    cleanup(TEMP_DB2);
}

void test_store_and_search() {
    std::println("  test_store_and_search...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("The Eiffel Tower is located in Paris, France.");
    mem.store("Photosynthesis converts sunlight into energy in plants.");

    auto resp = mem.search("Where is the Eiffel Tower?", 5, 0.0f);
    assert(!resp.results.empty());
    assert(resp.results[0].text.find("Eiffel") != std::string::npos);

    mem.close();
    cleanup_all();
    std::println(" OK");
}

void test_store_with_tags() {
    std::println("  test_store_with_tags...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("Reference: HTTP status 200 means OK.", {{"tags", {"reference"}}});
    mem.store("Memory: I had coffee this morning.", {{"tags", {"diary"}}});

    // Lean v2 has no collection filter; verify tags round-trip + search works.
    auto resp = mem.search("HTTP status", 5, 0.0f);
    assert(!resp.results.empty());
    assert(resp.results[0].text.find("HTTP") != std::string::npos);
    assert(resp.results[0].metadata["tags"] == "reference");

    mem.close();
    cleanup_all();
    std::println(" OK");
}


void test_count() {
    std::println("  test_count...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    assert(mem.count() == 0);

    mem.store("First item.");
    mem.store("Second item.");
    mem.store("Third item.");
    assert(mem.count() == 3);

    mem.close();
    cleanup_all();
    std::println(" OK");
}

void test_delete() {
    std::println("  test_delete...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    std::string id1 = mem.store("To be deleted.");
    mem.store("To be kept.");
    assert(mem.count() == 2);

    bool deleted = mem.delete_memory(std::stoi(id1));
    assert(deleted);
    assert(mem.count() == 1);

    mem.close();
    cleanup_all();
    std::println(" OK");
}


void test_empty_db_path_throws() {
    std::println("  test_empty_db_path_throws...");

    bool threw = false;
    try {
        ragger::RaggerMemory mem("");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::println(" OK");
}

int main() {
    ragger::init_config("");
    auto model_dir = ragger::config().resolved_model_dir();

    if (!fs::exists(model_dir + "/model.onnx")) {
        std::cerr << "Skipping memory tests: model not found at " << model_dir << "\n";
        std::println("test_memory: SKIPPED (no model)");
        return 0;
    }

    std::println("Running memory facade tests:");

    test_store_and_search();
    test_store_with_tags();
    test_count();
    test_delete();
    test_empty_db_path_throws();

    std::println("test_memory: all passed");
    return 0;
}
