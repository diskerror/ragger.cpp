/**
 * RaggerMemory facade tests
 *
 * Requires ONNX model files at the configured model_dir.
 */
#include "ragger/config.h"
#include "ragger/memory.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>

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
    std::cout << "  test_store_and_search..." << std::flush;
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("The Eiffel Tower is located in Paris, France.");
    mem.store("Photosynthesis converts sunlight into energy in plants.");

    auto resp = mem.search("Where is the Eiffel Tower?", 5, 0.0f);
    assert(!resp.results.empty());
    assert(resp.results[0].text.find("Eiffel") != std::string::npos);

    mem.close();
    cleanup_all();
    std::cout << " OK\n";
}

void test_store_with_collection() {
    std::cout << "  test_store_with_collection..." << std::flush;
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("Reference: HTTP status 200 means OK.", {{"collection", "reference"}});
    mem.store("Memory: I had coffee this morning.", {{"collection", "memory"}});

    auto resp = mem.search("HTTP status", 5, 0.0f, {"reference"});
    assert(!resp.results.empty());
    for (auto& r : resp.results) {
        assert(r.metadata["collection"] == "reference");
    }

    mem.close();
    cleanup_all();
    std::cout << " OK\n";
}

void test_search_merging_dual_db() {
    std::cout << "  test_search_merging_dual_db..." << std::flush;
    cleanup_all();

    // user_db_path is 3rd param — TEMP_DB1 = common, TEMP_DB2 = user
    ragger::RaggerMemory mem(TEMP_DB1, "", TEMP_DB2);
    (void)0;

    // Store to common DB
    mem.store("The speed of light is 299792458 meters per second.", {}, true);
    // Store to user DB
    mem.store("My favorite color is blue.");

    // Count should include both
    assert(mem.count() == 2);

    // Search should find results from both DBs
    auto resp = mem.search("speed of light meters per second", 5, 0.0f);
    assert(!resp.results.empty());
    // At least one result should mention speed of light
    bool found_light = false;
    for (auto& r : resp.results) {
        if (r.text.find("speed of light") != std::string::npos) found_light = true;
    }
    assert(found_light);

    resp = mem.search("my favorite color is blue", 5, 0.0f);
    assert(!resp.results.empty());
    bool found_color = false;
    for (auto& r : resp.results) {
        if (r.text.find("favorite color") != std::string::npos) found_color = true;
    }
    assert(found_color);

    mem.close();
    cleanup_all();
    std::cout << " OK\n";
}

void test_count() {
    std::cout << "  test_count..." << std::flush;
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    assert(mem.count() == 0);

    mem.store("First item.");
    mem.store("Second item.");
    mem.store("Third item.");
    assert(mem.count() == 3);

    mem.close();
    cleanup_all();
    std::cout << " OK\n";
}

void test_delete() {
    std::cout << "  test_delete..." << std::flush;
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
    std::cout << " OK\n";
}

void test_collections() {
    std::cout << "  test_collections..." << std::flush;
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("Doc A.", {{"collection", "alpha"}});
    mem.store("Doc B.", {{"collection", "beta"}});
    mem.store("Doc C.", {{"collection", "gamma"}});

    auto colls = mem.collections();
    assert(colls.size() == 3);

    mem.close();
    cleanup_all();
    std::cout << " OK\n";
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
    test_store_with_collection();
    test_search_merging_dual_db();
    test_count();
    test_delete();
    test_collections();

    std::println("test_memory: all passed");
    return 0;
}
