/**
 * Server integration tests
 *
 * Basic C++ sanity checks for the Server object (pImpl construction against a
 * scratch DB). Full HTTP endpoint behavior is documented in
 * docs/http-api.md; the retired Python suite (tests/python/test_http_api.py)
 * was deleted in the v0.17 rework (S12) — it targeted the removed
 * ragger_memory package and was never wired into CTest.
 */

#include "config.h"
#include "embedder.h"
#include "memory.h"
#include "server.h"
#include "test_env.h"
#include <cassert>
#include <iostream>
#include <print>

void test_server_instantiation(const std::string& model_dir) {
    std::println("  test_server_instantiation...");

    // Verify we can create a Server object without crashing
    // This tests the basic pImpl pattern and constructor
    // (config is initialized by the TestEnv fixture in main; the model dir
    // comes from the fixture so it survives the base override)

    // Skip if no model
    if (!std::filesystem::exists(model_dir + "/model.onnx") && !std::filesystem::exists(model_dir + "/onnx/model.onnx")) {
        std::println(" SKIPPED (no model)");
        return;
    }

    std::string temp_db = "/tmp/ragger_test_server.db";
    ragger::RaggerMemory mem(temp_db);

    // Create server on a high port (won't actually run it)
    ragger::Server server(mem, "127.0.0.1", 18432);

    // Clean up
    mem.close();
    std::filesystem::remove(temp_db);
    std::filesystem::remove(temp_db + "-wal");
    std::filesystem::remove(temp_db + "-shm");

    std::println(" OK");
}

// TODO: Add more C++-only server tests here if we extract testable
// helper functions from server.cpp (auth logic, route handlers, etc.)

int main() {
    TestEnv env("server");  // must be first: isolates from ~/.ragger
    std::println("Running server tests:");

    test_server_instantiation(env.model_dir());

    std::println("test_server: basic checks passed");
    return 0;
}
