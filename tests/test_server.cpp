/**
 * Server integration tests
 *
 * NOTE: Full HTTP endpoint testing is handled by tests/python/test_http_api.py,
 * which works against both Python and C++ server implementations.
 *
 * This file provides basic C++ sanity checks that don't require full HTTP
 * integration (which would require threading Crow, managing ports, HTTP clients).
 *
 * To run full server tests:
 *   1. Build and start the C++ server: ./build/ragger --server
 *   2. Run: cd tests/python && python test_http_api.py
 *
 * The Python test suite covers:
 * - Health endpoint
 * - Store/search/count endpoints
 * - Auth token validation
 * - Collection filtering
 * - Min score filtering
 * - Cleanup endpoint
 * - Error handling (401, 404, etc.)
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
// For now, the Python integration tests provide comprehensive coverage.

int main() {
    TestEnv env("server");  // must be first: isolates from ~/.ragger
    std::println("Running server tests:");
    std::println("NOTE: Full HTTP endpoint testing is in tests/python/test_http_api.py");

    test_server_instantiation(env.model_dir());

    std::println("test_server: basic checks passed");
    std::println("For complete server testing, run: cd tests/python && python test_http_api.py");
    return 0;
}
