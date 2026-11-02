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
#include <arpa/inet.h>
#include <cassert>
#include <iostream>
#include <netinet/in.h>
#include <print>
#include <sys/socket.h>
#include <unistd.h>

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

// S18 (C M6): run() must report a busy port as a start failure (false)
// through normal return — std::exit inside run() skipped destructors with
// DB connections already open.
void p10_run_port_busy() {
    std::println("  p10_run_port_busy...");
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(18433);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    assert(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    assert(::listen(fd, 1) == 0);

    std::string temp_db = "/tmp/ragger_test_server_busy.db";
    {
        ragger::RaggerMemory mem(temp_db);
        ragger::Server server(mem, "127.0.0.1", 18433);
        // If this ever called std::exit the test process would die here
        // instead of asserting; reaching the assert proves normal return.
        assert(!server.run());
        mem.close();
    }
    std::filesystem::remove(temp_db);
    std::filesystem::remove(temp_db + "-wal");
    std::filesystem::remove(temp_db + "-shm");
    ::close(fd);
    std::println(" ok");
}

int main() {
    TestEnv env("server");  // must be first: isolates from ~/.ragger
    std::println("Running server tests:");

    test_server_instantiation(env.model_dir());
    p10_run_port_busy();

    std::println("test_server: basic checks passed");
    return 0;
}
