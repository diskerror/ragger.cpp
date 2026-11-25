/**
 * tests/test_server.cpp — construction-level checks on ragger::Server.
 *
 * Two checks, both in-process; neither speaks HTTP:
 *   1. test_server_instantiation — a Server can be constructed over a scratch
 *      RaggerMemory and destroyed cleanly (pImpl wiring; no port is bound).
 *      Skipped when no embedding model is installed.
 *   2. p10_run_port_busy — Server::run() against an occupied port reports the
 *      failure by RETURNING false. S18 (C M6): it used to std::exit() there,
 *      which skipped destructors with DB connections already open.
 *
 * No route is exercised here — HTTP endpoint coverage is S53's job. The old
 * Python HTTP suite was deleted in S12; tests/python/test_plugin_robustness.py
 * is a separate, self-contained script and is registered in CTest (S52).
 *
 * DB paths come from the TestEnv scratch base (RULES.md #17), so two concurrent
 * runs — or parallel ctest jobs — cannot collide on a shared /tmp name.
 * Assertions use CHECK, not assert(): the default Release build defines NDEBUG,
 * which used to compile the bind()/listen() calls in p10_run_port_busy away
 * entirely, leaving the test asserting nothing (see tests/check.h).
 */

#include "config.h"
#include "embedder.h"
#include "memory.h"
#include "server.h"
#include "test_env.h"
#include <arpa/inet.h>
#include <iostream>
#include <netinet/in.h>
#include <print>
#include <sys/socket.h>
#include <unistd.h>

void test_server_instantiation(const std::string& model_dir, const std::string& base) {
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

    const std::string temp_db = base + "/server.db";
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
void p10_run_port_busy(const std::string& base) {
    std::println("  p10_run_port_busy...");
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(18433);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    CHECK(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    CHECK(::listen(fd, 1) == 0);

    const std::string temp_db = base + "/server_busy.db";
    {
        ragger::RaggerMemory mem(temp_db);
        ragger::Server server(mem, "127.0.0.1", 18433);
        // If this ever called std::exit the test process would die here
        // instead of failing the CHECK; reaching it proves normal return.
        CHECK(!server.run());
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

    test_server_instantiation(env.model_dir(), env.base());
    p10_run_port_busy(env.base());

    std::println("test_server: basic checks passed");
    return 0;
}
