/**
 * http_fixture.h — one running Ragger daemon per test binary, on an ephemeral
 * port, inside the TestEnv scratch base (S53, C §7 item 1).
 *
 * Why header-only: like check.h and test_env.h it is test scaffolding, not a
 * library; only one test binary includes it today and a .cpp would buy nothing.
 *
 * Lifecycle (RAII, in this order):
 *   TestEnv          -> scratch base, never ~/.ragger (RULES.md #17)
 *   socket_enabled   -> false through the SAME persisted-config key the daemon
 *                       uses (src/server.cpp reads cfg.socket_enabled), so no
 *                       AF_UNIX listener is created under the base
 *   free_port()      -> bind port 0, read the assigned port, close and hand it
 *                       to Server. There is a theoretical race (another process
 *                       could take it between close and bind); on loopback in a
 *                       test run it has not bitten, and binding the real
 *                       listener first would leave Server unable to use it.
 *   Server::run()    -> on a std::jthread. run() returning false means it could
 *                       not start (S18: normal return, not std::exit) — that is
 *                       a fixture failure and aborts with the port in the
 *                       message rather than silently testing nothing.
 *   /health poll     -> up to 5 s; the thread is alive but the listener is not
 *                       guaranteed the instant run() is entered.
 *   ~HttpFixture     -> stop() then join (jthread joins on destruction), so no
 *                       test binary ever leaves a listener behind.
 */
#pragma once

#include "check.h"
#include "test_env.h"
#include "auth.h"
#include "config.h"
#include "config_access.h"
#include "memory.h"
#include "server.h"
#include "util/http_client.h"

#include <atomic>
#include <chrono>
#include <arpa/inet.h>
#include <functional>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <memory>
#include <print>
#include <string>

namespace ragger_test {

/// Ask the kernel for a free loopback port, then release it.
inline int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;                            // ephemeral
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    CHECK(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    CHECK(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    const int port = ntohs(addr.sin_port);
    ::close(fd);
    CHECK(port > 0);
    return port;
}

struct HttpFixture {
    /// `after_env` runs after the scratch base exists but before the token
    /// and the Server are created — the window for persisting config the
    /// daemon must see at startup (e.g. an inference endpoint pointing at a
    /// test stub, LR S91).
    explicit HttpFixture(const char* tag, const std::function<void()>& after_env = {})
        : env(tag) {
        // The daemon's own switch, persisted into the scratch base. The
        // settable key is `socket_enable` (the map in config_access.cpp is
        // the authority; the Config field is socket_enabled).
        auto r = ragger::set_config_persisted("socket_enable", "false");
        CHECK(r.has_value());
        CHECK(!ragger::config().socket_enabled);

        if (after_env) after_env();

        token = ragger::ensure_token();
        CHECK(!token.empty());

        port = free_port();
        mem = std::make_unique<ragger::RaggerMemory>(env.base() + "/http_test.db");
        server = std::make_unique<ragger::Server>(*mem, "127.0.0.1", port);

        // run() returns only after stop(), long after this ctor is gone, so
        // its result must land in a member, never a ctor-local.
        runner = std::jthread([this] { run_result = server->run(); });

        // /health needs no token: it is the readiness probe.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        int seen = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            auto r = client.get(url("/health"));
            seen = static_cast<int>(r.status);
            if (seen == 200) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        if (seen != 200) {
            std::println(stderr,
                         "http_fixture: server never became ready on port {} "
                         "(last /health status {}, run() returned {})",
                         port, seen, run_result.load());
            CHECK(false && "HTTP fixture could not start");
        }
        ready = true;
    }

    ~HttpFixture() {
        if (server) server->stop();
        if (runner.joinable()) runner.join();
        if (mem) mem->close();
    }

    HttpFixture(const HttpFixture&) = delete;
    HttpFixture& operator=(const HttpFixture&) = delete;

    std::string url(const std::string& path) const {
        return "http://127.0.0.1:" + std::to_string(port) + path;
    }

    /// Raw header lines for an authenticated request.
    std::vector<std::string> auth() const {
        return {ragger::util::bearer_header(token),
                "Content-Type: application/json"};
    }

    TestEnv env;
    std::string token;
    int port = 0;
    bool ready = false;
    std::atomic<bool> run_result{false};
    ragger::util::HttpClient client;
    std::unique_ptr<ragger::RaggerMemory> mem;
    std::unique_ptr<ragger::Server> server;
    std::jthread runner;
};

} // namespace ragger_test
