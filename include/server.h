/**
 * HTTP server for Ragger Memory
 *
 * Serves the REST API (httplib) for the daemon: agent routes, dashboard
 * config, and the SSE activity stream.
 */
#pragma once

#include <memory>
#include <string>

namespace ragger {

class RaggerMemory;

class Server {
public:
    explicit Server(RaggerMemory& memory,
                    const std::string& host = "127.0.0.1",
                    int port = 8432);
    ~Server();

    /// Start listening (blocks). S18 (C M6): true = ran and stopped
    /// normally; false = could not start (port busy) — the caller exits
    /// non-zero through normal cleanup instead of std::exit inside run().
    /// Whether a restart was requested is reported by restart_requested().
    bool run();

    /// Request stop (from signal handler or another thread).
    void stop();

    /// True if the last run() ended because a restart was requested.
    static bool restart_requested();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
