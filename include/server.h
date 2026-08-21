/**
 * HTTP server for Ragger Memory
 *
 * Mirrors ragger_memory/server.py from the Python version.
 * Uses httplib for routing and built-in JSON handling.
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

    /// Start listening (blocks). Returns true if a restart was requested
    /// (POST /restart) — the caller (main) should re-exec the daemon in place.
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
