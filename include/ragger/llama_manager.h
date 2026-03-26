/**
 * Llama.cpp subprocess manager
 *
 * Manages a llama-server child process for local inference.
 * Ragger starts/stops/monitors the subprocess; clients connect
 * directly to llama-server's OpenAI-compatible API.
 */
#pragma once

#include <string>
#include <functional>

namespace ragger {

struct LlamaStatus {
    bool running    = false;
    pid_t pid       = 0;
    std::string model;
    std::string host;
    int port        = 0;
    int exit_code   = -1;  // last exit code, -1 = never exited
};

class LlamaManager {
public:
    LlamaManager();
    ~LlamaManager();

    // Non-copyable
    LlamaManager(const LlamaManager&) = delete;
    LlamaManager& operator=(const LlamaManager&) = delete;

    /// Start llama-server subprocess using current config.
    /// Returns true if started successfully (health check passed).
    bool start();

    /// Stop the subprocess gracefully (SIGTERM, then SIGKILL after timeout).
    void stop();

    /// Restart (stop + start).
    bool restart();

    /// Check if the subprocess is running.
    bool is_running() const;

    /// Get current status.
    LlamaStatus status() const;

    /// Wait for the subprocess to exit. Blocks.
    int wait();

    /// Poll — check if child exited, reap zombie. Non-blocking.
    /// Returns true if child is still running.
    bool poll();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
