/**
 * Llama.cpp subprocess manager — implementation
 */

#include "ragger/llama_manager.h"
#include "ragger/config.h"
#include "ragger/logs.h"
#include "ragger/lang/en.h"

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <curl/curl.h>

namespace ragger {

struct LlamaManager::Impl {
    pid_t child_pid = 0;
    int last_exit_code = -1;
    bool started = false;

    /// Build argv from config.
    std::vector<std::string> build_args() const {
        const auto& cfg = config();
        std::vector<std::string> args;

        args.push_back(cfg.llama_binary);
        args.push_back("--host");     args.push_back(cfg.llama_host);
        args.push_back("--port");     args.push_back(std::to_string(cfg.llama_port));

        if (!cfg.llama_model.empty()) {
            args.push_back("--model");
            args.push_back(expand_path(cfg.llama_model));
        }

        if (cfg.llama_ctx_size > 0) {
            args.push_back("--ctx-size");
            args.push_back(std::to_string(cfg.llama_ctx_size));
        }

        args.push_back("--n-gpu-layers");
        args.push_back(std::to_string(cfg.llama_gpu_layers));

        if (cfg.llama_threads > 0) {
            args.push_back("--threads");
            args.push_back(std::to_string(cfg.llama_threads));
        }

        args.push_back("--batch-size");
        args.push_back(std::to_string(cfg.llama_batch_size));

        if (cfg.llama_parallel > 1) {
            args.push_back("--parallel");
            args.push_back(std::to_string(cfg.llama_parallel));
        }

        if (cfg.llama_flash_attn) {
            args.push_back("--flash-attn");
            args.push_back("on");
        }

        if (cfg.llama_mlock) {
            args.push_back("--mlock");
        }

        if (!cfg.llama_mmap) {
            args.push_back("--no-mmap");
        }

        args.push_back("--cache-type-k");
        args.push_back(cfg.llama_cache_type_k);
        args.push_back("--cache-type-v");
        args.push_back(cfg.llama_cache_type_v);

        // Parse extra args (space-separated pass-through)
        if (!cfg.llama_extra_args.empty()) {
            std::istringstream iss(cfg.llama_extra_args);
            std::string token;
            while (iss >> token) {
                args.push_back(token);
            }
        }

        return args;
    }

    /// Convert string vector to C-style argv.
    static std::vector<char*> to_argv(std::vector<std::string>& args) {
        std::vector<char*> argv;
        for (auto& a : args) {
            argv.push_back(a.data());
        }
        argv.push_back(nullptr);
        return argv;
    }

    /// Health check — try GET /health on llama-server.
    bool health_check() const {
        const auto& cfg = config();
        std::string url = "http://" + cfg.llama_host + ":"
                        + std::to_string(cfg.llama_port) + "/health";

        CURL* curl = curl_easy_init();
        if (!curl) return false;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 2L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD-like
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        // Suppress output
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
            +[](char*, size_t s, size_t n, void*) -> size_t { return s * n; });

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        }
        curl_easy_cleanup(curl);

        return (res == CURLE_OK && http_code == 200);
    }

    /// Wait for llama-server to become healthy.
    bool wait_for_healthy(int timeout_secs = 120) const {
        auto start = std::chrono::steady_clock::now();
        while (true) {
            if (health_check()) return true;

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_secs) return false;

            // Check if child died
            if (child_pid > 0) {
                int status;
                pid_t result = waitpid(child_pid, &status, WNOHANG);
                if (result == child_pid) return false;  // child exited
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};

LlamaManager::LlamaManager() : pImpl(std::make_unique<Impl>()) {}
LlamaManager::~LlamaManager() { stop(); }

bool LlamaManager::start() {
    const auto& cfg = config();

    if (pImpl->child_pid > 0 && is_running()) {
        log_info("llama-server already running (pid " + std::to_string(pImpl->child_pid) + ")");
        return true;
    }

    if (cfg.llama_model.empty()) {
        std::cerr << "Error: llama.model not set in config" << std::endl;
        return false;
    }

    // Check if port is already in use
    if (pImpl->health_check()) {
        std::cerr << "Error: port " << cfg.llama_port
                  << " already in use (llama-server may already be running)" << std::endl;
        return false;
    }

    auto args = pImpl->build_args();
    auto argv = Impl::to_argv(args);

    std::cout << "Starting llama-server on " << cfg.llama_host
              << ":" << cfg.llama_port << std::endl;
    std::cout << "  Model: " << expand_path(cfg.llama_model) << std::endl;

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Error: fork() failed: " << strerror(errno) << std::endl;
        return false;
    }

    if (pid == 0) {
        // Child — redirect stdout/stderr to log
        std::string log_path = cfg.resolved_log_dir() + "/llama-server.log";
        int fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }

        // Create new session so it doesn't get terminal signals
        setsid();

        execvp(argv[0], argv.data());
        // If we get here, exec failed
        _exit(127);
    }

    // Parent
    pImpl->child_pid = pid;
    pImpl->started = true;

    log_info("llama-server started (pid " + std::to_string(pid) + "), waiting for health...");

    if (!pImpl->wait_for_healthy()) {
        std::cerr << "Error: llama-server failed to start within timeout" << std::endl;
        stop();
        return false;
    }

    std::cout << "llama-server ready on port " << cfg.llama_port << std::endl;
    log_info("llama-server ready on port " + std::to_string(cfg.llama_port));
    return true;
}

void LlamaManager::stop() {
    if (pImpl->child_pid <= 0) return;

    log_info("Stopping llama-server (pid " + std::to_string(pImpl->child_pid) + ")");

    // Try SIGTERM first
    kill(pImpl->child_pid, SIGTERM);

    // Wait up to 10 seconds
    for (int i = 0; i < 100; ++i) {
        int status;
        pid_t result = waitpid(pImpl->child_pid, &status, WNOHANG);
        if (result == pImpl->child_pid) {
            pImpl->last_exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            pImpl->child_pid = 0;
            pImpl->started = false;
            log_info("llama-server stopped");
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Force kill
    log_info("llama-server didn't stop, sending SIGKILL");
    kill(pImpl->child_pid, SIGKILL);
    int status;
    waitpid(pImpl->child_pid, &status, 0);
    pImpl->last_exit_code = -1;
    pImpl->child_pid = 0;
    pImpl->started = false;
}

bool LlamaManager::restart() {
    stop();
    return start();
}

bool LlamaManager::is_running() const {
    if (pImpl->child_pid <= 0) return false;
    int status;
    pid_t result = waitpid(pImpl->child_pid, &status, WNOHANG);
    if (result == pImpl->child_pid) {
        pImpl->last_exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        pImpl->child_pid = 0;
        return false;
    }
    return true;
}

LlamaStatus LlamaManager::status() const {
    const auto& cfg = config();
    LlamaStatus s;
    s.running = is_running();
    s.pid = pImpl->child_pid;
    s.model = cfg.llama_model;
    s.host = cfg.llama_host;
    s.port = cfg.llama_port;
    s.exit_code = pImpl->last_exit_code;
    return s;
}

bool LlamaManager::poll() {
    return is_running();
}

int LlamaManager::wait() {
    if (pImpl->child_pid <= 0) return pImpl->last_exit_code;
    int status;
    waitpid(pImpl->child_pid, &status, 0);
    pImpl->last_exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    pImpl->child_pid = 0;
    return pImpl->last_exit_code;
}

} // namespace ragger
