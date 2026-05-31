/**
 * EmbedExecutor implementation — spawn `ragger embed` per call with a timeout.
 */
#include "ragger/embed_executor.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include "diskerror/logger.h"

#include "nlohmann_json.hpp"

#include <spawn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <format>
#include <mutex>
#include <thread>

extern char** environ;

namespace ragger {

EmbedExecutor::EmbedExecutor()
    : EmbedExecutor(config().embed_timeout_ms, config().embed_retries,
                    config().embed_max_workers, executable_path()) {}

EmbedExecutor::EmbedExecutor(int timeout_ms, int retries, int max_workers,
                             std::string exe_path)
    : timeout_ms_(timeout_ms > 0 ? timeout_ms : 5000),
      retries_(retries < 0 ? 0 : retries),
      max_workers_(max_workers < 1 ? 1 : max_workers),
      exe_path_(std::move(exe_path)) {
    // A child that dies mid-write would otherwise SIGPIPE the parent. Ignore
    // it once; the write loop already handles short writes / errors.
    static std::once_flag once;
    std::call_once(once, []() { std::signal(SIGPIPE, SIG_IGN); });
}

namespace {

// One spawn attempt. Returns the child's stdout on success, or nullopt on
// spawn failure / timeout / non-zero exit. Kills the child on timeout.
std::optional<std::string> run_once(const std::string& exe,
                                    const std::string& text, int timeout_ms) {
    int in_pipe[2]  = {-1, -1};   // parent writes child stdin
    int out_pipe[2] = {-1, -1};   // parent reads child stdout
    if (pipe(in_pipe) != 0) return std::nullopt;
    if (pipe(out_pipe) != 0) { close(in_pipe[0]); close(in_pipe[1]); return std::nullopt; }

    // Mark every pipe fd close-on-exec. Critical for batch(): without this, a
    // concurrently-spawning worker's child inherits THIS pipe's write end, so
    // the child here never sees EOF on stdin and hangs until timeout. The
    // dup2 file-actions below give the child fresh, non-CLOEXEC stdio.
    // (macOS lacks pipe2(O_CLOEXEC), so set it explicitly.)
    for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]})
        fcntl(fd, F_SETFD, FD_CLOEXEC);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in_pipe[0],  STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, out_pipe[0]);
    posix_spawn_file_actions_addclose(&fa, in_pipe[0]);
    posix_spawn_file_actions_addclose(&fa, out_pipe[1]);

    const char* verb = "embed";
    char* argv[] = { const_cast<char*>(exe.c_str()),
                     const_cast<char*>(verb), nullptr };

    pid_t pid = -1;
    int rc = posix_spawnp(&pid, exe.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(in_pipe[0]);
    close(out_pipe[1]);
    if (rc != 0) {
        close(in_pipe[1]);
        close(out_pipe[0]);
        return std::nullopt;
    }

    // Feed the text and signal EOF. Writes may block until the child drains
    // its stdin, which it does eagerly; SIGPIPE is ignored process-wide.
    {
        size_t off = 0;
        while (off < text.size()) {
            ssize_t w = write(in_pipe[1], text.data() + off, text.size() - off);
            if (w <= 0) break;       // child died / pipe error — read loop handles it
            off += static_cast<size_t>(w);
        }
        close(in_pipe[1]);
    }

    // Read stdout under a wall-clock deadline; kill the child if it overruns.
    std::string out;
    bool timed_out = false;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    char buf[8192];
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) { timed_out = true; break; }
        int wait_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        struct pollfd pfd { out_pipe[0], POLLIN, 0 };
        int pr = poll(&pfd, 1, wait_ms);
        if (pr == 0) { timed_out = true; break; }
        if (pr < 0) { if (errno == EINTR) continue; break; }
        ssize_t n = read(out_pipe[0], buf, sizeof(buf));
        if (n > 0) out.append(buf, static_cast<size_t>(n));
        else if (n == 0) break;       // EOF — child closed stdout
        else { if (errno == EINTR) continue; break; }
    }
    close(out_pipe[0]);

    if (timed_out) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        return std::nullopt;
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return std::nullopt;
    return out;
}

}  // namespace

std::optional<std::vector<float>>
EmbedExecutor::one(const std::string& text) const {
    if (text.empty()) return std::nullopt;
    for (int attempt = 0; attempt <= retries_; ++attempt) {
        auto out = run_once(exe_path_, text, timeout_ms_);
        if (!out) {
            Diskerror::logger::warn(std::format(lang::WARN_EMBED_SUBPROCESS,
                                                attempt + 1, retries_ + 1));
            continue;
        }
        auto j = nlohmann::json::parse(*out, nullptr, false);
        if (j.is_discarded() || !j.is_array()) continue;
        std::vector<float> v;
        v.reserve(j.size());
        for (const auto& e : j) {
            if (!e.is_number()) { v.clear(); break; }
            v.push_back(e.get<float>());
        }
        if (!v.empty()) return v;
    }
    return std::nullopt;
}

std::vector<std::optional<std::vector<float>>>
EmbedExecutor::batch(const std::vector<std::string>& texts) const {
    std::vector<std::optional<std::vector<float>>> results(texts.size());
    if (texts.empty()) return results;

    std::atomic<size_t> next{0};
    int workers = std::min<int>(max_workers_, static_cast<int>(texts.size()));
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&]() {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= texts.size()) return;
                results[i] = one(texts[i]);
            }
        });
    }
    for (auto& t : pool) t.join();
    return results;
}

} // namespace ragger
