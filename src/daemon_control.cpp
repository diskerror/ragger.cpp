/**
 * Daemon / service control implementation. See include/ragger/daemon_control.h.
 * Moved verbatim out of main.cpp (the launchctl/systemctl service plumbing).
 */
#include "daemon_control.h"
#include "lang.h"
#include "Logger.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pwd.h>
#include <unistd.h>
#include <sys/wait.h>

namespace fs = std::filesystem;

namespace ragger {

// Wrappers around launchctl (macOS) or systemctl --user (Linux) that
// operate on the user-level service installed by install.sh.
// The daemon process itself is still `ragger serve` (what launchd/systemd run).
//
// Design: idempotent + friendly.
//   start   → "already running" if it is, else launch it
//   stop    → "not running" if it isn't, else stop it
//   restart → cycle it whether up or down
//   status  → one-line summary (running pid / stopped / not loaded)
// -----------------------------------------------------------------------
namespace {

/// Run a shell command, return captured stdout. Silent on failure (returns "").
std::string capture_output(const std::string &cmd) {
    FILE *pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!pipe) return "";
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}

/// Run a shell command, discard output, return true iff exit code 0.
bool run_quiet(const std::string &cmd) {
    int rc = std::system((cmd + " >/dev/null 2>&1").c_str());
    return WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
}

/// Kill every `ragger` process owned by the current user except this one.
/// Used by `stop` and `restart` to sweep MCP servers and stray CLI
/// invocations — MCP is typically launched by a client (OpenClaw, Claude
/// Desktop) and has no service manager of its own, so we have to reap it here.
/// Sends SIGTERM, waits briefly, then SIGKILL to any survivors.
/// Returns the number of processes signaled.
int kill_other_ragger_instances() {
    pid_t self = getpid();
    auto out = capture_output(
        "pgrep -u " + std::to_string(getuid()) + " -x ragger");
    std::vector<pid_t> pids;
    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        try {
            pid_t pid = std::stoi(line);
            if (pid != self && pid > 1) pids.push_back(pid);
        }
        catch (...) {
            /* skip malformed */
        }
    }
    if (pids.empty()) return 0;

    for (pid_t p: pids) ::kill(p, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    for (pid_t p: pids) {
        if (::kill(p, 0) == 0) ::kill(p, SIGKILL); // still alive → force
    }
    return static_cast<int>(pids.size());
}

} // anonymous namespace

int daemon_control(const std::string &action) {
    struct passwd *pw = getpwuid(getuid());
    std::string home = pw ? pw->pw_dir : (std::getenv("HOME") ? std::getenv("HOME") : "");
    if (home.empty()) {
        Diskerror::Logger::error(ragger::lang::ERR_HOME_NOT_FOUND);
        return 1;
    }

#if defined(__APPLE__)
    const std::string label = "com.diskerror.ragger";
    const std::string plist = home + "/Library/LaunchAgents/" + label + ".plist";
    const std::string target = "gui/" + std::to_string(getuid()) + "/" + label;
    const std::string domain = "gui/" + std::to_string(getuid());

    // Loaded = bootstrapped into the gui domain. Running = has a live pid.
    auto is_loaded = [&]() { return run_quiet("launchctl print " + target); };
    auto get_pid = [&]() -> std::string {
        if (!is_loaded()) return "";
        auto out = capture_output("launchctl print " + target);
        auto pos = out.find("\n\tpid = ");
        if (pos == std::string::npos) pos = out.find("\n    pid = ");
        if (pos == std::string::npos) return "";
        auto s = out.find('=', pos) + 1;
        while (s < out.size() && out[s] == ' ') ++s;
        auto e = out.find('\n', s);
        return out.substr(s, e - s);
    };
    auto is_running = [&]() { return !get_pid().empty(); };

    auto ensure_plist = [&]() -> bool {
        if (fs::exists(plist)) return true;
        Diskerror::Logger::error(std::format(ragger::lang::ERR_PLIST_NOT_FOUND, plist));
        return false;
    };

    if (action == "start") {
        if (!ensure_plist()) return 1;
        if (is_running()) {
            std::cout << std::format(ragger::lang::MSG_ALREADY_RUNNING, get_pid()) << "\n";
            return 0;
        }
        if (is_loaded()) {
            if (!run_quiet("launchctl kickstart " + target)) {
                Diskerror::Logger::error(ragger::lang::ERR_LAUNCHCTL_KICKSTART);
                return 1;
            }
        }
        else {
            if (!run_quiet("launchctl bootstrap " + domain + " " + plist)) {
                Diskerror::Logger::error(ragger::lang::ERR_LAUNCHCTL_BOOTSTRAP);
                return 1;
            }
        }
        auto pid = get_pid();
        if (!pid.empty())
            std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        else
            std::cout << ragger::lang::MSG_STARTED << "\n";
        return 0;
    }

    if (action == "stop") {
        bool daemon_was_running = is_loaded() && is_running();
        if (daemon_was_running) {
            if (!run_quiet("launchctl bootout " + target)) {
                Diskerror::Logger::error(ragger::lang::ERR_LAUNCHCTL_BOOTOUT);
                return 1;
            }
        }
        // Sweep any other ragger processes this user owns (MCP servers
        // launched by clients, stray CLI runs, etc.).
        int extras = kill_other_ragger_instances();

        if (daemon_was_running && extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_N, extras) << "\n";
        }
        else if (daemon_was_running) {
            std::cout << ragger::lang::MSG_STOPPED << "\n";
        }
        else if (extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_N, extras) << "\n";
        }
        else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    if (action == "restart") {
        if (!ensure_plist()) return 1;
        bool was_running = is_running();
        if (is_loaded()) {
            run_quiet("launchctl bootout " + target);
            // bootout returns before launchd completes teardown — wait for
            // the domain to actually release the label (up to 3s).
            for (int i = 0; i < 30 && is_loaded(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        // Sweep MCP + stray instances before relaunching the daemon.
        // (Clients will need to reconnect their MCP subprocess themselves.)
        kill_other_ragger_instances();

        if (!run_quiet("launchctl bootstrap " + domain + " " + plist)) {
            Diskerror::Logger::critical(ragger::lang::ERR_LAUNCHCTL_BOOTSTRAP);
            return 1;
        }
        auto pid = get_pid();
        if (!pid.empty()) {
            if (was_running)
                std::cout << std::format(ragger::lang::MSG_RESTARTED_PID, pid) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        }
        else {
            std::cout << (was_running
                              ? ragger::lang::MSG_RESTARTED
                              : ragger::lang::MSG_STARTED) << "\n";
        }
        return 0;
    }

    if (action == "status") {
        if (!is_loaded()) {
            std::cout << ragger::lang::MSG_NOT_LOADED << "\n";
            return 0;
        }
        auto pid = get_pid();
        if (pid.empty()) {
            std::cout << ragger::lang::MSG_LOADED_NOT_RUNNING << "\n";
            return 0;
        }
        std::cout << std::format(ragger::lang::MSG_RUNNING_PID, pid) << "\n";
        return 0;
    }

    Diskerror::Logger::critical(std::format(ragger::lang::ERR_UNKNOWN_ACTION, action));
    return 1;

#elif defined(__linux__)
    const std::string unit = "ragger.service";

    auto is_active = [&]() {
        return run_quiet("systemctl --user is-active --quiet " + unit);
    };
    auto main_pid = [&]() -> std::string {
        auto out = capture_output(
            "systemctl --user show -p MainPID --value " + unit);
        // trim trailing newline/whitespace
        while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
            out.pop_back();
        if (out == "0" || out.empty()) return "";
        return out;
    };

    if (action == "start") {
        if (is_active()) {
            auto pid = main_pid();
            if (!pid.empty())
                std::cout << std::format(ragger::lang::MSG_ALREADY_RUNNING, pid) << "\n";
            else
                std::cout << ragger::lang::MSG_IS_RUNNING << "\n";
            return 0;
        }
        if (!run_quiet("systemctl --user start " + unit)) {
            Diskerror::Logger::critical(ragger::lang::ERR_SYSTEMCTL_START);
            return 1;
        }
        auto pid = main_pid();
        if (!pid.empty())
            std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        else
            std::cout << ragger::lang::MSG_STARTED << "\n";
        return 0;
    }

    if (action == "stop") {
        bool daemon_was_running = is_active();
        if (daemon_was_running) {
            if (!run_quiet("systemctl --user stop " + unit)) {
                Diskerror::Logger::critical(ragger::lang::ERR_SYSTEMCTL_STOP);
                return 1;
            }
        }
        int extras = kill_other_ragger_instances();

        if (daemon_was_running && extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_N, extras) << "\n";
        }
        else if (daemon_was_running) {
            std::cout << ragger::lang::MSG_STOPPED << "\n";
        }
        else if (extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_N, extras) << "\n";
        }
        else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    if (action == "restart") {
        bool was_active = is_active();
        // Stop daemon and sweep MCP + strays before relaunching.
        if (was_active) run_quiet("systemctl --user stop " + unit);
        kill_other_ragger_instances();
        if (!run_quiet("systemctl --user start " + unit)) {
            Diskerror::Logger::critical(ragger::lang::ERR_SYSTEMCTL_START);
            return 1;
        }
        auto pid = main_pid();
        if (!pid.empty()) {
            if (was_active)
                std::cout << std::format(ragger::lang::MSG_RESTARTED_PID, pid) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        }
        else {
            std::cout << (was_active
                              ? ragger::lang::MSG_RESTARTED
                              : ragger::lang::MSG_STARTED) << "\n";
        }
        return 0;
    }

    if (action == "status") {
        if (is_active()) {
            auto pid = main_pid();
            if (!pid.empty())
                std::cout << std::format(ragger::lang::MSG_RUNNING_PID, pid) << "\n";
            else
                std::cout << ragger::lang::MSG_IS_RUNNING << "\n";
        }
        else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    Diskerror::Logger::critical(std::format(ragger::lang::ERR_UNKNOWN_ACTION, action));
    return 1;
#else
    Diskerror::Logger::error(ragger::lang::ERR_DAEMON_UNSUPPORTED);
    return 1;
#endif
}

} // namespace ragger
