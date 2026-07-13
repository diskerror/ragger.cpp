//
// Created by Reid Woodbury Jr on 4/28/26.
//

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "diskerror/logger.h"
#include "ragger/lang.h"

namespace Diskerror {

namespace fs = std::filesystem;

std::mutex logger::logMutex;
std::string logger::logFileName;
long logger::maxSizeBytes = 0;
int  logger::maxAgeDays   = 0;
unsigned int logger::instanceCount = 0;

// Default the sinks to no-ops so logging is always safe, even before any
// logger instance is constructed (e.g. library consumers and tests that
// build a Server without first standing up the logger). The constructor
// reassigns these to real sinks; until then a log call is a silent no-op
// rather than a bad_function_call crash.
namespace { const auto kNoop = [](std::string const&) {}; }
std::function<void(std::string const&)> logger::trace    = kNoop;
std::function<void(std::string const&)> logger::debug    = kNoop;
std::function<void(std::string const&)> logger::info     = kNoop;
std::function<void(std::string const&)> logger::warn     = kNoop;
std::function<void(std::string const&)> logger::error    = kNoop;
std::function<void(std::string const&)> logger::critical = kNoop;

logger::logger(const std::string &logFileName_, const std::string &level,
               long maxSizeMb, int maxAgeDays_) {
    instanceCount++;
    if (instanceCount > 1) return; // Don't do anything if one already exists.

    logFileName  = logFileName_;
    maxSizeBytes = maxSizeMb  > 0 ? maxSizeMb * 1024L * 1024L : 0;
    maxAgeDays   = maxAgeDays_ > 0 ? maxAgeDays_ : 0;

    // Create the containing directory (e.g. ~/.ragger/logs) if it doesn't
    // exist yet, then create the log file itself if missing. Fail fast if
    // the path is genuinely unwritable (permission denied, etc.) rather
    // than silently dropping every log line for the rest of the process.
    std::error_code ec;
    fs::path path(logFileName);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path(), ec);
    }
    {
        std::ofstream probe(logFileName, std::ios::app);
        if (probe.fail()) {
            throw std::runtime_error(std::format(ragger::lang::ERR_LOG_OPEN, logFileName));
        }
    }

    trace = [](std::string const &) {};
    debug = [](std::string const &) {};
    info = [](std::string const &) {};
    warn = [](std::string const &) {};
    error = [](std::string const &) {};
    critical = [](std::string const &) {};

    std::string lower_str = level;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "trace") { goto trace; }
    else if (lower_str == "debug") { goto debug; }
    else if (lower_str == "info") { goto info; }
    else if (lower_str == "warn") { goto warn; }
    else if (lower_str == "error") { goto error; }
    else if (lower_str == "critical") { goto critical; }
    else { goto warn; }

trace:
    trace = [](std::string const &message) { append_line("TRACE", message); };

debug:
    debug = [](std::string const &message) { append_line("DEBUG", message); };

info:
    info = [](std::string const &message) { append_line("INFO ", message); };

warn:
    warn = [](std::string const &message) { append_line("WARN ", message); };

error:
    error = [](std::string const &message) {
        std::cerr << message << std::endl;
        append_line("ERROR", message);
    };

critical:
    critical = [](std::string const &message) {
        std::cerr << message << std::endl;
        append_line("CRITICAL", message);
    };

}

logger::~logger() {
    instanceCount--;
}

// Opens the log file, writes one line, closes it again — rather than
// holding a persistent handle for the process lifetime. Then checks
// whether the freshly-written file has crossed the size threshold and
// rotates (+ sweeps aged-out backups) if so. All under one lock so
// rotation/append/sweep never interleave across log calls.
void logger::append_line(const std::string& tag, const std::string& message) {
    const std::lock_guard<std::mutex> lock(logMutex);

    {
        std::ofstream f(logFileName, std::ios::app);
        if (f) {
            f << get_timestamp() << " [" << tag << "] " << message << std::endl;
        }
        // If open fails (e.g. disk full, directory removed externally),
        // silently drop the line rather than throwing out of a log call —
        // logging must never be allowed to crash the caller.
    }

    if (maxSizeBytes > 0) {
        std::error_code ec;
        auto size = fs::file_size(logFileName, ec);
        if (!ec && static_cast<long>(size) >= maxSizeBytes) {
            rotate_locked();
        }
    }
}

// Renames the live log to a timestamped backup in the same directory
// (same-filesystem rename, so this is atomic and cheap — no copy). The
// next append_line() reopens logFileName with ios::app, which recreates
// it fresh since the rename just emptied that path. On any rename failure
// (e.g. permission race), logging just continues to grow the current file
// rather than losing data.
void logger::rotate_locked() {
    fs::path path(logFileName);

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time_t, &tm);
    std::ostringstream suffix;
    suffix << std::put_time(&tm, "%Y%m%d-%H%M%S");

    fs::path backup = path;
    backup += "." + suffix.str();

    // Same-second rotation collision (extremely unlikely for a 1MB+ log,
    // but cheap to guard): append a numeric disambiguator.
    int n = 1;
    while (fs::exists(backup)) {
        fs::path candidate = path;
        candidate += "." + suffix.str() + "-" + std::to_string(n++);
        backup = candidate;
    }

    std::error_code ec;
    fs::rename(path, backup, ec);
    // ec ignored deliberately: if rotation fails, the live log just keeps
    // growing past the threshold rather than losing any log data.

    sweep_backups_locked();
}

// Deletes rotated backups (logFileName.<timestamp>[-n]) older than
// maxAgeDays. Never touches the live log file itself. Only called right
// after a rotation (i.e. at most once per log_max_size_mb worth of log
// growth) — a rotation is the only event that can produce a new aged-out
// backup, so there's no need for a separate timer/throttle.
void logger::sweep_backups_locked() {
    if (maxAgeDays <= 0) return;

    fs::path path(logFileName);
    fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
    std::string prefix = path.filename().string() + ".";

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;

    auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(24 * maxAgeDays);

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;  // not one of our backups

        std::error_code mtime_ec;
        auto ftime = fs::last_write_time(entry.path(), mtime_ec);
        if (mtime_ec) continue;
        // C++20 file_clock -> system_clock conversion (portable across
        // libstdc++/libc++, unlike hand-rolling the offset via now()/now()).
        auto sctp = std::chrono::file_clock::to_sys(ftime);
        if (sctp < cutoff) {
            fs::remove(entry.path(), ec);
        }
    }
}

std::string logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 10000;

    std::tm tm{};
    localtime_r(&time_t, &tm);

    // Using ostringstream for timestamp formatting as it's part of core log-line assembly
    // and requires complex date/time formatting that would be cumbersome with std::format
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S.") << std::setfill('0') << std::setw(4) << ms.count();
    return oss.str();
}

} // namespace Diskerror
