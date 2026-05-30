//
// Created by Reid Woodbury Jr on 4/28/26.
//

#include <algorithm>
#include <chrono>
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

std::mutex logger::logMutex;
std::ofstream logger::logFile;
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

logger::logger(const std::string &logFileName, const std::string &level) {
    instanceCount++;
    if (instanceCount > 1) return; // Don't do anything if one already exists.

    logFile.open(logFileName, std::ios::app);

    if (logFile.fail()) {
        throw std::runtime_error(std::format(ragger::lang::ERR_LOG_OPEN, logFileName));
    }

    trace = [](std::string const &) {};
    debug = [](std::string const &) {};
    info = [](std::string const &) {};
    warn = [](std::string const &) {};
    error = [](std::string const &) {};
    critical = [](std::string const &) {};

    std::string lower_str = level;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "trace") {} /* goto trace; */
    else if (lower_str == "debug") { goto debug; }
    else if (lower_str == "info") { goto info; }
    else if (lower_str == "warn") { goto warn; }
    else if (lower_str == "error") { goto error; }
    else if (lower_str == "critical") { goto critical; }
    else { goto warn; }

    // trace:
    trace = [](std::string const &message) {
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [TRACE] " << message << std::endl;
    };

debug:
    debug = [](std::string const &message) {
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [DEBUG] " << message << std::endl;
    };

info:
    info = [](std::string const &message) {
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [INFO ] " << message << std::endl;
    };

warn:
    warn = [](std::string const &message) {
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [WARN ] " << message << std::endl;
    };

error:
    error = [](std::string const &message) {
        std::cerr << message << std::endl;
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [ERROR] " << message << std::endl;
    };

critical:
    critical = [](std::string const &message) {
        std::cerr << message << std::endl;
        const std::lock_guard<std::mutex> lock(logMutex);
        logger::get_file() << logger::get_timestamp() << " [CRITICAL] " << message << std::endl;
    };

}

logger::~logger() {
    instanceCount--;
    if (instanceCount == 0) {
        try {
            const std::lock_guard<std::mutex> lock(logMutex);
            if (logFile.is_open()) {
                logFile.flush();
                logFile.close();
            }
        } catch (...) {
            // Ignore exceptions during cleanup
        }
    }
}

std::ofstream &logger::get_file() { return logFile; }

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
