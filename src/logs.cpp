/**
 * Logging for Ragger Memory (C++ port)
 *
 * Four log files (all optional except error.log):
 * - query.log   — search queries, scores, timing
 * - http.log    — HTTP requests/responses
 * - mcp.log     — MCP JSON-RPC interactions
 * - error.log   — errors from all components (always on)
 */

#include "ragger/logs.h"
#include "ragger/config.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

namespace ragger {

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------

struct LoggerState {
    std::unique_ptr<std::ofstream> query_log;
    std::unique_ptr<std::ofstream> http_log;
    std::unique_ptr<std::ofstream> mcp_log;
    std::unique_ptr<std::ofstream> error_log;

    std::mutex query_mutex;
    std::mutex http_mutex;
    std::mutex mcp_mutex;
    std::mutex error_mutex;

    bool initialized = false;
};

static LoggerState g_logger;

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;

    std::tm tm{};
    localtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

static void write_log_line(std::ofstream* file, std::mutex& mtx,
                          const std::string& level,
                          const std::string& message) {
    if (!file || !file->is_open()) return;

    std::lock_guard<std::mutex> lock(mtx);
    (*file) << get_timestamp() << " [" << level << "] " << message << "\n";
    file->flush();
}

static std::unique_ptr<std::ofstream> open_log_file(const std::string& dir,
                                                    const std::string& filename) {
    fs::create_directories(dir);
    auto path = fs::path(dir) / filename;
    auto file = std::make_unique<std::ofstream>(path, std::ios::app);
    if (!file->is_open()) {
        throw std::runtime_error("Failed to open log file: " + path.string());
    }
    return file;
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

void setup_logging(bool verbose, bool server_mode) {
    if (g_logger.initialized) {
        return;  // Already initialized
    }

    const auto& cfg = config();

    // Determine log directory
    std::string log_dir;
    if (server_mode) {
        log_dir = cfg.resolved_log_dir();
    } else {
        log_dir = expand_path("~/.ragger");
    }

    // Always open error.log
    g_logger.error_log = open_log_file(log_dir, "error.log");

    // Conditionally open other logs
    if (cfg.query_log_enabled) {
        g_logger.query_log = open_log_file(log_dir, "query.log");
    }

    if (cfg.http_log_enabled) {
        g_logger.http_log = open_log_file(log_dir, "http.log");
    }

    if (cfg.mcp_log_enabled) {
        g_logger.mcp_log = open_log_file(log_dir, "mcp.log");
    }

    g_logger.initialized = true;
}

void log_query(const std::string& message) {
    if (g_logger.query_log) {
        write_log_line(g_logger.query_log.get(), g_logger.query_mutex,
                      "INFO", message);
    }
}

void log_http(const std::string& message) {
    if (g_logger.http_log) {
        write_log_line(g_logger.http_log.get(), g_logger.http_mutex,
                      "INFO", message);
    }
}

void log_mcp(const std::string& message) {
    if (g_logger.mcp_log) {
        write_log_line(g_logger.mcp_log.get(), g_logger.mcp_mutex,
                      "INFO", message);
    }
}

void log_error(const std::string& message) {
    // Write to error.log
    write_log_line(g_logger.error_log.get(), g_logger.error_mutex,
                  "ERROR", message);

    // Also write to stderr
    std::cerr << get_timestamp() << " [ERROR] " << message << "\n";
}

void log_info(const std::string& message) {
    // Write to stderr only
    std::cout << get_timestamp() << " [INFO] " << message << "\n";
}

} // namespace ragger
