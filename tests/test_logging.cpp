/**
 * Logging tests
 *
 * Tests that log files are created and written to when enabled.
 * Uses a custom INI with log_dir pointing to a temp directory.
 * Note: setup_logging() can only be called once per process (static state),
 * so we test all logging behavior in a single pass.
 */
#include "ragger/config.h"
#include "ragger/logs.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <print>
#include <sstream>

namespace fs = std::filesystem;

static const std::string TEMP_DIR = "/tmp/ragger_test_logs";

static void cleanup() {
    fs::remove_all(TEMP_DIR);
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;  // Cannot use std::format with file streams
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    std::println("Running logging tests:");
    cleanup();

    // Create INI with all logging enabled, pointing to temp dir
    std::string ini_path = "/tmp/ragger_test_logging.ini";
    {
        std::ofstream f(ini_path);
        f << "[server]\nport = 8432\nsingle_user = true\n"
          << "[logging]\n"
          << "log_dir = " << TEMP_DIR << "\n"
          << "query_log = true\n"
          << "http_log = true\n"
          << "mcp_log = true\n";
    }
    ragger::init_config(ini_path, true);

    // Initialize logging (can only call once)
    ragger::setup_logging(false, true);

    // 1. Log files should exist
    std::cout << "  test_log_files_created..." << std::flush;
    assert(fs::exists(TEMP_DIR + "/error.log"));
    assert(fs::exists(TEMP_DIR + "/query.log"));
    assert(fs::exists(TEMP_DIR + "/http.log"));
    assert(fs::exists(TEMP_DIR + "/mcp.log"));
    std::cout << " OK\n";

    // 2. Query log writes
    std::cout << "  test_query_log_writes..." << std::flush;
    ragger::log_query("search query: test embedding lookup");
    auto content = read_file(TEMP_DIR + "/query.log");
    assert(content.find("test embedding lookup") != std::string::npos);
    assert(content.find("[INFO]") != std::string::npos);
    std::cout << " OK\n";

    // 3. HTTP log writes
    std::cout << "  test_http_log_writes..." << std::flush;
    ragger::log_http("POST /store 200");
    content = read_file(TEMP_DIR + "/http.log");
    assert(content.find("POST /store 200") != std::string::npos);
    std::cout << " OK\n";

    // 4. MCP log writes
    std::cout << "  test_mcp_log_writes..." << std::flush;
    ragger::log_mcp("tools/list request");
    content = read_file(TEMP_DIR + "/mcp.log");
    assert(content.find("tools/list request") != std::string::npos);
    std::cout << " OK\n";

    // 5. Error log writes
    std::cout << "  test_error_log_writes..." << std::flush;
    ragger::log_error("test error message");
    content = read_file(TEMP_DIR + "/error.log");
    assert(content.find("test error message") != std::string::npos);
    assert(content.find("[ERROR]") != std::string::npos);
    std::cout << " OK\n";

    // 6. Timestamp format (YYYY-MM-DD HH:MM:SS.mmm)
    std::cout << "  test_timestamp_format..." << std::flush;
    content = read_file(TEMP_DIR + "/query.log");
    // Should start with a date like "2026-"
    assert(content.find("202") != std::string::npos);  // year prefix
    // Should have milliseconds (period followed by 3 digits)
    auto dot_pos = content.find('.');
    assert(dot_pos != std::string::npos);
    // 3 digits after the dot, then space
    assert(content[dot_pos + 4] == ' ');
    std::cout << " OK\n";

    // 7. Multiple log entries accumulate
    std::cout << "  test_log_accumulation..." << std::flush;
    ragger::log_query("query one");
    ragger::log_query("query two");
    ragger::log_query("query three");
    content = read_file(TEMP_DIR + "/query.log");
    assert(content.find("query one") != std::string::npos);
    assert(content.find("query two") != std::string::npos);
    assert(content.find("query three") != std::string::npos);
    std::cout << " OK\n";

    cleanup();
    fs::remove(ini_path);

    std::println("test_logging: all passed");
    return 0;
}
