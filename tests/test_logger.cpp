/**
 * Logger class tests
 *
 * Tests the new RAII logger class: construction, level filtering,
 * file output, timestamps, and destructor cleanup.
 * Uses a temp file that is removed after each test group.
 */
#include "diskerror/logger.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <sstream>

namespace fs = std::filesystem;

static const std::string LOG_FILE = "/tmp/ragger_test_logger.log";

static void cleanup() {
    fs::remove(LOG_FILE);
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    std::println("Running logger class tests:");

    // 1. Constructor creates the log file
    std::println("  test_file_created...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("startup message");
    }
    assert(fs::exists(LOG_FILE));
    auto content = read_file(LOG_FILE);
    assert(content.find("startup message") != std::string::npos);
    std::println(" OK");

    // 2. Destructor closes the file (second logger can reopen it)
    std::println("  test_destructor_closes_file...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("first run");
    }
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("second run");
    }
    content = read_file(LOG_FILE);
    assert(content.find("first run") != std::string::npos);
    assert(content.find("second run") != std::string::npos);
    std::println(" OK");

    // 3. Level filtering — "warn" should suppress info and debug
    std::println("  test_level_filtering_warn...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "warn");
        Diskerror::logger::debug("should not appear");
        Diskerror::logger::info("should not appear either");
        Diskerror::logger::warn("visible warning");
        Diskerror::logger::error("visible error");
    }
    content = read_file(LOG_FILE);
    assert(content.find("should not appear") == std::string::npos);
    assert(content.find("should not appear either") == std::string::npos);
    assert(content.find("visible warning") != std::string::npos);
    assert(content.find("visible error") != std::string::npos);
    std::println(" OK");

    // 4. Level filtering — "trace" enables everything
    std::println("  test_level_filtering_trace...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "trace");
        Diskerror::logger::trace("t msg");
        Diskerror::logger::debug("d msg");
        Diskerror::logger::info("i msg");
        Diskerror::logger::warn("w msg");
        Diskerror::logger::error("e msg");
        Diskerror::logger::critical("c msg");
    }
    content = read_file(LOG_FILE);
    assert(content.find("[TRACE]") != std::string::npos);
    assert(content.find("[DEBUG]") != std::string::npos);
    assert(content.find("[INFO ]") != std::string::npos);
    assert(content.find("[WARN ]") != std::string::npos);
    assert(content.find("[ERROR]") != std::string::npos);
    assert(content.find("[CRITICAL]") != std::string::npos);
    std::println(" OK");

    // 5. Level filtering — "critical" suppresses everything below
    std::println("  test_level_filtering_critical...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "critical");
        Diskerror::logger::trace("no");
        Diskerror::logger::debug("no");
        Diskerror::logger::info("no");
        Diskerror::logger::warn("no");
        Diskerror::logger::error("no");
        Diskerror::logger::critical("yes critical");
    }
    content = read_file(LOG_FILE);
    assert(content.find("yes critical") != std::string::npos);
    // Only one line should be present (the critical one)
    // Count newlines — should be exactly 1
    auto lines = std::count(content.begin(), content.end(), '\n');
    assert(lines == 1);
    std::println(" OK");

    // 6. Timestamp format (YYYY-MM-DD HH:MM:SS.mmmm)
    std::println("  test_timestamp_format...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("timestamp check");
    }
    content = read_file(LOG_FILE);
    assert(content.find("202") != std::string::npos);
    auto dot_pos = content.find('.');
    assert(dot_pos != std::string::npos);
    // 4 digits after the dot (microseconds mod 10000), then space
    assert(content[dot_pos + 5] == ' ');
    std::println(" OK");

    // 7. Log tags present in output
    std::println("  test_log_tags...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("tag test");
    }
    content = read_file(LOG_FILE);
    assert(content.find("[INFO ]") != std::string::npos);
    std::println(" OK");

    // 8. Invalid level defaults to warn
    std::println("  test_invalid_level_defaults_to_warn...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "nonsense");
        Diskerror::logger::info("should be hidden");
        Diskerror::logger::warn("should be visible");
    }
    content = read_file(LOG_FILE);
    assert(content.find("should be hidden") == std::string::npos);
    assert(content.find("should be visible") != std::string::npos);
    std::println(" OK");

    // 9. Multiple entries accumulate
    std::println("  test_accumulation...");
    cleanup();
    {
        Diskerror::logger log(LOG_FILE, "info");
        Diskerror::logger::info("line one");
        Diskerror::logger::info("line two");
        Diskerror::logger::info("line three");
    }
    content = read_file(LOG_FILE);
    assert(content.find("line one") != std::string::npos);
    assert(content.find("line two") != std::string::npos);
    assert(content.find("line three") != std::string::npos);
    std::println(" OK");

    // 10. Bad file path throws
    std::println("  test_bad_path_throws...");
    {
        bool threw = false;
        try {
            Diskerror::logger log("/no/such/directory/bad.log", "info");
        } catch (const std::runtime_error& e) {
            threw = true;
            assert(std::string(e.what()).find("failed to open") != std::string::npos);
        }
        assert(threw);
    }
    std::println(" OK");

    // 11. Constructor creates a missing containing directory (e.g. ~/.ragger/logs)
    std::println("  test_creates_missing_directory...");
    {
        fs::remove_all("/tmp/ragger_test_logdir");
        std::string nested = "/tmp/ragger_test_logdir/nested/activity.log";
        assert(!fs::exists("/tmp/ragger_test_logdir"));
        {
            Diskerror::logger log(nested, "info");
            Diskerror::logger::info("dir autocreate check");
        }
        assert(fs::exists(nested));
        content = read_file(nested);
        assert(content.find("dir autocreate check") != std::string::npos);
        fs::remove_all("/tmp/ragger_test_logdir");
    }
    std::println(" OK");

    // 12. Size-based rotation: once the file crosses max_size, it's renamed
    // to a timestamped backup and a fresh empty file continues at the same path.
    std::println("  test_size_based_rotation...");
    {
        std::string dir = "/tmp/ragger_test_rotate";
        std::string path = dir + "/activity.log";
        fs::remove_all(dir);
        fs::create_directories(dir);
        // Seed the file past the 1MB threshold so the logger's very first
        // append (which checks size after writing) triggers rotation.
        {
            std::ofstream seed(path);
            seed << std::string(1024 * 1024 + 10, 'x');  // > 1MB
        }
        {
            Diskerror::logger log(path, "info", /*maxSizeMb=*/1, /*maxAgeDays=*/0);
            Diskerror::logger::info("triggers rotation");
        }
        int backups = 0;
        for (const auto& e : fs::directory_iterator(dir)) {
            if (e.path().filename().string().rfind("activity.log.", 0) == 0) ++backups;
        }
        assert(backups == 1);
        content = read_file(path);
        assert(content.find("triggers rotation") != std::string::npos);
        assert(content.size() < 1024 * 1024);  // fresh file, not the seeded giant one
        fs::remove_all(dir);
    }
    std::println(" OK");

    // 13. Age-based cleanup: a backup older than max_age_days is deleted on
    // the next rotation; one younger than the cutoff survives.
    std::println("  test_age_based_cleanup...");
    {
        std::string dir = "/tmp/ragger_test_agesweep";
        std::string path = dir + "/activity.log";
        fs::remove_all(dir);
        fs::create_directories(dir);

        std::string old_backup = dir + "/activity.log.20200101-000000";
        std::string young_backup = dir + "/activity.log.20990101-000000";  // far future, never "old"
        { std::ofstream f(old_backup); f << "old"; }
        { std::ofstream f(young_backup); f << "young"; }

        // Backdate old_backup's mtime well past the 1-day cutoff used below.
        auto ancient = std::chrono::file_clock::now() - std::chrono::hours(24 * 30);
        fs::last_write_time(old_backup, ancient);

        {
            std::ofstream seed(path);
            seed << std::string(1024 * 1024 + 10, 'x');
        }
        Diskerror::logger log(path, "info", /*maxSizeMb=*/1, /*maxAgeDays=*/1);
        Diskerror::logger::info("triggers rotation + sweep");

        assert(!fs::exists(old_backup));    // swept (30 days old, cutoff is 1 day)
        assert(fs::exists(young_backup));   // untouched (2099 mtime, not old)
        fs::remove_all(dir);
    }
    std::println(" OK");

    cleanup();
    std::println("test_logger: all passed");
    return 0;
}
