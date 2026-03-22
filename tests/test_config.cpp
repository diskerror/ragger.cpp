#include "ragger/config.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>

int main() {
    // expand_path with ~
    std::string expanded = ragger::expand_path("~/.ragger/memories.db");
    const char* home = std::getenv("HOME");
    assert(home != nullptr);
    assert(expanded.find('~') == std::string::npos);
    assert(expanded.find(home) == 0);

    // expand_path without ~
    assert(ragger::expand_path("/absolute/path") == "/absolute/path");
    assert(ragger::expand_path("relative/path") == "relative/path");

    // Test load_config directly with a temp file
    std::string tmp_conf = "/tmp/ragger_test.conf";
    {
        std::ofstream f(tmp_conf);
        f << "[server]\n"
          << "host = 0.0.0.0\n"
          << "port = 9999\n"
          << "\n"
          << "[embedding]\n"
          << "dimensions = 384\n"
          << "\n"
          << "[search]\n"
          << "bm25_enabled = false\n"
          << "default_min_score = 0.5\n";
    }

    auto cfg = ragger::load_config(tmp_conf);

    assert(cfg.host == "0.0.0.0");
    assert(cfg.port == 9999);
    assert(cfg.embedding_dimensions == 384);
    assert(cfg.bm25_enabled == false);
    assert(cfg.default_min_score == 0.5f);
    // Defaults for unspecified values
    assert(cfg.default_collection == "memory");
    assert(cfg.bm25_weight == 3.0f);
    assert(cfg.normalize_home_path == true);

    std::filesystem::remove(tmp_conf);

    // Test find_system_config — explicit path takes priority (and throws if missing)
    bool threw_find = false;
    try {
        ragger::find_system_config("/nonexistent/ragger.ini");
    } catch (const std::runtime_error&) {
        threw_find = true;
    }
    assert(threw_find);

    // Test find_system_config — no explicit path finds /etc/ragger.ini or ~/.ragger/ragger.ini (or bootstraps)
    auto found = ragger::find_system_config("");
    assert(!found.empty());

    // Test load_config with nonexistent file
    bool threw = false;
    try {
        ragger::load_config("/nonexistent/ragger.ini");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "test_config: all passed\n";
    return 0;
}
