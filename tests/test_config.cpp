#include "ragger/config.h"
#include "ragger/util/fs.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <print>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// New tests
// -----------------------------------------------------------------------

void test_server_locked_override() {
    std::print("  test_server_locked_override...");

    std::string sys_path = "/tmp/ragger_test_system.ini";
    std::string usr_path = "/tmp/ragger_test_user.ini";

    {
        std::ofstream f(sys_path);
        f << "[server]\nport = 8432\n"
          << "[search]\ndefault_limit = 5\nmax_search_limit = 0\n";
    }
    {
        std::ofstream f(usr_path);
        f << "[server]\nport = 9999\n"
          << "[search]\ndefault_limit = 20\n";
    }

    ragger::Config sys_cfg = ragger::load_config(sys_path).value();
    ragger::Config usr_cfg = ragger::load_config(usr_path).value();
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.port == 8432);  // SERVER_LOCKED — not overridden
    assert(sys_cfg.default_search_limit == 20);  // user-overridable

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::println(" OK");
}

void test_system_ceilings() {
    std::print("  test_system_ceilings...");

    std::string sys_path = "/tmp/ragger_test_ceil_sys.ini";
    std::string usr_path = "/tmp/ragger_test_ceil_usr.ini";

    {
        std::ofstream f(sys_path);
        f << "[server]\nport = 8432\n"
          << "[search]\nmax_search_limit = 10\ndefault_limit = 5\n";
    }
    {
        std::ofstream f(usr_path);
        f << "[search]\ndefault_limit = 50\n";
    }

    ragger::Config sys_cfg = ragger::load_config(sys_path).value();
    ragger::Config usr_cfg = ragger::load_config(usr_path).value();
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.default_search_limit == 10);  // clamped to ceiling

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::println(" OK");
}

void test_ceiling_zero_means_no_limit() {
    std::print("  test_ceiling_zero_means_no_limit...");

    std::string sys_path = "/tmp/ragger_test_ceil0_sys.ini";
    std::string usr_path = "/tmp/ragger_test_ceil0_usr.ini";

    {
        std::ofstream f(sys_path);
        f << "[server]\nport = 8432\n"
          << "[search]\nmax_search_limit = 0\ndefault_limit = 5\n";
    }
    {
        std::ofstream f(usr_path);
        f << "[search]\ndefault_limit = 999\n";
    }

    ragger::Config sys_cfg = ragger::load_config(sys_path).value();
    ragger::Config usr_cfg = ragger::load_config(usr_path).value();
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.default_search_limit == 999);  // no ceiling applied

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::println(" OK");
}

void test_housekeeping_section_parsing() {
    std::print("  test_housekeeping_section_parsing...");

    std::string path = "/tmp/ragger_test_housekeeping.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[housekeeping]\n"
          << "cleanup_max_age_hours = 168\n"
          << "housekeeping_interval = 30\n"
          << "catch_up_batch_size = 25\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.cleanup_max_age_hours == 168.0f);
    assert(cfg.housekeeping_interval == 30);
    assert(cfg.catch_up_batch_size == 25);

    fs::remove(path);
    std::println(" OK");
}

void test_catch_up_batch_size_default() {
    std::print("  test_catch_up_batch_size_default...");

    std::string path = "/tmp/ragger_test_catchup_default.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n";  // no [housekeeping] section at all
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.catch_up_batch_size == 10);  // shipped default

    fs::remove(path);
    std::println(" OK");
}

void test_catch_up_batch_size_ignores_nonpositive() {
    std::print("  test_catch_up_batch_size_ignores_nonpositive...");

    std::string path = "/tmp/ragger_test_catchup_zero.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[housekeeping]\ncatch_up_batch_size = 0\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.catch_up_batch_size == 10);  // 0 rejected, default retained

    fs::remove(path);
    std::println(" OK");
}

void test_project_gap_days_parsing() {
    std::print("  test_project_gap_days_parsing...");

    std::string path = "/tmp/ragger_test_project_gap.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[housekeeping]\nproject_gap_days = 14\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.project_gap_days == 14);

    fs::remove(path);
    std::println(" OK");
}

void test_project_gap_days_default() {
    std::print("  test_project_gap_days_default...");

    std::string path = "/tmp/ragger_test_project_gap_default.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n";  // no [housekeeping] section at all
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.project_gap_days == 7);  // shipped default

    fs::remove(path);
    std::println(" OK");
}

void test_project_gap_days_ignores_nonpositive() {
    std::print("  test_project_gap_days_ignores_nonpositive...");

    std::string path = "/tmp/ragger_test_project_gap_zero.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[housekeeping]\nproject_gap_days = 0\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.project_gap_days == 7);  // 0 rejected, default retained

    fs::remove(path);
    std::println(" OK");
}

void test_logging_section_parsing() {
    std::print("  test_logging_section_parsing...");

    std::string path = "/tmp/ragger_test_logging.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[logging]\n"
          << "log_level = debug\n"
          << "log_max_size_mb = 5\n"
          << "log_max_age_days = 30\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.log_level == "debug");
    assert(cfg.log_max_size_mb == 5);
    assert(cfg.log_max_age_days == 30);

    fs::remove(path);
    std::println(" OK");
}

void test_logging_section_defaults() {
    std::print("  test_logging_section_defaults...");

    std::string path = "/tmp/ragger_test_logging_default.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n";  // no [logging] section at all
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.log_level == "warn");
    assert(cfg.log_max_size_mb == 1);
    assert(cfg.log_max_age_days == 14);

    fs::remove(path);
    std::println(" OK");
}

void test_inference_endpoint_parsing() {
    std::print("  test_inference_endpoint_parsing...");

    std::string path = "/tmp/ragger_test_ep.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[inference]\nmodel = gpt-4\n"
          << "[inference.local]\n"
          << "api_url = http://localhost:1234/v1\n"
          << "api_key = lmstudio\n"
          << "models = qwen/*, llama/*\n"
          << "format = openai\n"
          << "max_context = 32000\n"
          << "[inference.anthropic]\n"
          << "api_url = https://api.anthropic.com/v1\n"
          << "api_key = sk-ant-xxx\n"
          << "models = claude-*\n"
          << "format = anthropic\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.inference_model == "gpt-4");
    assert(cfg.inference_endpoints.size() == 2);

    // Find endpoints by name (order may vary due to map)
    bool found_local = false, found_anthropic = false;
    for (auto& ep : cfg.inference_endpoints) {
        if (ep.name == "local") {
            found_local = true;
            assert(ep.api_url == "http://localhost:1234/v1");
            assert(ep.api_key == "lmstudio");
            assert(ep.models == "qwen/*, llama/*");
            assert(ep.format == "openai");
            assert(ep.max_context == 32000);
        } else if (ep.name == "anthropic") {
            found_anthropic = true;
            assert(ep.api_url == "https://api.anthropic.com/v1");
            assert(ep.api_key == "sk-ant-xxx");
            assert(ep.models == "claude-*");
            assert(ep.format == "anthropic");
        }
    }
    assert(found_local && found_anthropic);

    fs::remove(path);
    std::println(" OK");
}

void test_socket_bind_config() {
    std::print("  test_socket_bind_config...");

    // Test: socket only, no bind leaves bind_address empty
    std::string path1 = "/tmp/ragger_test_socket.ini";
    {
        std::ofstream f(path1);
        f << "[server]\n"
          << "socket = /tmp/test.sock\n";
    }

    ragger::Config cfg1 = ragger::load_config(path1).value();
    assert(cfg1.bind_address == "127.0.0.1");  // bind defaults to 127.0.0.1
    fs::remove(path1);

    // Test: bind + port populate both fields
    std::string path2 = "/tmp/ragger_test_bind.ini";
    {
        std::ofstream f(path2);
        f << "[server]\n"
          << "bind = 0.0.0.0\n"
          << "port = 9000\n";
    }

    ragger::Config cfg2 = ragger::load_config(path2).value();
    assert(cfg2.bind_address == "0.0.0.0");
    assert(cfg2.port == 9000);
    fs::remove(path2);

    std::println(" OK");
}

void test_bool_parsing_variants() {
    std::print("  test_bool_parsing_variants...");

    auto test = [](const std::string& val, bool expected) {
        std::string path = "/tmp/ragger_test_bool.ini";
        {
            std::ofstream f(path);
            f << "[server]\nport = 8432\n"
              << "[search]\nbm25_enabled = " << val << "\n";
        }
        ragger::Config cfg = ragger::load_config(path).value();
        assert(cfg.bm25_enabled == expected);
        fs::remove(path);
    };

    test("true", true);
    test("yes", true);
    test("1", true);
    test("false", false);
    test("no", false);
    test("0", false);

    std::println(" OK");
}

void test_inline_comments() {
    std::print("  test_inline_comments...");

    std::string path = "/tmp/ragger_test_inline.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432 # main port\nhost = 0.0.0.0 # bind all\n";
    }

    ragger::Config cfg = ragger::load_config(path).value();
    assert(cfg.port == 8432);
    (void)0;

    fs::remove(path);
    std::println(" OK");
}



void test_default_values() {
    std::print("  test_default_values...");

    std::string path = "/tmp/ragger_test_defaults.ini";
    {
        std::ofstream f(path);
        f << "[server]\n";  // minimal — just a section header
    }

    ragger::Config cfg = ragger::load_config(path).value();
    // All defaults from Config struct
    (void)0;
    assert(cfg.port == 8432);
    assert(true == true);
    assert(cfg.embedding_dimensions == 384);
    assert(cfg.default_search_limit == 5);
    assert(cfg.default_min_score == 0.4f);
    assert(cfg.bm25_enabled == true);
    assert(cfg.bm25_weight == 4.0f);
    assert(cfg.vector_weight == 8.0f);
    assert(cfg.phon_weight == 1.0f);
    assert(cfg.normalize_home_path == true);
    assert(cfg.cleanup_max_age_hours == 0.0f);  // 0 = keep forever (never delete raw conversations)
    assert(cfg.housekeeping_interval == 60);
    assert(cfg.capture_turns == true);
    assert(cfg.build_context == false);
    assert(cfg.embedding_vector_type == "f16");
    assert(cfg.max_search_limit == 0);
    assert(cfg.inference_max_tokens == 4096);
    assert(cfg.minimum_chunk_size == 300);

    fs::remove(path);
    std::println(" OK");
}

void test_ragger_base_default() {
    std::print("  test_ragger_base_default...");

    // No override set (default state at process start) — should be $HOME/.ragger
    ragger::set_ragger_base_override("");
    const char* home = std::getenv("HOME");
    assert(home != nullptr);
    assert(ragger::ragger_base_dir() == std::string(home) + "/.ragger");

    std::println(" OK");
}

void test_ragger_base_override() {
    std::print("  test_ragger_base_override...");

    ragger::set_ragger_base_override("/tmp/ragger-base-override-test");
    assert(ragger::ragger_base_dir() == "/tmp/ragger-base-override-test");

    // Resolved config paths must honor the override
    ragger::Config cfg;
    assert(cfg.resolved_db_path() == "/tmp/ragger-base-override-test/memories.db");
    assert(cfg.resolved_recipes_dir() == "/tmp/ragger-base-override-test/recipes");
    assert(cfg.resolved_formats_dir() == "/tmp/ragger-base-override-test/formats");

    // Reset so subsequent tests (and expand_path("~/.ragger/...") below) see the default.
    ragger::set_ragger_base_override("");

    std::println(" OK");
}

int main() {
    std::print("Running config tests:\n");

    test_server_locked_override();
    test_system_ceilings();
    test_ceiling_zero_means_no_limit();
    test_housekeeping_section_parsing();
    test_catch_up_batch_size_default();
    test_catch_up_batch_size_ignores_nonpositive();
    test_project_gap_days_parsing();
    test_project_gap_days_default();
    test_project_gap_days_ignores_nonpositive();
    test_logging_section_parsing();
    test_logging_section_defaults();
    test_inference_endpoint_parsing();
    test_socket_bind_config();
    test_bool_parsing_variants();
    test_inline_comments();

    test_default_values();

    test_ragger_base_default();
    test_ragger_base_override();

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

    ragger::Config cfg = ragger::load_config(tmp_conf).value();

    (void)0;
    assert(cfg.port == 9999);
    assert(cfg.embedding_dimensions == 384);
    assert(cfg.bm25_enabled == false);
    assert(cfg.default_min_score == 0.5f);
    // Defaults for unspecified values
    assert(cfg.bm25_weight == 4.0f);
    assert(cfg.normalize_home_path == true);

    std::filesystem::remove(tmp_conf);

    // Test find_system_config — explicit path takes priority (and returns error if missing)
    auto result_find = ragger::find_system_config("/nonexistent/settings.ini");
    assert(!result_find.has_value());
    assert(result_find.error() == ragger::ConfigError::NotFound);

    // Test find_system_config — no explicit path finds /etc/ragger.ini or ~/.ragger/settings.ini (or bootstraps)
    auto result_found = ragger::find_system_config("");
    assert(result_found.has_value());
    assert(!result_found->empty());

    // Test load_config with nonexistent file
    auto result_load = ragger::load_config("/nonexistent/settings.ini");
    assert(!result_load.has_value());
    assert(result_load.error() == ragger::ConfigError::IOError);

    std::println("test_config: all passed\n");
    return 0;
}
