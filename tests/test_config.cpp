#include "ragger/config.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// New tests
// -----------------------------------------------------------------------

void test_server_locked_override() {
    std::cout << "  test_server_locked_override..." << std::flush;

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

    auto sys_cfg = ragger::load_config(sys_path);
    auto usr_cfg = ragger::load_config(usr_path);
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.port == 8432);  // SERVER_LOCKED — not overridden
    assert(sys_cfg.default_search_limit == 20);  // user-overridable

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::cout << " OK\n";
}

void test_system_ceilings() {
    std::cout << "  test_system_ceilings..." << std::flush;

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

    auto sys_cfg = ragger::load_config(sys_path);
    auto usr_cfg = ragger::load_config(usr_path);
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.default_search_limit == 10);  // clamped to ceiling

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::cout << " OK\n";
}

void test_ceiling_zero_means_no_limit() {
    std::cout << "  test_ceiling_zero_means_no_limit..." << std::flush;

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

    auto sys_cfg = ragger::load_config(sys_path);
    auto usr_cfg = ragger::load_config(usr_path);
    ragger::apply_user_overrides(sys_cfg, usr_cfg);

    assert(sys_cfg.default_search_limit == 999);  // no ceiling applied

    fs::remove(sys_path);
    fs::remove(usr_path);
    std::cout << " OK\n";
}

void test_chat_section_parsing() {
    std::cout << "  test_chat_section_parsing..." << std::flush;

    std::string path = "/tmp/ragger_test_chat.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[chat]\n"
          << "store_turns = session\n"
          << "summarize_on_pause = false\n"
          << "pause_minutes = 15\n"
          << "summarize_on_quit = false\n"
          << "max_persona_chars = 500\n"
          << "max_memory_results = 10\n"
          << "persona_pct = 30\n"
          << "chars_per_token = 3.5\n";
    }

    auto cfg = ragger::load_config(path);
    assert(cfg.chat_store_turns == "session");
    assert(cfg.chat_summarize_on_pause == false);
    assert(cfg.chat_pause_minutes == 15);
    assert(cfg.chat_summarize_on_quit == false);
    assert(cfg.chat_max_persona_chars == 500);
    assert(cfg.chat_max_memory_results == 10);
    assert(cfg.chat_persona_pct == 30);
    assert(cfg.chat_chars_per_token == 3.5f);

    fs::remove(path);
    std::cout << " OK\n";
}

void test_inference_endpoint_parsing() {
    std::cout << "  test_inference_endpoint_parsing..." << std::flush;

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

    auto cfg = ragger::load_config(path);
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
    std::cout << " OK\n";
}

void test_single_user_parsing() {
    std::cout << "  test_single_user_parsing..." << std::flush;

    std::string path = "/tmp/ragger_test_su.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\nsingle_user = false\n";
    }

    auto cfg = ragger::load_config(path);
    assert(cfg.single_user == false);

    fs::remove(path);
    std::cout << " OK\n";
}

void test_bool_parsing_variants() {
    std::cout << "  test_bool_parsing_variants..." << std::flush;

    auto test = [](const std::string& val, bool expected) {
        std::string path = "/tmp/ragger_test_bool.ini";
        {
            std::ofstream f(path);
            f << "[server]\nport = 8432\n"
              << "[search]\nbm25_enabled = " << val << "\n";
        }
        auto cfg = ragger::load_config(path);
        assert(cfg.bm25_enabled == expected);
        fs::remove(path);
    };

    test("true", true);
    test("yes", true);
    test("1", true);
    test("false", false);
    test("no", false);
    test("0", false);

    std::cout << " OK\n";
}

void test_inline_comments() {
    std::cout << "  test_inline_comments..." << std::flush;

    std::string path = "/tmp/ragger_test_inline.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432 # main port\nhost = 0.0.0.0 # bind all\n";
    }

    auto cfg = ragger::load_config(path);
    assert(cfg.port == 8432);
    assert(cfg.host == "0.0.0.0");

    fs::remove(path);
    std::cout << " OK\n";
}

void test_common_db_path_parsing() {
    std::cout << "  test_common_db_path_parsing..." << std::flush;

    std::string path = "/tmp/ragger_test_cdb.ini";
    {
        std::ofstream f(path);
        f << "[server]\nport = 8432\n"
          << "[storage]\ncommon_db_path = /data/shared/memories.db\n";
    }

    auto cfg = ragger::load_config(path);
    assert(cfg.common_db_path == "/data/shared/memories.db");

    fs::remove(path);
    std::cout << " OK\n";
}

void test_default_values() {
    std::cout << "  test_default_values..." << std::flush;

    std::string path = "/tmp/ragger_test_defaults.ini";
    {
        std::ofstream f(path);
        f << "[server]\n";  // minimal — just a section header
    }

    auto cfg = ragger::load_config(path);
    // All defaults from Config struct
    assert(cfg.host == "127.0.0.1");
    assert(cfg.port == 8432);
    assert(cfg.single_user == true);
    assert(cfg.db_path.empty());  // resolved at runtime via resolved_db_path()
    assert(cfg.default_collection == "memory");
    assert(cfg.embedding_dimensions == 384);
    assert(cfg.default_search_limit == 5);
    assert(cfg.default_min_score == 0.4f);
    assert(cfg.bm25_enabled == true);
    assert(cfg.bm25_weight == 3.0f);
    assert(cfg.vector_weight == 7.0f);
    assert(cfg.normalize_home_path == true);
    assert(cfg.chat_store_turns == "true");
    assert(cfg.chat_summarize_on_pause == true);
    assert(cfg.chat_pause_minutes == 10);
    assert(cfg.chat_max_memory_results == 3);
    assert(cfg.max_search_limit == 0);
    assert(cfg.inference_max_tokens == 4096);
    assert(cfg.minimum_chunk_size == 300);

    fs::remove(path);
    std::cout << " OK\n";
}

int main() {
    std::cout << "Running config tests:\n";

    test_server_locked_override();
    test_system_ceilings();
    test_ceiling_zero_means_no_limit();
    test_chat_section_parsing();
    test_inference_endpoint_parsing();
    test_single_user_parsing();
    test_bool_parsing_variants();
    test_inline_comments();
    test_common_db_path_parsing();
    test_default_values();

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
