#include "config.h"
#include "config_access.h"
#include "user_store.h"
#include "config_schema.h"
#include "test_env.h"
#include "check.h"
#include "util/fs.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <print>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// New tests
// -----------------------------------------------------------------------




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
    assert(cfg.default_min_score == 0.6f);
    assert(cfg.bm25_enabled == true);
    assert(cfg.bm25_weight == 4.0f);
    assert(cfg.vector_weight == 8.0f);
    assert(cfg.normalize_home_path == true);
    assert(cfg.cleanup_max_age_hours == 0.0f);  // 0 = keep forever (never delete raw conversations)
    assert(cfg.housekeeping_interval == 60);
    assert(cfg.capture_turns == true);
    assert(cfg.build_context == false);
    assert(cfg.embedding_vector_type == "f16");
    assert(cfg.inference_max_tokens == 4096);
    assert(cfg.minimum_chunk_size == 800);

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

// API keys are environment variables: the settings table holds only the NAME
// of the variable, the value is resolved at runtime and never saved, and a
// pasted credential is rejected as a name.
void test_api_keys_come_from_environment() {
    std::print("  test_api_keys_come_from_environment...\n");
    using namespace ragger;

    for (const char* key : {"summarizer_api_key_env", "embedding_api_key_env"}) {
        const auto* meta = lang::config_meta(key);
        assert(meta != nullptr);
        assert(!meta->secret);                       // a name, not a credential
        // The old value-bearing keys are gone from the schema.
    }
    assert(lang::config_meta("summarizer_api_key") == nullptr);
    assert(lang::config_meta("embedding_external_api_key") == nullptr);

    ::setenv("RAGGER_TEST_SUMMARIZER_KEY", "s3cr3t-value", 1);
    ::unsetenv("RAGGER_TEST_UNSET_VAR");

    auto set = [](std::string_view k, std::string_view v) {
        auto r = set_config_persisted(k, v);
        assert(r.has_value());
    };

    // Naming a variable resolves the live key from the environment...
    set("summarizer_api_key_env", "RAGGER_TEST_SUMMARIZER_KEY");
    assert(config().summarizer_api_key == "s3cr3t-value");
    // ...the persisted/read value is the NAME, never the key...
    assert(get_config_value(config(), "summarizer_api_key_env").value_or("")
           == "RAGGER_TEST_SUMMARIZER_KEY");
    // ...an unset variable gives an empty key, not an error...
    set("summarizer_api_key_env", "RAGGER_TEST_UNSET_VAR");
    assert(config().summarizer_api_key.empty());
    // ...and a pasted credential is not a valid variable name.
    assert(!set_config_persisted("summarizer_api_key_env", "sk-abc-123.def").has_value());
    assert(!set_config_persisted("embedding_api_key_env", "9STARTS_WITH_DIGIT").has_value());
    set("embedding_api_key_env", "RAGGER_TEST_SUMMARIZER_KEY");
    assert(config().embedding_external_api_key == "s3cr3t-value");

    set("summarizer_api_key_env", "");   // leave clean (back to the default name)
    set("embedding_api_key_env", "");
    ::unsetenv("RAGGER_TEST_SUMMARIZER_KEY");
    std::println(" ok");
}

// PHASE9A: the shipped defaults must be applied, every live key accepted.
void p9a_test_shipped_defaults_applied() {
    std::print("  p9a_test_shipped_defaults_applied...");
    for (const auto& k : ragger::default_config_unknown_keys())
        std::println("\n    unknown default key: {}", k);
    CHECK(ragger::default_config_unknown_keys().empty());
    ragger::Config d = ragger::default_config();
    CHECK(d.summarizer_model == "qwen3-4b-instruct-2507");
    CHECK(d.summarizer_api_url == "http://localhost:1234/v1");
    CHECK(d.summarizer_max_tokens == 1024);
    CHECK(d.summarizer_target_pct == 40);
    CHECK(d.summarizer_max_pct == 60);
    CHECK(d.embedding_model == "sentence-transformers/all-MiniLM-L12-v2");
    CHECK(d.embedding_dimensions == 384);
    CHECK(d.embedding_vector_type == "f16");
    CHECK(d.summarizer_prompt.empty());   // empty = built-in default prompt
    std::println(" ok");
}

// PHASE9A: config() is an immutable snapshot; writers publish a new one.
void p9a_test_config_snapshots() {
    std::print("  p9a_test_config_snapshots...");
    const ragger::Config* before = &ragger::config();
    const std::string& held = ragger::config().summarizer_prompt;  // taken BEFORE the updates
    const std::string held_copy = held;

    std::atomic<bool> stop{false};
    std::atomic<long> torn{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            while (!stop.load()) {
                const std::string& s = ragger::config().summarizer_prompt;
                for (size_t i = 1; i < s.size(); ++i)
                    if (s[i] != s[0]) { ++torn; break; }   // a torn read mixes characters
            }
        });
    for (int i = 0; i < 300; ++i)
        CHECK(ragger::update_config([&](ragger::Config& c) {
            c.summarizer_prompt = (i & 1) ? std::string(5, 'a') : std::string(5000, 'b');
            return true;
        }));
    stop = true;
    for (auto& r : readers) r.join();

    CHECK(torn.load() == 0);
    CHECK(held == held_copy);                      // the old snapshot is untouched and still valid
    CHECK(&ragger::config() != before);            // a new snapshot was published
    CHECK(ragger::config().summarizer_prompt == std::string(5, 'a'));   // i = 299 is odd

    const ragger::Config* cur = &ragger::config();
    CHECK(!ragger::update_config([](ragger::Config& c) { c.summarizer_prompt = "zzz"; return false; }));
    CHECK(&ragger::config() == cur);               // false = nothing published

    ragger::update_config([&](ragger::Config& c) { c.summarizer_prompt = held_copy; return true; });
    std::println(" ok");
}

// PHASE9B step 3b: summarizer_prompt must be a valid vformat pattern with
// two args; other Text keys are untouched.
void p9b_test_summarizer_prompt_validation() {
    std::print("  p9b_test_summarizer_prompt_validation...\n");
    auto ok = [](std::string_view k, std::string_view v) {
        return ragger::validate_config_value(k, v).has_value();
    };
    CHECK(ok("summarizer_prompt", ""));   // empty = reset to default
    CHECK(ok("summarizer_prompt", " "));
    CHECK(ok("summarizer_prompt", "summarize in {} words, max {}"));
    CHECK(ok("summarizer_prompt", "{0} {1}"));
    CHECK(ok("summarizer_prompt", "{{literal braces}}"));
    CHECK(!ok("summarizer_prompt", "{"));
    CHECK(!ok("summarizer_prompt", "}"));
    CHECK(!ok("summarizer_prompt", "{2}"));
    CHECK(!ok("summarizer_prompt", "{:d"));
    // Other Text keys keep accepting anything.
    CHECK(ok("default_recipe", "plain"));
    std::println(" OK");
}

// PHASE9B step 7: the episode keys are read per use by summarizer_service,
// so a SIGHUP reload must apply them (they used to be silently dropped).
void p9b_test_reload_config_episode_keys() {
    std::print("  p9b_test_reload_config_episode_keys...\n");
    const std::string db = ragger::config().resolved_db_path();
    ragger::UserStore store(db);

    // Nothing written yet: a reload with no change reports 0.
    CHECK(ragger::reload_config() == 0);

    store.set_setting("episode_threshold_base", "0.42");
    store.set_setting("episode_threshold_cap", "0.91");
    store.set_setting("episode_threshold_step", "0.07");
    store.set_setting("episode_step_minutes", "13");

    CHECK(ragger::reload_config() == 4);
    const auto& cfg = ragger::config();
    CHECK(std::abs(cfg.episode_threshold_base - 0.42f) < 1e-6f);
    CHECK(std::abs(cfg.episode_threshold_cap  - 0.91f) < 1e-6f);
    CHECK(std::abs(cfg.episode_threshold_step - 0.07f) < 1e-6f);
    CHECK(cfg.episode_step_minutes == 13);

    // Idempotent: the same rows are not counted as changes again.
    CHECK(ragger::reload_config() == 0);

    // Reset to defaults so later tests in this binary see the shipped values.
    for (const char* k : {"episode_threshold_base", "episode_threshold_cap",
                          "episode_threshold_step", "episode_step_minutes"})
        store.set_setting(k, "");
    CHECK(ragger::reload_config() == 4);
    std::println(" OK");
}

int main() {
    TestEnv env("config");  // must be first: set_config_persisted publishes through update_config
    std::print("Running config tests:\n");
    test_housekeeping_section_parsing();
    p9a_test_shipped_defaults_applied();
    p9b_test_summarizer_prompt_validation();
    p9b_test_reload_config_episode_keys();
    p9a_test_config_snapshots();
    test_catch_up_batch_size_default();
    test_catch_up_batch_size_ignores_nonpositive();
    test_logging_section_parsing();
    test_logging_section_defaults();
    test_inference_endpoint_parsing();
    test_socket_bind_config();
    test_bool_parsing_variants();
    test_inline_comments();

    test_default_values();
    test_api_keys_come_from_environment();

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

    // Test load_config with nonexistent file
    auto result_load = ragger::load_config("/nonexistent/settings.ini");
    assert(!result_load.has_value());
    assert(result_load.error() == ragger::ConfigError::IOError);

    std::println("test_config: all passed\n");
    return 0;
}
