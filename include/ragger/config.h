/**
 * Configuration for Ragger Memory (C++ port)
 *
 * Loaded from ragger.ini at runtime.
 * Search order: --config= → /etc/ragger.ini → ~/.ragger/ragger.ini
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

namespace ragger {

struct Config {
    // --- Server ---
    std::string host           = "127.0.0.1";
    int         port           = 8432;

    // --- Storage ---
    std::string db_path        = "~/.ragger/memories.db";
    std::string default_collection = "memory";
    std::string formats_dir    = "/var/ragger/formats";

    // --- Embedding ---
    std::string embedding_model = "all-MiniLM-L6-v2";
    int         embedding_dimensions = 384;
    std::string model_dir;   // empty = default (~/.ragger/models)

    // --- Search ---
    int   default_search_limit = 5;
    float default_min_score    = 0.4f;
    bool  bm25_enabled         = true;
    float bm25_weight          = 3.0f;
    float vector_weight        = 7.0f;
    float bm25_k1              = 1.5f;
    float bm25_b               = 0.75f;

    // --- Inference ---
    struct InferenceEndpointConfig {
        std::string name;
        std::string api_url;
        std::string api_key;
        std::string models = "*";
        std::string format;  // API format name (openai, anthropic, etc.)
    };
    std::string inference_model = "claude-sonnet-4-5";
    std::string inference_default = "";
    std::string inference_api_url = "";
    std::string inference_api_key = "";
    int inference_max_tokens = 4096;
    std::vector<InferenceEndpointConfig> inference_endpoints;

    // --- Logging ---
    std::string log_dir        = "~/.ragger";
    bool query_log_enabled     = true;
    bool http_log_enabled      = true;
    bool mcp_log_enabled       = true;

    // --- Paths ---
    bool normalize_home_path   = true;

    // --- Import ---
    int  minimum_chunk_size    = 300;

    /// Resolved paths (~ expanded)
    std::string resolved_db_path() const;
    std::string resolved_log_dir() const;
    std::string resolved_model_dir() const;
};

/// Expand ~ to $HOME in a path string.
std::string expand_path(const std::string& path);

/// Find system config file using search order. Returns path or throws.
/// @param cli_path  Path from --config (empty if not given)
std::string find_system_config(const std::string& cli_path = "");

/// Find user config file. Returns empty string if not found.
std::string find_user_config();

/// Load config from an INI file.
Config load_config(const std::string& path);

/// Apply user overrides to a config. Only allows specific fields.
void apply_user_overrides(Config& cfg, const Config& user);

/// Global config instance. Must call init_config() before use.
const Config& config();

/// Initialize global config. Call once at startup.
void init_config(const std::string& cli_config_path = "");

} // namespace ragger
