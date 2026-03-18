/**
 * Configuration for Ragger Memory (C++ port)
 *
 * Loaded from ragger.conf at runtime.
 * Search order: /etc/ragger.conf → ~/.ragger/ragger.conf → --config-file=
 */
#pragma once

#include <string>
#include <stdexcept>

namespace ragger {

struct Config {
    // --- Server ---
    std::string host           = "127.0.0.1";
    int         port           = 8432;

    // --- Storage ---
    std::string db_path        = "~/.ragger/memories.db";
    std::string default_collection = "memory";

    // --- Embedding ---
    std::string embedding_model = "all-MiniLM-L6-v2";
    int         embedding_dimensions = 384;
    std::string model_dir;   // empty = default (~/.ragger/models)

    // --- Search ---
    int   default_search_limit = 5;
    float default_min_score    = 0.4f;
    bool  bm25_enabled         = true;
    float bm25_weight          = 0.3f;
    float vector_weight        = 0.7f;
    float bm25_k1              = 1.5f;
    float bm25_b               = 0.75f;

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

/// Find config file using search order. Returns path or throws.
/// @param cli_path  Path from --config-file (empty if not given)
std::string find_config_file(const std::string& cli_path = "");

/// Load config from an INI file.
Config load_config(const std::string& path);

/// Global config instance. Must call init_config() before use.
const Config& config();

/// Initialize global config. Call once at startup.
void init_config(const std::string& cli_config_path = "");

} // namespace ragger
