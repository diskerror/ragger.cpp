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
    std::string server_name;   // hostname for Crow (e.g. "ragger.local")
    bool        single_user    = true;

    // --- Storage ---
    std::string db_path;     // empty = resolved at runtime (single_user: ~/.ragger/memories.db)
    std::string common_db_path = "/var/ragger/memories.db";
    std::string default_collection = "memory";
    std::string formats_dir    = "/var/ragger/formats";

    // --- Embedding ---
    std::string embedding_model = "all-MiniLM-L6-v2";
    int         embedding_dimensions = 384;
    std::string model_dir;   // empty = resolved at runtime (single_user: ~/.ragger/models, daemon: /var/ragger/models)

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
        int max_context = 0; // 0 = unknown. Context window in tokens.
        int max_tokens = 0;  // 0 = use global default
    };
    std::string inference_model = "claude-sonnet-4-5";
    std::string inference_default = "";
    std::string inference_api_url = "";
    std::string inference_api_key = "";
    int inference_max_tokens = 4096;
    std::vector<InferenceEndpointConfig> inference_endpoints;

    // --- Logging ---
    std::string log_dir;       // empty = resolved at runtime (single_user: ~/.ragger, daemon: /var/log/ragger)
    bool query_log_enabled     = true;
    bool http_log_enabled      = true;
    bool mcp_log_enabled       = true;

    // --- Paths ---
    bool normalize_home_path   = true;

    // --- Web UI ---
    std::string web_root;      // empty = use bundled web/ dir next to binary

    // --- TLS ---
    std::string tls_cert;      // path to certificate chain (PEM)
    std::string tls_key;       // path to private key (PEM)

    // --- Import ---
    int  minimum_chunk_size    = 300;

    // --- Llama.cpp (subprocess inference) ---
    // [llama] removed — use external inference providers (LM Studio, Ollama, etc.)

    // --- Model aliases ---
    std::map<std::string, std::string> model_aliases;  // short name → full name or .gguf filename

    // --- Auth ---
    int  token_rotation_minutes = 1440;  // 1440 = 24h, 0 = never

    // --- Chat ---
    std::string chat_store_turns = "true";  // "true", "session", "false"
    bool chat_summarize_on_pause = true;
    int  chat_pause_minutes      = 10;
    bool chat_summarize_on_quit  = true;
    int  chat_max_turn_retention_minutes = 60;
    int  chat_max_turns_stored   = 100;
    float cleanup_max_age_hours  = 336.0f;  // 2 weeks
    int   housekeeping_interval  = 60;      // seconds; 0 = disabled, <10 clamped to 10
    int  chat_max_persona_chars  = 0;   // 0 = unlimited
    int  chat_max_memory_results = 3;
    int  chat_persona_pct        = 25;  // % of context for persona
    float chat_chars_per_token   = 4.0f;

    // --- System ceilings (0 = no limit) ---
    int  max_search_limit             = 0;
    int  chat_max_persona_chars_limit = 0;
    int  chat_max_memory_results_limit = 0;

    /// Resolved paths (~ expanded)
    std::string resolved_db_path() const;
    std::string resolved_common_db_path() const;
    std::string resolved_log_dir() const;
    std::string resolved_model_dir() const;

    /// Resolve a model name: check aliases, prepend model_dir for .gguf files.
    std::string resolve_model(const std::string& name) const;
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
void init_config(const std::string& cli_config_path = "", bool quiet = false);

/// Reload config from INI file(s). Updates hot-reloadable values in-place.
/// Returns number of values changed. Logs restart-required changes without applying.
int reload_config();

} // namespace ragger
