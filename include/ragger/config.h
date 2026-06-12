/**
 * Configuration for Ragger Memory (C++ port)
 *
 * Loaded from settings.ini at runtime.
 * 
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <expected>

namespace ragger {

// Configuration error types
enum class ConfigError {
    NotFound,
    ParseError,
    IOError
};

struct Config {
    // --- Server ---
    // Listener selection: each listener is enabled only when its field is
    // non-empty after config load. Both may be set simultaneously (dual
    // listener). If both are empty after load, load_config() falls bind back
    // to "127.0.0.1" so the daemon always has at least one listener.
    std::string socket_path    = "";  // empty = no AF_UNIX listener
    std::string bind_address   = "";  // empty = no TCP listener
    int         port           = 8432;  // only meaningful when bind_address is set
    std::string server_name;   // hostname for cpp-httplib (e.g. "ragger.local")

    // --- Storage ---
    std::string formats_dir    = "/var/ragger/formats";

    // --- Embedding ---
    std::string embedding_model = "all-MiniLM-L6-v2";
    int         embedding_dimensions = 384;
    // On-disk vector storage precision: "f16" (half, default — half the size)
    // or "f32" (full). In-memory math is always f32; this only affects the
    // blob written to the DB. Tracked in the settings table for drift
    // protection — a whole DB must use one type (model+type are paired).
    std::string embedding_vector_type = "f16";
    std::string model_dir;   // empty = resolved at runtime (single_user: ~/.ragger/models, daemon: /var/ragger/models)

    // --- Search ---
    int   default_search_limit = 5;
    float default_min_score    = 0.4f;
    bool  bm25_enabled         = true;
    float bm25_weight          = 3.0f;
    float vector_weight        = 7.0f;

    // --- Embed (subprocess) ---
    int embed_timeout_ms  = 10000;
    int embed_retries     = 1;
    int embed_max_workers = 8;   // cap concurrent `ragger embed` subprocesses

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
    std::string inference_model = "";
    std::string inference_default = "";
    // Default to the LM Studio convention. Setting it lets the daemon's
    // summarizer find a local model out of the box; override per user.
    std::string inference_api_url = "http://localhost:1234/v1";
    std::string inference_api_key = "";
    int inference_max_tokens = 4096;
    std::vector<InferenceEndpointConfig> inference_endpoints;

    // --- [summarizer] section — model, endpoint, and prompt for L2/L3 ---
    std::string summarizer_model   = "";
    std::string summarizer_api_url = "";
    std::string summarizer_api_key = "";
    int         summarizer_max_tokens = 0;  // 0 = inherit from [inference] max_tokens
    // System-prompt sent to the summarizer model. Empty (or missing in the
    // INI) uses the built-in default. Any non-empty value is passed through
    // as-is — set to a single space " " to suppress the system prompt entirely.
    // Two positional placeholders {} {} are required (target_chars, hard_cap)
    // when overriding; the default already contains them.
    static constexpr std::string_view kDefaultSummarizerPrompt =
        "Summarize this conversation into a concise memory entry. "
        "Extract key facts, decisions, questions asked, topics discussed. "
        "Write in third person past tense. Target about {} characters; "
        "never exceed {}. If the exchange is trivial (a single command "
        "like /exit, a one-line greeting), respond with a short phrase, "
        "not a full sentence.";
    std::string summarizer_prompt = "";

    // --- Logging ---
    std::string log_file;   //  ~/.ragger/activity.log
    std::string log_level   = "warn";   //  trace, debug, info, warn, error, and critical
    std::string log_dir;    // empty = resolved at runtime (single_user: ~/.ragger, daemon: /var/log/ragger)
    bool query_log_enabled  = true;
    bool http_log_enabled   = true;
    bool mcp_log_enabled    = true;
    bool debug_log_enabled  = false;  // opt-in verbose tracing (per-chunk, etc.)

    // --- Paths ---
    bool normalize_home_path   = true;

    // --- TLS ---
    std::string tls_cert;      // path to certificate chain (PEM)
    std::string tls_key;       // path to private key (PEM)

    // --- Import ---
    int  minimum_chunk_size    = 300;

    // --- Turn handling ---
    // capture_turns (write side): when true, the capture_turn entry point
    // (MCP tool + HTTP POST /turn) ingests agent-pushed raw turns into the
    // `turns` table. When false, capture_turn is a no-op.
    bool capture_turns = true;
    // build_context (read side): when true, the build_context entry point
    // (MCP tool + HTTP GET /session/<id>) assembles a session's turns into a
    // context payload for the agent to inject. Only meaningful when
    // capture_turns is also true — there's nothing to build from otherwise.
    // Agent-driven search/store tools are unaffected by either flag.
    bool build_context = false;
    // Recipe name applied when the caller doesn't specify one. Recipes are
    // loaded from `recipes_dir` (JSON files); built-ins cover the case where
    // the directory is missing or empty.
    std::string default_recipe = "natural_fading";
    std::string recipes_dir;   // empty = ~/.ragger/recipes

    // --- Model aliases ---
    std::map<std::string, std::string> model_aliases;  // short name → full name or .gguf filename

    // --- Housekeeping / retention ---
    float cleanup_max_age_hours  = 336.0f;  // 2 weeks
    int   housekeeping_interval  = 60;      // seconds; 0 = disabled, <10 clamped to 10
    int   summary_pause_minutes  = 20;      // idle gap that closes a session's running (L3) summary

    // --- System ceilings (0 = no limit) ---
    int  max_search_limit             = 0;

    /// Resolved paths (~ expanded)
    std::string resolved_db_path() const;
    std::string resolved_log_dir() const;
    std::string resolved_model_dir() const;

    /// Resolve a model name: check aliases, prepend model_dir for .gguf files.
    std::string resolve_model(const std::string& name) const;
};

/// Expand ~ to $HOME in a path string.
std::string expand_path(const std::string& path);

/// Find system config file using search order. Returns path or throws.
/// @param cli_path  Path from --config (empty if not given)
std::expected<std::string, ConfigError> find_system_config(const std::string& cli_path = "");

/// Find user config file. Returns empty string if not found.
std::expected<std::string, ConfigError> find_user_config();

/// Load config from an INI file.
std::expected<Config, ConfigError> load_config(const std::string& path);

/// Apply user overrides to a config. Only allows specific fields.
void apply_user_overrides(Config& cfg, const Config& user);

/// Global config instance. Must call init_config() before use.
const Config& config();
/// Mutable access for CLI overrides applied at startup.
Config& mutable_config();

/// Initialize global config. Call once at startup.
void init_config(const std::string& cli_config_path = "");

/// Reload config from INI file(s). Updates hot-reloadable values in-place.
/// Returns number of values changed. Logs restart-required changes without applying.
int reload_config();

/// Absolute path to this `ragger` executable, used to spawn `ragger embed`
/// subprocesses. Set once at startup from argv[0]; falls back to "ragger"
/// (resolved via PATH) if never set or unresolvable.
void set_executable_path(const std::string& argv0);
const std::string& executable_path();

} // namespace ragger
