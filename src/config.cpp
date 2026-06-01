/**
 * Configuration loader — INI file parser
 */
#include "ragger/config.h"
#include "ragger/lang.h"

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <pwd.h>
#include <unistd.h>
#include <format>


namespace ragger {
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------
std::string normalize_lm_proxy_url(std::string url) {
    // Drop trailing slashes
    while (!url.empty() && url.back() == '/') url.pop_back();
    // Drop one trailing "/v1" segment if present
    static constexpr std::string_view v1_suffix = "/v1";
    if (url.size() >= v1_suffix.size() &&
        url.compare(url.size() - v1_suffix.size(), v1_suffix.size(), v1_suffix) == 0) {
        url.erase(url.size() - v1_suffix.size());
    }
    return url;
}

std::string expand_path(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) {
        // Fallback: look up home directory from passwd entry
        struct passwd* pw = getpwuid(getuid());
        if (pw && pw->pw_dir) home = pw->pw_dir;
    }
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

std::string Config::resolved_db_path() const {
    if (db_path.empty()) {
        return expand_path("~/.ragger/memories.db");
    }
    return expand_path(db_path);
}

std::string Config::resolved_log_dir() const {
    if (log_dir.empty()) {
        return expand_path("~/.ragger/logs");
    }
    return expand_path(log_dir);
}

std::string Config::resolved_model_dir() const {
    std::string base = model_dir.empty()
        ? expand_path("~/.ragger/models")
        : expand_path(model_dir);
    return base + "/" + resolve_model(embedding_model);
}

std::string Config::resolve_model(const std::string& name) const {
    // Check aliases first
    auto it = model_aliases.find(name);
    return (it != model_aliases.end()) ? it->second : name;
}

// -----------------------------------------------------------------------
// Default config (embedded)
// -----------------------------------------------------------------------
static constexpr const char* DEFAULT_CONFIG = R"(# settings.ini — Ragger Memory configuration
#
# Search order:
#   1. --config=<path>          (explicit override)
#   2. ~/.ragger/settings.ini    (per-user default)
#
# First file found wins. Created automatically on first run.

[server]
socket = ~/.ragger/ragger.sock
port = 8432

[storage]
db_path = ~/.ragger/memories.db
formats_dir = ~/.ragger/formats

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
# model_dir: path to ONNX model files (default: /var/ragger/models)
# All users on the same system must use the same embedding model
# for vector search to work correctly.

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 3
vector_weight = 7

[inference]
# Default model for chat
model =
max_tokens = 4096

# Single endpoint (simple setup):
# api_url = http://localhost:1234/v1
# api_key = lmstudio-local

# Multiple endpoints (advanced setup):
# Use [inference.<name>] sections for multiple endpoints
# Model routing: first matching glob pattern wins

# [inference.local]
# api_url = http://localhost:1234/v1
# api_key = lmstudio-local
# models = qwen/*, llama/*, mistral/*

# [inference.anthropic]
# api_url = https://api.anthropic.com/v1
# api_key = sk-ant-...
# models = claude-*

# Memory model: a cheaper/local model for summarization. Falls back to the
# main chat model when unset. api_url is optional — omit it if the model is
# already served by one of the endpoints above.
# [inference.memory]
# model   = qwen2.5:7b
# api_url = http://localhost:11434/v1
# api_key =

[logging]
log_file = ~/.ragger/activity.log
# Log Levels: trace, debug, info, warn, error, and critical
log_level = warn

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300

[embed]
timeout_ms = 5000
retries = 1
max_workers = 4
)";

// -----------------------------------------------------------------------
// Bootstrap ~/.ragger/ on first run
// -----------------------------------------------------------------------
static std::string bootstrap_user_config() {
    std::string ragger_dir = expand_path("~/.ragger");
    std::string conf_path  = ragger_dir + "/settings.ini";

    fs::create_directories(ragger_dir);

    // Never clobber an existing config — just return its path.
    if (fs::exists(conf_path)) {
        return conf_path;
    }

    std::ofstream out(conf_path);
    if (!out.is_open()) {
        throw std::runtime_error(std::format(lang::ERR_CONFIG_OPEN, conf_path));
    }
    out << DEFAULT_CONFIG;
    out.close();

//  std::cout << " [INFO] " << lang::MSG_CONFIG_CREATED << conf_path << std::endl;
    return conf_path;
}

// -----------------------------------------------------------------------
// Config file search
// -----------------------------------------------------------------------
std::expected<std::string, ConfigError> find_system_config(const std::string& cli_path) {
    // 1. Explicit --config= takes highest priority
    if (!cli_path.empty()) {
        std::string resolved = expand_path(cli_path);
        if (!fs::exists(resolved)) {
            return std::unexpected(ConfigError::NotFound);
        }
        return resolved;
    }

    // 2. First run — bootstrap default user config (acts as system config)
    try {
        return bootstrap_user_config();
    } catch (...) {
        return std::unexpected(ConfigError::IOError);
    }
}

std::expected<std::string, ConfigError> find_user_config() {
    std::string user_conf = expand_path("~/.ragger/settings.ini");
    try {
        if (fs::exists(user_conf)) {
            return user_conf;
        }
    } catch (const fs::filesystem_error&) {
        // Permission denied or inaccessible — skip user config silently
    }
    return std::unexpected(ConfigError::NotFound);
}

// Server infrastructure keys — system config always wins (blacklist)
// Match the Python SERVER_LOCKED set
struct ServerLockedKey {
    const char* section;
    const char* key;
};

static const ServerLockedKey SERVER_LOCKED[] = {
    {"server", "port"},
    {"storage", "db_path"},
    {"storage", "formats_dir"},
    {"logging", "log_dir"},
    {"embedding", "model"},
    {"embedding", "dimensions"},
    {"embedding", "model_dir"},
    // System ceilings
    {"chat", "max_persona_chars_limit"},
    {"chat", "max_memory_results_limit"},
    {"search", "max_search_limit"},
    // Inference endpoints (user can only pick model)
    {"inference", "api_url"},
    {"inference", "api_key"},
    {"inference", "provider"},
    {"inference", "max_tokens"},
};

/// Clamp a value to a ceiling. Ceiling of 0 = no limit.
/// Value of 0 = "unlimited" — ceiling applies (becomes ceiling).
static void clamp_to_ceiling(int& value, int ceiling) {
    if (ceiling <= 0) return;
    if (value <= 0 || value > ceiling) value = ceiling;
}

void apply_user_overrides(Config& cfg, const Config& user) {
    // New pattern: user config overrides everything EXCEPT server-locked fields
    // Server-locked fields stay as loaded from system config
    
    // User can override everything except SERVER_LOCKED:
    // ✗ server.host, server.port
    // ✗ storage.db_path, storage.formats_dir
    // ✗ logging.log_dir (always locked — one server, one log location)
    // ✗ embedding.model, embedding.dimensions, embedding.model_dir
    // ✓ Everything else
    
    // Search (all user-overridable)
    cfg.default_search_limit = user.default_search_limit;
    cfg.default_min_score = user.default_min_score;
    cfg.bm25_enabled = user.bm25_enabled;
    cfg.bm25_weight = user.bm25_weight;
    cfg.vector_weight = user.vector_weight;

    // Inference (user can only pick model)
    cfg.inference_model = user.inference_model;
    cfg.inference_default = user.inference_default;
    
    // Logging (query_log, http_log, mcp_log are user-overridable; log_dir is SERVER_LOCKED)
    cfg.query_log_enabled = user.query_log_enabled;
    cfg.http_log_enabled = user.http_log_enabled;
    cfg.mcp_log_enabled = user.mcp_log_enabled;
    
    // Paths (all user-overridable)
    cfg.normalize_home_path = user.normalize_home_path;
    
    // Import (all user-overridable)
    cfg.minimum_chunk_size = user.minimum_chunk_size;

    // Chat (user-overridable, but ceilings apply)
    cfg.chat_store_turns = user.chat_store_turns;
    cfg.chat_summarize_on_pause = user.chat_summarize_on_pause;
    cfg.chat_pause_minutes = user.chat_pause_minutes;
    cfg.chat_summarize_on_quit = user.chat_summarize_on_quit;
    cfg.chat_max_persona_chars = user.chat_max_persona_chars;
    cfg.chat_max_memory_results = user.chat_max_memory_results;
    cfg.chat_persona_pct = user.chat_persona_pct;
    cfg.chat_chars_per_token = user.chat_chars_per_token;
    cfg.chat_verbatim_turns = user.chat_verbatim_turns;
    cfg.chat_verbatim_minutes = user.chat_verbatim_minutes;
    cfg.chat_summary_count = user.chat_summary_count;
    cfg.chat_summary_minutes = user.chat_summary_minutes;
    cfg.chat_search_min_score = user.chat_search_min_score;
    cfg.chat_budget_persona_pct = user.chat_budget_persona_pct;
    cfg.chat_budget_session_pct = user.chat_budget_session_pct;
    cfg.chat_budget_summaries_pct = user.chat_budget_summaries_pct;
    cfg.chat_budget_search_pct = user.chat_budget_search_pct;
    cfg.chat_budget_reserve_pct = user.chat_budget_reserve_pct;
    // Note: max_turn_retention_minutes, max_turns_stored, and all *_limit
    // keys are SERVER_LOCKED — they stay as loaded from system config.

    // Apply system ceilings
    clamp_to_ceiling(cfg.default_search_limit, cfg.max_search_limit);
    clamp_to_ceiling(cfg.chat_max_memory_results, cfg.chat_max_memory_results_limit);
    clamp_to_ceiling(cfg.chat_max_persona_chars, cfg.chat_max_persona_chars_limit);
}

// -----------------------------------------------------------------------
// INI parser
// -----------------------------------------------------------------------
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static bool parse_bool(const std::string& val) {
    std::string lower = val;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "true" || lower == "yes" || lower == "1";
}

std::expected<Config, ConfigError> load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::unexpected(ConfigError::IOError);
    }

    Config cfg;
    std::string section;
    std::string line;

    // Temporary storage for inference endpoint sections
    std::map<std::string, Config::InferenceEndpointConfig> endpoint_map;

    while (std::getline(file, line)) {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;

        // Section header
        if (line[0] == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        // Key = value
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        // Strip inline comments (but not inside quoted values)
        if (val[0] != '"' && val[0] != '\'') {
            auto hash = val.find('#');
            if (hash != std::string::npos) {
                val = trim(val.substr(0, hash));
            }
        }

        // Map to config fields
        if (section == "server") {
            if      (key == "socket") cfg.socket_path = val;
            else if (key == "bind") cfg.bind_address = val;
            else if (key == "port") cfg.port = std::stoi(val);
            else if (key == "server_name" || key == "hostname") cfg.server_name = val;
        }
        else if (section == "storage") {
            if      (key == "db_path") cfg.db_path = val;
            else if (key == "formats_dir")        cfg.formats_dir = val;
        }
        else if (section == "embedding") {
            if      (key == "model")      cfg.embedding_model = val;
            else if (key == "dimensions") cfg.embedding_dimensions = std::stoi(val);
            else if (key == "model_dir")  cfg.model_dir = val;
        }
        else if (section == "search") {
            if      (key == "default_limit")    cfg.default_search_limit = std::stoi(val);
            else if (key == "default_min_score") cfg.default_min_score = std::stof(val);
            else if (key == "bm25_enabled")     cfg.bm25_enabled = parse_bool(val);
            else if (key == "bm25_weight")      cfg.bm25_weight = std::stof(val);
            else if (key == "vector_weight")    cfg.vector_weight = std::stof(val);
            else if (key == "max_search_limit") cfg.max_search_limit = std::stoi(val);
        }
        else if (section == "inference") {
            if      (key == "model")      cfg.inference_model = val;
            else if (key == "api_url")    cfg.inference_api_url = val;
            else if (key == "api_key")    cfg.inference_api_key = val;
            else if (key == "max_tokens") cfg.inference_max_tokens = std::stoi(val);
            else if (key == "default")    cfg.inference_default = val;
            else if (key == "lm_proxy_url") cfg.lm_proxy_url = normalize_lm_proxy_url(val);
        }
        else if (section == "inference.memory") {
            // The memory model (summarization/routing). Captured into dedicated
            // fields rather than a generic endpoint; from_config() builds the
            // endpoint from these when api_url is set.
            if      (key == "model")   cfg.inference_memory_model   = val;
            else if (key == "api_url") cfg.inference_memory_api_url = val;
            else if (key == "api_key") cfg.inference_memory_api_key = val;
        }
        else if (section.substr(0, 10) == "inference.") {
            // Named endpoint section: [inference.local], [inference.anthropic], etc.
            std::string ep_name = section.substr(10);
            auto& ep = endpoint_map[ep_name];
            ep.name = ep_name;

            if      (key == "api_url")     ep.api_url = val;
            else if (key == "api_key")     ep.api_key = val;
            else if (key == "models")      ep.models = val;
            else if (key == "format")      ep.format = val;
            else if (key == "max_context") ep.max_context = std::stoi(val);
            else if (key == "max_tokens") ep.max_tokens = std::stoi(val);
        }
        else if (section == "logging") {
            if      (key == "log_dir")   cfg.log_dir = val;
            else if (key == "log_file")  cfg.log_file = expand_path(val);
            else if (key == "log_level") cfg.log_level = val;
            else if (key == "query_log") cfg.query_log_enabled = parse_bool(val);
            else if (key == "http_log")  cfg.http_log_enabled = parse_bool(val);
            else if (key == "mcp_log")   cfg.mcp_log_enabled = parse_bool(val);
            else if (key == "debug_log") cfg.debug_log_enabled = parse_bool(val);
        }
        else if (section == "paths") {
            if (key == "normalize_home") cfg.normalize_home_path = parse_bool(val);
        }
        else if (section == "web") {
            if (key == "root" || key == "web_root") cfg.web_root = val;
        }
        else if (section == "tls" || section == "ssl") {
            if (key == "cert" || key == "tls_cert") cfg.tls_cert = val;
            else if (key == "key" || key == "tls_key") cfg.tls_key = val;
        }
        else if (section == "import") {
            if (key == "minimum_chunk_size") cfg.minimum_chunk_size = std::stoi(val);
        }
        else if (section == "embed") {
            if (key == "timeout_ms") cfg.embed_timeout_ms = std::stoi(val);
            else if (key == "retries") cfg.embed_retries = std::stoi(val);
            else if (key == "max_workers") cfg.embed_max_workers = std::stoi(val);
        }
        else if (section == "chat") {
            if (key == "store_turns") cfg.chat_store_turns = val;
            else if (key == "summarize_on_pause") cfg.chat_summarize_on_pause = parse_bool(val);
            else if (key == "pause_minutes") cfg.chat_pause_minutes = std::stoi(val);
            else if (key == "summarize_on_quit") cfg.chat_summarize_on_quit = parse_bool(val);
            else if (key == "cleanup_max_age_hours") cfg.cleanup_max_age_hours = std::stof(val);
            else if (key == "housekeeping_interval") {
                int v = std::stoi(val);
                cfg.housekeeping_interval = (v == 0) ? 0 : std::max(v, 10);
            }
            else if (key == "max_persona_chars") cfg.chat_max_persona_chars = std::stoi(val);
            else if (key == "max_memory_results") cfg.chat_max_memory_results = std::stoi(val);
            else if (key == "persona_pct") cfg.chat_persona_pct = std::stoi(val);
            else if (key == "chars_per_token") cfg.chat_chars_per_token = std::stof(val);
            else if (key == "verbatim_turns")   cfg.chat_verbatim_turns = std::stoi(val);
            else if (key == "verbatim_minutes") cfg.chat_verbatim_minutes = std::stoi(val);
            else if (key == "summary_count")    cfg.chat_summary_count = std::stoi(val);
            else if (key == "summary_minutes")  cfg.chat_summary_minutes = std::stoi(val);
            else if (key == "search_min_score") cfg.chat_search_min_score = std::stof(val);
            else if (key == "budget_persona_pct")   cfg.chat_budget_persona_pct = std::stoi(val);
            else if (key == "budget_session_pct")   cfg.chat_budget_session_pct = std::stoi(val);
            else if (key == "budget_summaries_pct") cfg.chat_budget_summaries_pct = std::stoi(val);
            else if (key == "budget_search_pct")    cfg.chat_budget_search_pct = std::stoi(val);
            else if (key == "budget_reserve_pct")   cfg.chat_budget_reserve_pct = std::stoi(val);
            else if (key == "stream_flush_seconds") cfg.chat_stream_flush_seconds = std::stoi(val);
            else if (key == "max_persona_chars_limit") cfg.chat_max_persona_chars_limit = std::stoi(val);
            else if (key == "max_memory_results_limit") cfg.chat_max_memory_results_limit = std::stoi(val);
            else if (key == "system_prompt_file") cfg.system_prompt_file = val;
            else if (key == "persona_dir") cfg.persona_dir = val;
            else if (key == "proxy_system_prompt") cfg.proxy_system_prompt = val;
        }
        else if (section == "models") {
            cfg.model_aliases[key] = val;
        }
        else if (section == "llama") {
            // [llama] section removed — use external inference providers
            // Silently ignore for backward compatibility with old configs
        }
    }

    // Convert endpoint map to vector
    for (auto& [name, ep] : endpoint_map) {
        cfg.inference_endpoints.push_back(ep);
    }

    // Validate: at least one of socket_path or bind_address must be set
    std::string expanded_socket = expand_path(cfg.socket_path);
    if (expanded_socket.empty() && cfg.bind_address.empty()) {
        throw std::runtime_error(lang::ERR_CONFIG_SOCKET_OR_BIND);
    }

    // Expand socket_path and bind_address if they start with ~
    if (!cfg.socket_path.empty() && cfg.socket_path[0] == '~') {
        cfg.socket_path = expanded_socket;
    }
    if (!cfg.bind_address.empty() && cfg.bind_address[0] == '~') {
        // For bind_address, expand ~ to home but keep the host part
        std::string home = expand_path("~");
        if (home.back() != '/') home += '/';
        cfg.bind_address = home + cfg.bind_address.substr(1);
    }

    return cfg;
}

// -----------------------------------------------------------------------
// Global singleton
// -----------------------------------------------------------------------
static Config* g_config = nullptr;

const Config& config() {
    if (!g_config) {
        throw std::runtime_error(lang::ERR_CONFIG_NOT_INIT);
    }
    return *g_config;
}

Config& mutable_config() {
    if (!g_config) {
        throw std::runtime_error(lang::ERR_CONFIG_NOT_INIT);
    }
    return *g_config;
}

void init_config(const std::string& cli_config_path) {
    // Load system config first
    auto system_result = find_system_config(cli_config_path);
    if (!system_result.has_value()) {
        throw std::runtime_error(lang::ERR_CONFIG_SYSTEM_NOT_FOUND);
    }
    std::string system_path = *system_result;
    
    auto sys_cfg_result = load_config(system_path);
    if (!sys_cfg_result.has_value()) {
        switch (sys_cfg_result.error()) {
            case ConfigError::IOError:
                throw std::runtime_error(std::format(lang::ERR_CONFIG_SYSTEM_LOAD, system_path));
            case ConfigError::ParseError:
                throw std::runtime_error(std::format(lang::ERR_CONFIG_SYSTEM_PARSE, system_path));
            default:
                throw std::runtime_error(std::format(lang::ERR_CONFIG_SYSTEM_UNKNOWN, system_path));
        }
    }
    static Config cfg = *sys_cfg_result;
//  std::cout << " [INFO] " << lang::MSG_CONFIG_LOADED << system_path << std::endl;
    
    // Then overlay user-specific overrides
    auto user_result = find_user_config();
    if (user_result.has_value()) {
        std::string user_path = *user_result;
        if (user_path != system_path) {
            auto user_cfg_result = load_config(user_path);
            if (user_cfg_result.has_value()) {
                Config user_cfg = *user_cfg_result;
                apply_user_overrides(cfg, user_cfg);
            } else {
                // Log but don't fail on user config errors
            }
        }
    }

    if (cfg.log_file.empty())
        cfg.log_file = expand_path("~/.ragger/activity.log");

    g_config = &cfg;
}

int reload_config() {
    if (!g_config) return 0;

    // Re-read from the same files
    std::string sys_path, user_path;
    auto sys_result = find_system_config("");
    if (!sys_result.has_value()) {
        return 0;
    }
    sys_path = *sys_result;
    
    auto user_result = find_user_config();
    if (user_result.has_value()) {
        user_path = *user_result;
    }

    auto fresh_result = load_config(sys_path);
    if (!fresh_result.has_value()) {
        return 0;
    }
    Config fresh = *fresh_result;
    if (user_result.has_value()) {
        std::string user_path = *user_result;
        if (!user_path.empty() && user_path != sys_path) {
            auto user_cfg_result = load_config(user_path);
            if (user_cfg_result.has_value()) {
                Config user_cfg = *user_cfg_result;
                apply_user_overrides(fresh, user_cfg);
            }
        }
    }

    Config& cfg = *g_config;
    int changes = 0;

    // Keys that require restart (log but don't apply)
    auto warn_restart = [&](const std::string& name, bool changed) {
        if (changed) {
            std::cerr << std::format(ragger::lang::WARN_CONFIG_RESTART, name) << "\n";
        }
    };
    warn_restart("port", fresh.port != cfg.port);
    warn_restart("db_path", fresh.db_path != cfg.db_path);
    warn_restart("tls_cert", fresh.tls_cert != cfg.tls_cert);
    warn_restart("tls_key", fresh.tls_key != cfg.tls_key);
    warn_restart("embedding_model", fresh.embedding_model != cfg.embedding_model);
    warn_restart("embedding_dimensions", fresh.embedding_dimensions != cfg.embedding_dimensions);

    // Hot-reloadable fields
    #define RELOAD(field) do { \
        if (cfg.field != fresh.field) { \
            cfg.field = fresh.field; \
            ++changes; \
        } \
    } while(0)

    // Search
    RELOAD(default_search_limit);
    RELOAD(default_min_score);
    RELOAD(bm25_enabled);
    RELOAD(bm25_weight);
    RELOAD(vector_weight);

    // Inference
    RELOAD(inference_model);
    RELOAD(inference_default);
    RELOAD(inference_api_url);
    RELOAD(inference_api_key);
    RELOAD(inference_max_tokens);
    RELOAD(inference_memory_model);
    RELOAD(inference_memory_api_url);
    RELOAD(inference_memory_api_key);
    RELOAD(lm_proxy_url);
    // Endpoints: replace entirely if different
    if (cfg.inference_endpoints.size() != fresh.inference_endpoints.size()) {
        cfg.inference_endpoints = fresh.inference_endpoints;
        ++changes;
    } else {
        for (size_t i = 0; i < cfg.inference_endpoints.size(); ++i) {
            auto& a = cfg.inference_endpoints[i];
            auto& b = fresh.inference_endpoints[i];
            if (a.name != b.name || a.api_url != b.api_url || a.api_key != b.api_key ||
                a.models != b.models || a.format != b.format || a.max_context != b.max_context) {
                cfg.inference_endpoints = fresh.inference_endpoints;
                ++changes;
                break;
            }
        }
    }

    // Logging
    RELOAD(query_log_enabled);
    RELOAD(http_log_enabled);
    RELOAD(mcp_log_enabled);
    RELOAD(debug_log_enabled);

    // Paths
    RELOAD(normalize_home_path);

    // Web
    RELOAD(web_root);

    // Import
    RELOAD(minimum_chunk_size);

    // Embed (subprocess settings)
    RELOAD(embed_timeout_ms);
    RELOAD(embed_retries);
    RELOAD(embed_max_workers);

    // Model aliases
    if (cfg.model_aliases != fresh.model_aliases) {
        cfg.model_aliases = fresh.model_aliases;
        ++changes;
    }

    // Chat / housekeeping
    RELOAD(chat_store_turns);
    RELOAD(chat_summarize_on_pause);
    RELOAD(chat_pause_minutes);
    RELOAD(chat_summarize_on_quit);
    RELOAD(cleanup_max_age_hours);
    RELOAD(housekeeping_interval);
    RELOAD(chat_max_persona_chars);
    RELOAD(chat_max_memory_results);
    RELOAD(chat_persona_pct);
    RELOAD(chat_chars_per_token);
    RELOAD(chat_verbatim_turns);
    RELOAD(chat_verbatim_minutes);
    RELOAD(chat_summary_count);
    RELOAD(chat_summary_minutes);
    RELOAD(chat_search_min_score);
    RELOAD(chat_budget_persona_pct);
    RELOAD(chat_budget_session_pct);
    RELOAD(chat_budget_summaries_pct);
    RELOAD(chat_budget_search_pct);
    RELOAD(chat_budget_reserve_pct);
    RELOAD(chat_stream_flush_seconds);
    RELOAD(system_prompt_file);
    RELOAD(persona_dir);
    RELOAD(proxy_system_prompt);

    // System ceilings
    RELOAD(max_search_limit);
    RELOAD(chat_max_persona_chars_limit);
    RELOAD(chat_max_memory_results_limit);

    #undef RELOAD

    return changes;
}

// --- executable path (for spawning `ragger embed`) ---------------------
static std::string g_executable_path;

void set_executable_path(const std::string& argv0) {
    if (argv0.empty()) return;
    // Resolve to an absolute path when possible; realpath handles a relative
    // or PATH-less invocation. Fall back to the raw argv0.
    char buf[4096];
    if (::realpath(argv0.c_str(), buf)) g_executable_path = buf;
    else g_executable_path = argv0;
}

const std::string& executable_path() {
    static const std::string fallback = "ragger";  // resolved via PATH
    return g_executable_path.empty() ? fallback : g_executable_path;
}

} // namespace ragger
