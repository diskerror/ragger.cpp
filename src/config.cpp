/**
 * Configuration loader — INI file parser
 */
#include "ragger/config.h"
#include "ragger/config_access.h"
#include "ragger/lang.h"
#include "ragger/util/fs.h"

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
#include <arpa/inet.h>
#include <cctype>


namespace ragger {
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------
// expand_path() only expands a leading "~" to the real $HOME — for
// user-supplied paths (CLI --db/--config args, settings.ini socket_path /
// bind_address). Ragger's own on-disk footprint (DB, settings.ini, logs,
// models, recipes, formats, socket default, token, stats.db,
// agent-memory-instructions.md) is never built through here — each has a
// dedicated Config::resolved_XX() below, hardcoded relative to
// ragger_base_dir() (see util/fs.h), so pointing --ragger-base at a
// throwaway directory relocates the entire footprint with nothing else to
// configure.
std::string expand_path(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    std::string home = home_dir();
    if (home.empty()) return path;
    return home + path.substr(1);
}

std::string Config::resolved_db_path() const {
    return ragger_base_dir() + "/memories.db";
}

std::string Config::resolved_model_dir() const {
    return ragger_base_dir() + "/models/" + resolve_model(embedding_model);
}

std::string Config::resolved_recipes_dir() const {
    return ragger_base_dir() + "/recipes";
}

std::string Config::resolved_formats_dir() const {
    return ragger_base_dir() + "/formats";
}

std::string Config::resolved_settings_path() const {
    return ragger_base_dir() + "/settings.ini";
}

std::string Config::resolved_token_path() const {
    return ragger_base_dir() + "/token";
}

std::string Config::resolved_stats_db_path() const {
    return ragger_base_dir() + "/stats.db";
}

std::string Config::resolved_agent_instructions_path() const {
    return ragger_base_dir() + "/agent-memory-instructions.md";
}

std::string Config::resolved_log_file_path() const {
    return ragger_base_dir() + "/logs/activity.log";
}

std::string Config::resolved_socket_path() const {
    return ragger_base_dir() + "/ragger.sock";
}

std::string Config::resolve_model(const std::string& name) const {
    // [models] aliasing removed — pass-through kept so call sites (embedding
    // dir resolution, summarizer model) don't need to change if aliasing is
    // reintroduced later.
    return name;
}

// -----------------------------------------------------------------------
// Default config (embedded) — generated at build time from
// default-settings.txt (single source of truth); see cmake/embed_ini.cmake.
// -----------------------------------------------------------------------
#include "default_config.inc"

// -----------------------------------------------------------------------
// Bootstrap ~/.ragger/ on first run
// -----------------------------------------------------------------------
static std::string bootstrap_user_config() {
    std::string ragger_dir = ragger_base_dir();
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
    std::string user_conf = ragger_base_dir() + "/settings.ini";
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
    {"embedding", "model"},
    {"embedding", "dimensions"},
    {"embedding", "vector_type"},
    // System ceilings
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

/// Validate a bind address: must be a valid IPv4/IPv6 literal, or a
/// hostname made of alnum/hyphen/dot characters. Throws std::runtime_error
/// on anything else — a malformed bind value is a startup error, not a
/// silent fallback, since silently ignoring it could bind somewhere the
/// user didn't intend.
static void validate_bind_address(const std::string& addr) {
    if (addr.empty()) return;  // empty = TCP listener disabled, always fine

    struct in_addr  a4{};
    struct in6_addr a6{};
    if (inet_pton(AF_INET, addr.c_str(), &a4) == 1) return;
    if (inet_pton(AF_INET6, addr.c_str(), &a6) == 1) return;

    // Hostname: alnum, '-', '.' only; must not start/end with '.' or '-'.
    bool looks_like_hostname = !addr.empty() &&
        std::all_of(addr.begin(), addr.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '.';
        }) &&
        addr.front() != '.' && addr.front() != '-' &&
        addr.back()  != '.' && addr.back()  != '-';

    if (!looks_like_hostname) {
        throw std::runtime_error(
            "invalid [server] bind address: \"" + addr +
            "\" (expected an IPv4/IPv6 literal or hostname)");
    }
}

void apply_user_overrides(Config& cfg, const Config& user) {
    // New pattern: user config overrides everything EXCEPT server-locked fields
    // Server-locked fields stay as loaded from system config
    
    // User can override everything except SERVER_LOCKED:
    // ✗ server.host, server.port
    // ✗ embedding.model, embedding.dimensions, embedding.vector_type
    // ✓ Everything else
    
    // Search (all user-overridable)
    cfg.default_search_limit = user.default_search_limit;
    cfg.default_min_score = user.default_min_score;
    cfg.bm25_enabled = user.bm25_enabled;
    cfg.bm25_weight = user.bm25_weight;
    cfg.vector_weight = user.vector_weight;
    cfg.phon_weight = user.phon_weight;

    // Inference (user can only pick model)
    cfg.inference_model = user.inference_model;
    cfg.inference_default = user.inference_default;
    
    // Paths (all user-overridable)
    cfg.normalize_home_path = user.normalize_home_path;
    
    // Import (all user-overridable)
    cfg.minimum_chunk_size = user.minimum_chunk_size;

    // Search (reserved toggle)
    cfg.inject_data = user.inject_data;

    // Apply system ceilings
    clamp_to_ceiling(cfg.default_search_limit, cfg.max_search_limit);
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

    // Alias tracking: episode_idle_minutes supersedes the deprecated
    // summary_pause_minutes. If the new key is never set explicitly but the
    // old one is present, fall back to the old value after the parse loop.
    bool episode_idle_set = false;
    bool summary_pause_set = false;

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

        // Surrounding quotes around a value are stripped so that
        //   socket = ''        (or "")
        // reads as an empty string — same as a missing/commented line.
        auto strip_quotes = [](std::string s) -> std::string {
            if (s.size() >= 2 &&
                ((s.front() == '"'  && s.back() == '"') ||
                 (s.front() == '\'' && s.back() == '\''))) {
                return s.substr(1, s.size() - 2);
            }
            return s;
        };

        // Map to config fields
        if (section == "server") {
            if      (key == "socket_enable") cfg.socket_enabled = parse_bool(val);
            else if (key == "socket") cfg.socket_enabled = !strip_quotes(val).empty();  // legacy key, back-compat
            else if (key == "tcp_enable") cfg.tcp_enabled = parse_bool(val);
            else if (key == "bind") cfg.bind_address = strip_quotes(val);
            else if (key == "port") cfg.port = std::stoi(val);
            else if (key == "server_name" || key == "hostname") cfg.server_name = val;
            else if (key == "capture_turns") cfg.capture_turns = parse_bool(val);
            else if (key == "build_context") cfg.build_context = parse_bool(val);
            else if (key == "default_recipe") cfg.default_recipe = val;
            else if (key == "cert" || key == "tls_cert") cfg.tls_cert = val;
            else if (key == "key" || key == "tls_key") cfg.tls_key = val;
        }
        else if (section == "embedding") {
            if      (key == "model")      cfg.embedding_model = val;
            else if (key == "dimensions") cfg.embedding_dimensions = std::stoi(val);
            else if (key == "vector_type") cfg.embedding_vector_type = val;
        }
        else if (section == "search") {
            if      (key == "default_limit")    cfg.default_search_limit = std::stoi(val);
            else if (key == "default_min_score") cfg.default_min_score = std::stof(val);
            else if (key == "bm25_enabled")     cfg.bm25_enabled = parse_bool(val);
            else if (key == "bm25_weight")      cfg.bm25_weight = std::stof(val);
            else if (key == "vector_weight")    cfg.vector_weight = std::stof(val);
            else if (key == "phon_weight")      cfg.phon_weight = std::stof(val);
            else if (key == "max_search_limit") cfg.max_search_limit = std::stoi(val);
            else if (key == "inject_data")      cfg.inject_data = parse_bool(val);
        }
        else if (section == "inference") {
            if      (key == "model")      cfg.inference_model = val;
            else if (key == "api_url")    cfg.inference_api_url = val;
            else if (key == "api_key")    cfg.inference_api_key = val;
            else if (key == "max_tokens") cfg.inference_max_tokens = std::stoi(val);
            else if (key == "default")    cfg.inference_default = val;
        }
        else if (section == "summarizer") {
            if      (key == "model")        cfg.summarizer_model        = val;
            else if (key == "api_url")      cfg.summarizer_api_url      = val;
            else if (key == "api_key")      cfg.summarizer_api_key      = val;
            else if (key == "max_tokens")   cfg.summarizer_max_tokens   = std::stoi(val);
            else if (key == "target_pct")   cfg.summarizer_target_pct  = std::stoi(val);
            else if (key == "max_pct")      cfg.summarizer_max_pct     = std::stoi(val);
            else if (key == "prompt")       cfg.summarizer_prompt       = val;
            else if (key == "episode_idle_minutes") {
                int v = std::stoi(val);
                if (v > 0) { cfg.episode_idle_minutes = v; episode_idle_set = true; }
            }
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
            if (key == "log_level") cfg.log_level = val;
            else if (key == "log_max_size_mb") {
                long v = std::stol(val);
                if (v >= 0) cfg.log_max_size_mb = v;
            }
            else if (key == "log_max_age_days") {
                int v = std::stoi(val);
                if (v >= 0) cfg.log_max_age_days = v;
            }
        }
        else if (section == "paths") {
            if (key == "normalize_home") cfg.normalize_home_path = parse_bool(val);
        }
        else if (section == "tls" || section == "ssl") {
            // Legacy standalone section — kept silently for back-compat.
            // TLS now lives under [server] (cert=/key=); see above.
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
        else if (section == "housekeeping") {
            if (key == "cleanup_max_age_hours") cfg.cleanup_max_age_hours = std::stof(val);
            else if (key == "housekeeping_interval") {
                int v = std::stoi(val);
                cfg.housekeeping_interval = (v == 0) ? 0 : std::max(v, 10);
            }
            else if (key == "summary_pause_minutes") {
                cfg.summary_pause_minutes = std::stoi(val);
                summary_pause_set = true;
            }
            else if (key == "episode_idle_minutes") {
                int v = std::stoi(val);
                if (v > 0) { cfg.episode_idle_minutes = v; episode_idle_set = true; }
            }
            else if (key == "episode_threshold_base") {
                cfg.episode_threshold_base = std::stof(val);
            }
            else if (key == "episode_threshold_cap") {
                cfg.episode_threshold_cap = std::stof(val);
            }
            else if (key == "episode_threshold_step") {
                cfg.episode_threshold_step = std::stof(val);
            }
            else if (key == "episode_step_minutes") {
                cfg.episode_step_minutes = std::stof(val);
            }
            else if (key == "catch_up_batch_size") {
                int v = std::stoi(val);
                if (v > 0) cfg.catch_up_batch_size = v;
            }
            else if (key == "project_gap_days") {
                int v = std::stoi(val);
                if (v > 0) cfg.project_gap_days = v;
            }
            else if (key == "max_turn_failures") {
                int v = std::stoi(val);
                if (v >= 0) cfg.max_turn_failures = v;
            }
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

    // Deprecated-alias fallback: if episode_idle_minutes was never set but the
    // old summary_pause_minutes is present, adopt it. Keep summary_pause_minutes
    // itself in sync with the effective idle gap so any lingering reader agrees.
    if (!episode_idle_set && summary_pause_set && cfg.summary_pause_minutes > 0) {
        cfg.episode_idle_minutes = cfg.summary_pause_minutes;
    } else {
        cfg.summary_pause_minutes = cfg.episode_idle_minutes;
    }

    // Fallback: if neither listener is configured, bring up TCP on loopback
    // so the daemon always has at least one listener.
    if (!cfg.socket_enabled && !cfg.tcp_enabled) {
        cfg.tcp_enabled = true;
        if (cfg.bind_address.empty()) cfg.bind_address = "127.0.0.1";
    }

    // Expand bind_address if it starts with ~ (rare, but historically
    // supported for host-as-path oddities; kept for back-compat).
    if (!cfg.bind_address.empty() && cfg.bind_address[0] == '~') {
        // For bind_address, expand ~ to home but keep the host part
        std::string home = expand_path("~");
        if (home.back() != '/') home += '/';
        cfg.bind_address = home + cfg.bind_address.substr(1);
    }

    // A malformed bind address is a startup error, not a silent fallback.
    validate_bind_address(cfg.bind_address);

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

    cfg.log_file = cfg.resolved_log_file_path();

    // DB is the source of truth for user config: overlay any rows present in
    // the settings table on top of the INI-seeded values. No-op if the DB
    // doesn't exist yet (first run before serve creates it).
    overlay_settings_from_db(cfg, cfg.resolved_db_path());

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
    warn_restart("socket_enabled", fresh.socket_enabled != cfg.socket_enabled);
    warn_restart("tcp_enabled", fresh.tcp_enabled != cfg.tcp_enabled);
    warn_restart("bind_address", fresh.bind_address != cfg.bind_address);
    warn_restart("tls_cert", fresh.tls_cert != cfg.tls_cert);
    warn_restart("tls_key", fresh.tls_key != cfg.tls_key);
    warn_restart("embedding_model", fresh.embedding_model != cfg.embedding_model);
    warn_restart("embedding_dimensions", fresh.embedding_dimensions != cfg.embedding_dimensions);
    warn_restart("embedding_vector_type", fresh.embedding_vector_type != cfg.embedding_vector_type);

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
    RELOAD(phon_weight);
    RELOAD(inject_data);

    // Inference
    RELOAD(inference_model);
    RELOAD(inference_default);
    RELOAD(inference_api_url);
    RELOAD(inference_api_key);
    RELOAD(inference_max_tokens);
    RELOAD(summarizer_model);
    RELOAD(summarizer_api_url);
    RELOAD(summarizer_api_key);
    RELOAD(summarizer_max_tokens);
    RELOAD(summarizer_target_pct);
    RELOAD(summarizer_max_pct);
    RELOAD(summarizer_prompt);
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
    // (log_level itself is not hot-reloadable — Diskerror::logger is
    // constructed once at startup with the level fixed.)

    // Paths
    RELOAD(normalize_home_path);

    // Import
    RELOAD(minimum_chunk_size);

    // Embed (subprocess settings)
    RELOAD(embed_timeout_ms);
    RELOAD(embed_retries);
    RELOAD(embed_max_workers);

    // Housekeeping / retention
    RELOAD(cleanup_max_age_hours);
    RELOAD(housekeeping_interval);
    RELOAD(summary_pause_minutes);
    RELOAD(episode_idle_minutes);
    RELOAD(catch_up_batch_size);
    RELOAD(project_gap_days);
    RELOAD(max_turn_failures);
    RELOAD(capture_turns);
    RELOAD(build_context);
    RELOAD(default_recipe);

    // System ceilings
    RELOAD(max_search_limit);

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

// --- full argv (for in-place re-exec / restart) ------------------------
static std::vector<std::string> g_full_argv;

void set_full_argv(int argc, char** argv) {
    g_full_argv.clear();
    for (int i = 0; i < argc; ++i) g_full_argv.emplace_back(argv[i]);
}

const std::vector<std::string>& full_argv() { return g_full_argv; }

} // namespace ragger
