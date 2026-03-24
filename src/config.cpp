/**
 * Configuration loader — INI file parser
 */
#include "ragger/config.h"
#include "ragger/lang.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace ragger {
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------
std::string expand_path(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

std::string Config::resolved_db_path()   const { return expand_path(db_path); }
std::string Config::resolved_log_dir()   const { return expand_path(log_dir); }
std::string Config::resolved_model_dir() const {
    return expand_path(model_dir.empty() ? "~/.ragger/models" : model_dir);
}

// -----------------------------------------------------------------------
// Default config (embedded)
// -----------------------------------------------------------------------
static constexpr const char* DEFAULT_CONFIG = R"(# ragger.ini — Ragger Memory configuration
#
# Search order:
#   1. --config=<path>          (explicit override)
#   2. ~/.ragger/ragger.ini    (per-user default)
#
# First file found wins. Created automatically on first run.

[server]
host = 127.0.0.1
port = 8432

[storage]
db_path = ~/.ragger/memories.db
default_collection = memory
formats_dir = /var/ragger/formats

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384
# model_dir: path to ONNX model files (default: ~/.ragger/models)
# model_dir = ~/.ragger/models

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 3
vector_weight = 7
bm25_k1 = 1.5
bm25_b = 0.75

[inference]
# Default model for chat
model = claude-sonnet-4-5
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

[logging]
log_dir = ~/.ragger
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300
)";

// -----------------------------------------------------------------------
// Bootstrap ~/.ragger/ on first run
// -----------------------------------------------------------------------
static std::string bootstrap_user_config() {
    std::string ragger_dir = expand_path("~/.ragger");
    std::string conf_path  = ragger_dir + "/ragger.ini";

    fs::create_directories(ragger_dir);

    std::ofstream out(conf_path);
    if (!out.is_open()) {
        throw std::runtime_error(
            std::string(lang::ERR_CONFIG_OPEN) + conf_path);
    }
    out << DEFAULT_CONFIG;
    out.close();

    std::cerr << lang::MSG_CONFIG_CREATED << conf_path << std::endl;
    return conf_path;
}

// -----------------------------------------------------------------------
// Config file search
// -----------------------------------------------------------------------
std::string find_system_config(const std::string& cli_path) {
    // 1. Explicit --config= takes highest priority
    if (!cli_path.empty()) {
        std::string resolved = expand_path(cli_path);
        if (!fs::exists(resolved)) {
            throw std::runtime_error(
                std::string(lang::ERR_CONFIG_FILE_MISSING) + resolved);
        }
        return resolved;
    }

    // 2. /etc/ragger.ini
    if (fs::exists("/etc/ragger.ini")) {
        return "/etc/ragger.ini";
    }

    // 3. First run — bootstrap default user config (acts as system config)
    return bootstrap_user_config();
}

std::string find_user_config() {
    std::string user_conf = expand_path("~/.ragger/ragger.ini");
    if (fs::exists(user_conf)) {
        return user_conf;
    }
    return "";
}

// Server infrastructure keys — system config always wins (blacklist)
// Match the Python SERVER_LOCKED set
struct ServerLockedKey {
    const char* section;
    const char* key;
};

static const ServerLockedKey SERVER_LOCKED[] = {
    {"server", "host"},
    {"server", "port"},
    {"storage", "db_path"},
    {"storage", "formats_dir"},
    {"logging", "log_dir"},
    {"embedding", "model"},
    {"embedding", "dimensions"},
    {"embedding", "model_dir"},
    // System ceilings
    {"chat", "max_turn_retention_minutes"},
    {"chat", "max_turns_stored"},
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
    // ✗ logging.log_dir
    // ✗ embedding.model, embedding.dimensions, embedding.model_dir
    // ✓ Everything else
    
    // Storage (non-locked)
    cfg.default_collection = user.default_collection;
    
    // Search (all user-overridable)
    cfg.default_search_limit = user.default_search_limit;
    cfg.default_min_score = user.default_min_score;
    cfg.bm25_enabled = user.bm25_enabled;
    cfg.bm25_weight = user.bm25_weight;
    cfg.vector_weight = user.vector_weight;
    cfg.bm25_k1 = user.bm25_k1;
    cfg.bm25_b = user.bm25_b;
    
    // Inference (user can only pick model)
    cfg.inference_model = user.inference_model;
    cfg.inference_default = user.inference_default;
    
    // Logging (query_log, http_log, mcp_log are user-overridable)
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

Config load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(std::string(lang::ERR_CONFIG_OPEN) + path);
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
            if      (key == "host") cfg.host = val;
            else if (key == "port") cfg.port = std::stoi(val);
            else if (key == "single_user") cfg.single_user = parse_bool(val);
        }
        else if (section == "storage") {
            if      (key == "db_path")            cfg.db_path = val;
            else if (key == "default_collection") cfg.default_collection = val;
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
            else if (key == "bm25_k1")          cfg.bm25_k1 = std::stof(val);
            else if (key == "bm25_b")           cfg.bm25_b = std::stof(val);
            else if (key == "max_search_limit") cfg.max_search_limit = std::stoi(val);
        }
        else if (section == "inference") {
            if      (key == "model")      cfg.inference_model = val;
            else if (key == "api_url")    cfg.inference_api_url = val;
            else if (key == "api_key")    cfg.inference_api_key = val;
            else if (key == "max_tokens") cfg.inference_max_tokens = std::stoi(val);
            else if (key == "default")    cfg.inference_default = val;
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
        }
        else if (section == "logging") {
            if      (key == "log_dir")   cfg.log_dir = val;
            else if (key == "query_log") cfg.query_log_enabled = parse_bool(val);
            else if (key == "http_log")  cfg.http_log_enabled = parse_bool(val);
            else if (key == "mcp_log")   cfg.mcp_log_enabled = parse_bool(val);
        }
        else if (section == "paths") {
            if (key == "normalize_home") cfg.normalize_home_path = parse_bool(val);
        }
        else if (section == "import") {
            if (key == "minimum_chunk_size") cfg.minimum_chunk_size = std::stoi(val);
        }
        else if (section == "chat") {
            if (key == "store_turns") cfg.chat_store_turns = val;
            else if (key == "summarize_on_pause") cfg.chat_summarize_on_pause = parse_bool(val);
            else if (key == "pause_minutes") cfg.chat_pause_minutes = std::stoi(val);
            else if (key == "summarize_on_quit") cfg.chat_summarize_on_quit = parse_bool(val);
            else if (key == "max_turn_retention_minutes") cfg.chat_max_turn_retention_minutes = std::stoi(val);
            else if (key == "max_turns_stored") cfg.chat_max_turns_stored = std::stoi(val);
            else if (key == "max_persona_chars") cfg.chat_max_persona_chars = std::stoi(val);
            else if (key == "max_memory_results") cfg.chat_max_memory_results = std::stoi(val);
            else if (key == "persona_pct") cfg.chat_persona_pct = std::stoi(val);
            else if (key == "chars_per_token") cfg.chat_chars_per_token = std::stof(val);
            else if (key == "max_persona_chars_limit") cfg.chat_max_persona_chars_limit = std::stoi(val);
            else if (key == "max_memory_results_limit") cfg.chat_max_memory_results_limit = std::stoi(val);
        }
    }

    // Convert endpoint map to vector
    for (auto& [name, ep] : endpoint_map) {
        cfg.inference_endpoints.push_back(ep);
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

void init_config(const std::string& cli_config_path) {
    // Load system config first
    std::string system_path = find_system_config(cli_config_path);
    static Config cfg = load_config(system_path);
    std::cerr << lang::MSG_CONFIG_LOADED << system_path << std::endl;
    
    // Then overlay user-specific overrides
    std::string user_path = find_user_config();
    if (!user_path.empty() && user_path != system_path) {
        Config user_cfg = load_config(user_path);
        apply_user_overrides(cfg, user_cfg);
        std::cerr << "Applied user overrides from " << user_path << std::endl;
    }

    // single_user mode: force logs to user directory
    if (cfg.single_user) {
        cfg.log_dir = "~/.ragger";
    }
    
    g_config = &cfg;
}

} // namespace ragger
