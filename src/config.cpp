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
// Config file search
// -----------------------------------------------------------------------
std::string find_config_file(const std::string& cli_path) {
    // 1. /etc/ragger.conf
    if (fs::exists("/etc/ragger.conf")) {
        return "/etc/ragger.conf";
    }

    // 2. ~/.ragger/ragger.conf
    std::string user_conf = expand_path("~/.ragger/ragger.conf");
    if (fs::exists(user_conf)) {
        return user_conf;
    }

    // 3. --config-file= (required at this point)
    if (cli_path.empty()) {
        throw std::runtime_error(lang::ERR_CONFIG_NOT_FOUND);
    }

    std::string resolved = expand_path(cli_path);
    if (!fs::exists(resolved)) {
        throw std::runtime_error(
            std::string(lang::ERR_CONFIG_FILE_MISSING) + resolved);
    }

    return resolved;
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
        }
        else if (section == "storage") {
            if      (key == "db_path")            cfg.db_path = val;
            else if (key == "default_collection") cfg.default_collection = val;
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
    std::string path = find_config_file(cli_config_path);
    static Config cfg = load_config(path);
    g_config = &cfg;
    std::cerr << lang::MSG_CONFIG_LOADED << path << std::endl;
}

} // namespace ragger
