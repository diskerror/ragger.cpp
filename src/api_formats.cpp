/**
 * API format loader and request/response transformers
 */
#include "ragger/api_formats.h"
#include "ragger/lang.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>
#include <format>

namespace ragger {
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Hardcoded OpenAI default — the "no file needed" format
// -----------------------------------------------------------------------
static const ApiFormat OPENAI_FORMAT = {
    "/chat/completions",        // path
    "bearer",                   // auth
    "Authorization",            // auth_header
    "Bearer ",                  // auth_prefix
    {},                         // auth_extra
    "openai",                   // request_transform
    "choices[0].message.content", // response_content
    "choices[0].delta.content", // stream_content
    "",                         // stream_type_field
    "",                         // stream_type_value
    "[DONE]"                    // stream_stop
};

// -----------------------------------------------------------------------
// Format search dirs
// -----------------------------------------------------------------------
static std::string _system_formats_dir = "/var/ragger/formats";

void init_formats_dir(const std::string& path) {
    _system_formats_dir = path;
    // Expand ~ if present
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            _system_formats_dir = std::string(home) + path.substr(1);
        }
    }
}

static std::vector<std::string> _format_search_dirs() {
    std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
    std::vector<std::string> dirs;
    
    if (!home.empty()) {
        dirs.push_back(home + "/.ragger/formats");
    }
    dirs.push_back(_system_formats_dir);
    
    // Package formats dir: try relative to executable and a few hardcoded paths
    // In production, this would be ../formats from the binary
    // For development, check the source tree
    std::string source_formats = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") 
                                 + "/CLionProjects/Ragger/formats";
    if (fs::exists(source_formats)) {
        dirs.push_back(source_formats);
    }
    
    return dirs;
}

// -----------------------------------------------------------------------
// Cache
// -----------------------------------------------------------------------
static std::map<std::string, ApiFormat> _cache;

/// Get a string from JSON, treating null as the default value.
static std::string json_string(const nlohmann::json& j, const std::string& key,
                               const std::string& def = "") {
    if (!j.contains(key) || j[key].is_null()) return def;
    return j[key].get<std::string>();
}

static std::optional<ApiFormat> _load_format_file(const std::string& name) {
    for (const auto& dir : _format_search_dirs()) {
        std::string path = dir + "/" + name + ".json";
        if (fs::exists(path)) {
            try {
                std::ifstream file(path);
                nlohmann::json j;
                file >> j;
                
                ApiFormat fmt;
                fmt.path = json_string(j, "path", OPENAI_FORMAT.path);
                fmt.auth = json_string(j, "auth", OPENAI_FORMAT.auth);
                fmt.auth_header = json_string(j, "auth_header", OPENAI_FORMAT.auth_header);
                fmt.auth_prefix = json_string(j, "auth_prefix", OPENAI_FORMAT.auth_prefix);
                fmt.request_transform = json_string(j, "request_transform", OPENAI_FORMAT.request_transform);
                fmt.response_content = json_string(j, "response_content", OPENAI_FORMAT.response_content);
                fmt.stream_content = json_string(j, "stream_content", OPENAI_FORMAT.stream_content);
                fmt.stream_type_field = json_string(j, "stream_type_field", "");
                fmt.stream_type_value = json_string(j, "stream_type_value", "");
                fmt.stream_stop = json_string(j, "stream_stop", OPENAI_FORMAT.stream_stop);
                
                // Load auth_extra as a map
                if (j.contains("auth_extra") && j["auth_extra"].is_object()) {
                    for (auto& [key, val] : j["auth_extra"].items()) {
                        if (val.is_string()) {
                            fmt.auth_extra[key] = val.get<std::string>();
                        }
                    }
                }
                
                // Loaded silently
                return fmt;
            } catch (const std::exception& e) {
                std::cerr << std::format(ragger::lang::WARN_FORMAT_LOAD_FAILED,
                                         path, e.what()) << std::endl;
            }
        }
    }
    return std::nullopt;
}

ApiFormat get_format(const std::string& name) {
    std::string use_name = name.empty() ? "openai" : name;
    
    // Check cache
    auto it = _cache.find(use_name);
    if (it != _cache.end()) {
        return it->second;
    }
    
    // Try to load
    auto fmt = _load_format_file(use_name);
    if (!fmt) {
        if (use_name == "openai") {
            // Fallback to hardcoded default
            _cache[use_name] = OPENAI_FORMAT;
            return OPENAI_FORMAT;
        }
        
        std::stringstream ss;
        ss << "Unknown API format '" << use_name << "'. No " << use_name 
           << ".json found in search dirs.";
        throw std::runtime_error(ss.str());
    }
    
    _cache[use_name] = *fmt;
    return *fmt;
}

std::vector<std::string> list_formats() {
    std::set<std::string> names;
    names.insert("openai");
    
    for (const auto& dir : _format_search_dirs()) {
        if (fs::exists(dir) && fs::is_directory(dir)) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    names.insert(entry.path().stem().string());
                }
            }
        }
    }
    
    return std::vector<std::string>(names.begin(), names.end());
}

std::string detect_format(const std::string& api_url) {
    if (api_url.find("anthropic.com") != std::string::npos) {
        return "anthropic";
    }
    return "openai";
}

// -----------------------------------------------------------------------
// Request builders
// -----------------------------------------------------------------------
static nlohmann::json _build_openai_body(
    const std::vector<ApiMessage>& messages,
    const std::string& model,
    int max_tokens,
    bool stream
) {
    nlohmann::json body;
    body["model"] = model;
    body["max_tokens"] = max_tokens;
    body["stream"] = stream;
    
    nlohmann::json msgs = nlohmann::json::array();
    for (const auto& msg : messages) {
        msgs.push_back({
            {"role", msg.role},
            {"content", msg.content}
        });
    }
    body["messages"] = msgs;
    
    return body;
}

static nlohmann::json _build_anthropic_body(
    const std::vector<ApiMessage>& messages,
    const std::string& model,
    int max_tokens,
    bool stream
) {
    std::string system_text;
    nlohmann::json chat_messages = nlohmann::json::array();
    
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            if (!system_text.empty()) {
                system_text += "\n\n";
            }
            system_text += msg.content;
        } else {
            chat_messages.push_back({
                {"role", msg.role},
                {"content", msg.content}
            });
        }
    }
    
    nlohmann::json body;
    body["model"] = model;
    body["messages"] = chat_messages;
    body["max_tokens"] = max_tokens;
    
    if (!system_text.empty()) {
        body["system"] = system_text;
    }
    if (stream) {
        body["stream"] = true;
    }
    
    return body;
}

nlohmann::json build_request_body(
    const ApiFormat& fmt,
    const std::vector<ApiMessage>& messages,
    const std::string& model,
    int max_tokens,
    bool stream
) {
    if (fmt.request_transform == "anthropic") {
        return _build_anthropic_body(messages, model, max_tokens, stream);
    } else {
        return _build_openai_body(messages, model, max_tokens, stream);
    }
}

// -----------------------------------------------------------------------
// Path extraction for dot-bracket paths
// -----------------------------------------------------------------------
struct PathPart {
    enum Type { KEY, INDEX };
    Type type;
    std::string key;
    int index;
};

static std::vector<PathPart> _split_path(const std::string& path) {
    std::vector<PathPart> parts;
    std::string current;
    
    for (size_t i = 0; i < path.size(); ++i) {
        char c = path[i];
        
        if (c == '.') {
            if (!current.empty()) {
                parts.push_back({PathPart::KEY, current, 0});
                current.clear();
            }
        } else if (c == '[') {
            if (!current.empty()) {
                parts.push_back({PathPart::KEY, current, 0});
                current.clear();
            }
            // Extract index
            std::string idx_str;
            i++;
            while (i < path.size() && path[i] != ']') {
                idx_str += path[i];
                i++;
            }
            parts.push_back({PathPart::INDEX, "", std::stoi(idx_str)});
        } else {
            current += c;
        }
    }
    
    if (!current.empty()) {
        parts.push_back({PathPart::KEY, current, 0});
    }
    
    return parts;
}

static std::optional<nlohmann::json> _extract_path(
    const nlohmann::json& data,
    const std::string& path
) {
    nlohmann::json obj = data;
    
    for (const auto& part : _split_path(path)) {
        if (obj.is_null()) {
            return std::nullopt;
        }
        
        if (part.type == PathPart::INDEX) {
            if (obj.is_array() && part.index >= 0 && 
                static_cast<size_t>(part.index) < obj.size()) {
                obj = obj[part.index];
            } else {
                return std::nullopt;
            }
        } else {
            if (obj.is_object() && obj.contains(part.key)) {
                obj = obj[part.key];
            } else {
                return std::nullopt;
            }
        }
    }
    
    return obj;
}

std::string extract_content(const ApiFormat& fmt, const nlohmann::json& response) {
    auto result = _extract_path(response, fmt.response_content);
    if (result && result->is_string()) {
        return result->get<std::string>();
    }
    return "";
}

std::string extract_stream_delta(const ApiFormat& fmt, const nlohmann::json& chunk) {
    // Check type filter if specified
    if (!fmt.stream_type_field.empty() && !fmt.stream_type_value.empty()) {
        if (!chunk.contains(fmt.stream_type_field) ||
            chunk[fmt.stream_type_field] != fmt.stream_type_value) {
            return "";
        }
    }
    
    auto result = _extract_path(chunk, fmt.stream_content);
    if (result && result->is_string()) {
        return result->get<std::string>();
    }
    return "";
}

bool is_stream_stop(const ApiFormat& fmt, const std::string& line, const nlohmann::json* chunk) {
    // Check line format: "data: [DONE]"
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    
    if (trimmed == "data: " + fmt.stream_stop) {
        return true;
    }
    
    // Check chunk type field
    if (chunk && chunk->is_object()) {
        std::string type_field = fmt.stream_type_field.empty() ? "type" : fmt.stream_type_field;
        if (chunk->contains(type_field) && (*chunk)[type_field] == fmt.stream_stop) {
            return true;
        }
    }
    
    return false;
}

// -----------------------------------------------------------------------
// Header builder
// -----------------------------------------------------------------------
std::map<std::string, std::string> build_headers(const ApiFormat& fmt, const std::string& api_key) {
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    
    if (!api_key.empty()) {
        if (fmt.auth == "bearer") {
            headers[fmt.auth_header] = fmt.auth_prefix + api_key;
        } else if (fmt.auth == "header") {
            headers[fmt.auth_header] = fmt.auth_prefix + api_key;
        }
    }
    
    // Add extra headers
    for (const auto& [key, val] : fmt.auth_extra) {
        headers[key] = val;
    }
    
    return headers;
}

} // namespace ragger
