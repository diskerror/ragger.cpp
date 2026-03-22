/**
 * API format definitions for inference endpoints
 *
 * OpenAI-compatible is the hardcoded default — no file needed.
 * All other formats are loaded from JSON files.
 *
 * Search order for format files:
 *   1. ~/.ragger/formats/<name>.json       (user override)
 *   2. <formats_dir>/<name>.json           (from config, default /var/ragger/formats)
 *   3. <package>/formats/<name>.json       (shipped with Ragger)
 *
 * Custom formats: drop a .json file in ~/.ragger/formats/ or the configured
 * formats_dir and reference it by name (without .json): format = myformat
 */
#pragma once

#include "nlohmann_json.hpp"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace ragger {

/// API format definition
struct ApiFormat {
    std::string path;              // "/chat/completions" or "/messages"
    std::string auth;              // "bearer" or "header"
    std::string auth_header;       // "Authorization" or "x-api-key"
    std::string auth_prefix;       // "Bearer " or ""
    std::map<std::string, std::string> auth_extra;  // e.g. {"anthropic-version": "2023-06-01"}
    std::string request_transform; // "openai" or "anthropic"
    std::string response_content;  // "choices[0].message.content"
    std::string stream_content;    // "choices[0].delta.content"
    std::string stream_type_field; // "type" or empty
    std::string stream_type_value; // "content_block_delta" or empty
    std::string stream_stop;       // "[DONE]" or "message_stop"
};

/// Message for request building
struct ApiMessage {
    std::string role;
    std::string content;
};

/// Set the system formats directory from config. Call once at startup.
void init_formats_dir(const std::string& path);

/// Get a format definition by name. Returns hardcoded OpenAI if not found.
ApiFormat get_format(const std::string& name);

/// List available format names (openai + any .json files found)
std::vector<std::string> list_formats();

/// Auto-detect format from URL. Returns 'openai' as default.
std::string detect_format(const std::string& api_url);

/// Build request body according to the format's transform
nlohmann::json build_request_body(
    const ApiFormat& fmt,
    const std::vector<ApiMessage>& messages,
    const std::string& model,
    int max_tokens,
    bool stream = false
);

/// Extract content from a non-streaming response
std::string extract_content(const ApiFormat& fmt, const nlohmann::json& response);

/// Extract text from a streaming delta chunk, respecting type filters
std::string extract_stream_delta(const ApiFormat& fmt, const nlohmann::json& chunk);

/// Check if a stream line/chunk signals completion
bool is_stream_stop(const ApiFormat& fmt, const std::string& line, const nlohmann::json* chunk = nullptr);

/// Build HTTP headers for a request
std::map<std::string, std::string> build_headers(const ApiFormat& fmt, const std::string& api_key);

} // namespace ragger
