/**
 * Inference client implementation
 */
#include "inference.h"
#include "config.h"
#include "api_formats.h"
#include "lang.h"
#include "util/time.h"
#include "nlohmann_json.hpp"

#include "util/http_client.h"
#include <fnmatch.h>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <functional>
#include <string_view>

namespace fs = std::filesystem;

namespace ragger {

// -----------------------------------------------------------------------
// Endpoint
// -----------------------------------------------------------------------
Endpoint::Endpoint(const std::string& name_,
                   const std::string& api_url_,
                   const std::string& api_key_,
                   const std::string& models_,
                   const std::string& format_,
                   int max_tokens_)
    : name(name_), api_url(api_url_), api_key(api_key_), models(models_),
      format_name(format_), max_tokens(max_tokens_) {
    // Strip trailing slash from URL
    if (!api_url.empty() && api_url.back() == '/') {
        api_url = api_url.substr(0, api_url.size() - 1);
    }
    
    // Auto-detect format if not specified
    if (format_name.empty()) {
        format_name = detect_format(api_url);
    }
}

bool Endpoint::matches(const std::string& model) const {
    // Split comma-separated patterns
    std::vector<std::string> patterns;
    std::stringstream ss(models);
    std::string pattern;
    while (std::getline(ss, pattern, ',')) {
        // Trim whitespace
        pattern.erase(0, pattern.find_first_not_of(" \t"));
        pattern.erase(pattern.find_last_not_of(" \t") + 1);
        if (!pattern.empty()) {
            patterns.push_back(pattern);
        }
    }

    // Match against any pattern
    for (const auto& p : patterns) {
        if (fnmatch(p.c_str(), model.c_str(), 0) == 0) {
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------
// InferenceClient
// -----------------------------------------------------------------------
InferenceClient::InferenceClient(const std::vector<Endpoint>& endpoints,
                                 const std::string& model_,
                                 int max_tokens_)
    : model(model_), max_tokens(max_tokens_), endpoints(endpoints) {}

InferenceClient InferenceClient::from_config(const Config& cfg) {
    std::vector<Endpoint> endpoints;

    // Multi-endpoint from config
    for (const auto& ep : cfg.inference_endpoints) {
        endpoints.push_back(Endpoint(ep.name, ep.api_url, ep.api_key, ep.models, ep.format, ep.max_tokens));
    }

    // Single endpoint fallback from main [inference] section
    if (endpoints.empty() && !cfg.inference_api_url.empty()) {
        endpoints.push_back(Endpoint("default", cfg.inference_api_url,
                                     cfg.inference_api_key, "*", ""));
    }

    // If a default is named, move it to the end (fallback position)
    if (!cfg.inference_default.empty() && endpoints.size() > 1) {
        std::vector<Endpoint> named, others;
        for (auto& ep : endpoints) {
            if (ep.name == cfg.inference_default) {
                named.push_back(ep);
            } else {
                others.push_back(ep);
            }
        }
        endpoints = others;
        endpoints.insert(endpoints.end(), named.begin(), named.end());
    }

    // Summarizer endpoint (optional). When [summarizer] gives its own
    // api_url, prepend an endpoint so the summarizer model's glob is matched
    // before the general endpoints (first match wins). When no api_url is
    // given, the summarizer model is assumed served by one of the endpoints above.
    if (!cfg.summarizer_api_url.empty()) {
        std::string mm = cfg.summarizer_model.empty()
                             ? std::string("*") : cfg.summarizer_model;
        endpoints.insert(endpoints.begin(),
                         Endpoint("memory", cfg.summarizer_api_url,
                                  cfg.summarizer_api_key, mm, "",
                                  cfg.summarizer_max_tokens));
    }

    InferenceClient client(endpoints, cfg.inference_model, cfg.inference_max_tokens);
    client.memory_model = cfg.summarizer_model.empty()
                              ? client.model : cfg.summarizer_model;
    return client;
}

// -----------------------------------------------------------------------
// Model auto-load (LM Studio v1 API)
// -----------------------------------------------------------------------

std::string InferenceClient::ensure_model_loaded(const std::string& model_override) {
    std::string use_model = model_override.empty() ? model : model_override;
    if (use_model.empty()) return "";

    Endpoint* ep = nullptr;
    try {
        ep = &resolve_endpoint(use_model);
    } catch (...) {
        return "";  // no endpoint, fail open
    }

    // Derive management API from OpenAI-compat URL
    // e.g. http://localhost:1234/v1 → http://localhost:1234/api/v1
    std::string base = ep->api_url;
    // Strip trailing slash
    while (!base.empty() && base.back() == '/') base.pop_back();
    auto pos = base.rfind("/v1");
    if (pos == std::string::npos) return "";  // not a recognized local engine
    std::string mgmt_base = base.substr(0, pos) + "/api/v1";

    util::HttpClient http;

    // Check if model is loaded
    std::string list_url = mgmt_base + "/models";
    auto list_resp = http.get(list_url, {}, /*timeout_sec=*/5);

    if (!list_resp.ok()) {
        if (list_resp.result == CURLE_COULDNT_CONNECT || list_resp.result == CURLE_OPERATION_TIMEDOUT) {
            return std::format(lang::ERR_ENGINE_UNREACHABLE, mgmt_base);
        }
        return "";  // fail open
    }

    try {
        auto data = nlohmann::json::parse(list_resp.body);
        auto models = data.contains("models") ? data["models"] : data.value("data", nlohmann::json::array());

        for (const auto& m : models) {
            std::string key = m.value("key", m.value("id", std::string("")));
            bool loaded = false;
            if (m.contains("loaded_instances") && m["loaded_instances"].is_array()) {
                loaded = !m["loaded_instances"].empty();
            }
            if (key == use_model && loaded) {
                return "";  // already loaded
            }
        }
    } catch (...) {
        return "";  // can't parse, fail open
    }

    // Model not loaded — trigger load
    std::string load_url = mgmt_base + "/models/load";
    nlohmann::json load_body = {{"model", use_model}};
    std::string body_str = load_body.dump();

    auto load_resp = http.post(load_url, body_str,
                                {"Content-Type: application/json"}, /*timeout_sec=*/120);

    if (load_resp.result == CURLE_OPERATION_TIMEDOUT) {
        return std::format(lang::ERR_MODEL_LOAD_TIMEOUT, use_model, load_url);
    }
    if (!load_resp.ok()) {
        return std::format(lang::ERR_MODEL_LOAD_FAILED, use_model, load_url, load_resp.error);
    }

    return "";  // loaded successfully
}

void InferenceClient::set_forced_endpoint(const std::string& name) {
    if (name.empty()) {
        forced_endpoint_.clear();
        return;
    }
    for (auto& ep : endpoints) {
        if (ep.name == name) {
            forced_endpoint_ = name;
            return;
        }
    }
    throw std::runtime_error(std::format(lang::ERR_UNKNOWN_ENDPOINT, name));
}

Endpoint& InferenceClient::resolve_endpoint(const std::string& model_name) {
    // If forced, use that endpoint
    if (!forced_endpoint_.empty()) {
        for (auto& ep : endpoints) {
            if (ep.name == forced_endpoint_) return ep;
        }
    }
    for (auto& ep : endpoints) {
        if (ep.matches(model_name)) {
            return ep;
        }
    }
    // Fallback to last endpoint if no match
    if (!endpoints.empty()) {
        return endpoints.back();
    }
    throw std::runtime_error(lang::ERR_NO_ENDPOINTS);
}

// -----------------------------------------------------------------------
// Endpoint health / model listing
// -----------------------------------------------------------------------
bool Endpoint::is_reachable() const {
    // Try /v1/models first, fall back to base URL
    std::string url = api_url;
    auto pos = url.rfind("/v1");
    if (pos != std::string::npos)
        url = url.substr(0, pos + 3) + "/models";
    else
        url += "/models";

    std::vector<std::string> headers;
    if (!api_key.empty()) {
        headers.push_back("Authorization: *** " + api_key);
    }

    util::HttpClient http;
    auto resp = http.get(url, headers, /*timeout_sec=*/3);
    return resp.ok() && resp.status < 400;
}

bool Endpoint::is_local() const {
    // Local if URL contains localhost, 127.x, 192.168.x, 10.x, or has /v1 without an API key
    return api_url.find("localhost") != std::string::npos
        || api_url.find("127.0.0.") != std::string::npos
        || api_url.find("192.168.") != std::string::npos
        || api_url.find("10.") != std::string::npos
        || api_url.find("0.0.0.0") != std::string::npos;
}

std::vector<std::string> Endpoint::list_models() const {
    std::vector<std::string> result;

    std::string url = api_url;
    auto pos = url.rfind("/v1");
    if (pos != std::string::npos)
        url = url.substr(0, pos + 3) + "/models";
    else
        url += "/models";

    std::vector<std::string> headers;
    if (!api_key.empty()) {
        headers.push_back("Authorization: *** " + api_key);
    }

    util::HttpClient http;
    auto resp = http.get(url, headers, /*timeout_sec=*/5);

    if (!resp.ok() || resp.status >= 400) return result;

    try {
        auto json = nlohmann::json::parse(resp.body);
        if (json.contains("data") && json["data"].is_array()) {
            for (const auto& m : json["data"]) {
                if (m.contains("id") && m["id"].is_string()) {
                    result.push_back(m["id"].get<std::string>());
                }
            }
        }
    } catch (...) {}

    std::sort(result.begin(), result.end());
    return result;
}

// -----------------------------------------------------------------------
// SSE streaming parse helper (used by chat_stream, defined below)
// -----------------------------------------------------------------------
struct StreamCallbackData {
    std::function<void(const std::string&)>* on_token;
    std::string buffer;
    ApiFormat* format;
    std::string raw_response;   // accumulates every raw SSE line for dump
};

static size_t stream_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    auto* data = static_cast<StreamCallbackData*>(userdata);
    data->buffer.append(ptr, total);

    // Process complete lines (SSE format: "data: {...}\n")
    size_t pos;
    while ((pos = data->buffer.find('\n')) != std::string::npos) {
        std::string line = data->buffer.substr(0, pos);
        data->buffer.erase(0, pos + 1);

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r"));
        line.erase(line.find_last_not_of(" \t\r") + 1);

        if (line.empty()) continue;

        // Capture every raw SSE line for payload dump
        data->raw_response += line + "\n";

        // Check for stream stop
        if (is_stream_stop(*data->format, line)) continue;
        
        if (line.substr(0, 6) != "data: ") continue;

        std::string json_str = line.substr(6);
        try {
            auto chunk = nlohmann::json::parse(json_str);
            
            // Extract delta content using format
            std::string content = extract_stream_delta(*data->format, chunk);
            if (!content.empty() && data->on_token) {
                (*data->on_token)(content);
            }
        } catch (...) {
            // Skip malformed JSON
        }
    }

    return total;
}

std::string InferenceClient::chat_memory(const std::vector<Message>& messages) {
    return chat(messages, memory_model);
}

std::string InferenceClient::chat(const std::vector<Message>& messages,
                                  const std::string& model_override) {
    std::string use_model = model_override.empty() ? model : model_override;
    auto& endpoint = resolve_endpoint(use_model);

    // Load API format for this endpoint
    ApiFormat fmt = get_format(endpoint.format_name);
    
    std::string url = endpoint.api_url + fmt.path;

    // Convert Message to ApiMessage
    std::vector<ApiMessage> api_messages;
    for (const auto& msg : messages) {
        api_messages.push_back({msg.role, msg.content});
    }

    // Build request payload using format (endpoint max_tokens overrides global)
    int use_max_tokens = endpoint.max_tokens > 0 ? endpoint.max_tokens : max_tokens;
    nlohmann::json payload = build_request_body(fmt, api_messages, use_model, use_max_tokens, false);
    std::string body = payload.dump();

    // Build headers using format
    std::vector<std::string> headers;
    for (const auto& [key, value] : build_headers(fmt, endpoint.api_key)) {
        headers.push_back(key + ": " + value);
    }

    util::HttpClient http;
    auto resp = http.post(url, body, headers, /*timeout_sec=*/0);

    if (!resp.ok()) {
        throw std::runtime_error(std::format(lang::ERR_HTTP_REQUEST, resp.error));
    }

    if (resp.status != 200) {
        throw std::runtime_error(std::format(lang::ERR_INFERENCE_API, resp.status, resp.body));
    }

    // Parse response using format
    try {
        auto response = nlohmann::json::parse(resp.body);
        return extract_content(fmt, response);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::format(lang::ERR_PARSE_RESPONSE, e.what()));
    }
}



} // namespace ragger
