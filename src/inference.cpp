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
        headers.push_back(util::bearer_header(api_key));
    }

    util::HttpClient http;
    auto resp = http.get(url, headers, /*timeout_sec=*/3);
    return resp.ok() && resp.status < 400;
}

bool Endpoint::is_local() const {
    // S14 (C m15): parse the host out of the URL — substring matching used
    // to call e.g. https://host110.example.com "local" because "10." appears
    // in the middle. Schemeless or empty URLs are never local.
    static constexpr std::string_view sep = "://";
    auto p = api_url.find(sep);
    if (p == sep.npos) return false;
    std::string authority = api_url.substr(p + sep.size());
    if (auto e = authority.find_first_of("/?#"); e != std::string::npos)
        authority.erase(e);
    // Strip userinfo: everything up to the last '@' of the authority.
    if (auto at = authority.rfind('@'); at != std::string::npos)
        authority.erase(0, at + 1);

    std::string host = authority;
    if (!host.empty() && host.front() == '[') {
        // IPv6 literal: the host is the bracket contents.
        auto close = host.find(']');
        host = close == std::string::npos ? host.substr(1)
                                          : host.substr(1, close - 1);
    } else if (auto c = host.find(':'); c != std::string::npos) {
        host.erase(c);  // strip :port
    }

    if (host == "localhost" || host == "0.0.0.0" || host == "::1")
        return true;
    if (host.starts_with("127.") || host.starts_with("10.") ||
        host.starts_with("192.168.")) {
        // Must be a real dotted quad: four numeric labels, each 0-255.
        size_t labels = 0;
        size_t start = 0;
        while (true) {
            auto dot = host.find('.', start);
            auto part = host.substr(start,
                dot == std::string::npos ? std::string::npos : dot - start);
            if (part.empty() || part.size() > 3 ||
                part.find_first_not_of("0123456789") != std::string::npos)
                return false;
            if (std::stoi(part) > 255) return false;
            ++labels;
            if (dot == std::string::npos) break;
            start = dot + 1;
        }
        return labels == 4;
    }
    return false;
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
        headers.push_back(util::bearer_header(api_key));
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

std::vector<Endpoint::TypedModel> Endpoint::list_models_lmstudio_typed() const {
    std::vector<TypedModel> result;

    // LM Studio's native REST API lives at {scheme}://{host}:{port}/api/v0/models
    // — a sibling of the OpenAI-compatible /v1 tree, not a path under it.
    // Strip any trailing /v1(/...) to get the server base, then append
    // /api/v0/models.
    std::string url = api_url;
    auto pos = url.find("/v1");
    std::string base = (pos != std::string::npos) ? url.substr(0, pos) : url;
    // Trim any trailing slash before appending.
    while (!base.empty() && base.back() == '/') base.pop_back();
    url = base + "/api/v0/models";

    std::vector<std::string> headers;
    if (!api_key.empty()) {
        headers.push_back(util::bearer_header(api_key));
    }

    util::HttpClient http;
    auto resp = http.get(url, headers, /*timeout_sec=*/5);

    // Non-LM-Studio servers (llama.cpp, Ollama, vLLM) will 404/timeout/refuse
    // here — that's expected and callers must fall back to heuristics.
    if (!resp.ok() || resp.status >= 400) return result;

    try {
        auto json = nlohmann::json::parse(resp.body);
        if (json.contains("data") && json["data"].is_array()) {
            for (const auto& m : json["data"]) {
                if (!m.contains("id") || !m["id"].is_string()) continue;
                TypedModel tm;
                tm.id = m["id"].get<std::string>();
                tm.type = m.value("type", "");
                result.push_back(std::move(tm));
            }
        }
    } catch (...) {
        result.clear();
    }

    return result;
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
    auto resp = http.post(url, body, headers, timeout_sec);

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
