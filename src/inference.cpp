/**
 * Inference client implementation
 */
#include "ragger/inference.h"
#include "ragger/config.h"
#include "ragger/api_formats.h"
#include "nlohmann_json.hpp"

#include <curl/curl.h>
#include <fnmatch.h>
#include <sstream>
#include <stdexcept>
#include <iostream>

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
    : model(model_), max_tokens(max_tokens_), _endpoints(endpoints) {}

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

    return InferenceClient(endpoints, cfg.inference_model, cfg.inference_max_tokens);
}

// -----------------------------------------------------------------------
// Model auto-load (LM Studio v1 API)
// -----------------------------------------------------------------------

/// CURL write callback for simple string collection
static size_t _simple_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

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

    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string response_buf;

    // Check if model is loaded
    std::string list_url = mgmt_base + "/models";
    curl_easy_setopt(curl, CURLOPT_URL, list_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _simple_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        if (res == CURLE_COULDNT_CONNECT || res == CURLE_OPERATION_TIMEDOUT) {
            return "Inference engine not reachable at " + mgmt_base;
        }
        return "";  // fail open
    }

    try {
        auto data = nlohmann::json::parse(response_buf);
        auto models = data.contains("models") ? data["models"] : data.value("data", nlohmann::json::array());

        for (const auto& m : models) {
            std::string key = m.value("key", m.value("id", std::string("")));
            bool loaded = false;
            if (m.contains("loaded_instances") && m["loaded_instances"].is_array()) {
                loaded = !m["loaded_instances"].empty();
            }
            if (key == use_model && loaded) {
                curl_easy_cleanup(curl);
                return "";  // already loaded
            }
        }
    } catch (...) {
        curl_easy_cleanup(curl);
        return "";  // can't parse, fail open
    }

    // Model not loaded — trigger load
    std::string load_url = mgmt_base + "/models/load";
    nlohmann::json load_body = {{"model", use_model}};
    std::string body_str = load_body.dump();
    response_buf.clear();

    curl_easy_setopt(curl, CURLOPT_URL, load_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res == CURLE_OPERATION_TIMEDOUT) {
        return "Model " + use_model + " load timed out";
    }
    if (res != CURLE_OK) {
        return "Failed to load model " + use_model;
    }

    return "";  // loaded successfully
}

void InferenceClient::set_forced_endpoint(const std::string& name) {
    if (name.empty()) {
        forced_endpoint_.clear();
        return;
    }
    for (auto& ep : _endpoints) {
        if (ep.name == name) {
            forced_endpoint_ = name;
            return;
        }
    }
    throw std::runtime_error("Unknown endpoint: " + name);
}

Endpoint& InferenceClient::resolve_endpoint(const std::string& model_name) {
    // If forced, use that endpoint
    if (!forced_endpoint_.empty()) {
        for (auto& ep : _endpoints) {
            if (ep.name == forced_endpoint_) return ep;
        }
    }
    for (auto& ep : _endpoints) {
        if (ep.matches(model_name)) {
            return ep;
        }
    }
    // Fallback to last endpoint if no match
    if (!_endpoints.empty()) {
        return _endpoints.back();
    }
    throw std::runtime_error("No inference endpoints configured");
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

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string buf;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
        +[](char* p, size_t s, size_t n, void* ud) -> size_t {
            static_cast<std::string*>(ud)->append(p, s * n);
            return s * n;
        });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

    if (!api_key.empty()) {
        struct curl_slist* hdrs = nullptr;
        std::string auth = "Authorization: Bearer " + api_key;
        hdrs = curl_slist_append(hdrs, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
        CURLcode res = curl_easy_perform(curl);
        long code = 0;
        if (res == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(curl);
        return res == CURLE_OK && code < 400;
    }

    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    if (res == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);
    return res == CURLE_OK && code < 400;
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

    CURL* curl = curl_easy_init();
    if (!curl) return result;

    std::string buf;
    struct curl_slist* hdrs = nullptr;
    if (!api_key.empty()) {
        std::string auth = "Authorization: Bearer " + api_key;
        hdrs = curl_slist_append(hdrs, auth.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
        +[](char* p, size_t s, size_t n, void* ud) -> size_t {
            static_cast<std::string*>(ud)->append(p, s * n);
            return s * n;
        });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    if (res == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (hdrs) curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || code >= 400) return result;

    try {
        auto json = nlohmann::json::parse(buf);
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
// HTTP helpers with libcurl
// -----------------------------------------------------------------------
struct WriteCallbackData {
    std::string* buffer;
};

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    auto* data = static_cast<WriteCallbackData*>(userdata);
    data->buffer->append(ptr, total);
    return total;
}

struct StreamCallbackData {
    std::function<void(const std::string&)>* on_token;
    std::string buffer;
    ApiFormat* format;
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

    // Setup libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    std::string response_buffer;
    WriteCallbackData write_data{&response_buffer};

    // Build headers using format
    struct curl_slist* headers = nullptr;
    for (const auto& [key, value] : build_headers(fmt, endpoint.api_key)) {
        std::string header = key + ": " + value;
        headers = curl_slist_append(headers, header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_data);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("HTTP request failed: ") +
                                 curl_easy_strerror(res));
    }

    if (http_code != 200) {
        throw std::runtime_error("Inference API error " + std::to_string(http_code) +
                                 ": " + response_buffer);
    }

    // Parse response using format
    try {
        auto response = nlohmann::json::parse(response_buffer);
        return extract_content(fmt, response);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse response: ") + e.what());
    }
}

void InferenceClient::chat_stream(const std::vector<Message>& messages,
                                  std::function<void(const std::string&)> on_token,
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
    nlohmann::json payload = build_request_body(fmt, api_messages, use_model, use_max_tokens, true);
    std::string body = payload.dump();

    // Setup libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    StreamCallbackData stream_data{&on_token, "", &fmt};

    // Build headers using format
    struct curl_slist* headers = nullptr;
    for (const auto& [key, value] : build_headers(fmt, endpoint.api_key)) {
        std::string header = key + ": " + value;
        headers = curl_slist_append(headers, header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // CURLE_PARTIAL_FILE is normal for SSE streams — server closes after [DONE]
    if (res != CURLE_OK && res != CURLE_PARTIAL_FILE) {
        throw std::runtime_error(std::string("HTTP request failed: ") +
                                 curl_easy_strerror(res));
    }

    if (http_code != 200) {
        throw std::runtime_error("Inference API error " + std::to_string(http_code));
    }
}

// -----------------------------------------------------------------------
// LM Proxy: OpenAI-compatible pass-through (if configured)
// -----------------------------------------------------------------------
namespace {
    InferenceClient::ProxyResponse forward_request(
            const std::string& lm_proxy_url,
            const std::string& path,
            const std::string& method,
            const std::string& body) {
        if (lm_proxy_url.empty()) {
            throw std::runtime_error("LM proxy not configured");
        }

        // Strip trailing slash from base URL
        std::string base = lm_proxy_url;
        if (!base.empty() && base.back() == '/') {
            base.pop_back();
        }

        std::string url = base + path;

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize libcurl for proxy");
        }

        std::string response_buffer;
        WriteCallbackData write_data{&response_buffer};

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_data);

        if (method == "POST" || method == "PUT") {
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        }

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error("Proxy connection failed");
        }

        return {http_code, std::move(response_buffer)};
    }
} // anonymous namespace

InferenceClient::ProxyResponse InferenceClient::proxy_request(
        const std::string& path,
        const std::string& method,
        const std::string& body) {
    if (lm_proxy_url_.empty()) {
        throw std::runtime_error("LM proxy not configured");
    }
    return forward_request(lm_proxy_url_, path, method, body);
}

std::vector<std::string> InferenceClient::proxy_list_models() {
    auto resp = proxy_request("/v1/models", "GET", "");
    if (resp.status_code != 200) {
        throw std::runtime_error("Upstream returned status " + std::to_string(resp.status_code));
    }
    try {
        auto json_response = nlohmann::json::parse(resp.body);
        std::vector<std::string> models;
        if (json_response.contains("data") && json_response["data"].is_array()) {
            for (const auto& item : json_response["data"]) {
                if (item.is_object() && item.contains("id")) {
                    models.push_back(item["id"].get<std::string>());
                }
            }
        }
        return models;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse upstream model list");
    }
}

} // namespace ragger
