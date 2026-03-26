/**
 * Inference client for OpenAI-compatible APIs
 *
 * Supports multi-endpoint routing via glob patterns.
 * Uses libcurl for HTTP (blocking + streaming SSE).
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>

namespace ragger {

/// Chat message
struct Message {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

/// One inference endpoint with its URL, key, and model glob patterns
struct Endpoint {
    std::string name;
    std::string api_url;
    std::string api_key;
    std::string models;  // comma-separated glob patterns
    std::string format_name;  // API format name (openai, anthropic, etc.)

    Endpoint(const std::string& name_,
             const std::string& api_url_,
             const std::string& api_key_ = "",
             const std::string& models_ = "*",
             const std::string& format_ = "");

    /// Check if this endpoint handles the given model name (fnmatch-style glob)
    bool matches(const std::string& model) const;

    /// Check if endpoint is reachable (GET /models or /health, 3s timeout)
    bool is_reachable() const;

    /// Query available models from endpoint (GET /v1/models)
    std::vector<std::string> list_models() const;
};

/// Multi-endpoint inference client
class InferenceClient {
public:
    std::string model;
    int max_tokens;
    std::vector<Endpoint> _endpoints;

    InferenceClient(const std::vector<Endpoint>& endpoints,
                    const std::string& model = "",
                    int max_tokens = 4096);

    /// Build from config
    static InferenceClient from_config(const struct Config& cfg);

    /// Blocking chat (returns full response text)
    std::string chat(const std::vector<Message>& messages,
                     const std::string& model = "");

    /// Streaming chat (calls callback for each token)
    void chat_stream(const std::vector<Message>& messages,
                     std::function<void(const std::string&)> on_token,
                     const std::string& model = "");

    /// Force a specific endpoint by name (empty = auto-route)
    void set_forced_endpoint(const std::string& name);

    /// Get the currently forced endpoint name (empty = auto)
    const std::string& forced_endpoint() const { return forced_endpoint_; }

private:
    std::string forced_endpoint_;
    Endpoint& resolve_endpoint(const std::string& model);
};

} // namespace ragger
