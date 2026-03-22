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
};

/// Multi-endpoint inference client
class InferenceClient {
public:
    std::string model;
    int max_tokens;
    std::vector<Endpoint> _endpoints;

    InferenceClient(const std::vector<Endpoint>& endpoints,
                    const std::string& model = "claude-sonnet-4-5",
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

private:
    Endpoint& resolve_endpoint(const std::string& model);
};

} // namespace ragger
