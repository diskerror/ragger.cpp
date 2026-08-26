/**
 * Embedder — generate embeddings via ONNX Runtime (internal) or a remote
 * OpenAI-compatible /v1/embeddings endpoint (external).
 *
 * Internal mode mirrors ragger_memory/embedding.py from the Python version.
 * Uses ONNX models from ~/.ragger/models/ + HuggingFace tokenizer.
 *
 * External mode sends text to a remote endpoint and reads the returned vector.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ragger {

class Embedder {
public:
    /// Internal (ONNX) mode: construct with path to model directory
    /// (containing model.onnx + tokenizer.json).
    explicit Embedder(const std::string& model_dir);

    /// External mode: construct with endpoint details. Embeddings are
    /// computed by calling POST /v1/embeddings on the remote server.
    Embedder(const std::string& host, int port,
             const std::string& model_name,
             const std::string& api_key = "",
             int dimensions = 0);

    ~Embedder();

    /// Encode text to a normalized embedding vector.
    std::vector<float> encode(const std::string& text) const;

    /// Embedding dimensions. For internal mode, read from config/model.
    /// For external mode, determined by probing or from constructor arg.
    int dimensions() const;

    /// True if this embedder uses an external endpoint.
    bool is_external() const;

    /// (External only) Query the remote endpoint's /v1/models and return
    /// available model names. Returns empty vector on error or if internal.
    std::vector<std::string> list_remote_models() const;

    /// (External only) Probe the endpoint by sending a test embedding
    /// request. Returns the vector dimensionality, or 0 on failure.
    int probe_dimensions() const;

    /// (External only) The model name the server reported in its last
    /// embedding response. May differ from the requested model when the
    /// endpoint (llama-swap, LM Studio) serves whatever is loaded.
    std::string last_served_model() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
