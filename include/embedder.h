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
#include <optional>
#include <string>
#include <vector>

#include "embed_provider.h"

namespace ragger {

class Embedder : public IEmbedProvider {
public:
    /// Disabled mode. Constructs successfully but cannot embed: ready() is
    /// false, dimensions() is 0, and encode() returns an empty vector.
    ///
    /// This exists so a bad embedding-model configuration degrades instead of
    /// killing the daemon. Ragger keeps accepting and recording conversations;
    /// the rows are stored with a NULL embedding and the housekeeping backfill
    /// picks them up once the configuration is corrected. Callers that write
    /// embeddings must check ready() and defer when it is false — never store
    /// the empty vector, which would encode as a zero-length blob and read
    /// back as a real (but meaningless) embedding.
    Embedder();

    /// Internal (ONNX) mode: construct with path to model directory
    /// (containing model.onnx + tokenizer.json).
    explicit Embedder(const std::string& model_dir);

    /// External mode: construct with endpoint details. Embeddings are
    /// computed by calling POST /v1/embeddings on the remote server.
    Embedder(const std::string& host, int port,
             const std::string& model_name,
             const std::string& api_key = "",
             int dimensions = 0);

    ~Embedder() override;

    /// Encode text to a normalized embedding vector. Returns an empty
    /// vector on failure (disabled instance, or — external mode — a
    /// request error).
    std::vector<float> encode(const std::string& text) const;

    /// Embedding dimensions. For internal mode, read from config/model.
    /// For external mode, determined by probing or from constructor arg.
    int dimensions() const override;

    /// True if this embedder can actually produce embeddings. False only for
    /// the disabled instance built by the default constructor.
    bool ready() const override;

    // --- IEmbedProvider adapter ---------------------------------------
    // Thin wrappers over encode(): translate its empty-vector-on-failure
    // convention into IEmbedProvider's std::optional-on-failure contract.

    /// Embed one text; std::nullopt if !ready() or encode() fails.
    std::optional<std::vector<float>> embed(const std::string& text) const override;

    /// Embed many texts sequentially (Embedder has no subprocess pool to
    /// bound, unlike EmbedExecutor — batching here is just a loop).
    std::vector<std::optional<std::vector<float>>>
    embed_batch(const std::vector<std::string>& texts) const override;

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
