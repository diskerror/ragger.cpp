/**
 * IEmbedProvider — common abstract interface for text-embedding providers.
 *
 * `Embedder` (in-process ONNX Runtime, or an HTTP call to an external
 * OpenAI-compatible endpoint) is the implementation. The interface was cut
 * when a second one, EmbedExecutor (a `ragger embed` subprocess pool), also
 * existed; that one was retired in 0.17. The contract it brought stays:
 * std::optional on failure (nullopt == failure), which is safer at call
 * sites than an empty-vector convention. Embedder adapts its encode()/
 * ready() to it through embed()/embed_batch().
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ragger {

class IEmbedProvider {
public:
    virtual ~IEmbedProvider() = default;

    /// Embed one text. Returns std::nullopt on any failure (disabled
    /// provider, model/endpoint error, subprocess timeout, etc). Never
    /// throws.
    virtual std::optional<std::vector<float>> embed(const std::string& text) const = 0;

    /// Embed many texts. Result is positional; failed items are
    /// std::nullopt. Implementations may parallelize internally.
    virtual std::vector<std::optional<std::vector<float>>>
    embed_batch(const std::vector<std::string>& texts) const = 0;

    /// True if this provider can currently produce embeddings.
    virtual bool ready() const = 0;

    /// Embedding dimensionality, or 0 if unknown/not yet determined.
    virtual int dimensions() const = 0;
};

} // namespace ragger
