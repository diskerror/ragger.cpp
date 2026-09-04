/**
 * IEmbedProvider — common abstract interface for text-embedding providers.
 *
 * Ragger has two ways to turn text into a vector: `Embedder` (in-process
 * ONNX Runtime, or an HTTP call to an external OpenAI-compatible endpoint)
 * and `EmbedExecutor` (spawns a `ragger embed` subprocess so a hung/crashed
 * model can't take the daemon down — issue #41). Call sites that only need
 * "give me a vector for this text" should depend on IEmbedProvider rather
 * than on either concrete class, so a future config knob could pick either
 * implementation without the call site changing.
 *
 * The two concrete classes did not agree on signatures before this
 * interface existed:
 *   - Embedder::encode()  returns std::vector<float>, empty vector == failure
 *   - EmbedExecutor::one() returns std::optional<std::vector<float>>,
 *     nullopt == failure (timeout / spawn / parse error)
 * EmbedExecutor's optional-based, explicit-failure signature is the
 * genuinely safer contract (an empty-vs-nullopt convention is easy to get
 * wrong at call sites), so the interface adopts it; Embedder gains a thin
 * `embed()`/`embed_batch()` adapter on top of its existing encode()/ready()
 * rather than changing encode()'s existing return type (which callers such
 * as memory.cpp already rely on).
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
