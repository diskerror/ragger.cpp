/**
 * EmbedExecutor — run embedding in spawned `ragger embed` subprocesses.
 *
 * The embedding model (ONNX) is isolated in a child process so a hung or
 * crashed model can be killed without taking down the parent (issue #41).
 * Each call spawns `ragger embed`, pipes the text to its stdin, and reads
 * the JSON vector from stdout under a timeout; on timeout the child is
 * SIGKILL'd and the call fails (caller leaves the row's embedding NULL for a
 * later retry). `batch()` caps concurrency to avoid overwhelming the box when
 * a large document is imported.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "embed_provider.h"

namespace ragger {

class EmbedExecutor : public IEmbedProvider {
public:
    /// timeout_ms / retries / max_workers default to the [embed] config.
    EmbedExecutor();
    EmbedExecutor(int timeout_ms, int retries, int max_workers,
                  std::string exe_path);

    /// Embed one text. Returns std::nullopt on timeout / spawn / parse
    /// failure (after `retries`). Never throws.
    std::optional<std::vector<float>> one(const std::string& text) const;

    /// Embed many texts with at most `max_workers` concurrent subprocesses.
    /// Result is positional; failed items are std::nullopt.
    std::vector<std::optional<std::vector<float>>>
    batch(const std::vector<std::string>& texts) const;

    // --- IEmbedProvider ---------------------------------------------
    // one()/batch() predate the interface and are kept as the primary,
    // more discoverable API; embed()/embed_batch() just forward to them.
    std::optional<std::vector<float>> embed(const std::string& text) const override {
        return one(text);
    }
    std::vector<std::optional<std::vector<float>>>
    embed_batch(const std::vector<std::string>& texts) const override {
        return batch(texts);
    }
    /// Subprocess executor has no persistent model state to be "not ready";
    /// individual calls fail (nullopt) instead. Always true.
    bool ready() const override { return true; }
    /// Not known ahead of a call (the subprocess reports it); no cheap way
    /// to determine without spawning. 0 = unknown, consistent with
    /// IEmbedProvider's documented contract.
    int dimensions() const override { return 0; }

    int max_workers() const { return max_workers_; }


private:
    int         timeout_ms_;
    int         retries_;
    int         max_workers_;
    std::string exe_path_;
};

} // namespace ragger
