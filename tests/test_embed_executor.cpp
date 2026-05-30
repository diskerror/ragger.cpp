/**
 * EmbedExecutor tests — spawn `ragger embed` with a timeout + bounded batch.
 *
 * RAGGER_BIN is injected by CMake ($<TARGET_FILE:ragger>) so the test drives
 * the freshly-built binary. Skips if the embedding model isn't present.
 */
#include "ragger/config.h"
#include "ragger/embed_executor.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <print>

namespace fs = std::filesystem;

#ifndef RAGGER_BIN
#define RAGGER_BIN "ragger"
#endif

int main() {
    ragger::init_config("");
    auto model_dir = ragger::config().resolved_model_dir();
    if (!fs::exists(model_dir + "/model.onnx")) {
        std::println("test_embed_executor: SKIPPED (no model)");
        return 0;
    }
    if (!fs::exists(RAGGER_BIN)) {
        std::println("test_embed_executor: SKIPPED (ragger binary not built at {})",
                     RAGGER_BIN);
        return 0;
    }

    std::println("Running embed executor tests:");

    const int dims = ragger::config().embedding_dimensions;

    // one(): a normal embed returns a dims-length vector.
    {
        std::println("  one() embeds...");
        ragger::EmbedExecutor exec(5000, 1, 4, RAGGER_BIN);
        auto v = exec.one("The quick brown fox.");
        assert(v.has_value());
        assert(static_cast<int>(v->size()) == dims);
        std::println("   OK ({} dims)", v->size());
    }

    // one(""): empty input → nullopt (no spawn).
    {
        std::println("  one(\"\") → nullopt...");
        ragger::EmbedExecutor exec(5000, 0, 1, RAGGER_BIN);
        assert(!exec.one("").has_value());
        std::println("   OK");
    }

    // Timeout: an impossibly short deadline kills the child → nullopt.
    {
        std::println("  timeout kills child → nullopt...");
        ragger::EmbedExecutor exec(/*timeout_ms=*/1, /*retries=*/0, 1, RAGGER_BIN);
        assert(!exec.one("model load alone exceeds 1ms").has_value());
        std::println("   OK");
    }

    // batch(): bounded concurrency, positional results, all embedded.
    {
        std::println("  batch() embeds all (bounded workers)...");
        ragger::EmbedExecutor exec(5000, 1, /*max_workers=*/2, RAGGER_BIN);
        std::vector<std::string> texts = {
            "apples and oranges", "the speed of light",
            "photosynthesis in plants", "a lazy dog sleeps",
            "quantum entanglement"
        };
        auto res = exec.batch(texts);
        assert(res.size() == texts.size());
        for (auto& r : res) {
            assert(r.has_value());
            assert(static_cast<int>(r->size()) == dims);
        }
        // Empty batch → empty result.
        assert(exec.batch({}).empty());
        std::println("   OK ({} texts)", texts.size());
    }

    std::println("test_embed_executor: all passed");
    return 0;
}
