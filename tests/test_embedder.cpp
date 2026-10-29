/**
 * Embedder tests
 *
 * Tests the ONNX Runtime embedder contract:
 * output shape, determinism, differentiation, normalization.
 * Requires ONNX model files at the configured model_dir.
 */
#include "config.h"
#include "embedder.h"
#include "test_env.h"
#include "check.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <print>

void test_output_shape(ragger::Embedder& emb) {
    std::println("  test_output_shape...");
    auto vec = emb.encode("test text");
    CHECK(vec.size() == 384);
    CHECK(emb.dimensions() == 384);
    std::println(" OK");
}

void test_deterministic(ragger::Embedder& emb) {
    std::println("  test_deterministic...");
    auto v1 = emb.encode("hello world");
    auto v2 = emb.encode("hello world");
    CHECK(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        CHECK(v1[i] == v2[i]);
    }
    std::println(" OK");
}

void test_different_texts_different_vectors(ragger::Embedder& emb) {
    std::println("  test_different_texts_different_vectors...");
    auto v1 = emb.encode("hello world");
    auto v2 = emb.encode("goodbye moon");
    // At least some dimensions must differ
    bool any_different = false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::fabs(v1[i] - v2[i]) > 1e-6f) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
    std::println(" OK");
}

void test_unit_normalized(ragger::Embedder& emb) {
    std::println("  test_unit_normalized...");
    auto vec = emb.encode("test normalization");
    float sum_sq = 0.0f;
    for (float v : vec) sum_sq += v * v;
    float norm = std::sqrt(sum_sq);
    CHECK(std::fabs(norm - 1.0f) < 1e-4f);
    std::println(" OK");
}

void test_empty_string(ragger::Embedder& emb) {
    std::println("  test_empty_string...");
    auto vec = emb.encode("");
    CHECK(vec.size() == 384);
    // Should still produce a valid vector (not NaN/inf)
    for (float v : vec) {
        CHECK(!std::isnan(v));
        CHECK(!std::isinf(v));
    }
    std::println(" OK");
}

void test_long_text(ragger::Embedder& emb) {
    std::println("  test_long_text...");
    // MiniLM has 512 token limit; verify it handles longer input gracefully
    std::string long_text;
    for (int i = 0; i < 1000; ++i) long_text += "word ";
    auto vec = emb.encode(long_text);
    CHECK(vec.size() == 384);
    float sum_sq = 0.0f;
    for (float v : vec) sum_sq += v * v;
    float norm = std::sqrt(sum_sq);
    CHECK(std::fabs(norm - 1.0f) < 1e-4f);
    std::println(" OK");
}

void test_semantic_similarity(ragger::Embedder& emb) {
    std::println("  test_semantic_similarity...");
    auto v_cat   = emb.encode("The cat sat on the mat");
    auto v_kitten = emb.encode("A kitten rested on the rug");
    auto v_sql    = emb.encode("SELECT * FROM users WHERE id = 1");

    // Cosine similarity (vectors are unit-normalized, so dot product = cosine)
    auto dot = [](const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    };

    float sim_related   = dot(v_cat, v_kitten);
    float sim_unrelated = dot(v_cat, v_sql);

    // Related sentences should be more similar than unrelated ones
    CHECK(sim_related > sim_unrelated);
    std::println(" OK");
}

// PHASE9A: padding must not leak into the embedding. Every shipped
// tokenizer.json pads to a fixed 128 tokens; with an all-ones attention mask
// the model pooled over ~115 [PAD] positions and unrelated texts scored ~0.9.
// Values below were measured with all-MiniLM-L12-v2 after the mask fix; before
// it: hello/zzqxv 0.9681, FTS5/cat 0.8612, FTS5/paraphrase 0.8638, cat/SQL 0.8104.
void p9a_test_padding_is_masked(ragger::Embedder& emb) {
    std::println("  p9a_test_padding_is_masked...");
    auto cos = [&](const std::string& a, const std::string& b) {
        auto x = emb.encode(a);
        auto y = emb.encode(b);
        float d = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) d += x[i] * y[i];
        return d;   // vectors are unit-normalized
    };

    CHECK(cos("hello world", "zzqxv plork") < 0.5f);            // measured 0.1719
    const float para = cos("custom inverted index instead of FTS5",
        "We decided to build our own inverted-index search layer rather than use SQLite FTS5");
    const float unrel = cos("custom inverted index instead of FTS5", "The cat sat on the mat.");
    CHECK(para > 0.6f);                                          // measured 0.7280
    CHECK(unrel < 0.3f);                                         // measured 0.0253
    CHECK(para - unrel > 0.4f);                                  // the gap the bug collapsed to 0.002
    CHECK(cos("The cat sat on the mat", "A kitten rested on the rug") > 0.4f);       // 0.6073
    CHECK(cos("The cat sat on the mat", "SELECT * FROM users WHERE id = 1") < 0.2f); // -0.0551

    // Reference vector, so a future change to tokenization or pooling is noticed.
    // Only meaningful for the model the numbers were taken from.
    if (ragger::Config{}.embedding_model == "sentence-transformers/all-MiniLM-L12-v2") {
        const float ref[8] = {-0.04617f, 0.02210f, 0.00533f, -0.04767f,
                              0.04166f, -0.09757f, 0.07874f, 0.06726f};
        auto v = emb.encode("hello world");
        for (int i = 0; i < 8; ++i) CHECK(std::fabs(v[i] - ref[i]) < 1e-3f);
    }
    std::println(" OK");
}

int main() {
    TestEnv env("embedder");  // must be first: isolates from ~/.ragger
    std::println("Running embedder tests:");

    auto cfg = ragger::config();
    auto model_dir = env.model_dir();
    if (!std::filesystem::exists(model_dir + "/model.onnx") && !std::filesystem::exists(model_dir + "/onnx/model.onnx")) {
        std::cerr << "Skipping embedder tests: model not found at " << model_dir << "\n";
        return 0;
    }
    ragger::Embedder emb(model_dir);

    test_output_shape(emb);
    test_deterministic(emb);
    test_different_texts_different_vectors(emb);
    test_unit_normalized(emb);
    test_empty_string(emb);
    test_long_text(emb);
    test_semantic_similarity(emb);
    p9a_test_padding_is_masked(emb);

    std::println("test_embedder: all passed");
    return 0;
}
