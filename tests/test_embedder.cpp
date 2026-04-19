/**
 * Embedder tests
 *
 * Tests the ONNX Runtime embedder contract:
 * output shape, determinism, differentiation, normalization.
 * Requires ONNX model files at the configured model_dir.
 */
#include "ragger/config.h"
#include "ragger/embedder.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <print>

void test_output_shape(ragger::Embedder& emb) {
    std::println("  test_output_shape...");
    auto vec = emb.encode("test text");
    assert(vec.size() == 384);
    assert(emb.dimensions() == 384);
    std::println(" OK");
}

void test_deterministic(ragger::Embedder& emb) {
    std::println("  test_deterministic...");
    auto v1 = emb.encode("hello world");
    auto v2 = emb.encode("hello world");
    assert(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        assert(v1[i] == v2[i]);
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
    assert(any_different);
    std::println(" OK");
}

void test_unit_normalized(ragger::Embedder& emb) {
    std::println("  test_unit_normalized...");
    auto vec = emb.encode("test normalization");
    float sum_sq = 0.0f;
    for (float v : vec) sum_sq += v * v;
    float norm = std::sqrt(sum_sq);
    assert(std::fabs(norm - 1.0f) < 1e-4f);
    std::println(" OK");
}

void test_empty_string(ragger::Embedder& emb) {
    std::println("  test_empty_string...");
    auto vec = emb.encode("");
    assert(vec.size() == 384);
    // Should still produce a valid vector (not NaN/inf)
    for (float v : vec) {
        assert(!std::isnan(v));
        assert(!std::isinf(v));
    }
    std::println(" OK");
}

void test_long_text(ragger::Embedder& emb) {
    std::println("  test_long_text...");
    // MiniLM has 512 token limit; verify it handles longer input gracefully
    std::string long_text;
    for (int i = 0; i < 1000; ++i) long_text += "word ";
    auto vec = emb.encode(long_text);
    assert(vec.size() == 384);
    float sum_sq = 0.0f;
    for (float v : vec) sum_sq += v * v;
    float norm = std::sqrt(sum_sq);
    assert(std::fabs(norm - 1.0f) < 1e-4f);
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
    assert(sim_related > sim_unrelated);
    std::println(" OK");
}

int main() {
    std::println("Running embedder tests:");

    ragger::init_config("", true);
    auto cfg = ragger::config();
    ragger::Embedder emb(cfg.model_dir);

    test_output_shape(emb);
    test_deterministic(emb);
    test_different_texts_different_vectors(emb);
    test_unit_normalized(emb);
    test_empty_string(emb);
    test_long_text(emb);
    test_semantic_similarity(emb);

    std::println("test_embedder: all passed");
    return 0;
}
