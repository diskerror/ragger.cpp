/**
 * Embedder — generate 384-dim embeddings via ONNX Runtime
 *
 * Mirrors ragger_memory/embedding.py from the Python version.
 * Uses the all-MiniLM-L6-v2 ONNX model + HuggingFace tokenizer.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ragger {

class Embedder {
public:
    /// Construct with path to model directory (containing model.onnx + tokenizer.json).
    explicit Embedder(const std::string& model_dir);
    ~Embedder();

    /// Encode text to a normalized embedding vector (384 floats).
    std::vector<float> encode(const std::string& text) const;

    /// Embedding dimensions (384 for MiniLM).
    int dimensions() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
