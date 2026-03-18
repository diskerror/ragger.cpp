/**
 * Thin wrapper around tokenizers-cpp for HuggingFace tokenizer.json
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ragger {

class TokenizerWrapper {
public:
    explicit TokenizerWrapper(const std::string& tokenizer_json_path);
    ~TokenizerWrapper();

    /// Encode text to token IDs.
    std::vector<int32_t> encode(const std::string& text) const;

    /// Encode and return attention mask too.
    struct Encoded {
        std::vector<int32_t> input_ids;
        std::vector<int32_t> attention_mask;
    };
    Encoded encode_with_mask(const std::string& text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ragger
