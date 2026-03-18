/**
 * TokenizerWrapper implementation using tokenizers-cpp
 */
#include "tokenizer_wrapper.h"
#include <tokenizers_cpp.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ragger {

// PIMPL implementation
struct TokenizerWrapper::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    
    explicit Impl(const std::string& tokenizer_json_path) {
        // Read tokenizer.json into string
        std::ifstream file(tokenizer_json_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer.json: " + tokenizer_json_path);
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_content = buffer.str();
        
        if (json_content.empty()) {
            throw std::runtime_error("Empty tokenizer.json file: " + tokenizer_json_path);
        }
        
        // Create tokenizer from JSON blob
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_content);
        if (!tokenizer) {
            throw std::runtime_error("Failed to create tokenizer from JSON");
        }
    }
};

TokenizerWrapper::TokenizerWrapper(const std::string& tokenizer_json_path)
    : pImpl(std::make_unique<Impl>(tokenizer_json_path)) {
}

TokenizerWrapper::~TokenizerWrapper() = default;

std::vector<int32_t> TokenizerWrapper::encode(const std::string& text) const {
    if (!pImpl->tokenizer) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return pImpl->tokenizer->Encode(text);
}

TokenizerWrapper::Encoded TokenizerWrapper::encode_with_mask(const std::string& text) const {
    Encoded result;
    result.input_ids = encode(text);
    
    // Attention mask: all 1s for real tokens
    result.attention_mask.resize(result.input_ids.size(), 1);
    
    return result;
}

} // namespace ragger
