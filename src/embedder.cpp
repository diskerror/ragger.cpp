/**
 * Embedder implementation using ONNX Runtime
 */
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "tokenizer_wrapper.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <cmath>
#include <filesystem>

namespace ragger {

// PIMPL implementation
struct Embedder::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<TokenizerWrapper> tokenizer;
    
    explicit Impl(const std::string& model_dir)
        : env(ORT_LOGGING_LEVEL_WARNING, "ragger") {
        
        // Construct paths
        std::filesystem::path model_path = std::filesystem::path(model_dir) / "model.onnx";
        std::filesystem::path tokenizer_path = std::filesystem::path(model_dir) / "tokenizer.json";
        
        // Check files exist
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path.string());
        }
        if (!std::filesystem::exists(tokenizer_path)) {
            throw std::runtime_error("Tokenizer file not found: " + tokenizer_path.string());
        }
        
        // Load tokenizer
        tokenizer = std::make_unique<TokenizerWrapper>(tokenizer_path.string());
        
        // Load ONNX model
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    }
    
    std::vector<float> encode(const std::string& text) const {
        // 1. Tokenize
        auto encoded = tokenizer->encode_with_mask(text);
        const auto& input_ids = encoded.input_ids;
        const auto& attention_mask = encoded.attention_mask;
        
        size_t seq_len = input_ids.size();
        if (seq_len == 0) {
            throw std::runtime_error("Empty tokenization result");
        }
        
        // 2. Create ONNX tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Input shapes: {1, seq_len}
        std::vector<int64_t> shape = {1, static_cast<int64_t>(seq_len)};
        
        // Convert int32_t vectors to int64_t for ONNX
        std::vector<int64_t> input_ids_i64(input_ids.begin(), input_ids.end());
        std::vector<int64_t> attention_mask_i64(attention_mask.begin(), attention_mask.end());
        std::vector<int64_t> token_type_ids_i64(seq_len, 0);  // All zeros
        
        // Create tensors
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_i64.data(), input_ids_i64.size(), shape.data(), shape.size());
        
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_i64.data(), attention_mask_i64.size(), shape.data(), shape.size());
        
        Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, token_type_ids_i64.data(), token_type_ids_i64.size(), shape.data(), shape.size());
        
        // 3. Run inference
        const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char* output_names[] = {"last_hidden_state"};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));
        input_tensors.push_back(std::move(token_type_ids_tensor));
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 3,
            output_names, 1
        );
        
        // 4. Extract output tensor
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Shape should be {1, seq_len, 384}
        if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[2] != EMBEDDING_DIMENSIONS) {
            throw std::runtime_error("Unexpected output shape from model");
        }
        
        size_t output_seq_len = output_shape[1];
        size_t hidden_size = output_shape[2];
        
        // 5. Mean pooling with attention mask
        std::vector<float> pooled(hidden_size, 0.0f);
        float mask_sum = 0.0f;
        
        for (size_t i = 0; i < output_seq_len; ++i) {
            float mask_value = static_cast<float>(attention_mask[i]);
            mask_sum += mask_value;
            
            for (size_t j = 0; j < hidden_size; ++j) {
                pooled[j] += output_data[i * hidden_size + j] * mask_value;
            }
        }
        
        if (mask_sum > 0.0f) {
            for (auto& val : pooled) {
                val /= mask_sum;
            }
        }
        
        // 6. L2 normalization
        float norm = 0.0f;
        for (float val : pooled) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-12f) {
            for (auto& val : pooled) {
                val /= norm;
            }
        }
        
        return pooled;
    }
};

Embedder::Embedder(const std::string& model_dir)
    : pImpl(std::make_unique<Impl>(model_dir)) {
}

Embedder::~Embedder() = default;

std::vector<float> Embedder::encode(const std::string& text) const {
    return pImpl->encode(text);
}

int Embedder::dimensions() const {
    return EMBEDDING_DIMENSIONS;
}

} // namespace ragger
