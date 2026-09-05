/**
 * Embedder implementation — ONNX (internal) and HTTP (external) modes.
 */
#include "embedder.h"
#include "config.h"
#include "lang.h"
#include "tokenizer_wrapper.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <filesystem>
#include <format>
#include <stdexcept>

// --- External mode: HTTP client + JSON -----------------------------------------
#include "util/http_client.h"
#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

// ====================================================================
// PIMPL — holds either an ONNX session or HTTP endpoint details.
// ====================================================================
struct Embedder::Impl {
    bool external = false;
    bool disabled = false;   // default-constructed: cannot embed, never throws

    // ---- Internal (ONNX) state ----
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ragger"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<TokenizerWrapper> tokenizer;
    int onnx_dims = 0;

    // ---- External (HTTP) state ----
    std::string ext_host;
    int         ext_port = 0;
    std::string ext_model;
    std::string ext_api_key;
    int         ext_dims = 0;       // 0 = not yet known
    mutable std::string last_served_model;  // model reported by the last response

    // ---- Disabled constructor ----
    Impl() : disabled(true) {}

    // ---- Internal constructor ----
    explicit Impl(const std::string& model_dir) {
        std::filesystem::path model_path = std::filesystem::path(model_dir) / "model.onnx";
        // Some HuggingFace repos put the ONNX model in an onnx/ subdirectory.
        if (!std::filesystem::exists(model_path)) {
            model_path = std::filesystem::path(model_dir) / "onnx" / "model.onnx";
        }
        std::filesystem::path tokenizer_path = std::filesystem::path(model_dir) / "tokenizer.json";

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(std::format(lang::ERR_MODEL_NOT_FOUND, model_path.string()));
        }
        if (!std::filesystem::exists(tokenizer_path)) {
            throw std::runtime_error(std::format(lang::ERR_TOKENIZER_NOT_FOUND, tokenizer_path.string()));
        }

        tokenizer = std::make_unique<TokenizerWrapper>(tokenizer_path.string());
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        onnx_dims = config().embedding_dimensions;
    }

    // ---- External constructor ----
    Impl(const std::string& host, int port, const std::string& model,
         const std::string& api_key, int dims)
        : external(true), ext_host(host), ext_port(port),
          ext_model(model), ext_api_key(api_key), ext_dims(dims) {}

    // ---- ONNX encode ----
    std::vector<float> encode_onnx(const std::string& text) const {
        auto encoded = tokenizer->encode_with_mask(text);
        const auto& input_ids = encoded.input_ids;
        const auto& attention_mask = encoded.attention_mask;

        size_t seq_len = input_ids.size();
        if (seq_len == 0) {
            throw std::runtime_error(lang::ERR_EMPTY_TOKENIZATION);
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape = {1, static_cast<int64_t>(seq_len)};

        std::vector<int64_t> input_ids_i64(input_ids.begin(), input_ids.end());
        std::vector<int64_t> attention_mask_i64(attention_mask.begin(), attention_mask.end());
        std::vector<int64_t> token_type_ids_i64(seq_len, 0);

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_i64.data(), input_ids_i64.size(), shape.data(), shape.size());
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_i64.data(), attention_mask_i64.size(), shape.data(), shape.size());
        Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, token_type_ids_i64.data(), token_type_ids_i64.size(), shape.data(), shape.size());

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

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[2] != onnx_dims) {
            throw std::runtime_error(lang::ERR_OUTPUT_SHAPE);
        }

        size_t output_seq_len = output_shape[1];
        size_t hidden_size = output_shape[2];

        // Mean pooling with attention mask
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
            for (auto& val : pooled) val /= mask_sum;
        }

        // L2 normalization
        float norm = 0.0f;
        for (float val : pooled) norm += val * val;
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            for (auto& val : pooled) val /= norm;
        }

        return pooled;
    }

    // ---- build base URL ----
    std::string base_url() const {
        return "http://" + ext_host + ":" + std::to_string(ext_port);
    }

    // ---- External encode via /v1/embeddings ----
    std::vector<float> encode_external(const std::string& text) const {
        json body = {
            {"input", text},
            {"model", ext_model}
        };
        // Pass dimensions if explicitly set (MRL / truncated models).
        if (ext_dims > 0) {
            body["dimensions"] = ext_dims;
        }
        std::string payload = body.dump();

        std::string url = base_url() + "/v1/embeddings";

        std::vector<std::string> headers{"Content-Type: application/json"};
        if (!ext_api_key.empty()) {
            headers.push_back("Authorization: *** " + ext_api_key);
        }

        util::HttpClient http;
        auto resp = http.post(url, payload, headers, /*timeout_sec=*/30);

        if (!resp.ok()) {
            throw std::runtime_error(std::format("Embedding request failed: {}", resp.error));
        }

        auto j = json::parse(resp.body, nullptr, false);
        if (j.is_discarded()) {
            throw std::runtime_error("Failed to parse embedding response");
        }

        // OpenAI format: { "data": [ { "embedding": [...] } ] }
        if (!j.contains("data") || !j["data"].is_array() || j["data"].empty()) {
            // Check for error
            if (j.contains("error")) {
                std::string msg = j["error"].is_string()
                    ? j["error"].get<std::string>()
                    : j["error"].value("message", "unknown error");
                throw std::runtime_error("Embedding API error: " + msg);
            }
            throw std::runtime_error("Unexpected embedding response format");
        }

        auto& emb = j["data"][0]["embedding"];
        if (!emb.is_array()) {
            throw std::runtime_error("Embedding response missing 'embedding' array");
        }

        // Record which model the server says actually served the request —
        // llama-swap / LM Studio may answer with a different loaded model
        // than the one requested. The probe route surfaces this mismatch.
        last_served_model = j.value("model", "");

        std::vector<float> vec;
        vec.reserve(emb.size());
        for (const auto& v : emb) {
            vec.push_back(v.get<float>());
        }
        return vec;
    }

    // ---- Dispatch ----
    std::vector<float> encode(const std::string& text) const {
        // Empty, not an exception: callers gate on ready() and defer the
        // write. Returning empty keeps a stray call from taking the daemon
        // down, but an empty vector must never reach bind_embedding().
        if (disabled) return {};
        if (external) return encode_external(text);
        return encode_onnx(text);
    }

    int dimensions() const {
        if (disabled) return 0;
        if (external) return ext_dims;
        return onnx_dims;
    }

    std::vector<std::string> list_remote_models() const {
        if (!external) return {};

        std::string url = base_url() + "/v1/models";

        std::vector<std::string> headers;
        if (!ext_api_key.empty()) {
            headers.push_back("Authorization: *** " + ext_api_key);
        }

        util::HttpClient http;
        auto resp = http.get(url, headers, /*timeout_sec=*/10);

        if (!resp.ok()) return {};

        auto j = json::parse(resp.body, nullptr, false);
        if (j.is_discarded() || !j.contains("data") || !j["data"].is_array())
            return {};

        std::vector<std::string> models;
        for (const auto& m : j["data"]) {
            if (m.contains("id") && m["id"].is_string()) {
                models.push_back(m["id"].get<std::string>());
            }
        }
        return models;
    }

    int probe_dimensions() const {
        if (!external) return 0;
        try {
            auto vec = encode_external("dimension probe");
            return static_cast<int>(vec.size());
        } catch (...) {
            return 0;
        }
    }
};

// ====================================================================
// Public API
// ====================================================================

Embedder::Embedder()
    : pImpl(std::make_unique<Impl>()) {}

Embedder::Embedder(const std::string& model_dir)
    : pImpl(std::make_unique<Impl>(model_dir)) {}

Embedder::Embedder(const std::string& host, int port,
                   const std::string& model_name,
                   const std::string& api_key, int dimensions)
    : pImpl(std::make_unique<Impl>(host, port, model_name, api_key, dimensions)) {}

Embedder::~Embedder() = default;

std::vector<float> Embedder::encode(const std::string& text) const {
    return pImpl->encode(text);
}

int Embedder::dimensions() const {
    return pImpl->dimensions();
}

bool Embedder::ready() const {
    return !pImpl->disabled;
}

bool Embedder::is_external() const {
    return pImpl->external;
}

std::vector<std::string> Embedder::list_remote_models() const {
    return pImpl->list_remote_models();
}

int Embedder::probe_dimensions() const {
    return pImpl->probe_dimensions();
}

std::string Embedder::last_served_model() const {
    return pImpl->last_served_model;
}

std::optional<std::vector<float>> Embedder::embed(const std::string& text) const {
    if (!ready()) return std::nullopt;
    auto vec = encode(text);
    if (vec.empty()) return std::nullopt;
    return vec;
}

std::vector<std::optional<std::vector<float>>>
Embedder::embed_batch(const std::vector<std::string>& texts) const {
    std::vector<std::optional<std::vector<float>>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) out.push_back(embed(t));
    return out;
}

} // namespace ragger
