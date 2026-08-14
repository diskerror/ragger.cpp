// Unit tests for ragger::vector_codec — round-trip fidelity across storage
// dtypes, self-describing header behavior, and legacy (headerless) blob
// decoding. main()-based per this project's test convention.

#include "ragger/vector_codec.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ragger::vector_codec;

static int failures = 0;

static void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// Mean absolute error of a round-trip through dtype `t`.
static float roundtrip_mae(VectorType t, const std::vector<float>& v) {
    std::vector<uint8_t> blob = encode(t, v);
    std::vector<float> back;
    bool ok = decode(blob.data(), static_cast<int>(blob.size()),
                     static_cast<int>(v.size()), back);
    if (!ok || back.size() != v.size()) return 1e9f;
    float acc = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) acc += std::fabs(v[i] - back[i]);
    return acc / static_cast<float>(v.size());
}

int main() {
    // --- string parsing / canonicalization -----------------------------
    check(parse("f32").value() == VectorType::F32, "parse f32");
    check(parse("F16").value() == VectorType::F16, "parse F16 (case-insensitive)");
    check(parse("bf16").value() == VectorType::BF16, "parse bf16");
    check(parse("int8").value() == VectorType::INT8, "parse int8");
    check(parse("bfloat16").value() == VectorType::BF16, "parse bfloat16 alias");
    check(!parse("f8").has_value(), "parse unknown -> nullopt");
    check(canonical("garbage") == "f16", "canonical fallback -> f16");
    check(canonical("BF16") == "bf16", "canonical normalizes case");
    check(is_supported("int8") && !is_supported("q4"), "is_supported");
    check(std::string(supported_csv()) == "f32,f16,bf16,int8", "supported_csv");

    // --- deterministic normalized-ish embedding ------------------------
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    std::vector<float> v(384);
    for (auto& x : v) x = nd(rng);

    // f32 must be exactly lossless.
    check(roundtrip_mae(VectorType::F32, v) == 0.0f, "f32 lossless round-trip");

    // f16/bf16/int8 lossy but bounded. Thresholds are generous but catch
    // gross breakage (wrong shift, endianness, missing dequant scale).
    float mae_f16  = roundtrip_mae(VectorType::F16, v);
    float mae_bf16 = roundtrip_mae(VectorType::BF16, v);
    float mae_int8 = roundtrip_mae(VectorType::INT8, v);
    std::printf("MAE  f16=%.6g  bf16=%.6g  int8=%.6g\n", mae_f16, mae_bf16, mae_int8);
    check(mae_f16  < 1e-3f, "f16 round-trip within tolerance");
    check(mae_bf16 < 5e-3f, "bf16 round-trip within tolerance");
    check(mae_int8 < 5e-3f, "int8 round-trip within tolerance");
    // f16 has more mantissa bits than bf16 -> should be more accurate here.
    check(mae_f16 < mae_bf16, "f16 more accurate than bf16 for small values");

    // --- blob sizes (header + payload) ---------------------------------
    check(encode(VectorType::F32,  v).size() == kHeaderBytes + 384u * 4, "f32 blob size");
    check(encode(VectorType::F16,  v).size() == kHeaderBytes + 384u * 2, "f16 blob size");
    check(encode(VectorType::BF16, v).size() == kHeaderBytes + 384u * 2, "bf16 blob size");
    check(encode(VectorType::INT8, v).size() == kHeaderBytes + 384u * 1, "int8 blob size");

    // --- f16 vs bf16 are distinguishable despite equal payload size ----
    // (the whole reason for the header). Encode as bf16, ensure it does NOT
    // decode as if it were f16.
    {
        std::vector<float> one = {2.5f};  // exactly representable in both
        auto bf = encode(VectorType::BF16, one);
        std::vector<float> back;
        check(decode(bf.data(), (int)bf.size(), 1, back) && std::fabs(back[0] - 2.5f) < 1e-6f,
              "bf16 header decodes as bf16");
    }

    // --- dimension-mismatch rejection ----------------------------------
    {
        auto blob = encode(VectorType::F16, v);
        std::vector<float> back;
        bool ok = decode(blob.data(), (int)blob.size(), 128 /*wrong*/, back);
        check(!ok, "dim mismatch rejected");
        check(back.size() == 128 && back[0] == 0.0f, "dim mismatch -> zero vector");
    }

    // --- null / empty ---------------------------------------------------
    {
        std::vector<float> back;
        check(!decode(nullptr, 0, 384, back), "null blob rejected");
        check(back.size() == 384, "null -> zero vector of expected dims");
    }

    // --- LEGACY headerless blobs (pre-header DBs) ----------------------
    // Raw f16: dims*2 bytes, host-order _Float16.
    {
        const int dims = 8;
        std::vector<float> src(dims);
        for (int i = 0; i < dims; ++i) src[i] = 0.1f * static_cast<float>(i - 4);
        std::vector<uint16_t> raw(dims);
        for (int i = 0; i < dims; ++i) {
            _Float16 h = static_cast<_Float16>(src[i]);
            std::memcpy(&raw[i], &h, sizeof(uint16_t));
        }
        std::vector<float> back;
        bool ok = decode(raw.data(), dims * 2, dims, back);
        check(ok, "legacy raw-f16 decodes");
        float mae = 0; for (int i=0;i<dims;++i) mae += std::fabs(src[i]-back[i]);
        check(mae / dims < 1e-3f, "legacy raw-f16 values correct");
    }
    // Raw f32: dims*4 bytes.
    {
        const int dims = 8;
        std::vector<float> src(dims);
        for (int i = 0; i < dims; ++i) src[i] = 0.1f * static_cast<float>(i - 4);
        std::vector<float> back;
        bool ok = decode(src.data(), dims * 4, dims, back);
        check(ok, "legacy raw-f32 decodes");
        float mae = 0; for (int i=0;i<dims;++i) mae += std::fabs(src[i]-back[i]);
        check(mae == 0.0f, "legacy raw-f32 lossless");
    }

    if (failures == 0) std::printf("all vector_codec tests passed\n");
    else               std::printf("%d vector_codec test(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
