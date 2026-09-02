// Unit tests for ragger::vector_codec — round-trip fidelity across storage
// dtypes, offset-parameterized format, and legacy blob decoding. main()-based
// per this project's test convention.

#include "vector_codec.h"

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

// Mean absolute error of a round-trip through dtype `t` with a 1-byte
// version-tag header (offset=1), matching the legacy on-disk format.
static float roundtrip_mae(VectorType t, const std::vector<float>& v,
                           uint8_t ver = 42) {
    std::vector<uint8_t> blob = encode(t, v, /*offset=*/1, ver);
    std::vector<float> back;
    bool ok = decode(blob.data(), static_cast<int>(blob.size()),
                     static_cast<int>(v.size()), t, back, /*offset=*/1);
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

    // --- blob sizes (version byte + payload, int8 gets +2 for f16 scale) --
    check(encode(VectorType::F32,  v, 1, 0).size() == 1u + 384u * 4, "f32 blob size");
    check(encode(VectorType::F16,  v, 1, 0).size() == 1u + 384u * 2, "f16 blob size");
    check(encode(VectorType::BF16, v, 1, 0).size() == 1u + 384u * 2, "bf16 blob size");
    check(encode(VectorType::INT8, v, 1, 0).size() == 1u + 384u * 1 + 2u, "int8 blob size");

    // --- expected_blob_size matches actual encode output ----------------
    for (auto t : {VectorType::F32, VectorType::F16, VectorType::BF16, VectorType::INT8}) {
        check(static_cast<int>(encode(t, v, 1, 7).size()) == expected_blob_size(t, 384, 1),
              "expected_blob_size matches encode");
    }

    // --- version byte is correctly stored and read ----------------------
    {
        auto blob = encode(VectorType::F16, v, 1, 99);
        check(blob_version(blob.data(), (int)blob.size()) == 99, "version byte stored");
        // Decode with correct dtype should succeed regardless of version value.
        std::vector<float> back;
        bool ok = decode(blob.data(), (int)blob.size(), 384, VectorType::F16, back, 1);
        check(ok, "decode succeeds for version-tagged blob");
    }

    // --- version byte 0 works (edge case) ------------------------------
    {
        auto blob = encode(VectorType::F16, v, 1, 0);
        check(blob_version(blob.data(), (int)blob.size()) == 0, "version 0 works");
    }

    // --- f16 vs bf16 are distinguishable via the caller knowing dtype ---
    // (caller gets dtype from settings, not from the blob itself)
    {
        std::vector<float> one = {2.5f};
        auto bf = encode(VectorType::BF16, one, 1, 1);
        std::vector<float> back;
        check(decode(bf.data(), (int)bf.size(), 1, VectorType::BF16, back, 1) &&
              std::fabs(back[0] - 2.5f) < 1e-6f,
              "bf16 decodes correctly with dtype hint");
        // Decoding bf16 blob as f16 should fail (size matches but produces wrong answer
        // only if dims match — with 1 dim both are 3 bytes so it succeeds; that's fine,
        // the point is the CALLER passes the correct dtype from settings).
    }

    // --- dimension-mismatch rejection ----------------------------------
    {
        auto blob = encode(VectorType::F16, v, 1, 5);
        std::vector<float> back;
        bool ok = decode(blob.data(), (int)blob.size(), 128 /*wrong*/, VectorType::F16, back, 1);
        check(!ok, "dim mismatch rejected");
        check(back.size() == 128 && back[0] == 0.0f, "dim mismatch -> zero vector");
    }

    // --- null / empty ---------------------------------------------------
    {
        std::vector<float> back;
        check(!decode(nullptr, 0, 384, VectorType::F16, back, 1), "null blob rejected");
        check(back.size() == 384, "null -> zero vector of expected dims");
    }

    // --- blob_version edge cases ----------------------------------------
    check(blob_version(nullptr, 0) == -1, "blob_version null -> -1");
    {
        uint8_t b = 255;
        check(blob_version(&b, 1) == 255, "blob_version 255");
    }

    // --- int8 f16 scale precision test ----------------------------------
    // Verify the f16 scale is sufficient for unit-norm embeddings.
    {
        // Normalize v to unit norm like real embeddings
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        std::vector<float> vn(v.size());
        for (size_t i = 0; i < v.size(); ++i) vn[i] = v[i] / norm;

        float mae = roundtrip_mae(VectorType::INT8, vn);
        std::printf("MAE  int8 unit-norm=%.6g\n", mae);
        check(mae < 5e-3f, "int8 unit-norm round-trip within tolerance");
    }

    // --- offset=0 (payload-only) encode/decode (Ragger db_version 0.15+:
    // version tag moved out of the blob into its own embedding_version
    // column) ------------------------------------------------------------
    {
        for (auto t : {VectorType::F32, VectorType::F16, VectorType::BF16, VectorType::INT8}) {
            // expected_blob_size(t, dims, 0) is exactly expected_blob_size(t,
            // dims, 1) minus 1 (the version byte) for every dtype.
            check(expected_blob_size(t, 384, 0) == expected_blob_size(t, 384, 1) - 1,
                  "offset=0 size == offset=1 size - 1");

            auto payload = encode(t, v, /*offset=*/0);
            check(static_cast<int>(payload.size()) == expected_blob_size(t, 384, 0),
                  "offset=0 encode size matches expected_blob_size");

            // offset=0 encode's bytes must equal offset=1 encode()'s bytes
            // with the leading version byte stripped -- same encoding, just
            // no tag.
            auto tagged = encode(t, v, /*offset=*/1, 123);
            check(std::vector<uint8_t>(tagged.begin() + 1, tagged.end()) == payload,
                  "offset=0 encode matches offset=1 encode() minus version byte");

            std::vector<float> back;
            bool ok = decode(payload.data(), static_cast<int>(payload.size()),
                             static_cast<int>(v.size()), t, back, /*offset=*/0);
            check(ok, "offset=0 decode succeeds");
            check(back.size() == v.size(), "offset=0 decode produces expected_dims");

            // Cross-check against offset=1 decode() on the tagged blob --
            // both must agree exactly (same payload bytes, same dequant math).
            std::vector<float> back_tagged;
            decode(tagged.data(), static_cast<int>(tagged.size()),
                  static_cast<int>(v.size()), t, back_tagged, /*offset=*/1);
            check(back == back_tagged,
                  "offset=0 decode agrees with offset=1 decode() on the equivalent tagged blob");
        }

        // Size-mismatch rejection mirrors decode()'s behavior.
        {
            auto payload = encode(VectorType::F16, v, /*offset=*/0);
            std::vector<float> back;
            bool ok = decode(payload.data(), static_cast<int>(payload.size()),
                             128 /*wrong*/, VectorType::F16, back, /*offset=*/0);
            check(!ok, "offset=0 decode dim mismatch rejected");
            check(back.size() == 128 && back[0] == 0.0f,
                  "offset=0 decode dim mismatch -> zero vector");
        }

        // Null blob rejection.
        {
            std::vector<float> back;
            check(!decode(nullptr, 0, 384, VectorType::F16, back, /*offset=*/0),
                  "offset=0 decode null blob rejected");
            check(back.size() == 384, "offset=0 decode null -> zero vector of expected dims");
        }
    }

    if (failures == 0) std::printf("all vector_codec tests passed\n");
    else               std::printf("%d vector_codec test(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
