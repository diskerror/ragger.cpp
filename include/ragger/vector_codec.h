// vector_codec.h — on-disk embedding (de)serialization across storage dtypes.
//
// Embeddings are computed and compared in float32. On disk we can trade
// precision for space: a 384-dim vector is 1536 B as f32, 768 B as f16/bf16,
// 384 B as int8. All in-memory math stays f32; this layer only governs the
// stored BLOB.
//
// SELF-DESCRIBING FORMAT
// ----------------------
// Blobs written by encode() carry a 12-byte little-endian header so the dtype
// is unambiguous on read (f16 and bf16 are both 2 B/dim — size alone can't
// tell them apart, and an int8 blob of N dims is byte-for-byte the size of an
// f16 blob of N/2 dims). The `dimensions` setting alone is therefore not
// enough; the header is authoritative.
//
//   offset  size  field
//   0       3     magic  'R','V','1'
//   3       1     dtype  (VectorType)
//   4       1     flags  (reserved, 0)
//   5       1     reserved (0)
//   6       2     dims   (uint16, sanity check vs caller's expected dims)
//   8       4     scale  (float32; INT8 dequant multiplier, 0 for float types)
//   12      ...   payload
//
// LEGACY BLOBS (pre-header DBs) have no magic and are exactly dims*2 (raw f16)
// or dims*4 (raw f32) bytes. decode() detects those by size FIRST, so old
// databases keep reading correctly until a `rebuild-embeddings` rewrites them
// in the headered form. (Real embedding models are >=16-dim, so the headered
// vs legacy size ranges never collide.)
//
// Endianness: header fields are written/read explicitly little-endian.
// float/half payloads are host-order memcpy, matching the legacy on-disk
// layout (Ragger targets LE hosts: Apple Silicon, x86-64).

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ragger::vector_codec {

// Stable on-disk dtype codes — DO NOT renumber (persisted in blob headers).
enum class VectorType : uint8_t {
    F32  = 0,  // IEEE 754 single precision (4 B/dim) — lossless baseline
    F16  = 1,  // IEEE 754 half (2 B/dim): 1s/5e/10m — good precision, small range
    BF16 = 2,  // bfloat16  (2 B/dim): 1s/8e/7m — f32's range, less mantissa
    INT8 = 3,  // symmetric per-vector int8 (1 B/dim + scale) — 4x smaller than f32
};

constexpr int kHeaderBytes = 12;

// Parse a config string ("f32"/"f16"/"bf16"/"int8", case-insensitive) to a
// VectorType. std::nullopt if unrecognized.
std::optional<VectorType> parse(std::string_view s);

// Canonical lowercase name of a VectorType ("f32"/"f16"/"bf16"/"int8").
std::string to_string(VectorType t);

// Canonicalize a user/config string. Returns the canonical spelling when
// recognized; otherwise returns `fallback` (default "f16"). This replaces the
// old `== "f32" ? "f32" : "f16"` collapse that silently coerced every
// non-f32 value to f16.
std::string canonical(std::string_view s, std::string_view fallback = "f16");

// True if `s` names a supported dtype.
bool is_supported(std::string_view s);

// Comma-separated list of supported dtypes, for config enum "allowed values"
// and validation ("f32,f16,bf16,int8").
std::string_view supported_csv();

// Encode a float32 vector into a headered on-disk blob of the given dtype.
std::vector<uint8_t> encode(VectorType t, const std::vector<float>& v);

// Decode an on-disk blob (headered OR legacy raw f16/f32) of `expected_dims`
// dimensions into `out` (always resized to expected_dims). Returns true on
// success. On any failure (null, size/dim mismatch, unknown dtype, corrupt
// header) `out` is filled with zeros and false is returned — the caller owns
// the warning policy.
bool decode(const void* blob, int blob_bytes, int expected_dims,
            std::vector<float>& out);

}  // namespace ragger::vector_codec
