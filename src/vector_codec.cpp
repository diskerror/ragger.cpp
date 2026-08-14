// vector_codec.cpp — see vector_codec.h for the on-disk format contract.

#include "ragger/vector_codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ragger::vector_codec {

// -----------------------------------------------------------------------
// Scalar dtype conversions. In-memory values are always f32; these convert
// to/from the packed on-disk representations.
// -----------------------------------------------------------------------

// IEEE 754 half. _Float16 is a native type on Apple Silicon (ARMv8.2-FP16)
// and modern x86 toolchains; the compiler emits correct round-to-nearest
// conversions, so we don't hand-roll the bit twiddling.
static inline uint16_t f32_to_f16(float f) {
    _Float16 h = static_cast<_Float16>(f);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}
static inline float f16_to_f32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

// bfloat16 = the high 16 bits of the float32 bit pattern. Same exponent range
// as f32 (8 exponent bits), only 7 mantissa bits. We round-to-nearest-even
// rather than truncate: add the rounding bias derived from the low 16 bits and
// the lsb of the retained half before shifting. NaN is preserved (never
// rounded into an infinity).
static inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        // NaN — keep it a NaN (set a mantissa bit in the top half).
        return static_cast<uint16_t>((x >> 16) | 0x0040u);
    }
    const uint32_t lsb          = (x >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}
static inline float bf16_to_f32(uint16_t bits) {
    uint32_t x = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// -----------------------------------------------------------------------
// Little-endian header (de)serialization helpers.
// -----------------------------------------------------------------------
static inline void put_u16le(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xff);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
}
static inline uint16_t get_u16le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static inline void put_f32le(uint8_t* p, float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    p[0] = static_cast<uint8_t>(x & 0xff);
    p[1] = static_cast<uint8_t>((x >> 8) & 0xff);
    p[2] = static_cast<uint8_t>((x >> 16) & 0xff);
    p[3] = static_cast<uint8_t>((x >> 24) & 0xff);
}
static inline float get_f32le(const uint8_t* p) {
    uint32_t x = static_cast<uint32_t>(p[0]) |
                 (static_cast<uint32_t>(p[1]) << 8) |
                 (static_cast<uint32_t>(p[2]) << 16) |
                 (static_cast<uint32_t>(p[3]) << 24);
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

static constexpr uint8_t kMagic0 = 'R';
static constexpr uint8_t kMagic1 = 'V';
static constexpr uint8_t kMagic2 = '1';

// Bytes-per-dimension of the packed payload for a given dtype.
static int payload_stride(VectorType t) {
    switch (t) {
        case VectorType::F32:  return 4;
        case VectorType::F16:  return 2;
        case VectorType::BF16: return 2;
        case VectorType::INT8: return 1;
    }
    return 0;
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

std::optional<VectorType> parse(std::string_view s) {
    // lowercase compare without allocating.
    auto eq = [&](std::string_view lit) {
        if (s.size() != lit.size()) return false;
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != lit[i]) return false;
        }
        return true;
    };
    if (eq("f32")  || eq("float32")) return VectorType::F32;
    if (eq("f16")  || eq("float16") || eq("half")) return VectorType::F16;
    if (eq("bf16") || eq("bfloat16")) return VectorType::BF16;
    if (eq("int8") || eq("i8") || eq("q8")) return VectorType::INT8;
    return std::nullopt;
}

std::string to_string(VectorType t) {
    switch (t) {
        case VectorType::F32:  return "f32";
        case VectorType::F16:  return "f16";
        case VectorType::BF16: return "bf16";
        case VectorType::INT8: return "int8";
    }
    return "f16";
}

std::string canonical(std::string_view s, std::string_view fallback) {
    if (auto t = parse(s)) return to_string(*t);
    return std::string(fallback);
}

bool is_supported(std::string_view s) { return parse(s).has_value(); }

std::string_view supported_csv() { return "f32,f16,bf16,int8"; }

std::vector<uint8_t> encode(VectorType t, const std::vector<float>& v) {
    const int dims   = static_cast<int>(v.size());
    const int stride = payload_stride(t);

    // Symmetric per-vector int8 quantization: scale = max|x| / 127. Storing
    // the scale in the header lets decode() dequantize without a second pass.
    float scale = 0.0f;
    if (t == VectorType::INT8) {
        float maxabs = 0.0f;
        for (float x : v) maxabs = std::max(maxabs, std::fabs(x));
        scale = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
    }

    std::vector<uint8_t> out(kHeaderBytes + static_cast<size_t>(dims) * stride);
    uint8_t* p = out.data();
    p[0] = kMagic0; p[1] = kMagic1; p[2] = kMagic2;
    p[3] = static_cast<uint8_t>(t);
    p[4] = 0;  // flags
    p[5] = 0;  // reserved
    put_u16le(p + 6, static_cast<uint16_t>(dims));
    put_f32le(p + 8, scale);

    uint8_t* payload = p + kHeaderBytes;
    switch (t) {
        case VectorType::F32:
            for (int i = 0; i < dims; ++i) put_f32le(payload + i * 4, v[i]);
            break;
        case VectorType::F16:
            for (int i = 0; i < dims; ++i)
                put_u16le(payload + i * 2, f32_to_f16(v[i]));
            break;
        case VectorType::BF16:
            for (int i = 0; i < dims; ++i)
                put_u16le(payload + i * 2, f32_to_bf16(v[i]));
            break;
        case VectorType::INT8: {
            const float inv = 1.0f / scale;
            for (int i = 0; i < dims; ++i) {
                float q = std::lround(v[i] * inv);
                q = std::clamp(q, -127.0f, 127.0f);
                payload[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
            }
            break;
        }
    }
    return out;
}

// Decode a legacy (headerless) raw blob: dims*2 => f16, dims*4 => f32.
static bool decode_legacy(const uint8_t* blob, int blob_bytes, int dims,
                          std::vector<float>& out) {
    if (blob_bytes == dims * 2) {
        for (int i = 0; i < dims; ++i)
            out[i] = f16_to_f32(get_u16le(blob + i * 2));
        return true;
    }
    if (blob_bytes == dims * 4) {
        for (int i = 0; i < dims; ++i)
            out[i] = get_f32le(blob + i * 4);
        return true;
    }
    return false;
}

bool decode(const void* blob, int blob_bytes, int expected_dims,
            std::vector<float>& out) {
    out.assign(static_cast<size_t>(expected_dims), 0.0f);
    if (blob == nullptr || expected_dims <= 0) return false;
    const uint8_t* p = static_cast<const uint8_t*>(blob);

    const bool has_header =
        blob_bytes >= kHeaderBytes &&
        p[0] == kMagic0 && p[1] == kMagic1 && p[2] == kMagic2;

    if (!has_header) {
        return decode_legacy(p, blob_bytes, expected_dims, out);
    }

    const auto t     = static_cast<VectorType>(p[3]);
    const int  dims  = static_cast<int>(get_u16le(p + 6));
    const float scale = get_f32le(p + 8);
    if (dims != expected_dims) return false;

    const int stride = payload_stride(t);
    if (stride == 0) return false;  // unknown dtype
    if (blob_bytes != kHeaderBytes + dims * stride) return false;

    const uint8_t* payload = p + kHeaderBytes;
    switch (t) {
        case VectorType::F32:
            for (int i = 0; i < dims; ++i) out[i] = get_f32le(payload + i * 4);
            return true;
        case VectorType::F16:
            for (int i = 0; i < dims; ++i)
                out[i] = f16_to_f32(get_u16le(payload + i * 2));
            return true;
        case VectorType::BF16:
            for (int i = 0; i < dims; ++i)
                out[i] = bf16_to_f32(get_u16le(payload + i * 2));
            return true;
        case VectorType::INT8:
            for (int i = 0; i < dims; ++i)
                out[i] = static_cast<float>(static_cast<int8_t>(payload[i])) * scale;
            return true;
    }
    return false;
}

}  // namespace ragger::vector_codec
