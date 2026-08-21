/**
 * TokenizerWrapper implementation using tokenizers-cpp
 */
#include "lang.h"
#include "tokenizer_wrapper.h"
#include <tokenizers_cpp.h>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ragger {

namespace {

// Scrub malformed UTF-8 before it ever reaches the Rust tokenizers-cpp FFI
// boundary. HFTokenizer::Encode() panics inside Rust on invalid UTF-8
// (`Result::unwrap()` on a `Utf8Error`), and that panic is marked
// non-unwinding across the FFI boundary — it aborts the *entire process*,
// taking the daemon down with it (observed: turn text with stray/truncated
// multi-byte sequences, likely from an import or a client-side encoding
// slip). Validate byte-by-byte per the UTF-8 spec (reject overlong
// encodings, surrogate code points D800-DFFF, and codepoints > 10FFFF) and
// replace each invalid byte with U+FFFD (the standard "best guess"
// replacement character), resyncing one byte at a time. This guarantees the
// tokenizer only ever sees well-formed UTF-8, at the cost of losing the
// handful of bad bytes (as U+FFFD) rather than losing the whole process.
std::string sanitize_utf8(const std::string& s) {
    static constexpr char kReplacement[] = "\xEF\xBF\xBD";  // U+FFFD
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    auto is_cont = [](unsigned char c) { return (c & 0xC0) == 0x80; };

    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {  // ASCII fast path
            out += static_cast<char>(c);
            ++i;
            continue;
        }

        int len = 0;
        unsigned char lead_mask = 0;
        unsigned int cp_min = 0;
        if ((c & 0xE0) == 0xC0)      { len = 2; lead_mask = 0x1F; cp_min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; lead_mask = 0x0F; cp_min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; lead_mask = 0x07; cp_min = 0x10000; }
        else {
            // Stray continuation byte or an invalid lead byte (0xF8-0xFF).
            out += kReplacement;
            ++i;
            continue;
        }

        if (i + static_cast<size_t>(len) > n) {
            // Truncated multi-byte sequence at end of string.
            out += kReplacement;
            ++i;
            continue;
        }

        unsigned int cp = c & lead_mask;
        bool valid = true;
        for (int k = 1; k < len; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if (!is_cont(cc)) { valid = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (valid && (cp < cp_min || cp > 0x10FFFF ||
                     (cp >= 0xD800 && cp <= 0xDFFF))) {
            valid = false;  // overlong encoding, out of range, or surrogate
        }

        if (valid) {
            out.append(s, i, static_cast<size_t>(len));
            i += static_cast<size_t>(len);
        } else {
            out += kReplacement;
            ++i;  // resync one byte at a time
        }
    }
    return out;
}

}  // namespace

// PIMPL implementation
struct TokenizerWrapper::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    
    explicit Impl(const std::string& tokenizer_json_path) {
        // Read tokenizer.json into string
        std::ifstream file(tokenizer_json_path);
        if (!file.is_open()) {
            throw std::runtime_error(std::format(ragger::lang::ERR_TOKENIZER_OPEN, tokenizer_json_path));
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_content = buffer.str();
        
        if (json_content.empty()) {
            throw std::runtime_error(std::format(ragger::lang::ERR_TOKENIZER_EMPTY, tokenizer_json_path));
        }
        
        // Create tokenizer from JSON blob
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_content);
        if (!tokenizer) {
            throw std::runtime_error(ragger::lang::ERR_TOKENIZER_CREATE);
        }
    }
};

TokenizerWrapper::TokenizerWrapper(const std::string& tokenizer_json_path)
    : pImpl(std::make_unique<Impl>(tokenizer_json_path)) {
}

TokenizerWrapper::~TokenizerWrapper() = default;

std::vector<int32_t> TokenizerWrapper::encode(const std::string& text) const {
    if (!pImpl->tokenizer) {
        throw std::runtime_error(ragger::lang::ERR_TOKENIZER_NOT_INIT);
    }
    return pImpl->tokenizer->Encode(sanitize_utf8(text));
}

TokenizerWrapper::Encoded TokenizerWrapper::encode_with_mask(const std::string& text) const {
    Encoded result;
    result.input_ids = encode(text);
    
    // Attention mask: all 1s for real tokens
    result.attention_mask.resize(result.input_ids.size(), 1);
    
    return result;
}

} // namespace ragger
