#pragma once
/**
 * utf8.h — UTF-8-safe string cutting (PHASE9B step 3c).
 *
 * Byte-count truncation of model output can land inside a multi-byte
 * sequence. SQLite TEXT and nlohmann::json both require valid UTF-8, so a
 * stored value ending in a lead-byte fragment makes every later read of that
 * row throw (this happened on the live DB: an episode cut at 5000 bytes ends
 * in a lone 0xE2). Any code that truncates text to a byte budget must cut on
 * a character boundary.
 */
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ragger {

/// The longest prefix of `s` that is at most `max_bytes` long and does not
/// end inside a multi-byte UTF-8 sequence. If `max_bytes >= s.size()` the
/// whole string is returned. For already-invalid input the result is still a
/// prefix of `s` of length <= max_bytes (never a crash, never an underflow):
/// a sequence at the cut point whose continuation bytes are missing or
/// malformed is excluded entirely.
inline std::string utf8_safe_prefix(std::string_view s, std::size_t max_bytes) {
    if (max_bytes >= s.size()) return std::string(s);
    std::size_t i = max_bytes;
    // Walk back over continuation bytes (10xxxxxx) to the sequence start.
    while (i > 0 && (static_cast<std::uint8_t>(s[i]) & 0xC0) == 0x80)
        --i;
    if (i == max_bytes) return std::string(s.substr(0, i));  // ASCII boundary
    // s[i] is a lead byte (or a lone continuation byte at i == 0). The
    // sequence starting at i reaches across the cut; include it only when
    // all of its bytes are present before max_bytes and well-formed.
    const auto lead = static_cast<std::uint8_t>(s[i]);
    std::size_t need = 0;
    if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    if (need >= 2 && i + need <= max_bytes) {
        for (std::size_t k = 1; k < need; ++k)
            if ((static_cast<std::uint8_t>(s[i + k]) & 0xC0) != 0x80)
                return std::string(s.substr(0, i));   // malformed: drop it
        return std::string(s.substr(0, i + need));    // complete: keep it
    }
    return std::string(s.substr(0, i));               // truncated/malformed: drop
}

} // namespace ragger
