/**
 * Strict integer parsing for user-facing inputs (C §1.10).
 *
 * Header-only. Same rule as parse_agent_handle's rowid suffix: a positive
 * integer is non-empty, ASCII digits only, first digit 1-9 (so "0" and any
 * leading-zero form fail), and must fit in int. std::stoi accepts "12abc",
 * " 1", "+1", and "007" — wrong for route parameters and query params.
 */
#pragma once

#include <charconv>
#include <string_view>

namespace ragger::util {

inline bool parse_positive_int(std::string_view s, int& out) {
    if (s.empty()) return false;
    if (s[0] < '1' || s[0] > '9') return false;  // rejects 0, +x, -x, space, zeros
    for (char c : s)
        if (c < '0' || c > '9') return false;
    int v = 0;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) return false;  // overflow
    out = v;
    return true;
}

} // namespace ragger::util
