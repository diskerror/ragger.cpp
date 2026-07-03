// double_metaphone.cpp — see double_metaphone.h.
//
// Faithful port of Lawrence Philips' Double Metaphone (CUJ, June 2000). The
// structure mirrors the canonical reference implementation: a cursor walks the
// upper-cased word, appending to a primary and an alternate key; the two keys
// diverge only where the algorithm sees a plausible second pronunciation.
//
// This file intentionally has zero Ragger dependencies so it can be unit-tested
// in isolation and reused anywhere.

#include "ragger/double_metaphone.h"

#include <array>
#include <cctype>

namespace ragger {

namespace {

class Encoder {
public:
    explicit Encoder(std::string_view in, size_t max_length)
        : max_length_(max_length) {
        // Upper-case, keep only A-Z. Non-alpha is dropped so punctuation and
        // digits never derail the cursor.
        word_.reserve(in.size());
        for (unsigned char c : in) {
            if (std::isalpha(c)) word_ += static_cast<char>(std::toupper(c));
        }
        length_ = static_cast<int>(word_.size());
        last_   = length_ - 1;
    }

    std::vector<std::string> run() {
        if (length_ == 0) return {};

        int current = 0;

        // Skip these silent initial letter pairs / letters.
        if (starts_at(0, {"GN", "KN", "PN", "WR", "PS"})) current += 1;

        // Initial 'X' is pronounced 'S' (e.g. "Xavier").
        if (word_[0] == 'X') { add("S"); current += 1; }

        while ((int)primary_.size() < (int)max_length_ ||
               (int)secondary_.size() < (int)max_length_) {
            if (current >= length_) break;
            current = step(current);
        }

        std::string p = primary_.substr(0, max_length_);
        std::string s = secondary_.substr(0, max_length_);
        std::vector<std::string> out;
        out.push_back(p);
        if (!s.empty() && s != p) out.push_back(s);
        return out;
    }

private:
    std::string word_;
    int length_ = 0;
    int last_   = 0;
    size_t max_length_ = 4;
    std::string primary_;
    std::string secondary_;

    void add(const char* both) { primary_ += both; secondary_ += both; }
    void add(const char* p, const char* s) { primary_ += p; secondary_ += s; }

    char at(int i) const {
        if (i < 0 || i >= length_) return '\0';
        return word_[i];
    }

    bool is_vowel(int i) const {
        char c = at(i);
        return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y';
    }

    // Does word_ contain `s` at position start?
    bool at_pos(int start, std::string_view s) const {
        if (start < 0) return false;
        if (start + (int)s.size() > length_) return false;
        return word_.compare(start, s.size(), s) == 0;
    }

    bool starts_at(int start, std::initializer_list<std::string_view> opts) const {
        for (auto o : opts) if (at_pos(start, o)) return true;
        return false;
    }

    // Slavo-Germanic words get special 'J' / 'W' handling.
    bool slavo_germanic() const {
        return word_.find('W') != std::string::npos ||
               word_.find('K') != std::string::npos ||
               word_.find("CZ") != std::string::npos ||
               word_.find("WITZ") != std::string::npos;
    }

    int step(int current);
};

int Encoder::step(int current) {
    char c = word_[current];
    switch (c) {
    case 'A': case 'E': case 'I': case 'O': case 'U': case 'Y':
        if (current == 0) add("A");   // only vowel at start counts
        return current + 1;

    case 'B':
        add("P");
        return (at(current + 1) == 'B') ? current + 2 : current + 1;

    case 'C':
        // Various Germanic / Italian / Spanish special cases.
        if (current > 1 && !is_vowel(current - 2) && at_pos(current - 1, "ACH") &&
            at(current + 2) != 'I' &&
            (at(current + 2) != 'E' || at_pos(current - 2, "BACHER") ||
             at_pos(current - 2, "MACHER"))) {
            add("K");
            return current + 2;
        }
        if (current == 0 && at_pos(current, "CAESAR")) { add("S"); return current + 2; }
        if (at_pos(current, "CHIA")) { add("K"); return current + 2; }
        if (at_pos(current, "CH")) {
            if (current > 0 && at_pos(current, "CHAE")) { add("K", "X"); return current + 2; }
            if (current == 0 &&
                (starts_at(current + 1, {"HARAC", "HARIS"}) ||
                 starts_at(current + 1, {"HOR", "HYM", "HIA", "HEM"})) &&
                !at_pos(0, "CHORE")) {
                add("K"); return current + 2;
            }
            if (starts_at(0, {"VAN ", "VON "}) || at_pos(0, "SCH") ||
                starts_at(current - 2, {"ORCHES", "ARCHIT", "ORCHID"}) ||
                (at(current + 2) == 'T' || at(current + 2) == 'S') ||
                ((current == 0 || starts_at(current - 1, {"A", "O", "U", "E"})) &&
                 (starts_at(current + 2, {"L", "R", "N", "M", "B", "H", "F", "V", "W", " "}) ||
                  current + 1 == last_))) {
                add("K");
            } else if (current > 0) {
                add(at_pos(0, "MC") ? "K" : "X", "K");
            } else {
                add("X");
            }
            return current + 2;
        }
        if (at_pos(current, "CZ") && !at_pos(current - 2, "WICZ")) {
            add("S", "X"); return current + 2;
        }
        if (at_pos(current + 1, "CIA")) { add("X"); return current + 3; }
        if (at_pos(current, "CC") && !(current == 1 && at(0) == 'M')) {
            if (starts_at(current + 2, {"I", "E", "H"}) && !at_pos(current + 2, "HU")) {
                if ((current == 1 && at(current - 1) == 'A') ||
                    starts_at(current - 1, {"UCCEE", "UCCES"})) {
                    add("KS");
                } else {
                    add("X");
                }
                return current + 3;
            }
            add("K"); return current + 2;
        }
        if (starts_at(current, {"CK", "CG", "CQ"})) { add("K"); return current + 2; }
        if (starts_at(current, {"CI", "CE", "CY"})) {
            if (starts_at(current, {"CIO", "CIE", "CIA"})) add("S", "X");
            else add("S");
            return current + 2;
        }
        add("K");
        if (starts_at(current + 1, {" C", " Q", " G"})) return current + 3;
        if (starts_at(current + 1, {"C", "K", "Q"}) && !starts_at(current + 1, {"CE", "CI"}))
            return current + 2;
        return current + 1;

    case 'D':
        if (at_pos(current, "DG")) {
            if (starts_at(current + 2, {"I", "E", "Y"})) { add("J"); return current + 3; }
            add("TK"); return current + 2;
        }
        if (starts_at(current, {"DT", "DD"})) { add("T"); return current + 2; }
        add("T"); return current + 1;

    case 'F':
        add("F");
        return (at(current + 1) == 'F') ? current + 2 : current + 1;

    case 'G':
        if (at(current + 1) == 'H') {
            if (current > 0 && !is_vowel(current - 1)) { add("K"); return current + 2; }
            if (current == 0) {
                if (at(current + 2) == 'I') add("J");
                else add("K");
                return current + 2;
            }
            if ((current > 1 && starts_at(current - 2, {"B", "H", "D"})) ||
                (current > 2 && starts_at(current - 3, {"B", "H", "D"})) ||
                (current > 3 && starts_at(current - 4, {"B", "H"}))) {
                return current + 2;
            }
            if (current > 2 && at(current - 1) == 'U' &&
                starts_at(current - 3, {"C", "G", "L", "R", "T"})) {
                add("F");
            } else if (current > 0 && at(current - 1) != 'I') {
                add("K");
            }
            return current + 2;
        }
        if (at(current + 1) == 'N') {
            if (current == 1 && is_vowel(0) && !slavo_germanic()) { add("KN", "N"); }
            else if (!at_pos(current + 2, "EY") && at(current + 1) != 'Y' && !slavo_germanic()) {
                add("N", "KN");
            } else {
                add("KN");
            }
            return current + 2;
        }
        if (at_pos(current + 1, "LI") && !slavo_germanic()) { add("KL", "L"); return current + 2; }
        if (current == 0 && (at(current + 1) == 'Y' ||
            starts_at(current + 1, {"ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE",
                                     "EI", "ER"}))) {
            add("K", "J"); return current + 2;
        }
        if ((at_pos(current + 1, "ER") || at(current + 1) == 'Y') &&
            !starts_at(0, {"DANGER", "RANGER", "MANGER"}) &&
            !starts_at(current - 1, {"E", "I"}) &&
            !starts_at(current - 1, {"RGY", "OGY"})) {
            add("K", "J"); return current + 2;
        }
        if (starts_at(current + 1, {"E", "I", "Y"}) || starts_at(current - 1, {"AGGI", "OGGI"})) {
            if (starts_at(0, {"VAN ", "VON "}) || at_pos(0, "SCH") || at_pos(current + 1, "ET")) {
                add("K");
            } else if (at_pos(current + 1, "IER ")) {
                add("J");
            } else {
                add("J", "K");
            }
            return current + 2;
        }
        add("K");
        return (at(current + 1) == 'G') ? current + 2 : current + 1;

    case 'H':
        if ((current == 0 || is_vowel(current - 1)) && is_vowel(current + 1)) {
            add("H"); return current + 2;
        }
        return current + 1;

    case 'J':
        if (at_pos(current, "JOSE") || starts_at(0, {"SAN "})) {
            if ((current == 0 && at(current + 4) == ' ') || starts_at(0, {"SAN "})) add("H");
            else add("J", "H");
            return current + 1;
        }
        if (current == 0 && !at_pos(current, "JOSE")) add("J", "A");
        else if (is_vowel(current - 1) && !slavo_germanic() &&
                 (at(current + 1) == 'A' || at(current + 1) == 'O')) {
            add("J", "H");
        } else if (current == last_) {
            add("J", "");
        } else if (!starts_at(current + 1, {"L", "T", "K", "S", "N", "M", "B", "Z"}) &&
                   !starts_at(current - 1, {"S", "K", "L"})) {
            add("J");
        }
        return (at(current + 1) == 'J') ? current + 2 : current + 1;

    case 'K':
        add("K");
        return (at(current + 1) == 'K') ? current + 2 : current + 1;

    case 'L':
        if (at(current + 1) == 'L') {
            if ((current == length_ - 3 &&
                 starts_at(current - 1, {"ILLO", "ILLA", "ALLE"})) ||
                ((starts_at(last_ - 1, {"AS", "OS"}) || starts_at(last_, {"A", "O"})) &&
                 at_pos(current - 1, "ALLE"))) {
                add("L", "");
                return current + 2;
            }
            add("L"); return current + 2;
        }
        add("L"); return current + 1;

    case 'M':
        if ((at_pos(current - 1, "UMB") &&
             (current + 1 == last_ || at_pos(current + 2, "ER"))) ||
            at(current + 1) == 'M') {
            add("M"); return current + 2;
        }
        add("M"); return current + 1;

    case 'N':
        add("N");
        return (at(current + 1) == 'N') ? current + 2 : current + 1;

    case 'P':
        if (at(current + 1) == 'H') { add("F"); return current + 2; }
        add("P");
        return (starts_at(current + 1, {"P", "B"})) ? current + 2 : current + 1;

    case 'Q':
        add("K");
        return (at(current + 1) == 'Q') ? current + 2 : current + 1;

    case 'R':
        if (current == last_ && !slavo_germanic() && at_pos(current - 2, "IE") &&
            !starts_at(current - 4, {"ME", "MA"})) {
            add("", "R");
        } else {
            add("R");
        }
        return (at(current + 1) == 'R') ? current + 2 : current + 1;

    case 'S':
        if (starts_at(current - 1, {"ISL", "YSL"})) return current + 1;
        if (current == 0 && at_pos(current, "SUGAR")) { add("X", "S"); return current + 1; }
        if (at_pos(current, "SH")) {
            if (starts_at(current + 1, {"HEIM", "HOEK", "HOLM", "HOLZ"})) add("S");
            else add("X");
            return current + 2;
        }
        if (at_pos(current, "SIO") || at_pos(current, "SIA") || at_pos(current, "SIAN")) {
            if (!slavo_germanic()) add("S", "X"); else add("S");
            return current + 3;
        }
        if ((current == 0 && starts_at(current + 1, {"M", "N", "L", "W"})) ||
            at(current + 1) == 'Z') {
            add("S", "X");
            return (at(current + 1) == 'Z') ? current + 2 : current + 1;
        }
        if (at_pos(current, "SC")) {
            if (at(current + 2) == 'H') {
                if (starts_at(current + 3, {"OO", "ER", "EN", "UY", "ED", "EM"})) {
                    if (starts_at(current + 3, {"ER", "EN"})) add("X", "SK");
                    else add("SK");
                } else if (current == 0 && !is_vowel(3) && at(3) != 'W') {
                    // word-initial SCH + consonant (German/Dutch): 'schmidt'
                    add("X", "S");
                } else {
                    add("X");
                }
                return current + 3;
            }
            if (starts_at(current + 2, {"I", "E", "Y"})) { add("S"); return current + 3; }
            add("SK"); return current + 3;
        }
        if (current == last_ && starts_at(current - 2, {"AI", "OI"})) add("", "S");
        else add("S");
        return (starts_at(current + 1, {"S", "Z"})) ? current + 2 : current + 1;

    case 'T':
        if (at_pos(current, "TION") || at_pos(current, "TIA") || at_pos(current, "TCH")) {
            if (at_pos(current, "TCH")) { add("X"); return current + 3; }
            add("X"); return current + 3;
        }
        if (at_pos(current, "TH") || at_pos(current, "TTH")) {
            if (starts_at(current + 2, {"OM", "AM"}) ||
                starts_at(0, {"VAN ", "VON "}) || at_pos(0, "SCH")) {
                add("T");
            } else {
                add("0", "T");   // '0' represents the 'th' voiceless sound
            }
            return current + 2;
        }
        add("T");
        return (starts_at(current + 1, {"T", "D"})) ? current + 2 : current + 1;

    case 'V':
        add("F");
        return (at(current + 1) == 'V') ? current + 2 : current + 1;

    case 'W':
        if (at_pos(current, "WR")) { add("R"); return current + 2; }
        if (current == 0 && (is_vowel(current + 1) || at_pos(current, "WH"))) {
            if (is_vowel(current + 1)) add("A", "F");
            else add("A");
        }
        if ((current == last_ && is_vowel(current - 1)) ||
            starts_at(current - 1, {"EWSKI", "EWSKY", "OWSKI", "OWSKY"}) ||
            at_pos(0, "SCH")) {
            add("", "F"); return current + 1;
        }
        if (starts_at(current, {"WICZ", "WITZ"})) { add("TS", "FX"); return current + 4; }
        return current + 1;

    case 'X':
        if (!(current == last_ &&
              (starts_at(current - 3, {"IAU", "EAU"}) ||
               starts_at(current - 2, {"AU", "OU"})))) {
            add("KS");
        }
        return (starts_at(current + 1, {"C", "X"})) ? current + 2 : current + 1;

    case 'Z':
        if (at(current + 1) == 'H') { add("J"); return current + 2; }
        if (starts_at(current + 1, {"ZO", "ZI", "ZA"}) ||
            (slavo_germanic() && current > 0 && at(current - 1) != 'T')) {
            add("S", "TS");
        } else {
            add("S");
        }
        return (at(current + 1) == 'Z') ? current + 2 : current + 1;

    default:
        return current + 1;
    }
}

} // namespace

std::vector<std::string> double_metaphone(std::string_view word, size_t max_length) {
    Encoder enc(word, max_length);
    return enc.run();
}

std::string phonize(std::string_view text) {
    std::string out;
    std::string word;
    auto flush = [&]() {
        if (word.empty()) return;
        for (const auto& code : double_metaphone(word)) {
            if (code.empty()) continue;
            if (!out.empty()) out += ' ';
            out += code;
        }
        word.clear();
    };
    for (unsigned char c : text) {
        // A "word" is a maximal run of letters plus intra-word apostrophes
        // (don't, it's) so contractions stay intact — the apostrophe is
        // stripped by the encoder anyway.
        if (std::isalpha(c) || c == '\'') word += static_cast<char>(c);
        else flush();
    }
    flush();
    return out;
}

} // namespace ragger
