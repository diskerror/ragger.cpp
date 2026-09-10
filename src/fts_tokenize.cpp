#include "fts_tokenize.h"
#include "lang/en.h"
#include "double_metaphone.h"

#include <algorithm>
#include <cctype>
#include <map>

// Include c_lib headers directly (no extern C needed for C++ functions)
#include "Stemmer.h"
#include "DoubleMetaphone.h"

namespace ragger::fts {

// ===== Text Segmentation =====

std::vector<std::string> split_sentences(std::string_view text) {
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        if (c == '.' || c == '!' || c == '?') {
            // Trim and add if non-empty
            std::string trimmed = current;
            trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
            trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
            if (!trimmed.empty()) {
                result.push_back(trimmed);
            }
            current.clear();
        }
    }

    // Handle remaining text (no final terminator)
    std::string trimmed = current;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    if (!trimmed.empty()) {
        result.push_back(trimmed);
    }

    return result;
}

// ===== Word Normalization & Contraction Expansion =====

std::vector<std::string> normalize_words(std::string_view sentence) {
    std::vector<std::string> result;

    // First pass: lowercase and split on non-alphanumeric, but preserve special symbols
    std::string lower_sent(sentence);
    std::transform(lower_sent.begin(), lower_sent.end(), lower_sent.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string current_word;
    for (size_t i = 0; i < lower_sent.size(); ++i) {
        char c = lower_sent[i];
        unsigned char uc = static_cast<unsigned char>(c);
        
        if (std::isalnum(c)) {
            current_word += c;
        } else if (c == '+' || c == '=' || c == '-') {
            // Preserve +, =, - as potential operator symbols
            if (!current_word.empty()) {
                result.push_back(current_word);
                current_word.clear();
            }
            result.push_back(std::string(1, c));
        } else if (uc == 0xC2 && i + 1 < lower_sent.size() && (unsigned char)lower_sent[i + 1] == 0xB1) {
            // UTF-8 plus-minus (±, U+00B1: C2 B1)
            if (!current_word.empty()) {
                result.push_back(current_word);
                current_word.clear();
            }
            result.push_back(std::string("\xC2\xB1"));
            i += 1;  // Skip next byte
        } else if (uc == 0xE2 && i + 2 < lower_sent.size() && 
                   (unsigned char)lower_sent[i + 1] == 0x89) {
            // UTF-8 multi-byte operators starting with E2 89:
            unsigned char byte3 = (unsigned char)lower_sent[i + 2];
            if (byte3 == 0xA0) {
                // ≠ (U+2260: E2 89 A0)
                if (!current_word.empty()) {
                    result.push_back(current_word);
                    current_word.clear();
                }
                result.push_back(std::string("\xE2\x89\xA0"));
                i += 2;
            } else if (byte3 == 0x88) {
                // ≈ (U+2248: E2 89 88)
                if (!current_word.empty()) {
                    result.push_back(current_word);
                    current_word.clear();
                }
                result.push_back(std::string("\xE2\x89\x88"));
                i += 2;
            } else if (byte3 == 0x92) {
                // − (U+2212: E2 88 92) — UTF-8 minus
                if (!current_word.empty()) {
                    result.push_back(current_word);
                    current_word.clear();
                }
                result.push_back(std::string("\xE2\x88\x92"));
                i += 2;
            } else {
                // Not a recognized operator, drop it
                if (!current_word.empty()) {
                    result.push_back(current_word);
                    current_word.clear();
                }
            }
        } else if (c == '\'' && i > 0 && i < lower_sent.size() - 1 && std::isalpha(lower_sent[i + 1])) {
            // Preserve apostrophe in contractions (e.g., "isn't", "don't")
            current_word += c;
        } else {
            if (!current_word.empty()) {
                result.push_back(current_word);
                current_word.clear();
            }
        }
    }
    if (!current_word.empty()) {
        result.push_back(current_word);
    }

    // Second pass: handle minus signs and operators before single digits
    // Detect operator followed by single digit and handle appropriately
    std::vector<std::string> with_operator_handling;
    for (size_t i = 0; i < result.size(); ++i) {
        const auto& word = result[i];
        bool is_minus_op = false;
        bool is_plus_op = false;
        
        // Check for minus-type operators
        if (word == "-" || word == "\xE2\x88\x92") {  // ASCII hyphen or UTF-8 minus
            is_minus_op = true;
        }
        // Check for plus-type operators
        if (word == "+" || word == "\xC2\xB1") {  // ASCII plus or UTF-8 plus-minus
            is_plus_op = true;
        }
        
        // If operator is followed by single digit, convert operator to word (no digit skip needed)
        if ((is_minus_op || is_plus_op) && i + 1 < result.size() && 
            result[i + 1].size() == 1 && std::isdigit(result[i + 1][0])) {
            // Replace operator with its word form
            if (is_minus_op) {
                with_operator_handling.push_back("minus");
            } else if (is_plus_op) {
                with_operator_handling.push_back("plus");
            }
        } else {
            with_operator_handling.push_back(word);
        }
    }

    // Third pass: replace operators and single digits with their word forms, and replace contractions
    std::vector<std::string> expanded;
    for (size_t i = 0; i < with_operator_handling.size(); ++i) {
        const auto& word = with_operator_handling[i];
        
        // Check if it's a single digit
        if (word.size() == 1 && std::isdigit(word[0])) {
            // Convert single digit to word
            bool digit_found = false;
            for (const auto& [digit, digit_word] : ragger::lang::SINGLE_DIGIT_WORDS) {
                if (word == digit) {
                    expanded.push_back(std::string(digit_word));
                    digit_found = true;
                    break;
                }
            }
            if (digit_found) {
                continue;
            }
        }
        
        // Check for math operators
        bool op_found = false;
        for (const auto& [op, op_word] : ragger::lang::MATH_SYMBOLS) {
            if (word == op) {
                expanded.push_back(std::string(op_word));
                op_found = true;
                break;
            }
        }
        if (op_found) {
            continue;
        }
        
        // Check for contractions
        bool found = false;
        for (const auto& [contraction, replacement] : ragger::lang::CONTRACTIONS_N_T) {
            if (word == contraction) {
                expanded.push_back(std::string(replacement));
                found = true;
                break;
            }
        }
        if (!found) {
            expanded.push_back(word);
        }
    }

    return expanded;
}

// ===== Stemming =====

std::string stem(std::string_view word) {
    return Diskerror::stem_en(word);
}

// ===== Metaphone =====

std::string metaphone(std::string_view stem) {
    if (stem.empty()) return "";

    // Call ragger::double_metaphone which returns vector<string> with primary and secondary
    auto codes = ragger::double_metaphone(stem);
    // Return ONLY the primary code (first element)
    if (!codes.empty() && !codes[0].empty()) {
        return codes[0];
    }
    return "";
}

// ===== Unigrams (literals + metaphone) =====

std::vector<UnigramToken> unigrams(const std::vector<std::string>& words,
                                   const StopSet& uni_stops) {
    std::vector<UnigramToken> result;

    for (const auto& word : words) {
        // Filter on RAW word
        if (uni_stops.find(word) != uni_stops.end()) {
            continue;  // stopword, skip
        }

        // Stem the survivor
        std::string stemmed = stem(word);
        if (stemmed.empty()) {
            continue;  // all-digit or malformed, skip
        }

        // Compute metaphone on the stem
        std::string meta = metaphone(stemmed);

        result.push_back({stemmed, meta});
    }

    return result;
}

// ===== Bigrams (literals + metaphone) =====

std::vector<BigramToken> bigrams(const std::vector<std::string>& words,
                                 const StopSet& bi_stops) {
    std::vector<BigramToken> result;

    if (words.size() < 2) {
        return result;  // need at least 2 words
    }

    for (size_t i = 0; i < words.size() - 1; ++i) {
        // Check if either raw word is a bigram stopword
        if (bi_stops.find(words[i]) != bi_stops.end() ||
            bi_stops.find(words[i + 1]) != bi_stops.end()) {
            continue;  // skip this pair (no bridging)
        }

        // Stem both
        std::string stem_a = stem(words[i]);
        std::string stem_b = stem(words[i + 1]);

        // Skip if either stem is empty
        if (stem_a.empty() || stem_b.empty()) {
            continue;
        }

        // Form literal bigram
        std::string literal = stem_a + "_" + stem_b;

        // Form metaphone bigram
        std::string meta_a = metaphone(stem_a);
        std::string meta_b = metaphone(stem_b);
        std::string meta_bigram;
        if (!meta_a.empty() && !meta_b.empty()) {
            meta_bigram = meta_a + "_" + meta_b;
        }
        // If either metaphone is empty, we still add the bigram (may have one part)

        result.push_back({literal, meta_bigram});
    }

    return result;
}

}  // namespace ragger::fts
