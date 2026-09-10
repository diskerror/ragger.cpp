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

    // First pass: lowercase and split on non-alphanumeric (but preserve apostrophes for contractions)
    std::string lower_sent(sentence);
    std::transform(lower_sent.begin(), lower_sent.end(), lower_sent.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::string current_word;
    for (size_t i = 0; i < lower_sent.size(); ++i) {
        char c = lower_sent[i];
        if (std::isalnum(c)) {
            current_word += c;
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

    // Second pass: replace contractions with "not" directly
    std::vector<std::string> expanded;
    for (const auto& word : result) {
        // Search the contraction map for this word
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
