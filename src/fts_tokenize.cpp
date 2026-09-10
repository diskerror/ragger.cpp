#include "fts_tokenize.h"
#include "lang/en.h"
#include "double_metaphone.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>

// Directly include c_lib headers since they're in the public include path
#include "Stemmer.h"

namespace ragger::fts {

// ===== Sentence Splitting =====

std::vector<std::string> split_sentences(std::string_view text) {
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        // Check for sentence terminators
        if (c == '.' || c == '!' || c == '?') {
            // Trim and store if non-empty
            std::string trimmed;
            for (char ch : current) {
                if (std::isspace(ch) && trimmed.empty()) continue;
                trimmed += ch;
            }
            // Remove trailing terminator
            if (!trimmed.empty() && (trimmed.back() == '.' || trimmed.back() == '!' || trimmed.back() == '?')) {
                trimmed.pop_back();
            }
            // Trim trailing whitespace
            while (!trimmed.empty() && std::isspace(trimmed.back())) {
                trimmed.pop_back();
            }
            if (!trimmed.empty()) {
                sentences.push_back(trimmed);
            }
            current.clear();
        }
    }

    // Handle any remaining text (no final terminator)
    if (!current.empty()) {
        std::string trimmed;
        for (char c : current) {
            if (std::isspace(c) && trimmed.empty()) continue;
            trimmed += c;
        }
        while (!trimmed.empty() && std::isspace(trimmed.back())) {
            trimmed.pop_back();
        }
        if (!trimmed.empty()) {
            sentences.push_back(trimmed);
        }
    }

    return sentences;
}

// ===== Word Normalization =====

std::vector<std::string> normalize_words(std::string_view sentence) {
    std::vector<std::string> words;
    std::string current;

    // Lowercase and process character by character
    for (size_t i = 0; i < sentence.size(); ++i) {
        char c = std::tolower(static_cast<unsigned char>(sentence[i]));

        if (std::isalnum(c) || c == '\'') {
            current += c;
        } else {
            // Non-alphanumeric delimiter
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }

    // Expand contractions (n't family critical)
    std::vector<std::string> expanded;
    for (const auto& word : words) {
        bool found = false;
        // Check n't contractions
        for (const auto& [contraction, expansion] : lang::CONTRACTIONS_N_T) {
            if (word == contraction) {
                expanded.push_back(std::string(expansion.first));
                expanded.push_back(std::string(expansion.second));
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

// ===== Unigrams =====

std::vector<std::string> unigrams(const std::vector<std::string>& words,
                                   const StopSet& uni_stops) {
    std::vector<std::string> result;

    for (const auto& word : words) {
        // Check stopword on raw word
        if (uni_stops.find(word) == uni_stops.end()) {
            // Not a stopword — stem it
            std::string stemmed = stem(word);
            if (!stemmed.empty()) {
                result.push_back(stemmed);
            }
        }
    }

    return result;
}

// ===== Bigrams =====

std::vector<std::string> bigrams(const std::vector<std::string>& words,
                                  const StopSet& bi_stops) {
    std::vector<std::string> result;

    // Form consecutive pairs from the raw word list
    for (size_t i = 0; i + 1 < words.size(); ++i) {
        const auto& word_a = words[i];
        const auto& word_b = words[i + 1];

        // Skip pair if EITHER raw word is a bigram stopword
        bool a_is_stop = (bi_stops.find(word_a) != bi_stops.end());
        bool b_is_stop = (bi_stops.find(word_b) != bi_stops.end());

        if (!a_is_stop && !b_is_stop) {
            // Both survive — stem and join
            std::string stem_a = stem(word_a);
            std::string stem_b = stem(word_b);
            if (!stem_a.empty() && !stem_b.empty()) {
                result.push_back(stem_a + "_" + stem_b);
            }
        }
    }

    return result;
}

// ===== Helper: Load Stopword Lists =====

StopSet stopword_set(const std::array<std::string_view, 132>& list) {
    StopSet result;
    for (auto word : list) {
        result.insert(word);
    }
    return result;
}

StopSet stopword_set(const std::array<std::string_view, 47>& list) {
    StopSet result;
    for (auto word : list) {
        result.insert(word);
    }
    return result;
}

}  // namespace ragger::fts
