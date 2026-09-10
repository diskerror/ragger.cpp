/**
 * FTS tokenizer — v0.16 custom index pipeline.
 *
 * Pure, testable functions for splitting text → sentences → normalized words →
 * stemmed unigrams and bigrams, with metaphone tokens. No DB access.
 *
 * Pipeline: split sentences → normalize/expand contractions → stem → unigrams/bigrams
 * → metaphone on each stem → emit both literal + metaphone tokens.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <unordered_set>

namespace ragger::fts {

using StopSet = std::unordered_set<std::string_view>;

// ===== Text Segmentation =====

/**
 * split_sentences: Break text on sentence terminators . ! ? : ;
 * Returns: vector of trimmed sentence strings (no empty sentences).
 */
std::vector<std::string> split_sentences(std::string_view text);

// ===== Word Normalization =====

/**
 * normalize_words: Lowercase, expand contractions (e.g. isn't -> is not),
 * split on non-alphanumeric, drop empties.
 * Returns: vector of normalized words (NO stopword filtering yet).
 */
std::vector<std::string> normalize_words(std::string_view sentence);

// ===== Stemming =====

/**
 * stem: Apply Porter stemmer (from c_lib) to a word.
 * Returns: stemmed form (may be empty for all-digit or malformed input).
 */
std::string stem(std::string_view word);

// ===== Metaphone Tokens =====

/**
 * metaphone: Apply Double Metaphone to a stem, returning ONLY the primary code.
 * Returns: primary metaphone code (may be empty for pure-digit or odd input).
 */
std::string metaphone(std::string_view stem);

// ===== Unigram Tokens (literals + metaphone) =====

/**
 * struct UnigramToken: One unigram with both literal (stemmed) and metaphone form.
 */
struct UnigramToken {
    std::string literal;    // the stemmed word
    std::string metaphone;  // primary DMP code (may be empty)
};

/**
 * unigrams: Filter stopwords on RAW words, stem survivors, emit metaphone.
 * Returns: vector of UnigramToken (both literal and metaphone codes present,
 *          but metaphone may be empty).
 */
std::vector<UnigramToken> unigrams(const std::vector<std::string>& words,
                                   const StopSet& uni_stops);

// ===== Bigram Tokens (literals + metaphone) =====

/**
 * struct BigramToken: One bigram with both literal and metaphone forms.
 */
struct BigramToken {
    std::string literal;    // stem(a) + "_" + stem(b)
    std::string metaphone;  // meta(a) + "_" + meta(b)  (may have empty components)
};

/**
 * bigrams: Form consecutive word pairs within the sentence; skip if either
 * raw word is in bi_stops; stem and metaphone both. Returns as flat token stream.
 * Returns: vector of BigramToken.
 */
std::vector<BigramToken> bigrams(const std::vector<std::string>& words,
                                 const StopSet& bi_stops);

// ===== Helper: Stopword Set =====

/**
 * stopword_set: Convert a constexpr array of string_view into an unordered_set.
 * Template allows any array type; typically constexpr string_view arrays.
 */
template <size_t N>
StopSet stopword_set(const std::array<std::string_view, N>& arr) {
    return StopSet(arr.begin(), arr.end());
}

}  // namespace ragger::fts
