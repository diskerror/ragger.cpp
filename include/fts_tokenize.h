/**
 * FTS tokenizer — v0.16 custom index pipeline.
 *
 * Pure, testable functions for splitting text → sentences → normalized words →
 * stemmed unigrams and bigrams. No DB access, no metaphone (Step 5 adds that).
 *
 * Pipeline order (per sentence):
 *   1. split_sentences(text) → vector<sentence_text>
 *   2. normalize_words(sentence) → vector<raw_word> (with contractions expanded)
 *   3. stopword filter on raw word
 *   4. stem() each survivor
 *   5. unigrams/bigrams from the result
 *
 * Key constraints:
 *   - Stopword check is BEFORE stemming (Porter list is raw words).
 *   - Bigrams are formed WITHIN a single sentence only (no cross-sentence pairs).
 *   - Bigram formation skips pairs where EITHER raw word is a bigram stopword.
 *   - Contraction expansion (isn't → is not) happens in normalize_words.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

namespace ragger::fts {

/// A set of stopwords (cheap using unordered_set<string_view>).
using StopSet = std::unordered_set<std::string_view>;

// ===== Sentence and Word Normalization =====

/**
 * split_sentences(text) → vector<string>
 *
 * Split text on sentence terminators (`.`, `!`, `?`).
 * Removes the terminators from the output.
 * Returns non-empty sentences only.
 *
 * Example: "Hello. World!" → ["Hello", "World"]
 */
std::vector<std::string> split_sentences(std::string_view text);

/**
 * normalize_words(sentence) → vector<string>
 *
 * Lowercase, expand contractions (e.g., "isn't" → "is not"),
 * split on non-alphanumeric, drop empties.
 * Returns raw words BEFORE stopword filtering or stemming.
 *
 * Contraction map lives in lang/en.h (English-specific).
 * Non-negating contractions (it's, we'll, I'm) are optional;
 * n't family is mandatory (isn't, don't, won't, etc.).
 */
std::vector<std::string> normalize_words(std::string_view sentence);

/**
 * stem(word) → string
 *
 * Porter stemmer via c_lib wrapper (Diskerror::stem_en).
 * Returns the stemmed form of the word.
 *
 * Example: "running" → "run", "flies" → "fli"
 */
std::string stem(std::string_view word);

// ===== Unigrams and Bigrams =====

/**
 * unigrams(words, uni_stops) → vector<string>
 *
 * Filter the raw word list by STOPWORDS_UNIGRAM, then stem survivors.
 * Each stemmed word becomes one unigram token.
 *
 * Example (given "is", "good", "running" and stopwords excluding "good", "running"):
 *   Input words: ["is", "good", "running"]
 *   Stopword filter: ["good", "running"] survive
 *   After stemming: ["good", "run"]
 *   Output: ["good", "run"]
 */
std::vector<std::string> unigrams(const std::vector<std::string>& words,
                                   const StopSet& uni_stops);

/**
 * bigrams(words, bi_stops) → vector<string>
 *
 * Form consecutive word pairs (word[i], word[i+1]) from the raw word list.
 * Skip any pair if EITHER raw word is in STOPWORDS_BIGRAM.
 * Stem both survivors, join with '_' → one bigram token.
 *
 * Bigrams are formed WITHIN ONE SENTENCE ONLY. This function assumes
 * the input word list is from a single sentence (use split_sentences
 * first to enforce this).
 *
 * Example (given "step", "forward", "slowly" and stopwords excluding "forward"):
 *   Input words: ["step", "forward", "slowly"]
 *   Candidate pairs: (step, forward), (forward, slowly)
 *   After stopword check: both survive (neither "forward" nor others are bigram-stops)
 *   After stemming: (step, forward), (forward, slowli)
 *   Output: ["step_forward", "forward_slowli"]
 */
std::vector<std::string> bigrams(const std::vector<std::string>& words,
                                  const StopSet& bi_stops);

/**
 * Helper: Load stopword list into a StopSet for efficient lookup.
 * Used at startup to pre-populate the sets.
 */
StopSet stopword_set(const std::array<std::string_view, 132>& list);
StopSet stopword_set(const std::array<std::string_view, 47>& list);

}  // namespace ragger::fts
