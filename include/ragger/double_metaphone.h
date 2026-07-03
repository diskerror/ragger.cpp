// double_metaphone.h — phonetic ("sounds-like") encoding for Ragger search.
//
// Lawrence Philips' Double Metaphone algorithm (CUJ, June 2000): maps a word to
// 1 or 2 four-character-ish phonetic keys. Two keys are produced when a word has
// a plausible alternate pronunciation (e.g. foreign-origin spellings); most
// words yield only the primary.
//
// This is the "dolphining" sounds-like signal: it lets search match on how a
// phrase *sounds*, not just how it's spelled or what it means. Fed into a third
// FTS5 column and blended alongside vector + BM25 (see config.phon_weight).
//
// No external dependencies. Self-contained, header + one .cpp.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace ragger {

// Compute the Double Metaphone codes for a single word.
//   returns {primary} or {primary, alternate}.
// The alternate is only present when it differs from the primary.
// Non-alpha input yields an empty vector. Codes are upper-case; the class
// classically caps each key at 4 chars — we keep that cap (max_length=4).
std::vector<std::string> double_metaphone(std::string_view word,
                                          size_t max_length = 4);

// Phonize an arbitrary text run into a space-joined stream of Double Metaphone
// codes, suitable for storing in the `phon` column and matching via FTS5.
//
// Deliberately NO stopword filtering: every word contributes, because even
// function words carry the rhythm and sound of a phrase (Reid's call). For each
// word both the primary and (when present) the alternate code are emitted, so a
// sounds-like query can match either pronunciation.
//
// Example: phonize("Don't panic") -> "TNT PNK"  (both codes when they differ).
std::string phonize(std::string_view text);

} // namespace ragger
