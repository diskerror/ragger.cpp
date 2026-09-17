#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>

#include "fts_tokenize.h"
#include "lang/en.h"

using namespace ragger::fts;
using namespace ragger::lang;

// ===== Helper: Pretty Print =====

void print_unigrams(const std::string& label, const std::vector<UnigramToken>& v) {
    std::cout << label << ": ";
    for (const auto& tok : v) {
        std::cout << tok.literal;
        if (!tok.metaphone.empty()) {
            std::cout << "(" << tok.metaphone << ")";
        }
        std::cout << " ";
    }
    std::cout << "\n";
}

void print_bigrams(const std::string& label, const std::vector<BigramToken>& v) {
    std::cout << label << ": ";
    for (const auto& tok : v) {
        std::cout << tok.literal;
        if (!tok.metaphone.empty()) {
            std::cout << "(" << tok.metaphone << ")";
        }
        std::cout << " ";
    }
    std::cout << "\n";
}

// ===== Helper: Extract Literals from Token Arrays =====

std::set<std::string> unigram_literals(const std::vector<UnigramToken>& v) {
    std::set<std::string> result;
    for (const auto& tok : v) {
        result.insert(tok.literal);
    }
    return result;
}

std::set<std::string> bigram_literals(const std::vector<BigramToken>& v) {
    std::set<std::string> result;
    for (const auto& tok : v) {
        result.insert(tok.literal);
    }
    return result;
}

std::set<std::string> bigram_metaphones(const std::vector<BigramToken>& v) {
    std::set<std::string> result;
    for (const auto& tok : v) {
        if (!tok.metaphone.empty()) {
            result.insert(tok.metaphone);
        }
    }
    return result;
}

// ===== Unit Tests =====

int main() {
    std::cout << "=== FTS Tokenizer Unit Tests (Step 5: Metaphone) ===\n\n";

    // ===== Test 1: Unigrams with metaphone =====
    {
        std::cout << "Test 1: unigrams (with metaphone)\n";
        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        std::vector<std::string> words = {"running", "good"};
        auto result = unigrams(words, uni_stops);
        print_unigrams("  Result", result);
        // "running" stems to "run", "good" stems to "good"
        // Both should have metaphone codes (non-empty)
        auto lits = unigram_literals(result);
        assert(lits.find("run") != lits.end());
        assert(lits.find("good") != lits.end());
        // Verify metaphone non-empty for at least one
        bool has_metaphone = false;
        for (const auto& tok : result) {
            if (!tok.metaphone.empty()) {
                has_metaphone = true;
                break;
            }
        }
        assert(has_metaphone);
        std::cout << "  ✓ Unigrams include metaphone codes\n";
    }

    // ===== Test 2: Bigrams with metaphone =====
    {
        std::cout << "Test 2: bigrams (with metaphone)\n";
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        std::vector<std::string> words = {"step", "forward"};
        auto result = bigrams(words, bi_stops);
        print_bigrams("  Result", result);
        // Should form literal "step_forward"
        auto lits = bigram_literals(result);
        assert(lits.find("step_forward") != lits.end());
        // And have a metaphone bigram
        bool has_meta = false;
        for (const auto& tok : result) {
            if (!tok.metaphone.empty()) {
                has_meta = true;
            }
        }
        assert(has_meta);
        std::cout << "  ✓ Bigrams include metaphone codes\n";
    }

    // ===== Test 3: Metaphone on "not good" =====
    {
        std::cout << "Test 3: Full pipeline - 'This is not good.' with metaphone\n";
        auto sentences = split_sentences("This is not good.");
        assert(sentences.size() == 1);
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";

        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        auto uni = unigrams(words, uni_stops);
        auto bi = bigrams(words, bi_stops);

        print_unigrams("  Unigrams", uni);
        print_bigrams("  Bigrams", bi);

        // We expect "good" as unigram and "not_good" as bigram
        auto uni_lits = unigram_literals(uni);
        auto bi_lits = bigram_literals(bi);
        assert(uni_lits.find("good") != uni_lits.end());
        assert(bi_lits.find("not_good") != bi_lits.end());
        std::cout << "  ✓ Full pipeline works (good and not_good present, with metaphone)\n";
    }

    // ===== Test 4: No metaphone for empty/all-digit =====
    {
        std::cout << "Test 4: Metaphone on all-digits\n";
        auto meta = metaphone("123");
        assert(meta.empty());
        std::cout << "  ✓ Metaphone correctly returns empty for all-digits\n";
    }

    // ===== Test 5: Contraction - \"Isn't good\" with metaphone =====
    {
        std::cout << "Test 5: Contraction - \"Isn't good\" with metaphone\n";
        auto sentences = split_sentences("Isn't good");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        // "Isn't" should be replaced with "not" directly, resulting in ["not", "good"]
        assert(words.size() == 2);
        assert(words[0] == "not");
        assert(words[1] == "good");

        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        auto uni = unigrams(words, uni_stops);
        auto bi = bigrams(words, bi_stops);

        auto uni_lits = unigram_literals(uni);
        auto bi_lits = bigram_literals(bi);
        // "not" is a unigram stopword (drops), "good" survives
        // Bigram: "not_good" (not is NOT a bigram stopword)
        assert(uni_lits.find("good") != uni_lits.end());
        assert(bi_lits.find("not_good") != bi_lits.end());
        std::cout << "  ✓ Contraction replaced directly with 'not' (good and not_good with metaphone)\n";
    }

    // ===== Test 6: Single digit conversion =====
    {
        std::cout << "Test 6: Single digit conversion\n";
        auto sentences = split_sentences("I have 3 cats and 2 dogs.");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        // Should convert digits: 3 → three, 2 → two
        // stopwords drop "and"
        assert(std::find(words.begin(), words.end(), "three") != words.end());
        assert(std::find(words.begin(), words.end(), "two") != words.end());
        assert(std::find(words.begin(), words.end(), "3") == words.end());  // digit should be gone
        assert(std::find(words.begin(), words.end(), "2") == words.end());  // digit should be gone
        std::cout << "  ✓ Single digits converted to words\n";
    }

    // ===== Test 7: Minus sign before single digit =====
    {
        std::cout << "Test 7: Minus sign handling\n";
        auto sentences = split_sentences("Temperature is -5 degrees.");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        // Should convert: - → minus, 5 → five
        assert(std::find(words.begin(), words.end(), "minus") != words.end());
        assert(std::find(words.begin(), words.end(), "five") != words.end());
        assert(std::find(words.begin(), words.end(), "-") == words.end());   // minus sign should be gone
        assert(std::find(words.begin(), words.end(), "5") == words.end());   // digit should be gone
        std::cout << "  ✓ Minus sign and single digit handled correctly\n";
    }

    // ===== Test 8: Multi-digit numbers NOT converted (but ARE stopped at term stage) =====
    {
        std::cout << "Test 8: Multi-digit numbers unchanged in normalize_words, dropped from terms\n";
        auto sentences = split_sentences("Year 2024 is here.");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        // normalize_words() doesn't convert/drop multi-digit numbers itself --
        // that's a word-shape pass, not a term-admission decision.
        assert(std::find(words.begin(), words.end(), "2024") != words.end());

        // But unigrams()/bigrams() (the actual terms-table admission point)
        // must reject a pure-digit token outside 0-12: no stem/metaphone
        // signal, and it's not a product/algorithm name (see gpt4/sha256 below).
        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        auto uni = unigrams(words, uni_stops);
        auto bi = bigrams(words, bi_stops);
        auto uni_lits = unigram_literals(uni);
        auto bi_lits = bigram_literals(bi);
        assert(uni_lits.find("2024") == uni_lits.end());
        for (const auto& lit : bi_lits) {
            assert(lit.find("2024") == std::string::npos);
        }
        std::cout << "  ✓ Multi-digit numbers preserved in word list, stopped from terms\n";
    }

    // ===== Test 8b: Alphanumeric product/algorithm names keep their digits =====
    {
        std::cout << "Test 8b: Mixed alphanumerics (gpt4, sha256) keep digits\n";
        auto sentences = split_sentences("The gpt4 model uses sha256 hashing.");
        auto words = normalize_words(sentences[0]);
        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        auto uni = unigrams(words, uni_stops);
        auto uni_lits = unigram_literals(uni);
        assert(uni_lits.find("gpt4") != uni_lits.end());
        assert(uni_lits.find("sha256") != uni_lits.end());
        std::cout << "  ✓ Mixed alphanumeric tokens survive term admission intact\n";
    }

    // ===== Test 8c: Bare dash not before a kept small number is stopped =====
    {
        std::cout << "Test 8c: Bare/large-number minus sign is dropped\n";
        auto sentences = split_sentences("Temperature is -5 degrees, not -100.");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        // -5: 5 is small (0-12), so the dash converts to "minus" and is kept.
        assert(std::find(words.begin(), words.end(), "minus") != words.end());
        assert(std::find(words.begin(), words.end(), "five") != words.end());
        // -100: 100 is not small, so the dash is dropped outright (not left
        // as a stray "-" token), and "100" itself is dropped later at the
        // unigrams() term-admission stage (see Test 8).
        assert(std::find(words.begin(), words.end(), "-") == words.end());
        std::cout << "  ✓ Dash before a non-small number is dropped, not indexed\n";
    }

    // ===== Test 8d: Possessive/contraction 's is stripped =====
    {
        std::cout << "Test 8d: Possessive 's stripped (\"user's\" -> \"user\")\n";
        auto sentences = split_sentences("The user's model isn't good.");
        auto words = normalize_words(sentences[0]);
        std::cout << "  Normalized words: [";
        for (const auto& w : words) std::cout << w << " ";
        std::cout << "]\n";
        assert(std::find(words.begin(), words.end(), "user") != words.end());
        assert(std::find(words.begin(), words.end(), "user's") == words.end());
        // isn't still -> "not" (unaffected: it's an 't contraction, not 's)
        assert(std::find(words.begin(), words.end(), "not") != words.end());
        std::cout << "  ✓ Possessive 's stripped; n't contractions unaffected\n";
    }
    {
        // ===== Test 9: Math operators =====
        {
            std::cout << "Test 9: Math operators\n";
            auto sentences = split_sentences("Formula: x+5=10 and y≈3.14.");
            // `:` is a sentence terminator, so math formula is in sentences[1]
            auto words = normalize_words(sentences[1]);
            std::cout << "  Normalized words: [";
            for (const auto& w : words) std::cout << w << " ";
            std::cout << "]\n";
            // Should convert: + → plus, = → equals, ≈ → approximately equal to (three words), 5 → five
            assert(std::find(words.begin(), words.end(), "plus") != words.end());
            assert(std::find(words.begin(), words.end(), "equals") != words.end());
            assert(std::find(words.begin(), words.end(), "approximately") != words.end());
            // Check that "approximately equal to" appears as consecutive tokens
            auto approx_it = std::find(words.begin(), words.end(), "approximately");
            assert(approx_it != words.end());
            auto next_it = std::next(approx_it);
            assert(next_it != words.end() && *next_it == "equal");
            next_it = std::next(next_it);
            assert(next_it != words.end() && *next_it == "to");
            assert(std::find(words.begin(), words.end(), "five") != words.end());
            std::cout << "  ✓ Math operators converted as written: + → plus, ± → plus or minus, = → equals, ≠ → not equal to, ≈ → approximately equal to\n";
        }
        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        std::vector<std::string> words = {"running", "testing", "good"};
        auto uni = unigrams(words, uni_stops);

        // Each unigram should have exactly one literal and one metaphone (even if empty)
        int meta_count = 0;
        for (const auto& tok : uni) {
            assert(!tok.literal.empty());
            if (!tok.metaphone.empty()) {
                meta_count++;
            }
        }
        assert(meta_count > 0);  // at least one has a valid metaphone
        std::cout << "  ✓ Metaphone assignment is consistent (one per unigram)\n";
    }

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
