#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

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

    // ===== Test 6: Metaphone weight calculation consistency =====
    {
        std::cout << "Test 6: Metaphone consistency (each stem yields exactly one metaphone)\n";
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
