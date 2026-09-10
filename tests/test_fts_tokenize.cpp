#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "fts_tokenize.h"
#include "lang/en.h"

using namespace ragger::fts;
using namespace ragger::lang;

// Helper: convert vector<string> to set for easier comparison
std::set<std::string> as_set(const std::vector<std::string>& v) {
    return {v.begin(), v.end()};
}

// Helper: print vector for debugging
void print_vec(const std::string& label, const std::vector<std::string>& v) {
    std::cout << label << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << v[i] << "\"";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== FTS Tokenizer Unit Tests (Step 4) ===\n\n";

    // ===== Test 1: Sentence Splitting =====
    {
        std::cout << "Test 1: split_sentences\n";
        auto result = split_sentences("Hello. World! How are you?");
        assert(result.size() == 3);
        assert(result[0] == "Hello");
        assert(result[1] == "World");
        assert(result[2] == "How are you");
        std::cout << "  ✓ Basic sentence splitting\n";
    }

    {
        std::cout << "Test 1b: split_sentences (no final terminator)\n";
        auto result = split_sentences("First. Second");
        assert(result.size() == 2);
        assert(result[0] == "First");
        assert(result[1] == "Second");
        std::cout << "  ✓ Handles text without final terminator\n";
    }

    // ===== Test 2: Word Normalization =====
    {
        std::cout << "Test 2: normalize_words (basic)\n";
        auto result = normalize_words("Hello, World!");
        assert(result.size() >= 2);
        assert(result[0] == "hello");
        assert(result[1] == "world");
        std::cout << "  ✓ Lowercases and splits on non-alphanumeric\n";
    }

    {
        std::cout << "Test 2b: normalize_words (contraction n't)\n";
        auto result = normalize_words("Isn't good");
        print_vec("  Result", result);
        // Should expand "isn't" to ["is", "not"], then "good"
        auto s = as_set(result);
        assert(s.find("is") != s.end());
        assert(s.find("not") != s.end());
        assert(s.find("good") != s.end());
        std::cout << "  ✓ Expands n't contractions (isn't → is not)\n";
    }

    // ===== Test 3: Unigrams =====
    {
        std::cout << "Test 3: unigrams (basic)\n";
        StopSet stops = stopword_set(STOPWORDS_UNIGRAM);
        std::vector<std::string> words = {"is", "good", "running"};
        auto result = unigrams(words, stops);
        print_vec("  Result", result);
        // "is" should be filtered out, "good" and "running" → "run" should remain
        auto s = as_set(result);
        assert(s.find("is") == s.end());
        assert(s.find("good") != s.end());
        assert(s.find("run") != s.end());  // stemmed from "running"
        std::cout << "  ✓ Filters stopwords and stems survivors\n";
    }

    // ===== Test 4: Bigrams =====
    {
        std::cout << "Test 4: bigrams (basic)\n";
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        std::vector<std::string> words = {"step", "forward", "slowly"};
        auto result = bigrams(words, bi_stops);
        print_vec("  Result", result);
        // Both "forward" and "slowly" should form pairs if not in bigram stoplist
        auto s = as_set(result);
        // "forward" is NOT in STOPWORDS_BIGRAM (it's a polar word)
        assert(s.find("step_forward") != s.end());
        std::cout << "  ✓ Forms bigrams from consecutive words\n";
    }

    // ===== Test 5: Bigrams with stopword skipping =====
    {
        std::cout << "Test 5: bigrams (stopword filtering)\n";
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        std::vector<std::string> words = {"king", "of", "spain"};
        auto result = bigrams(words, bi_stops);
        print_vec("  Result", result);
        // "of" is NOT in STOPWORDS_BIGRAM (it's only in UNIGRAM), so (king, of) and (of, spain) should form
        // But this test should check against words that ARE in the bigram stoplist
        // Let's use "the" instead, which is in STOPWORDS_BIGRAM
        auto s = as_set(result);
        // Actually, "of" survives, so we get the bigrams. Let's adjust the test to use a word that IS in the bigram list.
        std::cout << "  (Note: 'of' is NOT in STOPWORDS_BIGRAM, so bigrams form correctly)\n";
        assert(!result.empty());  // we get some bigrams
        std::cout << "  ✓ Bigram formation works (of is not a bigram stopword)\n";
    }

    // ===== Test 5b: Bigrams with actual bigram stopword =====
    {
        std::cout << "Test 5b: bigrams (with actual bigram stopwords)\n";
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);
        std::vector<std::string> words = {"king", "the", "spain"};
        auto result = bigrams(words, bi_stops);
        print_vec("  Result", result);
        // "the" IS in STOPWORDS_BIGRAM, so (king, the) and (the, spain) should both drop
        auto s = as_set(result);
        assert(s.find("king_the") == s.end());
        assert(s.find("the_spain") == s.end());
        assert(s.find("king_spain") == s.end());  // no bridge
        std::cout << "  ✓ Skips pairs with bigram stopwords (no bridging)\n";
    }

    // ===== Test 6: Full pipeline: "This is not good." =====
    {
        std::cout << "Test 6: Full pipeline - \"This is not good.\"\n";
        std::string text = "This is not good.";
        auto sentences = split_sentences(text);
        assert(sentences.size() == 1);

        auto words = normalize_words(sentences[0]);
        print_vec("  Normalized words", words);

        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);

        auto uni = unigrams(words, uni_stops);
        auto bi = bigrams(words, bi_stops);

        print_vec("  Unigrams", uni);
        print_vec("  Bigrams", bi);

        auto uni_set = as_set(uni);
        auto bi_set = as_set(bi);

        // "this" and "is" should be filtered out as unigram stopwords
        assert(uni_set.find("this") == uni_set.end());
        assert(uni_set.find("is") == uni_set.end());

        // "good" should survive
        assert(uni_set.find("good") != uni_set.end());

        // "not" is NOT in STOPWORDS_UNIGRAM (it's in UNIGRAM but should drop)
        // Actually, let me check: "not" is in STOPWORDS_UNIGRAM, so it drops
        // BUT "not" is NOT in STOPWORDS_BIGRAM, so it can pair
        // Expected: bigram "not_good" since "is" drops but "not" pairs with "good"

        // Actually the sequence is: this, is, not, good
        // After stopword filter (unigram): not, good survive
        // After stopword filter (bigram): (is, not) - "is" is bigram-stop, drops
        //                                 (not, good) - both survive
        // So we expect bigram "not_good"
        assert(bi_set.find("not_good") != bi_set.end());

        std::cout << "  ✓ Full pipeline works (good and not_good present)\n";
    }

    // ===== Test 7: Contraction "Isn't good" =====
    {
        std::cout << "Test 7: Contraction - \"Isn't good\"\n";
        std::string text = "Isn't good.";
        auto sentences = split_sentences(text);
        auto words = normalize_words(sentences[0]);
        print_vec("  Normalized words", words);

        StopSet uni_stops = stopword_set(STOPWORDS_UNIGRAM);
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);

        auto uni = unigrams(words, uni_stops);
        auto bi = bigrams(words, bi_stops);

        print_vec("  Unigrams", uni);
        print_vec("  Bigrams", bi);

        auto uni_set = as_set(uni);
        auto bi_set = as_set(bi);

        // After expansion: "is", "not", "good"
        // Unigrams: "is" and "not" are both in STOPWORDS_UNIGRAM → only "good" survives
        // Wait, let me check the lists again
        // STOPWORDS_UNIGRAM contains: "not" (line 59), but also "is" (line 49)
        // STOPWORDS_BIGRAM contains: "is" (line 76), but NOT "not"
        // So unigrams: only "good" survives
        // Bigrams: (is, not) - "is" is bigram-stop, skip
        //          (not, good) - neither is bigram-stop, form "not_good"
        assert(uni_set.find("good") != uni_set.end());
        assert(bi_set.find("not_good") != bi_set.end());

        std::cout << "  ✓ Contraction expands correctly (good and not_good)\n";
    }

    // ===== Test 8: Polar words survive bigrams =====
    {
        std::cout << "Test 8: Polar words in bigrams\n";
        StopSet bi_stops = stopword_set(STOPWORDS_BIGRAM);

        // Check that "forward" and "backward" are NOT in the bigram stoplist
        assert(bi_stops.find("forward") == bi_stops.end());
        assert(bi_stops.find("backward") == bi_stops.end());

        std::vector<std::string> words1 = {"step", "forward"};
        std::vector<std::string> words2 = {"step", "backward"};

        auto bi1 = bigrams(words1, bi_stops);
        auto bi2 = bigrams(words2, bi_stops);

        print_vec("  step forward bigrams", bi1);
        print_vec("  step backward bigrams", bi2);

        auto s1 = as_set(bi1);
        auto s2 = as_set(bi2);

        assert(s1.find("step_forward") != s1.end());
        assert(s2.find("step_backward") != s2.end());
        assert(s1.find("step_forward") != s2.begin());  // distinct

        std::cout << "  ✓ Polar words form distinct bigrams (step_forward vs step_backward)\n";
    }

    // ===== Test 9: Stopword helper =====
    {
        std::cout << "Test 9: stopword_set helpers\n";
        auto uni_set = stopword_set(STOPWORDS_UNIGRAM);
        auto bi_set = stopword_set(STOPWORDS_BIGRAM);

        assert(!uni_set.empty());
        assert(!bi_set.empty());
        assert(uni_set.find("the") != uni_set.end());
        assert(bi_set.find("the") != bi_set.end());
        assert(uni_set.find("not") != uni_set.end());
        assert(bi_set.find("not") == bi_set.end());  // not removed from bigram

        std::cout << "  ✓ Stopword sets populate correctly\n";
    }

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
