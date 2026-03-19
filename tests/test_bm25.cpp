/**
 * BM25 unit tests
 */
#include "ragger/bm25.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>

using ragger::BM25Index;

// -----------------------------------------------------------------------
// Tokenizer tests
// -----------------------------------------------------------------------

void test_basic_tokenization() {
    auto tokens = BM25Index::tokenize("Hello World foo bar");
    // "Hello" → "hello", "World" → "world", "foo" → "foo", "bar" → "bar"
    assert(tokens.size() == 4);
    assert(tokens[0] == "hello");
    assert(tokens[1] == "world");
}

void test_short_tokens_filtered() {
    // Tokens under 3 chars should be filtered
    auto tokens = BM25Index::tokenize("I am a big cat");
    // "I"(1), "am"(2), "a"(1) filtered; "big"(3), "cat"(3) kept
    for (auto& t : tokens) {
        assert(t.size() >= 3);
    }
    assert(std::find(tokens.begin(), tokens.end(), "big") != tokens.end());
    assert(std::find(tokens.begin(), tokens.end(), "cat") != tokens.end());
}

void test_hex_strings_filtered() {
    // Bare hex strings should be filtered
    auto tokens = BM25Index::tokenize("deadbeef abc123ff real_word 0xCAFE");
    // hex-like tokens removed, "real_word" should survive
    for (auto& t : tokens) {
        // None of the pure hex tokens should be present
        assert(t != "deadbeef");
        assert(t != "abc123ff");
    }
}

void test_numbers_kept() {
    // Numbers with 3+ digits should be kept
    auto tokens = BM25Index::tokenize("port 8432 has 42 connections and 100 users");
    assert(std::find(tokens.begin(), tokens.end(), "8432") != tokens.end());
    assert(std::find(tokens.begin(), tokens.end(), "100") != tokens.end());
    assert(std::find(tokens.begin(), tokens.end(), "port") != tokens.end());
    assert(std::find(tokens.begin(), tokens.end(), "connections") != tokens.end());
}

void test_empty_input() {
    auto tokens = BM25Index::tokenize("");
    assert(tokens.empty());
}

void test_punctuation_handling() {
    auto tokens = BM25Index::tokenize("hello, world! test-case foo_bar");
    // Should handle punctuation reasonably
    assert(!tokens.empty());
}

// -----------------------------------------------------------------------
// Index tests
// -----------------------------------------------------------------------

void test_build_and_score() {
    BM25Index idx;
    std::vector<std::string> docs = {
        "the quick brown fox jumps over the lazy dog",
        "sqlite database memory storage backend",
        "machine learning embeddings neural network",
        "quick search through memory documents"
    };
    idx.build(docs);
    assert(idx.is_built());

    // Search for "quick" — should rank docs 0 and 3 higher
    std::vector<int> all = {0, 1, 2, 3};
    auto scores = idx.score("quick search", all);
    assert(scores.size() == 4);

    // Doc 3 ("quick search through memory documents") should score highest
    // Doc 0 has "quick" too
    // Docs 1 and 2 should score lowest
    assert(scores[3] > scores[1]);
    assert(scores[3] > scores[2]);
}

void test_score_empty_query() {
    BM25Index idx;
    std::vector<std::string> docs = {"hello world test"};
    idx.build(docs);

    std::vector<int> all = {0};
    auto scores = idx.score("", all);
    // Empty query — all scores should be 0
    for (auto s : scores) {
        assert(s == 0.0f);
    }
}

void test_score_no_match() {
    BM25Index idx;
    std::vector<std::string> docs = {"hello world test document"};
    idx.build(docs);

    std::vector<int> all = {0};
    auto scores = idx.score("zzzznotfound xyzabc", all);
    for (auto s : scores) {
        assert(s == 0.0f);
    }
}

void test_filtered_indices() {
    BM25Index idx;
    std::vector<std::string> docs = {
        "apple banana cherry",
        "delta echo foxtrot",
        "apple cherry delta"
    };
    idx.build(docs);

    // Only score docs 0 and 2
    std::vector<int> subset = {0, 2};
    auto scores = idx.score("apple", subset);
    assert(scores.size() == 2);  // one score per filtered index
    // Both docs contain "apple"
    assert(scores[0] > 0.0f);  // doc 0
    assert(scores[1] > 0.0f);  // doc 2
}

void test_load_from_prebuilt() {
    BM25Index idx;

    // Simulate loading from bm25_index table
    std::vector<int> doc_ids   = {0, 0, 0, 1, 1, 1};
    std::vector<std::string> tokens = {"hello", "world", "test", "hello", "foo", "bar"};
    std::vector<int> freqs     = {1, 1, 1, 2, 1, 1};

    idx.load(doc_ids, tokens, freqs);
    assert(idx.is_built());

    std::vector<int> all = {0, 1};
    auto scores = idx.score("hello", all);
    // Both docs have "hello", but doc 1 has freq=2
    assert(scores[0] > 0.0f);
    assert(scores[1] > 0.0f);
}

void test_not_built() {
    BM25Index idx;
    assert(!idx.is_built());
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    test_basic_tokenization();
    test_short_tokens_filtered();
    test_hex_strings_filtered();
    test_numbers_kept();
    test_empty_input();
    test_punctuation_handling();
    test_build_and_score();
    test_score_empty_query();
    test_score_no_match();
    test_filtered_indices();
    test_load_from_prebuilt();
    test_not_built();

    std::cout << "test_bm25: all passed\n";
    return 0;
}
