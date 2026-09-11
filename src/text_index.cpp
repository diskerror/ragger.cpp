/**
 * text_index.cpp — engine-agnostic index algorithm.
 *
 * index_record(): tokenize a record's text and persist its term postings via
 * the storage hooks. score_query(): TF-IDF scoring over the same tokenizer,
 * per-table df/N (Decision C), metaphone-weighted (Decision D).
 */
#include "text_index.h"

#include <algorithm>
#include <cmath>

#include "fts_tokenize.h"
#include "lang/en.h"

namespace ragger {

namespace {

// Stopword sets are constructed once from the constexpr lang arrays.
const fts::StopSet& uni_stops() {
    static const fts::StopSet s = fts::stopword_set(ragger::lang::STOPWORDS_UNIGRAM);
    return s;
}
const fts::StopSet& bi_stops() {
    static const fts::StopSet s = fts::stopword_set(ragger::lang::STOPWORDS_BIGRAM);
    return s;
}

}  // namespace

void TextIndex::index_record(const std::string& table, int id,
                             const std::string& text) {
    std::unordered_map<std::string, int> counts_by_term;
    int unigram_count = 0;
    int bigram_count  = 0;

    for (const auto& sentence : fts::split_sentences(text)) {
        auto words = fts::normalize_words(sentence);

        for (const auto& t : fts::unigrams(words, uni_stops())) {
            if (!t.literal.empty())   { counts_by_term[t.literal]++; ++unigram_count; }
            if (!t.metaphone.empty())   counts_by_term[t.metaphone]++;
        }
        for (const auto& t : fts::bigrams(words, bi_stops())) {
            if (!t.literal.empty())   { counts_by_term[t.literal]++; ++bigram_count; }
            // Only emit a bigram metaphone when both halves produced a code.
            if (!t.metaphone.empty() && t.metaphone.find('_') != std::string::npos)
                counts_by_term[t.metaphone]++;
        }
    }

    // Resolve term strings to ids, folding duplicate ids (a literal and its
    // metaphone are distinct strings, so collisions are rare but harmless).
    std::unordered_map<int, int> counts_by_id;
    counts_by_id.reserve(counts_by_term.size());
    for (const auto& [term, c] : counts_by_term) {
        int tid = upsert_term(term);
        if (tid > 0) counts_by_id[tid] += c;
    }

    replace_record_terms(table, id, counts_by_id);
    set_counts(table, id, unigram_count, bigram_count);
}

std::vector<TextScore> TextIndex::score_query(const std::string& table,
                                              const std::string& query) {
    const int N = std::max(1, corpus_size(table));
    std::unordered_map<int, float> acc;

    // Accumulate TF-IDF contribution of one query term across its postings.
    auto add_term = [&](const std::string& term, bool is_meta, bool is_bi) {
        if (term.empty()) return;
        if (!is_meta && !weights_.literal_enabled) return;  // DMP-only mode
        int tid = lookup_term(term);
        if (tid == 0) return;
        int df = std::max(1, doc_freq(table, tid));
        float idf   = std::log(static_cast<float>(N) / static_cast<float>(df));
        float wtype = is_bi ? weights_.bigram : weights_.unigram;
        float m     = is_meta ? weights_.metaphone : 1.0f;
        for (const auto& p : postings(table, tid)) {
            int denom = is_bi ? p.bigram_count : p.unigram_count;
            if (denom <= 0) continue;
            acc[p.id] += wtype * (static_cast<float>(p.count) / denom) * idf * m;
        }
    };

    for (const auto& sentence : fts::split_sentences(query)) {
        auto words = fts::normalize_words(sentence);
        for (const auto& t : fts::unigrams(words, uni_stops())) {
            add_term(t.literal,   /*meta=*/false, /*bi=*/false);
            add_term(t.metaphone, /*meta=*/true,  /*bi=*/false);
        }
        for (const auto& t : fts::bigrams(words, bi_stops())) {
            add_term(t.literal, /*meta=*/false, /*bi=*/true);
            if (!t.metaphone.empty() && t.metaphone.find('_') != std::string::npos)
                add_term(t.metaphone, /*meta=*/true, /*bi=*/true);
        }
    }

    std::vector<TextScore> out;
    out.reserve(acc.size());
    for (const auto& [id, s] : acc) out.push_back({id, s});
    return out;
}

}  // namespace ragger
