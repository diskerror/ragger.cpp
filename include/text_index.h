/**
 * TextIndex — engine-agnostic custom inverted index (v0.16 "terms" model).
 *
 * The tokenize→term-multiset and TF-IDF scoring algorithm lives here as
 * concrete methods; the per-engine storage (insert term, replace junction
 * rows, COUNT postings) is supplied by the pure-virtual hooks below. A future
 * MariaDB/Mongo/Postgres backend implements only the hooks, reusing the whole
 * indexing + scoring pipeline. sqlite::TextIndex is the only concrete impl today.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ragger {

// One scored hit from the text index: record id + accumulated TF-IDF score.
struct TextScore { int id; float score; };

// TF-IDF blend weights. Defaults match the migration plan (unigram 0.3,
// bigram 0.7, metaphone 1.0). Migration Step 9 wires these to `settings`.
struct TextIndexWeights {
    float unigram   = 0.3f;
    float bigram    = 0.7f;
    float metaphone = 1.0f;
    bool  literal_enabled = true;   // false => DMP-only scoring (no reindex)
};

class TextIndex {
public:
    virtual ~TextIndex() = default;

    void set_weights(const TextIndexWeights& w) { weights_ = w; }

    // ---- engine-agnostic algorithm (implemented in text_index.cpp) --------

    // Tokenize `text`, replace this record's postings, update count columns.
    // Reused by full reindex, on-demand reindex, and the per-record write path.
    void index_record(const std::string& table, int id, const std::string& text);

    // Score every candidate record in `table` for `query` via TF-IDF over
    // literal + metaphone unigram/bigram terms. Returns id->score (unsorted).
    std::vector<TextScore> score_query(const std::string& table,
                                       const std::string& query);

protected:
    // ---- per-engine storage hooks (implemented per backend) ---------------

    // INSERT OR IGNORE a term string; return its stable term_id (>0).
    virtual int  upsert_term(const std::string& term) = 0;
    // Delete this record's junction rows, then insert the given (term_id,count).
    virtual void replace_record_terms(const std::string& table, int id,
        const std::unordered_map<int, int>& term_counts) = 0;
    // Write the TF denominators onto the record row.
    virtual void set_counts(const std::string& table, int id,
                            int unigram_count, int bigram_count) = 0;
    // Live corpus size N = COUNT(*) of the table.
    virtual int  corpus_size(const std::string& table) = 0;
    // Live df = COUNT(*) of records containing term_id in this table.
    virtual int  doc_freq(const std::string& table, int term_id) = 0;
    // Resolve a query term string to term_id (0 if unknown — skip it).
    virtual int  lookup_term(const std::string& term) = 0;
    // Postings for a term in a table, joined with the record's TF denominators
    // so score_query() needs no extra per-record round-trips.
    struct Posting { int id; int count; int unigram_count; int bigram_count; };
    virtual std::vector<Posting> postings(const std::string& table,
                                          int term_id) = 0;

    TextIndexWeights weights_{};
};

}  // namespace ragger
