/**
 * BM25 ranking index
 *
 * Mirrors ragger_memory/bm25.py from the Python version.
 * Pure C++ implementation — no external NLP dependencies.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ragger {

class BM25Index {
public:
    BM25Index(float k1 = 1.5f, float b = 0.75f);

    /// Build index from document texts. IDs are positional indices.
    void build(const std::vector<std::string>& texts);

    /// Load index from pre-tokenized data (SQLite bm25_index table).
    void load(const std::vector<int>& doc_ids,
              const std::vector<std::string>& tokens,
              const std::vector<int>& freqs);

    /// Score all indexed documents against a query. Returns scores aligned to filtered_indices.
    std::vector<float> score(const std::string& query,
                             const std::vector<int>& filtered_indices) const;

    bool is_built() const { return built_; }

    /// Simple whitespace + lowercase tokenizer with stop-word removal.
    static std::vector<std::string> tokenize(const std::string& text);

private:

    float k1_, b_;
    bool  built_ = false;

    // doc_id → { token → count }
    std::unordered_map<int, std::unordered_map<std::string, int>> doc_tokens_;
    // token → number of documents containing it
    std::unordered_map<std::string, int> df_;
    // doc_id → document length (in tokens)
    std::unordered_map<int, int> doc_lengths_;
    float avg_doc_length_ = 0.0f;
    int   num_docs_ = 0;
};

} // namespace ragger
