#include "ragger/bm25.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <unordered_set>

namespace ragger {

namespace {

// Common English stop words
const std::unordered_set<std::string> STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "not", "no", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "all", "also", "into", "over", "after"
};

// Check if token looks like bare hex (6+ consecutive hex chars, no 0x prefix)
bool is_bare_hex(const std::string& token) {
    if (token.length() < 6) return false;
    if (token.substr(0, 2) == "0x") return false; // prefixed hex is OK
    
    int hex_count = 0;
    for (char c : token) {
        if (std::isxdigit(c)) {
            ++hex_count;
        }
    }
    return hex_count >= 6 && hex_count == token.length();
}

// Check if token looks like base64 noise (20+ chars, mostly base64 alphabet)
bool is_base64_noise(const std::string& token) {
    if (token.length() < 20) return false;
    
    int b64_count = 0;
    for (char c : token) {
        if (std::isalnum(c) || c == '+' || c == '/' || c == '=') {
            ++b64_count;
        }
    }
    return b64_count >= 20;
}

// Check if token contains 3+ consecutive digits
bool has_meaningful_number(const std::string& token) {
    int digit_count = 0;
    for (char c : token) {
        if (std::isdigit(c)) {
            ++digit_count;
            if (digit_count >= 3) return true;
        } else {
            digit_count = 0;
        }
    }
    return false;
}

} // anonymous namespace

BM25Index::BM25Index(float k1, float b)
    : k1_(k1), b_(b) {}

std::vector<std::string> BM25Index::tokenize(const std::string& text) {
    // Lowercase
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // Split on non-alphanumeric, keep [a-z0-9]+
    std::regex token_re("[a-z0-9]+");
    std::sregex_iterator it(lower.begin(), lower.end(), token_re);
    std::sregex_iterator end;
    
    std::vector<std::string> tokens;
    for (; it != end; ++it) {
        std::string token = it->str();
        
        // Filter: minimum 3 chars
        if (token.length() < 3) continue;
        
        // Remove stop words
        if (STOP_WORDS.count(token)) continue;
        
        // Filter bare hex noise
        if (is_bare_hex(token)) continue;
        
        // Filter base64 noise
        if (is_base64_noise(token)) continue;
        
        // Keep tokens with 3+ digit numbers (years, ports, etc.)
        // or regular alphanumeric tokens
        tokens.push_back(token);
    }
    
    return tokens;
}

void BM25Index::build(const std::vector<std::string>& texts) {
    doc_tokens_.clear();
    df_.clear();
    doc_lengths_.clear();
    avg_doc_length_ = 0.0f;
    num_docs_ = static_cast<int>(texts.size());
    
    // Tokenize all documents and compute term frequencies
    int total_length = 0;
    for (int doc_id = 0; doc_id < num_docs_; ++doc_id) {
        auto tokens = tokenize(texts[doc_id]);
        
        std::unordered_map<std::string, int> tf;
        for (const auto& token : tokens) {
            ++tf[token];
        }
        
        doc_tokens_[doc_id] = tf;
        doc_lengths_[doc_id] = static_cast<int>(tokens.size());
        total_length += static_cast<int>(tokens.size());
        
        // Update document frequency
        for (const auto& [token, _] : tf) {
            ++df_[token];
        }
    }
    
    // Compute average document length
    if (num_docs_ > 0) {
        avg_doc_length_ = static_cast<float>(total_length) / num_docs_;
    }
    
    built_ = true;
}

void BM25Index::load(const std::vector<int>& doc_ids,
                     const std::vector<std::string>& tokens,
                     const std::vector<int>& freqs) {
    doc_tokens_.clear();
    df_.clear();
    doc_lengths_.clear();
    avg_doc_length_ = 0.0f;
    
    if (doc_ids.size() != tokens.size() || doc_ids.size() != freqs.size()) {
        return; // Invalid input
    }
    
    // Build doc_tokens_ from provided data
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        int doc_id = doc_ids[i];
        const std::string& token = tokens[i];
        int freq = freqs[i];
        
        doc_tokens_[doc_id][token] = freq;
        doc_lengths_[doc_id] += freq;
    }
    
    // Compute document frequency and stats
    std::unordered_set<int> unique_docs;
    for (const auto& [doc_id, tf_map] : doc_tokens_) {
        unique_docs.insert(doc_id);
        
        for (const auto& [token, _] : tf_map) {
            ++df_[token];
        }
    }
    
    num_docs_ = static_cast<int>(unique_docs.size());
    
    // Compute average document length
    int total_length = 0;
    for (const auto& [_, length] : doc_lengths_) {
        total_length += length;
    }
    if (num_docs_ > 0) {
        avg_doc_length_ = static_cast<float>(total_length) / num_docs_;
    }
    
    built_ = true;
}

std::vector<float> BM25Index::score(const std::string& query,
                                    const std::vector<int>& filtered_indices) const {
    if (!built_) {
        return std::vector<float>(filtered_indices.size(), 0.0f);
    }
    
    // Tokenize query
    auto query_tokens = tokenize(query);
    
    // Compute IDF for query tokens
    std::unordered_map<std::string, float> idf;
    for (const auto& token : query_tokens) {
        if (idf.count(token)) continue; // already computed
        
        auto it = df_.find(token);
        int df = (it != df_.end()) ? it->second : 0;
        
        // IDF formula: log((N - df + 0.5) / (df + 0.5) + 1.0)
        float idf_val = std::log((num_docs_ - df + 0.5f) / (df + 0.5f) + 1.0f);
        idf[token] = idf_val;
    }
    
    // Score each document in filtered_indices
    std::vector<float> scores;
    scores.reserve(filtered_indices.size());
    
    for (int doc_id : filtered_indices) {
        float score = 0.0f;
        
        auto doc_it = doc_tokens_.find(doc_id);
        if (doc_it == doc_tokens_.end()) {
            scores.push_back(0.0f);
            continue;
        }
        
        const auto& tf_map = doc_it->second;
        auto len_it = doc_lengths_.find(doc_id);
        int doc_len = (len_it != doc_lengths_.end()) ? len_it->second : 0;
        
        // BM25 formula
        for (const auto& token : query_tokens) {
            auto idf_it = idf.find(token);
            if (idf_it == idf.end()) continue;
            float idf_val = idf_it->second;
            
            auto tf_it = tf_map.find(token);
            int tf = (tf_it != tf_map.end()) ? tf_it->second : 0;
            
            if (tf == 0) continue;
            
            // BM25: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            float norm = 1.0f - b_ + b_ * (doc_len / avg_doc_length_);
            float tf_component = (tf * (k1_ + 1.0f)) / (tf + k1_ * norm);
            score += idf_val * tf_component;
        }
        
        scores.push_back(score);
    }
    
    return scores;
}

} // namespace ragger
