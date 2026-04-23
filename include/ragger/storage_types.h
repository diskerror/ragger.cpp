/**
 * Data types for storage operations (SearchResult, SearchResponse, AllEmbeddings, UserInfo)
 *
 * These types are logically tied to storage but separated from SqliteBackend
 * to reduce header dependencies for files that only need the data structures.
 */
#pragma once

#include <string>
#include <vector>

#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

struct SearchResult {
    int         id;
    std::string text;
    float       score;
    json        metadata;
    std::string timestamp;
};

struct SearchResponse {
    std::vector<SearchResult> results;
    json                      timing;
};

struct AllEmbeddings {
    std::vector<int>                ids;
    std::vector<std::string>        texts;
    std::vector<std::vector<float>> embeddings;
    std::vector<json>               metadata;
    std::vector<std::string>        timestamps;
};

/// User information (extracted from SqliteBackend for modularity)
struct UserInfo {
    int         id;
    std::string username;
    // is_admin removed — sudo is the admin gate
    std::string token_hash;
    std::string preferred_model;  // empty = system default
};


} // namespace ragger
