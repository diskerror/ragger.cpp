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

/// Full memory record including raw embedding blob — used for bulk export/import.
struct MemoryRecord {
    int64_t              id = 0;
    std::string          text;
    std::vector<uint8_t> embedding;   // raw blob (may be empty)
    std::string          metadata;    // JSON string
    std::string          timestamp;
    std::string          collection;
    std::string          category;
    std::string          tags;
};

/// Filter criteria for bulk memory export.  All fields are optional;
/// empty/default values mean "no filter on this field".
struct MemoryFilter {
    std::vector<int> ids;             // specific IDs (empty = all)
    std::string      collection;      // exact match
    std::string      category;        // exact match
    std::string      source_pattern;  // metadata $.source LIKE pattern
};

} // namespace ragger
