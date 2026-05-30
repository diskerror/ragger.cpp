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

/// Input for storing a Level 5 RAG document chunk (lean v2 documents schema).
/// `chunk_index` of 0 means "unchunked" (single-row document). `imported_at`
/// should be the same string for every chunk of one import so they share a
/// single timestamp (issue #48); empty = backend stamps now_iso() at insert.
///
/// `title` identifies the publication, `tags` is a comma-separated subject
/// list, `year` is the publish year, `path` is the origin. Only `text` is
/// embedded — title/tags/year are grouping/priority metadata, not embedded.
struct DocumentChunk {
    std::string text;
    std::string path;
    std::string title;
    std::string tags;
    int         year        = 0;
    int         chunk_index = 0;
    std::string imported_at;
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
