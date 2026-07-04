/**
 * Data types for storage operations.
 *
 * Separated from SqliteBackend/StorageBackend to reduce header dependencies
 * for files that only need the data structures.
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
    float       score;        // raw vector cosine similarity (reported score)
    json        metadata;
    std::string timestamp;

    // Per-signal ranking breakdown, captured for RAGGER_STATS instrumentation
    // so each search type's contribution can be judged (tune/keep/drop a weight).
    // Each *_score is the min-max-normalized [0,1] signal value that actually
    // fed the weighted blend; a value of -1 means that signal was inactive for
    // this search (bm25 disabled, or phon_weight==0). `blended` is the final
    // weighted ranking score (vector_weight*vec + bm25_weight*bm25 + phon_weight*phon).
    float vec_score  = -1.0f;
    float bm25_score = -1.0f;
    float phon_score = -1.0f;
    float blended    = 0.0f;
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

/// User information (extracted from SqliteBackend for modularity)
struct UserInfo {
    int         id;
    std::string username;
    // is_admin removed — sudo is the admin gate
    std::string token_hash;
};

/// A summary row that was written as a heuristic fallback ('draft') while
/// inference was unavailable. The summarizer rewrites these when it comes back.
struct DraftSummary {
    int         summary_id;
    std::string level;
    std::string session_guid;   // empty when session_id is NULL
    std::string timestamp;      // source-turn timestamp for L2
};

/// A summary row returned by session-level queries (turn or session summaries).
struct SummaryRecord {
    int         summary_id;
    std::string text;
    std::string status;
    std::string timestamp;
};


} // namespace ragger
