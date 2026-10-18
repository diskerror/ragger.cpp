/**
 * Data types for storage operations.
 *
 * Separated from sqlite::Backend/StorageBackend to reduce header dependencies
 * for files that only need the data structures.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann_json.hpp"

namespace ragger {

using json = nlohmann::json;

/// Input for a structured (skill-shaped) decision — agent API Phase 1.
/// text + trigger are required; everything else is optional.
struct DecisionInput {
    std::string text;                 ///< the ruling itself
    std::string trigger;              ///< "Use when ..." — future-query vocabulary
    std::string rationale;            ///< why; alternatives rejected ("" = NULL)
    std::string scope;                ///< what it does NOT apply to ("" = NULL)
    std::string tags;                 ///< comma-separated, stored verbatim
    std::string status = "current";   ///< current | provisional | pending
    std::optional<int> supersedes;    ///< decision_id this replaces
    std::string session_guid;         ///< source session GUID ("" = none)
    std::vector<int> evidence;        ///< turn_summary_ids backing the decision
};

/// Input to store_episode_close (db 0.17, PHASE5 B.1): one closed episode —
/// its text plus the recorded membership. span_start/span_end are epoch
/// seconds of the first/last member turn. members must be non-empty and all
/// belong to session_guid; any straggler fails the whole close.
struct EpisodeClose {
    std::string text;
    std::string model_name;
    std::string session_guid;
    int64_t span_start = 0;
    int64_t span_end = 0;
    std::vector<int> member_turn_summary_ids;
};

struct StructuredDecisionResult {
    int id = 0;                       ///< new decision_id
    std::optional<int> superseded;    ///< decision_id flipped to 'superseded'
    /// Flag-only hint (Phase 2 step 5): up to 3 OTHER active decisions whose
    /// embedding cosine with the new row is >= decision_supersede_threshold.
    /// Never a merge — the caller decides what to do with them.
    std::vector<std::pair<int,float>> possible_supersedes;
};

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
    // this search (bm25 disabled). `blended` is the final weighted ranking
    // score (vector_weight*vec + bm25_weight*bm25). phon_score is retained at
    // -1 for API/stats-schema compat only -- the old independent phonetic
    // signal was folded into bm25_score by TextIndex::score_query() (v0.16;
    // see fts_w_metaphone).
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

/// memory_search request (agent API Phase 2). Empty vectors/strings = no filter.
struct AgentSearchQuery {
    std::string query;
    std::vector<std::string> layers;   ///< subset of turn|turn_summary|episode|summary|document|decision; empty = all
    std::string since;                 ///< "YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", or relative "30d"/"12h"; "" = none
    std::string until;                 ///< same formats; "" = none
    std::string status = "active";     ///< decisions only: "active" (current+provisional) | "any" | one exact status
    std::string session;               ///< only this session GUID
    std::string exclude_session;       ///< drop this session GUID
    int limit = 10;
    int max_tokens = 0;                ///< 0 = no budget; else stop adding results past ~max_tokens (chars/4)
};

struct AgentHit {
    std::string id;          ///< typed handle "<layer>:<rowid>"
    std::string layer;
    float score = 0;
    std::string matched_by;  ///< "vector" | "lexical" | "both"
    std::string created;     ///< local "YYYY-MM-DD HH:MM:SS"
    std::string snippet;     ///< first <=240 chars of text, cut at a word boundary, "…" appended if cut
    json extra;              ///< layer-specific: decision → {status, trigger}; summary → {level}; turn_summary → {turn_id, session}
};

/// memory_get request (agent API Phase 2). Exactly one mode per call:
/// ids (mode 1) XOR session (mode 2) — both or neither is invalid_argument.
struct AgentGetQuery {
    std::vector<std::string> ids;   ///< typed handles; mode 1
    std::string expand = "none";    ///< none | context | sources | chain
    std::string session;            ///< mode 2: history range for this session GUID
    int after_turn = 0;             ///< mode 2: turns with turn_id > after_turn (0 = from start)
    int before_turn = 0;            ///< mode 2: turns with turn_id < before_turn (0 = to end)
    int limit = 20;                 ///< mode 2 page size, clamped [1,100]
};

/// memory_timeline request (agent API Phase 4): deterministic range scan,
/// newest first, key-paginated. No embeddings involved.
struct TimelineQuery {
    std::string session;           ///< GUID; "" = all
    std::string since, until;      ///< same formats as AgentSearchQuery
    std::string granularity = "episode";   ///< turn | episode | session
    int limit = 20;                ///< clamp [1,100]
    std::string cursor;            ///< opaque "<created_epoch>:<rowid>"; "" = newest first page
};

/// memory_update request (agent API Phase 4): metadata-only curation.
/// Text never changes in place — a text change is a new row with supersedes.
struct UpdateRequest {
    std::string id;                          ///< typed handle
    std::optional<std::string> status;
    std::optional<std::string> tags;         ///< replaces the whole tag string
    std::optional<std::string> text;         ///< present => always rejected
};

/// User information (extracted from sqlite::Backend for modularity)
struct UserInfo {
    int         id;
    std::string username;
    // is_admin removed — sudo is the admin gate
    std::string token_hash;
};

/// A summary row returned by session-level queries (turn/episode reads).
struct SummaryRecord {
    int         summary_id;
    std::string text;
    std::string status;
    std::string timestamp;
};

/// A candidate turn for similarity-based episode boundary detection.
/// Carries the raw turn embedding and turn-summary embedding alongside the
/// summary text so the caller can compute cosine similarities and split
/// into episodes without a second round-trip to the DB.
struct EpisodeCandidateTurn {
    int                turn_summary_id;
    std::string        summary_text;     // L2 turn-summary text
    std::string        timestamp;        // "%F %T" local-time string for store_episode
    int64_t            epoch;            // created_at epoch seconds (for gap math)
    std::vector<float> turn_embedding;   // from turns.embedding
    std::vector<float> summary_embedding;// from turn_summaries.embedding
};


/// A schema object from sqlite_master (table, index, trigger, view).
struct SchemaObject {
    std::string type;
    std::string name;
    std::string tbl_name;
    std::string sql;
};

/// A single cell in a generic table row (for export/introspection).
struct ExportCell {
    enum class Type { Null, Integer, Float, Text, Blob };
    Type        type      = Type::Null;
    int64_t     int_val   = 0;
    double      float_val = 0.0;
    std::string text_val;
    std::vector<uint8_t> blob_val;
};
using ExportRow = std::vector<ExportCell>;

} // namespace ragger
