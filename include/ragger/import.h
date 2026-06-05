/**
 * Import utilities — heading-aware paragraph chunking
 *
 * Extracted for testability. Used by main.cpp import verb.
 */
#pragma once

#include <string>
#include <vector>

namespace ragger {

struct ImportChunk {
    std::string text;
    std::string section;  // breadcrumb e.g. "Title » Subtitle"
};

/// Detect markdown heading level (1-6), or 0 if not a heading.
int heading_level(const std::string& line);

/// Extract heading text (strip the # prefix).
std::string heading_text(const std::string& line);

/// Split markdown text into heading-aware chunks.
/// min_chunk_size: minimum chars before starting a new chunk.
std::vector<ImportChunk> chunk_markdown(const std::string& text, int min_chunk_size);

// -------------------------------------------------------------------------
// Conversation import (Claude Code JSONL + claude.ai web export).
// -------------------------------------------------------------------------

/// One user+assistant exchange lifted from an exported conversation.
/// `timestamp` is the source-format string verbatim (typically ISO-8601);
/// the importer is responsible for converting it to db_timestamp form when
/// it stores rows.
struct ImportTurn {
    std::string timestamp;     // as parsed from source (ISO-8601)
    std::string user_text;
    std::string assistant_text;
    std::string session_id;
    std::string source;        // "claude-code" or "claude-web"
};

/// Optional filters applied while iterating turns.
struct TurnFilter {
    std::string session;            // exact session_id match (empty = any)
    std::string since;              // YYYY-MM-DD, inclusive (empty = no lower bound)
    std::string until;              // YYYY-MM-DD, exclusive (empty = no upper bound)
};

/// Parse Claude Code JSONL files. `path` may be a single .jsonl or a
/// directory of them. One ImportTurn per user+assistant pair.
std::vector<ImportTurn> parse_claude_code(const std::string& path);

/// Parse a claude.ai "Export Data" `conversations.json`.
std::vector<ImportTurn> parse_claude_web(const std::string& path);

/// Apply the filter in-place (returns the kept turns).
std::vector<ImportTurn> filter_turns(const std::vector<ImportTurn>& turns,
                                     const TurnFilter& f);

// -------------------------------------------------------------------------
// L4 summaries import.
// -------------------------------------------------------------------------

/// One hand-authored project-level summary to ingest as an L4 row.
struct SummaryImport {
    std::string text;
    std::string tags;        // comma-separated (empty allowed)
    std::string timestamp;   // optional ISO-8601 / db_timestamp; empty = now
};

/// Load one record per file. The filename stem becomes the default tag if
/// `tags` is empty. Files are read as UTF-8 text verbatim.
std::vector<SummaryImport> load_summary_files(const std::vector<std::string>& paths);

/// Parse a JSONL stream where each line is `{"text": "...", "tags"?: "...",
/// "timestamp"?: "..."}`. Blank lines and parse failures are skipped.
std::vector<SummaryImport> load_summary_jsonl(const std::string& path);

} // namespace ragger
