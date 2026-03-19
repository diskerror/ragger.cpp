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

} // namespace ragger
