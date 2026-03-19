/**
 * Import utilities — heading-aware paragraph chunking
 */
#include "ragger/import.h"

#include <regex>
#include <sstream>

namespace ragger {

int heading_level(const std::string& line) {
    int level = 0;
    while (level < 6 && level < (int)line.size() && line[level] == '#') ++level;
    if (level > 0 && level < (int)line.size() && line[level] == ' ') return level;
    return 0;
}

std::string heading_text(const std::string& line) {
    auto pos = line.find(' ');
    return (pos != std::string::npos) ? line.substr(pos + 1) : line;
}

std::vector<ImportChunk> chunk_markdown(const std::string& raw_text, int min_chunk_size) {
    // Clean the text
    std::string text = raw_text;

    // Strip base64 image data
    text = std::regex_replace(text, std::regex(R"(!\[[^\]]*\]\(data:[^)]+\))"), "");
    text = std::regex_replace(text, std::regex(R"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)"), "");

    // Collapse multi-space artifacts per line
    std::istringstream lines_stream(text);
    std::string line;
    std::string cleaned;
    while (std::getline(lines_stream, line)) {
        line = std::regex_replace(line, std::regex(R"(  +)"), " ");
        cleaned += line + "\n";
    }
    text = std::regex_replace(cleaned, std::regex(R"(\n{3,})"), "\n\n");

    // Split on paragraph boundaries
    std::vector<std::string> paragraphs;
    {
        std::istringstream ss(text);
        std::string buffer;
        while (std::getline(ss, line)) {
            if (line.empty() || (line.size() == 1 && line[0] == '\r')) {
                if (!buffer.empty()) {
                    while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r' || buffer.back() == ' '))
                        buffer.pop_back();
                    paragraphs.push_back(buffer);
                    buffer.clear();
                }
            } else {
                if (!buffer.empty()) buffer += "\n";
                buffer += line;
            }
        }
        if (!buffer.empty()) {
            while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r' || buffer.back() == ' '))
                buffer.pop_back();
            paragraphs.push_back(buffer);
        }
    }

    // Heading-aware chunking
    struct HeadingEntry { int level; std::string text; };
    std::vector<HeadingEntry> heading_stack;
    struct Annotated { std::string body; std::string heading_block; std::string section; };
    std::vector<Annotated> annotated;

    auto current_section = [&]() -> std::string {
        std::string s;
        for (auto& h : heading_stack) {
            if (!s.empty()) s += " \xC2\xBB ";  // UTF-8 »
            s += h.text;
        }
        return s;
    };

    auto current_heading_block = [&]() -> std::string {
        std::string s;
        for (auto& h : heading_stack) {
            if (!s.empty()) s += "\n\n";
            s += std::string(h.level, '#') + " " + h.text;
        }
        return s;
    };

    for (auto& para : paragraphs) {
        int level = heading_level(para);
        if (level > 0) {
            while (!heading_stack.empty() && heading_stack.back().level >= level)
                heading_stack.pop_back();
            heading_stack.push_back({level, ragger::heading_text(para)});
        } else {
            annotated.push_back({para, current_heading_block(), current_section()});
        }
    }

    // Merge short paragraphs into chunks
    std::vector<ImportChunk> chunks;
    std::string current;
    std::string current_sec;

    for (auto& a : annotated) {
        if (current.empty()) {
            current = a.heading_block.empty() ? a.body : (a.heading_block + "\n\n" + a.body);
            current_sec = a.section;
        } else if ((int)current.size() >= min_chunk_size) {
            chunks.push_back({current, current_sec});
            current = a.heading_block.empty() ? a.body : (a.heading_block + "\n\n" + a.body);
            current_sec = a.section;
        } else {
            if (a.section != current_sec && !a.heading_block.empty()) {
                current += "\n\n" + a.heading_block + "\n\n" + a.body;
            } else {
                current += "\n\n" + a.body;
            }
            current_sec = a.section;
        }
    }
    if (!current.empty()) {
        chunks.push_back({current, current_sec});
    }

    return chunks;
}

} // namespace ragger
