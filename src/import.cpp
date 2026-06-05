/**
 * Import utilities — heading-aware paragraph chunking and conversation /
 * summary ingest from external sources.
 */
#include "ragger/import.h"
#include "nlohmann_json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
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

// =========================================================================
// Conversation parsers
// =========================================================================

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

/// Trim ASCII whitespace from both ends.
std::string trim(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back()))  s.pop_back();
    return s;
}

/// Extract text from a Claude message body. The body may be a raw string or
/// an array of content blocks; only `type=="text"` blocks contribute (tool
/// blocks / thinking blocks aren't conversational and would pollute search).
std::string extract_claude_text(const json& content) {
    if (content.is_string()) return trim(content.get<std::string>());
    if (!content.is_array()) return "";
    std::vector<std::string> parts;
    for (const auto& block : content) {
        if (!block.is_object()) continue;
        if (block.value("type", "") == "text") {
            auto t = trim(block.value("text", ""));
            if (!t.empty()) parts.push_back(t);
        }
    }
    std::string joined;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) joined += "\n\n";
        joined += parts[i];
    }
    return joined;
}

/// "2024-10-15" → "2024-10-15 00:00:00" for lexical comparison against the
/// stored ts. The export timestamps are ISO-8601 (with "T" + zone); since we
/// only need <, ==, > we can compare the first 10 characters directly.
std::string date_only(const std::string& iso) {
    if (iso.size() >= 10) return iso.substr(0, 10);
    return iso;
}

} // anonymous

std::vector<ImportTurn> parse_claude_code(const std::string& path_str) {
    std::vector<ImportTurn> out;
    fs::path root(path_str);
    std::vector<fs::path> files;
    if (fs::is_directory(root)) {
        for (auto& e : fs::directory_iterator(root)) {
            if (e.is_regular_file() && e.path().extension() == ".jsonl") {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(root);
    }

    struct Pending { std::string ts, text, sid; };

    for (const auto& jf : files) {
        std::ifstream in(jf);
        if (!in) continue;
        std::string line;
        std::optional<Pending> pending;
        const std::string default_sid = jf.stem().string();

        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty()) continue;
            json ev;
            try { ev = json::parse(line); } catch (...) { continue; }

            auto etype = ev.value("type", "");
            if (etype != "user" && etype != "assistant") continue;

            json msg = ev.value("message", json::object());
            std::string text = extract_claude_text(msg.value("content", json()));
            if (text.empty()) continue;

            std::string ts  = ev.value("timestamp", "");
            std::string sid = ev.value("sessionId", default_sid);

            if (etype == "user") {
                if (pending) {
                    out.push_back({pending->ts, pending->text, "", pending->sid, "claude-code"});
                }
                pending = Pending{ts, text, sid};
            } else { // assistant
                if (pending) {
                    out.push_back({pending->ts, pending->text, text, pending->sid, "claude-code"});
                    pending.reset();
                }
            }
        }
        if (pending) {
            out.push_back({pending->ts, pending->text, "", pending->sid, "claude-code"});
        }
    }
    return out;
}

std::vector<ImportTurn> parse_claude_web(const std::string& path_str) {
    std::vector<ImportTurn> out;
    std::ifstream in(path_str);
    if (!in) return out;
    json doc;
    try { in >> doc; } catch (...) { return out; }

    // Some exports wrap the array in {"conversations": [...]}.
    json convs;
    if (doc.is_array()) convs = doc;
    else if (doc.is_object() && doc.contains("conversations")) convs = doc["conversations"];
    else return out;

    struct Pending { std::string ts, text; };

    for (const auto& conv : convs) {
        if (!conv.is_object()) continue;
        std::string sid = conv.value("uuid", "");
        json msgs = conv.contains("chat_messages") ? conv["chat_messages"]
                  : conv.contains("messages")      ? conv["messages"]
                                                   : json::array();

        std::optional<Pending> pending;
        for (const auto& m : msgs) {
            if (!m.is_object()) continue;
            std::string sender = m.value("sender", m.value("role", ""));
            std::string text;
            if (m.contains("text") && m["text"].is_string()) text = trim(m.value("text", ""));
            else text = extract_claude_text(m.value("content", json()));
            if (text.empty()) continue;
            std::string ts = m.value("created_at", m.value("timestamp", ""));

            if (sender == "human" || sender == "user") {
                if (pending) out.push_back({pending->ts, pending->text, "", sid, "claude-web"});
                pending = Pending{ts, text};
            } else if (sender == "assistant" || sender == "claude") {
                if (pending) {
                    out.push_back({pending->ts, pending->text, text, sid, "claude-web"});
                    pending.reset();
                }
            }
        }
        if (pending) out.push_back({pending->ts, pending->text, "", sid, "claude-web"});
    }
    return out;
}

std::vector<ImportTurn> filter_turns(const std::vector<ImportTurn>& turns,
                                     const TurnFilter& f) {
    std::vector<ImportTurn> out;
    out.reserve(turns.size());
    for (const auto& t : turns) {
        if (!f.session.empty() && t.session_id != f.session) continue;
        if (!f.since.empty() || !f.until.empty()) {
            std::string d = date_only(t.timestamp);
            if (d.empty()) continue;
            if (!f.since.empty() && d <  f.since) continue;
            if (!f.until.empty() && d >= f.until) continue;
        }
        out.push_back(t);
    }
    return out;
}

// =========================================================================
// L4 summary import
// =========================================================================

std::vector<SummaryImport> load_summary_files(const std::vector<std::string>& paths) {
    std::vector<SummaryImport> out;
    for (const auto& p : paths) {
        std::ifstream in(p);
        if (!in) continue;
        std::stringstream ss;
        ss << in.rdbuf();
        std::string body = trim(ss.str());
        if (body.empty()) continue;
        SummaryImport s;
        s.text = body;
        s.tags = fs::path(p).stem().string();
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<SummaryImport> load_summary_jsonl(const std::string& path) {
    std::vector<SummaryImport> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        json j;
        try { j = json::parse(t); } catch (...) { continue; }
        if (!j.is_object() || !j.contains("text")) continue;
        SummaryImport s;
        s.text      = trim(j.value("text", ""));
        s.tags      = j.value("tags", "");
        s.timestamp = j.value("timestamp", "");
        if (s.text.empty()) continue;
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace ragger
