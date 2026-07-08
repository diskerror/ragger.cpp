/**
 * Import utilities — heading-aware paragraph chunking and conversation /
 * summary ingest from external sources.
 */
#include "ragger/import.h"
#include "nlohmann_json.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace ragger {

using json = nlohmann::json;

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

namespace {

// Structural/metadata keys we never want as prose — skip their string
// values entirely rather than dumping ids/timestamps/flags into the text.
bool is_noise_key(const std::string& key) {
    static const std::vector<std::string> noise = {
        "uuid", "id", "account_uuid", "session_id", "chat_id", "user_id",
        "type", "status", "sender", "role", "created_at", "updated_at",
        "timestamp", "start_timestamp", "stop_timestamp", "date",
        "is_private", "is_starter_project", "verified_phone_number",
        "email_address", "phone_number", "attachments", "files", "citations",
        "parent_message_uuid", "flags", "prompt_template"
    };
    std::string lower;
    lower.reserve(key.size());
    for (char c : key) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    for (const auto& n : noise) if (lower == n) return true;
    return false;
}

// Recursively collect string leaves worth keeping. `key` is the parent
// object key this value was found under (empty at the array/root level).
void collect_json_text(const json& node, const std::string& key,
                       std::vector<std::string>& out) {
    if (node.is_string()) {
        if (!key.empty() && is_noise_key(key)) return;
        std::string s = node.get<std::string>();
        // Skip short/opaque strings (e.g. stray uuids under unlisted keys,
        // "human"/"assistant" sender labels that slipped through) — prose
        // worth chunking is generally more than a few words.
        if (s.size() < 3) return;
        out.push_back(std::move(s));
        return;
    }
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            collect_json_text(it.value(), it.key(), out);
        }
        return;
    }
    if (node.is_array()) {
        for (const auto& elem : node) collect_json_text(elem, key, out);
        return;
    }
    // numbers/bools/null: never prose, always skip.
}

} // namespace

std::string extract_text_from_json(const std::string& text) {
    json doc;
    try {
        doc = json::parse(text);
    } catch (...) {
        return "";  // not JSON — caller treats input as plain text/markdown
    }
    std::vector<std::string> pieces;
    collect_json_text(doc, "", pieces);
    std::string out;
    for (auto& p : pieces) {
        if (!out.empty()) out += "\n\n";
        out += p;
    }
    return out;
}

namespace {

// Local trim — the shared `trim()` helper lives further down in this file
// (used by the conversation parsers) and isn't visible yet at this point.
std::string trim_ws(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back()))  s.pop_back();
    return s;
}

// True if `line` (trimmed) is entirely wrapped in **bold** or *italic*
// markers with nothing else — i.e. it's a bare section header line like
// "**Work context**" or "*Recent months*", not prose that merely
// contains bold/italic spans. Returns the marker length (2 for **, 1 for
// *) via `marker_len`, or 0 if the line isn't a bare header.
int bare_emphasis_marker_len(const std::string& line) {
    if (line.size() >= 4 && line.substr(0, 2) == "**" &&
        line.substr(line.size() - 2) == "**" &&
        line.find("**", 2) == line.size() - 2) {
        return 2;
    }
    if (line.size() >= 2 && line.front() == '*' && line.back() == '*' &&
        line.find('*', 1) == line.size() - 1) {
        return 1;
    }
    return 0;
}

std::string strip_emphasis(const std::string& line, int marker_len) {
    if ((int)line.size() < 2 * marker_len) return line;
    return line.substr(marker_len, line.size() - 2 * marker_len);
}

} // namespace

std::vector<MemoryChunk> split_memory_narrative(const std::string& text) {
    std::vector<MemoryChunk> out;
    std::string bold_heading;    // current **bold** section
    std::string italic_heading;  // current *italic* sub-section

    auto breadcrumb = [&]() -> std::string {
        if (bold_heading.empty()) return italic_heading;
        if (italic_heading.empty()) return bold_heading;
        return bold_heading + " \xC2\xBB " + italic_heading;  // UTF-8 »
    };

    std::istringstream in(text);
    std::string line;
    std::string current;
    auto flush = [&]() {
        std::string t = trim_ws(current);
        if (!t.empty()) out.push_back({t, breadcrumb()});
        current.clear();
    };
    while (std::getline(in, line)) {
        std::string t = trim_ws(line);
        if (t.empty()) { flush(); continue; }
        int mlen = bare_emphasis_marker_len(t);
        if (mlen > 0) {
            flush();
            std::string heading = trim_ws(strip_emphasis(t, mlen));
            if (mlen == 2) { bold_heading = heading; italic_heading.clear(); }
            else           { italic_heading = heading; }
            continue;
        }
        if (!current.empty()) current += " ";
        current += t;
    }
    flush();
    return out;
}

bool memory_chunk_is_decision_like(const std::string& heading_path,
                                   const std::string& text,
                                   bool default_decision_like) {
    std::string h = heading_path;
    for (char& c : h) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    // Heading signal first.
    static const std::vector<std::string> session_headings = {
        "history", "recent", "earlier", "background", "timeline"
    };
    static const std::vector<std::string> decision_headings = {
        "context", "top of mind", "preference"
    };
    for (auto& k : session_headings)  if (h.find(k) != std::string::npos) return false;
    for (auto& k : decision_headings) if (h.find(k) != std::string::npos) return true;

    // Fall back to a keyword scan of the text itself.
    std::string t = text;
    for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static const std::vector<std::string> session_verbs = {
        "worked on", "explored", "set up", "asked about", "troubleshot",
        "troubleshooting", "iterating", "iterated", "debugging", "investigated",
        "started with", "began", "set out"
    };
    static const std::vector<std::string> decision_verbs = {
        "prefers", "preferring", "favors", "favoring", "is a ", "is an ",
        "uses ", "runs a ", "works as", "has a strong interest"
    };
    int session_hits = 0, decision_hits = 0;
    for (auto& k : session_verbs)  if (t.find(k) != std::string::npos) ++session_hits;
    for (auto& k : decision_verbs) if (t.find(k) != std::string::npos) ++decision_hits;
    if (session_hits != decision_hits) return decision_hits > session_hits;
    // No signal either way: fall back to the caller's default. For
    // Claude's memories.json narrative this stays decision-like (a
    // curated memory blob skews toward stable facts). For flat daily-log
    // memory files this is false — a dated log entry with no other
    // signal is an episode that never got a session id, not a standing
    // fact, so it belongs with session summaries.
    return default_decision_like;
}

std::string extract_date_from_filename(const std::string& filename) {
    static const std::regex date_re(R"(^(\d{4}-\d{2}-\d{2}))");
    std::smatch m;
    if (std::regex_search(filename, m, date_re)) return m[1].str();
    return "";
}

bool is_agent_scaffolding_filename(const std::string& filename) {
    std::string lower = filename;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static const std::vector<std::string> scaffolding = {
        "soul.md", "user.md", "agents.md", "tools.md",
        "identity.md", "heartbeat.md", "bootstrap.md"
    };
    for (auto& s : scaffolding) if (lower == s) return true;
    return false;
}

std::vector<FlatMemoryChunk> parse_flat_markdown_memory(const std::string& path,
                                                        int min_chunk_size,
                                                        int min_chunk_chars) {
    std::vector<FlatMemoryChunk> out;
    namespace fs = std::filesystem;
    fs::path p(path);
    std::string filename = p.filename().string();
    if (is_agent_scaffolding_filename(filename)) return out;

    std::ifstream in(path, std::ios::binary);
    if (!in) return out;
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    if (text.empty()) return out;

    std::string date = extract_date_from_filename(filename);
    if (date.empty()) {
        // No date in the filename (e.g. a MEMORY.md snapshot) — fall back
        // to the file's mtime, truncated to a bare date. No time-of-day
        // component: we don't actually know one, so don't invent it.
        auto ftime = fs::last_write_time(path);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
        std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&tt));
        date = buf;
    }

    for (auto& chunk : chunk_markdown(text, min_chunk_size)) {
        if (chunk.text.empty()) continue;
        // chunk_markdown() bakes heading lines (e.g. "# Session: 2026-04-13
        // 21:43:46 UTC" / "## Conversation Summary") into the literal text
        // — not just a leading block, but also mid-chunk whenever its
        // merge pass stitches paragraphs from different sub-sections
        // together. Useful for chunk_markdown's own general-purpose
        // callers, but here it's pure duplication: the row's created_at
        // already carries the date, and the section breadcrumb is kept
        // separately below. Strip every '#'-prefixed heading line
        // (anywhere in the chunk) so none of this gets embedded/FTS-
        // indexed as content — repeated across every chunk in a file,
        // it would dilute search.
        std::string body;
        body.reserve(chunk.text.size());
        {
            std::istringstream ls(chunk.text);
            std::string line;
            bool first_line = true;
            while (std::getline(ls, line)) {
                if (heading_level(line) > 0) continue;
                if (!first_line) body += "\n";
                body += line;
                first_line = false;
            }
        }
        // Collapse any now-consecutive blank lines left behind by removed
        // headings, and trim.
        body = std::regex_replace(body, std::regex(R"(\n{3,})"), "\n\n");
        while (!body.empty() && (body.front() == '\n' || body.front() == ' '))
            body.erase(body.begin());
        while (!body.empty() && (body.back() == '\n' || body.back() == ' '))
            body.pop_back();
        if (body.empty()) continue;
        if ((int)body.size() < min_chunk_chars) continue;
        FlatMemoryChunk fc;
        fc.text = body;
        fc.heading_path = chunk.section;
        fc.date = date;
        fc.is_decision_like = memory_chunk_is_decision_like(chunk.section, body,
                                                            /*default_decision_like=*/false);
        out.push_back(std::move(fc));
    }
    return out;
}

ConversationFormat detect_conversation_format(const std::string& path) {
    std::ifstream in(path);
    if (!in) return ConversationFormat::Unknown;

    // JSONL check first: peek at the first non-blank line and see if it
    // parses standalone as a Claude Code event — a whole-file json::parse
    // would fail on real JSONL (multiple top-level values), so this has
    // to happen before attempting the full-document parse below.
    {
        std::string first_line;
        while (std::getline(in, first_line)) {
            auto t = first_line;
            size_t a = t.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;  // blank, keep scanning
            t = t.substr(a);
            try {
                json ev = json::parse(t);
                if (ev.is_object() && ev.contains("type") &&
                    (ev.value("type", "") == "user" || ev.value("type", "") == "assistant") &&
                    ev.contains("message")) {
                    return ConversationFormat::ClaudeCode;
                }
            } catch (...) {
                // not a standalone-parseable line; fall through to whole-
                // document JSON parse below.
            }
            break;
        }
    }

    in.clear();
    in.seekg(0);
    json doc;
    try {
        in >> doc;
    } catch (...) {
        return ConversationFormat::Unknown;
    }

    // Claude web export: array of conversation objects, or {"conversations":[...]}.
    json convs;
    if (doc.is_array()) convs = doc;
    else if (doc.is_object() && doc.contains("conversations") && doc["conversations"].is_array())
        convs = doc["conversations"];
    if (convs.is_array() && !convs.empty() && convs[0].is_object() &&
        (convs[0].contains("chat_messages") || convs[0].contains("messages")) &&
        convs[0].contains("uuid")) {
        return ConversationFormat::ClaudeWeb;
    }

    if (doc.is_object()) {
        // Telegram: single-chat export (top-level "messages") or full
        // multi-chat export ("chats":{"list":[...]}).
        if (doc.contains("messages") && doc["messages"].is_array()) {
            return ConversationFormat::Telegram;
        }
        if (doc.contains("chats") && doc["chats"].is_object() &&
            doc["chats"].contains("list")) {
            return ConversationFormat::Telegram;
        }
        // Claude memories.json: single object, no message/conversation
        // structure at all, just a narrative blob + account id.
        if (doc.contains("conversations_memory")) {
            return ConversationFormat::ClaudeMemories;
        }
    }
    if (doc.is_array() && !doc.empty() && doc[0].is_object() &&
        doc[0].contains("conversations_memory")) {
        return ConversationFormat::ClaudeMemories;
    }

    return ConversationFormat::Unknown;
}

// =========================================================================
// Conversation parsers
// =========================================================================

namespace fs = std::filesystem;

namespace {

/// Trim ASCII whitespace from both ends.
std::string trim(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back()))  s.pop_back();
    return s;
}

/// Strip inline tool-call trace lines that some Claude clients/hooks render
/// as literal text inside an assistant text block — lines like
/// `⚙️ ragger_search: "..."` or `🔍 session_search: "..."`. These are
/// UI chrome for a tool invocation, not conversational content, and
/// duplicate what the actual tool_use/tool_result blocks already carry
/// (which extract_claude_text already excludes). Matched by the leading
/// glyph + "name: " shape rather than a fixed tool-name list, so it covers
/// any current or future tool without needing updates here.
std::string strip_tool_trace_lines(const std::string& text) {
    static const std::regex trace_line(
        R"(^[ \t]*(?:⚙️|🔍|🔧|🛠️)[ \t]*[A-Za-z0-9_]+:.*$\n?)",
        std::regex::multiline);
    return std::regex_replace(text, trace_line, "");
}

/// Extract text from a Claude message body. The body may be a raw string or
/// an array of content blocks; only type=="text" blocks contribute (tool
/// blocks / thinking blocks aren't conversational and would pollute search).
std::string extract_claude_text(const json& content) {
    if (content.is_string()) return trim(strip_tool_trace_lines(content.get<std::string>()));
    if (!content.is_array()) return "";
    std::vector<std::string> parts;
    for (const auto& block : content) {
        if (!block.is_object()) continue;
        if (block.value("type", "") == "text") {
            auto t = trim(strip_tool_trace_lines(block.value("text", "")));
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

namespace {

/// Flatten a Telegram "text" field. It's either a plain string or an array
/// mixing plain strings with `{"type":..,"text":..}` entity objects (bold,
/// link, code, etc.) — we just want the concatenated text content.
std::string extract_telegram_text(const json& text_field) {
    if (text_field.is_string()) return trim(text_field.get<std::string>());
    if (!text_field.is_array()) return "";
    std::string out;
    for (const auto& piece : text_field) {
        if (piece.is_string()) {
            out += piece.get<std::string>();
        } else if (piece.is_object()) {
            out += piece.value("text", "");
        }
    }
    return trim(out);
}

/// True if `text` is a bare control command that should never be stored as
/// conversation content: a slash-command (e.g. "/stop", "/model", "/reset",
/// with or without trailing arguments) or one of the bare no-slash keywords
/// users type as an interrupt ("stop", "wait" — case-insensitive, exact
/// match only, so "please wait" or "stopwatch" are NOT treated as commands).
bool is_bare_command(const std::string& text) {
    if (text.empty()) return false;
    if (text[0] == '/') return true;
    std::string lower = text;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "stop" || lower == "wait";
}

/// Parse one chat object (has "messages"; may or may not have "name")
/// into ImportTurns, appending onto `out`. Session identity: Telegram has
/// no session concept, but the user's `/new` command messages mark exact
/// boundaries where the live agent minted a fresh session. We mint
/// synthetic GUIDs (telegram-import-0001, -0002, ...) and roll to the
/// next on every `/new`. The `/new` message itself is a command, not
/// conversation — it is dropped. Other bare commands (/stop, /model, plain
/// "stop"/"wait", etc.) are dropped too but do NOT roll the session.
void parse_telegram_chat(const json& chat, const std::string& self_name,
                          std::vector<ImportTurn>& out) {
    if (!chat.is_object() || !chat.contains("messages")) return;

    int session_no = 1;
    auto session_guid = [&]() {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "telegram-import-%04d", session_no);
        return std::string(buf);
    };

    struct Block { bool is_self; std::string ts; std::string text; std::string sid; };
    std::vector<Block> blocks;

    for (const auto& m : chat["messages"]) {
        if (!m.is_object()) continue;
        if (m.value("type", "") != "message") continue;  // skip service events
        std::string text = extract_telegram_text(m.value("text", json("")));
        if (text.empty()) continue;
        std::string from = m.value("from", "");
        bool is_self = (from == self_name);
        std::string ts = m.value("date", "");

        // Session boundary: a user message that is the /new command.
        if (is_self && (text == "/new" || text.rfind("/new ", 0) == 0)) {
            ++session_no;
            continue;  // command, not conversation
        }

        // Other bare commands (/stop, /model, /reset, plain "stop"/"wait",
        // etc.) are control chatter, not conversation content — drop them
        // without touching session numbering.
        if (is_self && is_bare_command(text)) continue;

        // Merge consecutive same-sender bubbles — but never across a
        // session boundary (sid mismatch breaks the merge naturally).
        if (!blocks.empty() && blocks.back().is_self == is_self &&
            blocks.back().sid == session_guid()) {
            blocks.back().text += "\n" + text;
        } else {
            blocks.push_back({is_self, ts, text, session_guid()});
        }
    }

    std::optional<Block> pending_user;
    for (const auto& b : blocks) {
        if (b.is_self) {
            if (pending_user) {
                // Two user blocks in a row shouldn't happen (merged above),
                // but guard anyway: flush the stale one with no reply.
                out.push_back({pending_user->ts, pending_user->text, "", pending_user->sid, "telegram"});
            }
            pending_user = b;
        } else {
            if (pending_user) {
                // A reply from a different session than its prompt means a
                // /new landed between them — don't pair across it.
                if (pending_user->sid == b.sid) {
                    out.push_back({pending_user->ts, pending_user->text, b.text, b.sid, "telegram"});
                } else {
                    out.push_back({pending_user->ts, pending_user->text, "", pending_user->sid, "telegram"});
                }
                pending_user.reset();
            }
            // Leading assistant-only blocks (no prior user turn) are dropped —
            // nothing to anchor them to.
        }
    }
    if (pending_user) {
        out.push_back({pending_user->ts, pending_user->text, "", pending_user->sid, "telegram"});
    }
}

} // anonymous

std::vector<ImportTurn> parse_telegram(const std::string& path_str, const std::string& self_name) {
    std::vector<ImportTurn> out;
    std::ifstream in(path_str);
    if (!in) return out;
    json doc;
    try { in >> doc; } catch (...) { return out; }
    if (!doc.is_object()) return out;

    if (doc.contains("messages")) {
        // Single-chat export: the document itself is the chat object.
        parse_telegram_chat(doc, self_name, out);
    } else if (doc.contains("chats") && doc["chats"].is_object() &&
               doc["chats"].contains("list")) {
        for (const auto& chat : doc["chats"]["list"]) {
            parse_telegram_chat(chat, self_name, out);
        }
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

bool is_synthetic_session_guid(const std::string& guid) {
    if (guid.empty()) return true;
    std::string lower;
    lower.reserve(guid.size());
    for (char c : guid) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (lower == "telegram-import" || lower.rfind("telegram-import-", 0) == 0) return true;
    if (lower.find("test") != std::string::npos) return true;
    if (lower.find("debug") != std::string::npos) return true;
    if (lower.find("synthetic") != std::string::npos) return true;
    return false;
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
