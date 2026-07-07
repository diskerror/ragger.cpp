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

/// Walk a JSON document and concatenate every string leaf worth keeping
/// into plain text (double-newline separated), preferring values under
/// keys that read as prose ("text", "content", "description",
/// "conversations_memory", "name", etc.) over structural keys
/// (uuid/id/type/status/...) which are skipped as noise. Falls back to an
/// empty string if `text` doesn't parse as JSON at all.
/// Used specifically by `import-conversations`'s memories.json handling,
/// where the shape is known in advance. NOT used by import-docs — general
/// "guess what's prose in arbitrary JSON" is exactly the kind of implicit
/// heuristic that silently mangles data; convert JSON to Markdown with a
/// script or a tool like docling and inspect it before importing instead.
std::string extract_text_from_json(const std::string& text);

/// One paragraph-level chunk split from a Claude memories.json narrative,
/// carrying the breadcrumb of **bold**/*italic* section headers it fell
/// under (e.g. "Brief history » Recent months") — the memory file's own
/// section syntax, distinct from `#`-style Markdown headings which
/// chunk_markdown expects. `heading_path` is used both as a classification
/// signal (session-like vs. decision-like) and to preserve provenance.
struct MemoryChunk {
    std::string text;
    std::string heading_path;  // e.g. "Brief history » Recent months"
};

/// Split a Claude memories.json narrative into paragraph-level chunks,
/// tracking **bold** (top-level) and *italic* (sub-level) section headers
/// as a breadcrumb. Blank-line-separated paragraphs are the chunk
/// granularity; a paragraph that is itself only a heading line updates
/// the breadcrumb and produces no chunk of its own.
std::vector<MemoryChunk> split_memory_narrative(const std::string& text);

/// Heuristic classification of a memory chunk: true if it reads as a
/// standing fact/preference (decision-like — L5/L6 `decisions`), false if
/// it reads as episodic narrative of a specific work session (session-like
/// — L3 `summaries`). Checks `heading_path` first (headings containing
/// "history"/"recent"/"earlier"/"background"/"timeline" → session-like;
/// "context"/"top of mind"/"preference" → decision-like), then falls back
/// to a verb/keyword scan of `text` (stative "prefers"/"is a"/"uses X as"
/// → decision-like; episodic "worked on"/"explored"/"set up"/"asked about"
/// → session-like) when the heading gives no signal.
bool memory_chunk_is_decision_like(const std::string& heading_path,
                                   const std::string& text);


/// Auto-detected shape of a conversation-import source, from sniffing its
/// JSON structure (never from filename). Used by `import-conversations` so
/// `--format=` is an override, not a requirement.
enum class ConversationFormat {
    ClaudeWeb,      // conversations.json: array/{"conversations":[...]}
                     // of {..., "chat_messages":[{sender,text,created_at}]}
    ClaudeCode,      // JSONL: {"type":"user"|"assistant","message":{...}}
    Telegram,        // {"messages":[...]} or {"chats":{"list":[...]}}
    ClaudeMemories,   // {"conversations_memory": "...", "account_uuid": "..."}
                      // — rejected by import-conversations (no derivable
                      // date without --fdate/--date=), handled instead by
                      // import-conversations's memories path or import-docs.
    Unknown
};

/// Sniff a single file's structure (reads just enough to identify the
/// shape; JSONL formats peek at the first non-blank line). Returns
/// Unknown for non-JSON or unrecognized JSON.
ConversationFormat detect_conversation_format(const std::string& path);


// -------------------------------------------------------------------------
// Conversation import (Claude Code JSONL + claude.ai web export).
// -------------------------------------------------------------------------

/// One user+assistant exchange lifted from an exported conversation.
/// `timestamp` is the source-format string verbatim (typically ISO-8601);
/// the importer is responsible for converting it to db_timestamp form when
/// it stores rows. `session_id` is the source conversation id for
/// code/web; for telegram it is a synthetic /new-boundary GUID
/// (telegram-import-NNNN) — the chat id is never carried (personal
/// identifier, must not land in the DB).
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

/// Parse a Telegram Desktop "Export Chat History" JSON — either a
/// single-chat `result.json` (top-level `messages` array) or a full
/// multi-chat export (top-level `chats.list[]`, each with its own
/// `messages`). Consecutive messages from the same sender are merged into
/// one block before pairing, since Telegram commonly splits one thought
/// into several bubbles. `self_name` identifies which sender's messages
/// become `user_text` (matched against the message's `from` field,
/// case-sensitively); everything else is `assistant_text`. User messages
/// that are the `/new` command mark session boundaries: turns get
/// synthetic session GUIDs (telegram-import-NNNN) that roll on each /new,
/// and the /new message itself is dropped.
std::vector<ImportTurn> parse_telegram(const std::string& path, const std::string& self_name);

/// Apply the filter in-place (returns the kept turns).
std::vector<ImportTurn> filter_turns(const std::vector<ImportTurn>& turns,
                                     const TurnFilter& f);

/// True if `guid` looks like a placeholder/synthetic session id rather
/// than a real one an agent assigned: empty, "telegram-import" or
/// "telegram-import-*", or containing "test"/"debug"/"synthetic"
/// (case-insensitive). Shared gate used by every importer that reconciles
/// session ids across sources (see main.cpp's conversation-import loop) —
/// a synthetic guid is safe to overwrite on a cross-source text match; a
/// real one (agent-assigned or from another authoritative import) is not.
bool is_synthetic_session_guid(const std::string& guid);

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
