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

/// Final line-level cleanup applied to a chunk's text right before it's
/// stored in the `documents` table (issue: Markdown heading/list noise
/// bloating L5 rows). For every line: strip leading spaces and '#'
/// characters (in any mix — "  ## Heading" -> "Heading", "### " ->
/// "", a bare "#" -> ""), and strip trailing spaces. Newlines and blank
/// lines are preserved as structure; this only touches the run of
/// leading/trailing characters on each line, never the line count.
/// Deliberately separate from chunk_markdown() itself: the chunker's own
/// heading-block text is still needed verbatim for section-breadcrumb
/// logic and is covered by existing tests, so this is a distinct pass
/// callers apply only when the destination is a stored document body.
std::string clean_document_text(const std::string& text);

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
/// → session-like) when the heading gives no signal. If neither signal
/// fires, returns `default_decision_like` — true for Claude's
/// memories.json (a narrative that really does skew toward standing
/// facts/preferences), false for flat daily-log-style memory files
/// (dated entries are episodes-without-a-session-id by construction, so
/// the safer unsignaled default there is session-like, not decision-like).
bool memory_chunk_is_decision_like(const std::string& heading_path,
                                   const std::string& text,
                                   bool default_decision_like = true);


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

/// Extract a leading "YYYY-MM-DD" date from a filename, e.g.
/// "2026-03-26.md" -> "2026-03-26", "2026-04-13-http-400-foo.md" ->
/// "2026-04-13". Returns empty string if the filename has no such
/// prefix. Used by import-conversations' flat-Markdown-memory-log path
/// (OpenClaw/nanobot-style daily notes and MEMORY.md snapshots), where
/// the filename is frequently the only date signal available.
std::string extract_date_from_filename(const std::string& filename);

/// Basenames (case-insensitive) that are agent persona/scaffolding files,
/// not memory content — SOUL.md, USER.md, AGENTS.md, TOOLS.md,
/// IDENTITY.md, HEARTBEAT.md, BOOTSTRAP.md. import-conversations' flat-
/// Markdown-memory path skips these automatically when walking a
/// directory, since they're prompt boilerplate repeated verbatim across
/// every agent workspace, not conversational memory.
bool is_agent_scaffolding_filename(const std::string& filename);

/// One heading-aware chunk lifted from a flat OpenClaw/nanobot/zeroclaw-
/// style Markdown memory file (dated daily log like "2026-03-26.md", or
/// a curated "MEMORY.md" snapshot) — plain `#`/`##`/`###` headings, no
/// JSON wrapper. `date` is "YYYY-MM-DD" only (never a full timestamp):
/// dated daily logs carry no time-of-day signal beyond what's in their
/// own heading text (which is prose, not parsed), and MEMORY.md snapshots
/// are dated by file mtime, which is itself just "whenever this snapshot
/// happened to be saved" — not a real moment worth pretending to
/// second-granularity precision.
struct FlatMemoryChunk {
    std::string text;
    std::string heading_path;   // e.g. "Ragger Architecture Simplification"
    std::string date;           // "YYYY-MM-DD"
    bool is_decision_like;
};

/// Parse one flat Markdown memory file into heading-aware chunks, classify
/// each via memory_chunk_is_decision_like, and date every chunk the same
/// way: filename-embedded "YYYY-MM-DD" prefix if present (daily logs),
/// else the file's mtime truncated to a date (MEMORY.md snapshots with no
/// date in their name). Returns an empty vector for scaffolding filenames
/// (see is_agent_scaffolding_filename) or files with no headings/content.
/// `min_chunk_chars` drops any final chunk shorter than this after
/// chunk_markdown's merge pass — chunk_markdown's own min_chunk_size only
/// governs *merge* boundaries, not a guaranteed final size (the very last
/// paragraph in a file, or one that starts its own chunk immediately, can
/// still come out tiny — e.g. a bare "please continue" reply, or a
/// stand-alone "Session Key: ..." metadata line). Those aren't worth a
/// DB row on their own.
std::vector<FlatMemoryChunk> parse_flat_markdown_memory(const std::string& path,
                                                        int min_chunk_size = 200,
                                                        int min_chunk_chars = 120);


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
