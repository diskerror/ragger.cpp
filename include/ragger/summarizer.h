/**
 * Conversation summarization — shared by the CLI chat (orphan-turn recovery)
 * and the HTTP server (idle-session housekeeping), which previously each
 * inlined their own transcript build + (drifted) summary prompt + model call.
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace ragger {

class InferenceClient;

/// Summarize a conversation (user/assistant text pairs) into one concise
/// memory entry. Runs on the configured memory model via
/// InferenceClient::chat_memory(), which falls back to the main model when no
/// separate memory model is set. Returns "" for empty input or on error; the
/// caller decides where/whether to store the result.
std::string summarize_transcript(
    InferenceClient& inference,
    const std::vector<std::pair<std::string, std::string>>& turns);

/// Summarize a flat list of text blobs (e.g. L2 summaries → L3, or L3
/// summaries → L4). Each entry is treated as an equal-weight fragment;
/// the same percentage-based size contract as summarize_transcript applies,
/// measured against the total character count of all input texts.
/// Returns "" for empty input or on error.
std::string summarize_texts(
    InferenceClient& inference,
    const std::vector<std::string>& texts);

/// True when the user side of a turn is *entirely* a Hermes system-injected
/// annotation rather than something the human typed — model-switch notes,
/// context-compaction handoffs, interruption notices, background-process
/// completion reports. These carry no conversational content worth indexing,
/// so the summarizer skips them deterministically (no inference call, no L2
/// summary row). The raw turn is still persisted upstream by capture_turn, so
/// the system event remains auditable; only the searchable summary is omitted.
/// Detection is prefix-based on the stable bracketed markers Hermes emits.
bool is_system_injected_turn(const std::string& user_text);

/// Strip any leading Hermes system-injected annotation block(s) from a user
/// turn, returning only the real human message that follows (trimmed). Hermes
/// prepends notes like "[System note: ...] \n\n stop" — the human typed only
/// "stop". Returns "" when the text is nothing but system annotation. Applied
/// at capture time so the raw `turns` row stores clean user content. Handles
/// balanced-bracket markers (System note, model-switch Note, bg-process
/// IMPORTANT) and the CONTEXT COMPACTION block (stripped through its
/// END-OF-SUMMARY sentinel). Idempotent; leaves non-system text untouched.
std::string strip_system_injected_prefix(const std::string& user_text);

/// Strip <think>…</think> blocks and bare "Thinking Process:\n…\n\n" preambles
/// from summarizer model output. Applied before any result is stored so
/// reasoning-model thinking traces never land in the DB.
std::string strip_thinking(const std::string& result);

} // namespace ragger
