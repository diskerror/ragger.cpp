/**
 * Conversation summarization — used by the HTTP server's daemon-resident
 * summarizer (L2 turn / L3 session, idle-triggered housekeeping).
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

/// LR S62 (shorter-text rule): whitespace-token count of the trimmed
/// string. A single-token source measures 1.
size_t count_words(const std::string& s);

/// LR S62 (shorter-text rule, addenda 2/3): decide between the source text
/// and the model's summary. Measurement is the whitespace-token count of the
/// trimmed strings; when the source has no whitespace at all (one token —
/// words are meaningless there) both sides are compared by Unicode code
/// points instead. The summary wins only when it measures STRICTLY less:
/// longer summaries and ties keep the source (a copy). An empty summary is
/// not a valid summary — it is returned with copied=false so the caller
/// treats it as a failure, exactly as today.
struct SummaryChoice {
    std::string text;
    bool        copied;   // true = the source was kept, not the model's text
};
SummaryChoice choose_summary(const std::string& source, const std::string& summary);

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
