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

/// True when the user side of a turn is *entirely* a Hermes system-injected
/// annotation rather than something the human typed — model-switch notes,
/// context-compaction handoffs, interruption notices, background-process
/// completion reports. These carry no conversational content worth indexing,
/// so the summarizer skips them deterministically (no inference call, no L2
/// summary row). The raw turn is still persisted upstream by capture_turn, so
/// the system event remains auditable; only the searchable summary is omitted.
/// Detection is prefix-based on the stable bracketed markers Hermes emits.
bool is_system_injected_turn(const std::string& user_text);

} // namespace ragger
