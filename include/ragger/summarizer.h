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

} // namespace ragger
