/**
 * HTTP chat session manager
 *
 * Manages conversation state for /chat endpoint.
 * Sessions track message history and expire after inactivity.
 */
#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "ragger/inference.h"

namespace ragger {

// Forward declaration
class StorageBackend;

// --- Budget-aware payload assembly (issue #23) -------------------------
// A candidate piece of the inference payload. Lower `priority` = more
// important (shed last; 1=highest .. 9=lowest). SYSTEM pieces are merged into
// the leading system message in priority order, each under its `label`;
// TURN pieces are appended as conversation messages in the order added.
// `keep` pieces are never shed (the new user message, the previous raw turn).
enum class PieceKind { System, Turn };

struct PayloadPiece {
    PieceKind   kind     = PieceKind::System;
    int         priority = 5;   // shed importance: 1=highest..9=lowest
    int         order    = 0;   // SYSTEM positional order in messages[0] (asc);
                                // decoupled from priority (e.g. decisions sit
                                // last but outrank summaries in importance)
    bool        keep     = false;
    std::string role;      // TURN pieces: "user" / "assistant"
    std::string label;     // SYSTEM pieces: header, e.g. "## Session summary"
    std::string content;
};

struct AssembledPayload {
    std::vector<Message> messages;
    int  estimated_tokens = 0;
    int  shed_count       = 0;
    bool fit              = true;   // false if even keep pieces exceed budget
};

/// Rough token estimate (chars / chars_per_token, min 1 for non-empty).
int estimate_tokens(const std::string& text, float chars_per_token);

/// Fit prioritized pieces into `token_budget` by shedding the lowest-
/// importance non-keep pieces first (issue #23 shrinking). SYSTEM pieces are
/// concatenated into messages[0] in priority order; TURN pieces follow in
/// order. `token_budget` is the post-reserve budget (caller subtracts the
/// response reserve from max_context).
AssembledPayload assemble_payload(std::vector<PayloadPiece> pieces,
                                  int token_budget, float chars_per_token);

struct ChatSession {
    std::string session_id;
    std::string username;
    std::vector<Message> messages;  // conversation history
    std::vector<std::pair<std::string, std::string>> unsummarized_turns;
    std::chrono::system_clock::time_point last_activity;
    std::chrono::system_clock::time_point created_at;

    ChatSession() = default;
    ChatSession(const std::string& id, const std::string& user);

    void add_user_message(const std::string& text);
    void add_assistant_message(const std::string& text);
    int idle_seconds() const;

    /// Build full message array with system prompt + memory context + history
    std::vector<Message> build_messages(const std::string& system_prompt,
                                         const std::string& memory_context,
                                         int max_turns = 200) const;
};

class ChatSessionManager {
public:
    /// Get existing session or create a new one (optionally restore from DB)
    ChatSession& get_or_create(const std::string& session_id, const std::string& username,
                               StorageBackend* backend = nullptr);

    /// Generate a new session ID
    static std::string generate_id();

    /// Load workspace/persona files for system prompt
    static std::string load_workspace_files();

    /// Expired session data (username + unsummarized turns)
    struct ExpiredSession {
        std::string username;
        std::string session_id;
        std::vector<std::pair<std::string, std::string>> turns;  // (user_text, assistant_text)
    };

    /// Remove expired sessions (returns data for summarization)
    std::vector<ExpiredSession> cleanup_expired(int pause_minutes);

private:
    std::map<std::string, ChatSession> sessions_;
    std::mutex mutex_;
};

} // namespace ragger
