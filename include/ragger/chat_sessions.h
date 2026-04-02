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
class SqliteBackend;

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
                               SqliteBackend* backend = nullptr);

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
