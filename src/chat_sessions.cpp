/**
 * HTTP chat session manager implementation
 */
#include "ragger/chat_sessions.h"
#include "ragger/config.h"
#include "ragger/sqlite_backend.h"
#include "nlohmann_json.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace ragger {

// -----------------------------------------------------------------------
// ChatSession
// -----------------------------------------------------------------------
ChatSession::ChatSession(const std::string& id, const std::string& user)
    : session_id(id), username(user)
    , last_activity(std::chrono::system_clock::now())
    , created_at(std::chrono::system_clock::now())
{}

void ChatSession::add_user_message(const std::string& text) {
    messages.push_back({"user", text});
    last_activity = std::chrono::system_clock::now();
}

void ChatSession::add_assistant_message(const std::string& text) {
    messages.push_back({"assistant", text});
    if (messages.size() >= 2) {
        unsummarized_turns.push_back({
            messages[messages.size() - 2].content,  // user
            text  // assistant
        });
    }
    last_activity = std::chrono::system_clock::now();
}

int ChatSession::idle_seconds() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now - last_activity).count();
}

std::vector<Message> ChatSession::build_messages(
    const std::string& system_prompt,
    const std::string& memory_context,
    int max_turns) const
{
    std::vector<Message> result;

    // System prompt with memory context
    std::string system_content = system_prompt;
    if (!memory_context.empty()) {
        system_content += "\n\n## Relevant memories:\n\n" + memory_context;
    }
    if (!system_content.empty()) {
        result.push_back({"system", system_content});
    }

    // Conversation history (bounded)
    int max_messages = max_turns * 2;
    int start = (int)messages.size() > max_messages
        ? (int)messages.size() - max_messages : 0;
    for (int i = start; i < (int)messages.size(); ++i) {
        result.push_back(messages[i]);
    }

    return result;
}

// -----------------------------------------------------------------------
// ChatSessionManager
// -----------------------------------------------------------------------
std::string ChatSessionManager::generate_id() {
    // Simple UUID-like random string
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char hex[] = "0123456789abcdef";

    std::string id;
    id.reserve(36);
    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) id += '-';
        id += hex[dis(gen)];
    }
    return id;
}

ChatSession& ChatSessionManager::get_or_create(
    const std::string& session_id, const std::string& username,
    StorageBackend* backend)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Check in-memory cache first
    if (!session_id.empty()) {
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            it->second.last_activity = std::chrono::system_clock::now();
            return it->second;
        }
    }

    // Try to restore from database
    if (!session_id.empty() && backend) {
        auto messages_json = backend->get_chat_session(session_id);
        if (messages_json) {
            try {
                auto messages_array = json::parse(*messages_json);
                ChatSession restored(session_id, username);
                // Parse messages array into ChatSession
                for (const auto& msg : messages_array) {
                    if (msg.contains("role") && msg.contains("content")) {
                        std::string role = msg["role"];
                        std::string content = msg["content"];
                        if (role == "user") {
                            restored.add_user_message(content);
                        } else if (role == "assistant") {
                            restored.add_assistant_message(content);
                        }
                    }
                }
                sessions_.emplace(session_id, std::move(restored));
                return sessions_[session_id];
            } catch (const std::exception&) {
                // Fall through to create new session
            }
        }
    }

    // Create new session
    std::string id = session_id.empty() ? generate_id() : session_id;
    sessions_.emplace(id, ChatSession(id, username));
    return sessions_[id];
}

std::string ChatSessionManager::load_workspace_files() {
    // Load persona/workspace files in priority order.
    // The combined content serves as the system prompt for Ragger-owned sessions
    // (CLI and browser chat). Proxy sessions pass through client system prompts unchanged.
    const auto& cfg = config();
    std::string user_dir = expand_path("~/.ragger");
    std::string result;

    auto load_file = [&](const std::string& path, const std::string& label) {
        if (!fs::exists(path)) return;
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        auto end = content.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) content = content.substr(0, end + 1);
        if (content.empty()) return;
        if (!result.empty()) result += "\n\n---\n\n";
        result += "## " + label + "\n\n" + content;
    };

    // system_prompt_file first (configurable; default ~/.ragger/SYSTEM.md)
    std::string sys_path = expand_path(cfg.system_prompt_file);
    load_file(sys_path, fs::path(sys_path).filename().string());

    // Persona files from OpenClaw / Ragger workspace
    for (const auto& fname : {"SOUL.md", "USER.md", "MEMORY.md"}) {
        std::string fpath = user_dir + "/" + fname;
        // skip if this is the same file as system_prompt_file (avoid duplicates)
        if (fs::path(fpath).lexically_normal() == fs::path(sys_path).lexically_normal()) continue;
        load_file(fpath, fname);
    }

    return result;
}

std::vector<ChatSessionManager::ExpiredSession>
ChatSessionManager::cleanup_expired(int pause_minutes) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ExpiredSession> expired;
    std::vector<std::string> to_remove;

    for (auto& [sid, session] : sessions_) {
        if (session.idle_seconds() > pause_minutes * 60) {
            if (!session.unsummarized_turns.empty()) {
                expired.push_back({
                    session.username,
                    session.session_id,
                    session.unsummarized_turns
                });
            }
            to_remove.push_back(sid);
        }
    }

    for (const auto& sid : to_remove) {
        sessions_.erase(sid);
    }

    return expired;
}

} // namespace ragger
