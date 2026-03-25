/**
 * Chat module — conversation REPL with persistence
 *
 * Features:
 * - Turn storage (per-turn or session mode)
 * - Turn expiration (age + count limits)
 * - Pause detection and summarization
 * - Quit summarization
 * - Dynamic context sizing with priority-based persona loading
 */
#pragma once

#include <string>
#include <vector>
#include <chrono>
#include "ragger/memory.h"
#include "ragger/inference.h"

namespace ragger {

class Chat {
public:
    /// Construct chat session with memory and inference client
    Chat(RaggerMemory& memory, InferenceClient& inference, const std::string& model);

    /// Run the chat REPL
    void run();

private:
    RaggerMemory& memory_;
    InferenceClient& inference_;
    std::string model_;
    
    // Conversation state
    std::vector<Message> messages_;
    std::vector<std::pair<std::string, std::string>> unsummarized_turns_;  // (user, assistant) pairs
    std::chrono::system_clock::time_point last_activity_;
    int session_memory_id_ = -1;  // for "session" mode
    
    // Config cache
    std::string store_turns_;
    bool summarize_on_pause_;
    int pause_minutes_;
    bool summarize_on_quit_;
    int max_turn_retention_minutes_;
    int max_turns_stored_;
    int max_memory_results_;
    
    /// Load workspace files with dynamic context sizing
    std::string load_workspace_files(int max_context);
    
    /// Store a single exchange to memory
    void store_turn(const std::string& user_text, const std::string& assistant_text);
    
    /// Check for orphaned turns from previous session (crash recovery)
    void check_orphaned_turns();
    
    /// Expire old turns based on age and count limits
    void expire_old_turns();
    
    /// Fork a background process to summarize turns
    void bg_summarize(const std::vector<std::pair<std::string, std::string>>& turns);
    
    /// Check if pause has elapsed and summarize if needed
    void check_pause_summary();
    
    /// Summarize conversation on quit
    void quit_summary();
    
    /// Generate summary from conversation turns
    std::string summarize_conversation(const std::vector<std::pair<std::string, std::string>>& turns);
    
    /// Update last activity timestamp
    void update_activity() {
        last_activity_ = std::chrono::system_clock::now();
    }
    
    /// Get idle seconds since last activity
    int idle_seconds() const {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - last_activity_).count();
    }
};

} // namespace ragger
