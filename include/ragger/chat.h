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

/// A single piece of turn content. Today only `type == "text"` is used;
/// the struct exists so tool_use / tool_result blocks can be added later
/// without reshaping the turn buffer or storage path.
struct ContentBlock {
    std::string type = "text";
    std::string text;
};

/// One conversation turn (a user prompt OR an assistant reply, not a pair).
/// Carries role + content blocks + the memory row id it was persisted as,
/// so summarization can locate raw rows for deletion later.
struct Turn {
    std::string role;                  // "user" | "assistant"
    std::vector<ContentBlock> content;
    int memory_id = -1;                // -1 = not persisted
};

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
    std::vector<Turn> unsummarized_turns_;
    std::chrono::system_clock::time_point last_activity_;
    
    // Config cache
    std::string store_turns_;
    bool summarize_on_pause_;
    int pause_minutes_;
    bool summarize_on_quit_;
    int max_memory_results_;
    
    /// Load workspace files with dynamic context sizing
    std::string load_workspace_files(int max_context);
    
    /// Persist the user prompt immediately (before the LLM call) so it
    /// survives a crash mid-stream. Returns the memory row id, or -1 if
    /// turn storage is disabled. Caller passes this id to finalize_turn().
    int store_partial_turn(const std::string& user_text);

    /// After the assistant response is complete, replace the partial row
    /// (created by store_partial_turn) with the full user+assistant
    /// exchange. Re-embeds against the combined text for retrieval.
    /// Returns the new memory row id.
    int finalize_turn(int partial_id,
                      const std::string& user_text,
                      const std::string& assistant_text);

    /// Flatten `unsummarized_turns_` (per-role Turn entries) into the
    /// (user, assistant) pair shape that summarization expects.
    std::vector<std::pair<std::string, std::string>> turns_as_pairs() const;
    
    /// Check for orphaned turns from previous session (crash recovery)
    void check_orphaned_turns();
    
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
