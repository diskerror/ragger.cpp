/**
 * Conversation summarization implementation. See include/ragger/summarizer.h.
 */
#include "ragger/summarizer.h"

#include "ragger/inference.h"
#include "ragger/lang.h"
#include "diskerror/logger.h"

#include <format>

namespace ragger {

std::string summarize_transcript(
    InferenceClient& inference,
    const std::vector<std::pair<std::string, std::string>>& turns) {
    if (turns.empty()) return "";

    std::string conversation;
    for (const auto& [user_text, asst_text] : turns) {
        conversation += "**User:** " + user_text + "\n\n";
        conversation += "**Assistant:** " + asst_text + "\n\n";
    }

    std::vector<Message> messages = {
        {"system", "Summarize this conversation into a concise memory entry. "
                   "Extract: key facts, decisions, questions asked, topics discussed. "
                   "Write in third person past tense. Be brief — this will be stored "
                   "as a memory chunk for future retrieval."},
        {"user", conversation}
    };

    try {
        return inference.chat_memory(messages);
    } catch (const std::exception& e) {
        Diskerror::logger::warn(std::format(lang::WARN_SUMMARY, e.what()));
        return "";
    }
}

} // namespace ragger
