/**
 * Conversation summarization implementation. See include/ragger/summarizer.h.
 */
#include "ragger/summarizer.h"

#include "ragger/inference.h"
#include "ragger/lang.h"
#include "diskerror/logger.h"

#include <format>
#include <string_view>

namespace ragger {

namespace {

// Stable bracketed markers Hermes prepends to system-injected user turns.
// Matched case-sensitively against the trimmed start of the user text. Kept
// deliberately narrow: each is a phrase Hermes core controls, so a real human
// message is extraordinarily unlikely to begin with one. Note the OUT-OF-BAND
// USER MESSAGE marker is intentionally absent — that wraps a genuine human
// message and should be summarized normally.
constexpr std::string_view kSystemMarkers[] = {
    "[System note:",
    "[Note: model was just switched",
    "[CONTEXT COMPACTION",
    "[IMPORTANT: Background process",
};

std::string_view ltrim_view(std::string_view s) {
    size_t i = 0;
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        ++i;
    return s.substr(i);
}

} // namespace

bool is_system_injected_turn(const std::string& user_text) {
    const std::string_view body = ltrim_view(user_text);
    for (const auto marker : kSystemMarkers) {
        if (body.size() >= marker.size() &&
            body.compare(0, marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

namespace {

// Sentinel that closes a CONTEXT COMPACTION block. Everything from the
// "[CONTEXT COMPACTION" opener through the end of the line containing this
// sentinel is one annotation block. Matching the bracket alone is unsafe —
// the summary body is full of '[...]' lookalikes.
constexpr std::string_view kCompactionEnd = "END OF CONTEXT SUMMARY";

// Which marker (if any) does `body` start with? Returns index into
// kSystemMarkers, or -1.
int leading_marker(std::string_view body) {
    for (size_t k = 0; k < std::size(kSystemMarkers); ++k) {
        const auto m = kSystemMarkers[k];
        if (body.size() >= m.size() && body.compare(0, m.size(), m) == 0)
            return static_cast<int>(k);
    }
    return -1;
}

} // namespace

std::string strip_system_injected_prefix(const std::string& user_text) {
    std::string_view body = ltrim_view(user_text);

    // Strip leading annotation blocks repeatedly — Hermes can stack several
    // (e.g. a model-switch Note followed by a System note).
    for (;;) {
        const int idx = leading_marker(body);
        if (idx < 0) break;

        const std::string_view marker = kSystemMarkers[idx];
        if (marker == "[CONTEXT COMPACTION") {
            // Strip through the end of the line carrying the END sentinel.
            const size_t e = body.find(kCompactionEnd);
            if (e == std::string_view::npos) {
                // Malformed/truncated block — nothing trustworthy follows.
                return "";
            }
            size_t eol = body.find('\n', e);
            body = (eol == std::string_view::npos)
                       ? std::string_view{}
                       : ltrim_view(body.substr(eol + 1));
        } else {
            // Balanced-bracket marker: strip through the matching ']'.
            int depth = 0;
            size_t close = std::string_view::npos;
            for (size_t i = 0; i < body.size(); ++i) {
                if (body[i] == '[') ++depth;
                else if (body[i] == ']') {
                    if (--depth == 0) { close = i; break; }
                }
            }
            if (close == std::string_view::npos) {
                // Unterminated note — nothing trustworthy follows.
                return "";
            }
            body = ltrim_view(body.substr(close + 1));
        }
    }

    return std::string(body);
}

std::string summarize_transcript(
    InferenceClient& inference,
    const std::vector<std::pair<std::string, std::string>>& turns) {
    if (turns.empty()) return "";

    // Summary contract: never longer than the source, target ~1/4 of it.
    // Short exchanges (slash commands like /exit, single-line greetings)
    // can't be compressed — return them verbatim and skip inference.
    std::string conversation;
    std::size_t source_chars = 0;
    for (const auto& [user_text, asst_text] : turns) {
        // Omit an empty user side entirely (assistant-only turns, e.g. an
        // interrupted turn whose system-note user side was stripped). Emitting
        // a bare "**User:** " line made the model summarize the empty prompt
        // and produce degenerate output like "The assistant ".
        if (!user_text.empty())
            conversation += "**User:** " + user_text + "\n\n";
        conversation += "**Assistant:** " + asst_text + "\n\n";
        source_chars += user_text.size() + asst_text.size();
    }

    // Trivial-turn shortcut: a single short pair gets stored verbatim.
    // 120 chars ≈ a one-line command + a one-line reply.
    if (turns.size() == 1 && source_chars < 120) {
        const auto& [u, a] = turns.front();
        if (a.empty()) return u;
        if (u.empty()) return a;
        return u + " | " + a;
    }

    const std::size_t target_chars = std::max<std::size_t>(source_chars / 4, 40);
    const std::size_t hard_cap     = source_chars;

    std::vector<Message> messages = {
        {"system", std::format(
            "Summarize this conversation into a concise memory entry. "
            "Extract key facts, decisions, questions asked, topics discussed. "
            "Write in third person past tense. Target about {} characters; "
            "never exceed {}. If the exchange is trivial (a single command "
            "like /exit, a one-line greeting), respond with a short phrase, "
            "not a full sentence.",
            target_chars, hard_cap)},
        {"user", conversation}
    };

    std::string result;
    try {
        result = inference.chat_memory(messages);
    } catch (const std::exception& e) {
        Diskerror::logger::warn(std::format(lang::WARN_SUMMARY, e.what()));
        return "";
    }

    // Enforce the contract. If the model ignored the cap, prefer a
    // concatenation of the source turns (it's already shorter than the
    // bloated paraphrase and more useful for later retrieval).
    if (result.size() > hard_cap) {
        std::string fallback;
        for (const auto& [u, a] : turns) {
            if (!fallback.empty()) fallback += " | ";
            fallback += u;
            if (!a.empty()) fallback += " → " + a;
        }
        return (fallback.size() <= hard_cap) ? fallback : result.substr(0, hard_cap);
    }
    return result;
}

} // namespace ragger
