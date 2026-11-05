/**
 * LR S66/S67 — back-fill passes behind `ragger backfill <kind> [--dry-run]`.
 * See include/backfill.h for the contracts and the in-place column rules.
 */
#include "backfill.h"

#include "config.h"
#include "inference.h"
#include "storage_backend.h"
#include "summarizer.h"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace ragger {

namespace {

void tick(const ProgressFn& progress, const std::string& kind, int done, int total) {
    if (progress) progress(kind.c_str(), done, total);
}

/// The exact text the summarizer sees for a turn (mirrors handle_l2: the
/// system-injected user side is blanked; an assistant-only turn summarizes
/// the assistant alone). Doubles as the copy text for the S62 rules.
std::string l2_source(const std::string& user_text, const std::string& assistant_text) {
    const std::string eff_user =
        is_system_injected_turn(user_text) ? std::string{} : user_text;
    return eff_user.empty() ? assistant_text : eff_user + " | " + assistant_text;
}

/// Drafting prompts for the two kinds the L2/episode prompts do not fit:
/// a trigger is phrased in the future query's vocabulary (design §5.3a),
/// and a session title is one short line. Authored here (not in
/// summarizer.cpp) because they belong to the back-fill verb, not the live
/// summarizer flow; same strip_thinking + strictly-shorter contract as
/// every other summarizer output.
std::string draft_with_prompt(InferenceClient& inference,
                              const std::string& system_prompt,
                              const std::string& input) {
    std::vector<Message> messages{
        {"system", system_prompt},
        {"user", input}};
    try {
        auto out = strip_thinking(inference.chat_memory(messages));
        // Trim surrounding whitespace the same way the store paths do.
        const size_t a = out.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return {};
        const size_t b = out.find_last_not_of(" \t\r\n");
        return out.substr(a, b - a + 1);
    } catch (const std::exception&) {
        return {};  // unreachable endpoint / bad config -> caller counts failure
    }
}

constexpr const char* kTriggerDraftPrompt =
    "Write a TRIGGER for the stored memory below: one short line phrased as "
    "the question or situation that should bring this memory up in a future "
    "search. Use the vocabulary a future asker would use, not the memory's "
    "own wording where they differ. No quotation marks, no prefix, no "
    "trailing punctuation beyond a question mark. Never longer than the "
    "memory itself.";

constexpr const char* kSessionTitlePrompt =
    "Write a title for a conversation session from the summaries below: a "
    "few words naming the topic, no quotation marks, no trailing "
    "punctuation, no sentence.";

void require_inference_configured(InferenceClient& inference) {
    // S25 (C M10) parity: "not configured" is a refusal, not a failure
    // count. The CLI prints it and exits without touching the DB.
    if (inference.endpoints.empty())
        throw std::invalid_argument("inference not configured");
}

} // namespace

BackfillResult backfill_l2(StorageBackend& backend, InferenceClient& inference,
                           const std::string& kind, bool dry_run,
                           const std::string& model_name,
                           const std::vector<int>& ids,
                           const ProgressFn& progress) {
    if (kind != "null" && kind != "empty" && kind != "regenerate")
        throw std::invalid_argument("unknown l2 backfill kind '" + kind + "'");
    // Addendum 5: regeneration is eligible, never run by default.
    if (kind == "regenerate" && model_name.empty() && ids.empty())
        throw std::invalid_argument(
            "l2-regenerate requires an explicit filter (--model <name> or --ids)");
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.l2_backfill_candidates(kind, model_name, ids);
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    const std::string label = "l2-" + kind;
    const bool overwrite = kind == "regenerate";
    const size_t floor_words =
        static_cast<size_t>(std::max(0, config().summarizer_min_words));
    int done = 0;
    for (const auto& c : cands) {
        ++done;
        tick(progress, label, done, res.candidates);
        const std::string source = l2_source(c.user_text, c.assistant_text);
        if (source.empty()) { ++res.skipped; continue; }  // nothing to write
        try {
            std::string text;
            std::string model;
            if (count_words(source) <= floor_words) {
                // S62 floor: copy, no model call (works with inference down).
                text  = source;
                model = c.turn_model_name;   // copy provenance = the turn's model
            } else {
                const std::string summary =
                    summarize_transcript(inference, {{
                        is_system_injected_turn(c.user_text) ? std::string{} : c.user_text,
                        c.assistant_text }});
                if (summary.empty()) { ++res.failed; continue; }
                const auto choice = choose_summary(source, summary);
                text  = choice.text;
                model = choice.copied ? c.turn_model_name : inference.memory_model;
            }
            if (backend.apply_l2_text(c.turn_summary_id, text, model, overwrite))
                ++res.applied;
            else
                ++res.skipped;   // guard rejected (concurrent real write)
        } catch (const std::exception&) {
            ++res.failed;        // row unchanged; pass continues
        }
    }
    tick(progress, label, -1, res.candidates);
    if (res.applied > 0) backend.invalidate_all_caches();
    return res;
}

BackfillResult backfill_triggers(StorageBackend& backend,
                                 InferenceClient& inference, bool dry_run,
                                 const ProgressFn& progress) {
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.trigger_backfill_candidates();
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    int done = 0;
    for (const auto& c : cands) {
        ++done;
        tick(progress, "triggers", done, res.candidates);
        try {
            const std::string draft = draft_with_prompt(inference, kTriggerDraftPrompt, c.text);
            if (draft.empty()) { ++res.failed; continue; }
            // The draft must be a real trigger: strictly shorter than the
            // memory it points at (same measurement as the S62 rule). A
            // copy of the text is not a trigger.
            const auto choice = choose_summary(c.text, draft);
            if (choice.copied) { ++res.failed; continue; }
            if (backend.apply_trigger(c.knowledge_id, choice.text))
                ++res.applied;
            else
                ++res.skipped;   // guard rejected or row vanished
        } catch (const std::exception&) {
            ++res.failed;
        }
    }
    tick(progress, "triggers", -1, res.candidates);
    if (res.applied > 0) backend.invalidate_all_caches();
    return res;
}

BackfillResult backfill_stranded(StorageBackend& backend, bool dry_run,
                                 const ProgressFn& progress) {
    BackfillResult res;
    int ties = 0;
    const auto cands = backend.stranded_l2_candidates(ties);
    res.candidates = static_cast<int>(cands.size());
    res.ties = ties;
    // Open tail: content L2 rows with no episode whose time lies after the
    // last span of their session (or whose session has no episodes at all).
    // Counted, never touched — those spans have not been closed yet.
    res.open_tail = backend.stranded_open_tail_count();
    if (dry_run) return res;

    int done = 0;
    for (const auto& c : cands) {
        ++done;
        tick(progress, "stranded", done, res.candidates);
        try {
            if (backend.attach_stranded_l2(c.turn_summary_id, c.episode_id))
                ++res.applied;
            else
                ++res.skipped;
        } catch (const std::exception&) {
            ++res.failed;
        }
    }
    tick(progress, "stranded", -1, res.candidates);
    if (res.applied > 0) backend.invalidate_all_caches();
    return res;
}

BackfillResult backfill_session_names(StorageBackend& backend,
                                      InferenceClient& inference, bool dry_run,
                                      const ProgressFn& progress) {
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.unnamed_session_candidates();
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    int done = 0;
    for (const auto& c : cands) {
        ++done;
        tick(progress, "session-names", done, res.candidates);
        if (c.l2_texts.empty()) { ++res.skipped; continue; }
        try {
            // The L2 texts are the title corpus (D5). Join the way
            // summarize_texts joins them so the model sees one document.
            std::string corpus;
            for (const auto& t : c.l2_texts) {
                if (!corpus.empty()) corpus += "\n\n";
                corpus += t;
            }
            const std::string title = draft_with_prompt(inference, kSessionTitlePrompt, corpus);
            if (title.empty()) { ++res.failed; continue; }
            if (backend.apply_session_name(c.session_id, title, "summarizer"))
                ++res.applied;
            else
                ++res.skipped;   // named concurrently (user/agent name wins)
        } catch (const std::exception&) {
            ++res.failed;
        }
    }
    tick(progress, "session-names", -1, res.candidates);
    return res;
}

} // namespace ragger
