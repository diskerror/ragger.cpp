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
#include <map>
#include <stdexcept>

namespace ragger {

namespace {

void tick(const ProgressFn& progress, const std::string& kind, int done, int total) {
    if (progress) progress(kind.c_str(), done, total);
}

// LR S90: cancel between rows. Rows already applied stay applied (each
// write is its own transaction); the caches must not keep serving values
// the pass already replaced, so invalidate before throwing. A rerun
// re-selects only what is left — the pass is resumable by construction.
void check_cancel(const CancelFlag* cancel, const std::string& kind,
                  const BackfillResult& res, StorageBackend& backend) {
    if (!cancel || !cancel->load()) return;
    if (res.applied > 0) backend.invalidate_all_caches();
    throw cancelled_error(std::format(
        "backfill {}: cancelled after {} applied, {} failed",
        kind, res.applied, res.failed));
}

/// The exact text the summarizer sees for a turn (mirrors handle_turn_summary: the
/// system-injected user side is blanked; an assistant-only turn summarizes
/// the assistant alone). Doubles as the copy text for the S62 rules.
std::string turn_summary_source(const std::string& user_text, const std::string& assistant_text) {
    const std::string eff_user =
        is_system_injected_turn(user_text) ? std::string{} : user_text;
    return eff_user.empty() ? assistant_text : eff_user + " | " + assistant_text;
}

/// Drafting call for the two kinds the turn and episode prompts do not fit:
/// a trigger is phrased in the future query's vocabulary (design §5.3a),
/// and a session title is a few words. The prompts are settings
/// (summarizer_trigger_prompt / summarizer_title_prompt) with built-in
/// defaults in Config; same strip_thinking + strictly-shorter contract as
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

/// The configured prompt, or the built-in one when the setting is empty or
/// blank. Copies at once: config() may be swapped by a live reload.
std::string prompt_or_default(const std::string& configured, std::string_view builtin) {
    if (configured.find_first_not_of(" \t\r\n") == std::string::npos)
        return std::string(builtin);
    return configured;
}


void require_inference_configured(InferenceClient& inference) {
    // S25 (C M10) parity: "not configured" is a refusal, not a failure
    // count. The CLI prints it and exits without touching the DB.
    if (inference.endpoints.empty())
        throw std::invalid_argument("inference not configured");
}

} // namespace

BackfillResult backfill_turn_summary(StorageBackend& backend, InferenceClient& inference,
                           const std::string& kind, bool dry_run,
                           const std::string& model_name,
                           const std::vector<int>& ids,
                           const ProgressFn& progress,
                           const CancelFlag* cancel) {
    if (kind != "null" && kind != "empty" && kind != "regenerate")
        throw std::invalid_argument("unknown turn-summary backfill kind '" + kind + "'");
    // Addendum 5: regeneration is eligible, never run by default.
    if (kind == "regenerate" && model_name.empty() && ids.empty())
        throw std::invalid_argument(
            "turn-summary-regenerate requires an explicit filter (--model <name> or --ids)");
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.turn_summary_backfill_candidates(kind, model_name, ids);
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    const std::string label = "turn-summary-" + kind;
    const bool overwrite = kind == "regenerate";
    const size_t floor_words =
        static_cast<size_t>(std::max(0, config().summarizer_min_words));
    int done = 0;
    for (const auto& c : cands) {
        check_cancel(cancel, label, res, backend);
        ++done;
        tick(progress, label, done, res.candidates);
        const std::string source = turn_summary_source(c.user_text, c.assistant_text);
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
            if (backend.apply_turn_summary_text(c.turn_summary_id, text, model, overwrite))
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
                                 const ProgressFn& progress,
                                 const CancelFlag* cancel) {
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.trigger_backfill_candidates();
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    int done = 0;
    for (const auto& c : cands) {
        check_cancel(cancel, "triggers", res, backend);
        ++done;
        tick(progress, "triggers", done, res.candidates);
        try {
            const std::string draft = draft_with_prompt(
                inference, prompt_or_default(config().summarizer_trigger_prompt, Config::kDefaultTriggerPrompt),
                c.text);
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

BackfillResult backfill_stranded(StorageBackend& backend, InferenceClient& inference,
                                 bool dry_run,
                                 const ProgressFn& progress,
                                 const CancelFlag* cancel) {
    // X23: the sweep regenerates too (Reid, 2026-11-14). Attaching a
    // late turn-summary row without rebuilding the episode would leave the DERIVED
    // episode text describing the old membership. Like every real pass this
    // one requires inference; a dry run only counts and never needs it.
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    int ties = 0;
    const auto cands = backend.stranded_turn_summary_candidates(ties);
    res.candidates = static_cast<int>(cands.size());
    res.ties = ties;
    // Open tail: content turn-summary rows with no episode whose time lies after the
    // last span of their session (or whose session has no episodes at all).
    // Counted, never touched — those spans have not been closed yet.
    res.open_tail = backend.stranded_open_tail_count();

    // One unit of work per episode that gains at least one late row (the
    // same grouping the live catch-up pass uses). `episodes` is reported for
    // both modes: a dry run's whole output is its counts.
    std::map<int, std::vector<int>> by_episode;
    for (const auto& c : cands) by_episode[c.episode_id].push_back(c.turn_summary_id);
    res.episodes = static_cast<int>(by_episode.size());
    if (dry_run) return res;

    int done = 0;
    for (const auto& [episode_id, late_ids] : by_episode) {
        check_cancel(cancel, "stranded", res, backend);
        ++done;
        tick(progress, "stranded", done, res.episodes);
        try {
            // Exactly steps 2 and 4 of SummarizerService::handle_episode_regen
            // (no stale refresh here — the sweep attaches; keeping stale rows
            // fresh is the live pass's job): the episode's members plus the
            // late rows, in turn_datetime order.
            auto members = backend.episode_members(episode_id);
            std::vector<int> attach;
            for (const int ts_id : late_ids) {
                const auto lm = backend.turn_summary_member_by_id(ts_id);
                if (!lm || lm->episode_id != 0 || lm->text.empty()) continue;
                if (std::any_of(members.begin(), members.end(),
                                [ts_id](const auto& m) {
                                    return m.turn_summary_id == ts_id;
                                }))
                    continue;
                const auto at = std::lower_bound(
                    members.begin(), members.end(), *lm,
                    [](const auto& a, const auto& b) {
                        return a.turn_datetime < b.turn_datetime ||
                               (a.turn_datetime == b.turn_datetime &&
                                a.turn_summary_id < b.turn_summary_id);
                    });
                members.insert(at, *lm);
                attach.push_back(ts_id);
            }
            if (attach.empty()) { ++res.skipped; continue; }  // attached concurrently

            std::vector<std::string> texts;
            texts.reserve(members.size());
            std::string single_model;
            for (const auto& m : members) {
                texts.push_back(m.text);
                single_model = m.model_name;
            }
            const auto ms = summarize_member_texts(inference, texts, single_model);
            if (ms.text.empty()) { ++res.failed; continue; }

            // cleared = {} : the sweep never refreshes a member, so it clears
            // no stale counters. A row that is no longer attachable rolls the
            // whole unit back (episode text included) and counts as skipped.
            if (backend.replace_episode_text(episode_id, ms.text, ms.model, {}, attach))
                res.applied += static_cast<int>(attach.size());
            else
                ++res.skipped;   // lost the attach race
        } catch (const std::exception&) {
            ++res.failed;
        }
    }
    tick(progress, "stranded", -1, res.episodes);
    if (res.applied > 0) backend.invalidate_all_caches();
    return res;
}

BackfillResult backfill_session_names(StorageBackend& backend,
                                      InferenceClient& inference, bool dry_run,
                                      const ProgressFn& progress,
                                      const CancelFlag* cancel) {
    if (!dry_run) require_inference_configured(inference);

    BackfillResult res;
    const auto cands = backend.unnamed_session_candidates();
    res.candidates = static_cast<int>(cands.size());
    if (dry_run) return res;

    int done = 0;
    for (const auto& c : cands) {
        check_cancel(cancel, "session-names", res, backend);
        ++done;
        tick(progress, "session-names", done, res.candidates);
        if (c.turn_summary_texts.empty()) { ++res.skipped; continue; }
        try {
            // The turn-summary texts are the title corpus (D5). Join the way
            // summarize_texts joins them so the model sees one document.
            std::string corpus;
            for (const auto& t : c.turn_summary_texts) {
                if (!corpus.empty()) corpus += "\n\n";
                corpus += t;
            }
            const std::string title = draft_with_prompt(
                inference, prompt_or_default(config().summarizer_title_prompt, Config::kDefaultTitlePrompt),
                corpus);
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
