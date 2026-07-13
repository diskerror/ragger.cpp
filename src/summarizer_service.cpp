/**
 * SummarizerService implementation. See include/ragger/summarizer_service.h.
 */

#include "ragger/summarizer_service.h"

#include "ragger/config.h"
#include "ragger/inference.h"
#include "ragger/lang.h"
#include "ragger/storage_backend.h"
#include "ragger/summarizer.h"
#include "diskerror/logger.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace ragger {

namespace {

// Cap the per-side text in the heuristic draft. Picked to keep the row
// small and embedding-cheap; real summaries replace it later.
constexpr size_t kDraftSideCap = 300;

std::string trim_to(const std::string& s, size_t cap) {
    if (s.size() <= cap) return s;
    return s.substr(0, cap) + "...";
}

} // namespace

SummarizerService::SummarizerService(StorageBackend& backend,
                                     InferenceClient* inference)
    : backend_(backend), inference_(inference) {}

SummarizerService::~SummarizerService() {
    stop();
}

void SummarizerService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;

    enqueue_catch_up();
    worker_thread_      = std::thread([this] { worker_loop(); });
    pause_timer_thread_ = std::thread([this] { pause_timer_loop(); });
}

void SummarizerService::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;

    cv_.notify_all();
    if (worker_thread_.joinable())      worker_thread_.join();
    if (pause_timer_thread_.joinable()) pause_timer_thread_.join();

    Diskerror::logger::info(lang::MSG_SUMMARIZER_STOP);
}

void SummarizerService::enqueue_turn(int turn_id,
                                     const std::string& user_text,
                                     const std::string& assistant_text,
                                     const std::string& model_name,
                                     const std::string& session_guid,
                                     const std::string& source_timestamp) {
    if (!running_.load() || assistant_text.empty()) return;
    Job j{JobKind::L2Turn, turn_id, -1,
          user_text, assistant_text, model_name, session_guid, source_timestamp};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
    }
    cv_.notify_one();
}

void SummarizerService::enqueue_catch_up() {
    auto* backend = &backend_;

    // Re-entrancy guard: if the queue still has work from the previous
    // catch-up, don't rescan — with a large backlog (e.g. a bulk import
    // of thousands of turns) the periodic mop-up tick would otherwise
    // re-enqueue the same unsummarized turns every cycle, exploding the
    // queue with duplicates while the worker is still draining it.
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!queue_.empty()) return;
    }

    // Per-tick batch cap. Bounds memory and keeps each catch-up pass
    // short; the mop-up timer picks up the next slice once the queue
    // drains. Configurable via [housekeeping] catch_up_batch_size in
    // settings.ini (default 10) — small on purpose: the main use case
    // beyond normal live-turn trickle is a deliberate manual resummarize
    // (nulling model_id on a batch of `summaries` rows via direct SQL),
    // which should redo a handful of inference calls per tick, not
    // hundreds at once.
    const int catch_up_cap = std::max(1, config().catch_up_batch_size);

    size_t turn_n = 0, draft_n = 0, sess_n = 0;

    // L2 catch-up: unsummarized turns (newest first), capped per tick.
    // `session_guid` is empty for anonymous turns (turns.session_id NULL);
    // we still summarize them — the L2 row just lands without a session.
    for (const auto& t : backend->unsummarized_turns(catch_up_cap)) {
        if (t.assistant_text.empty()) continue;  // partial turn, skip
        Job j{JobKind::L2Turn, t.turn_id, -1,
              t.user_text, t.assistant_text, t.model_name,
              t.session_guid, t.timestamp};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            queue_.push_back(std::move(j));
        }
        ++turn_n;
    }

    // Draft retry: draft-tagged rows (housekeeping retry), same cap.
    for (const auto& d : backend->draft_summaries(catch_up_cap)) {
        Job j{JobKind::DraftRetry, -1, d.summary_id,
              {}, {}, {}, d.session_guid, d.timestamp};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++draft_n;
    }

    // Phase 2: session/project rollups are driven by episode boundaries
    // (handle_episode_close cascades into a session rollup + project rollup),
    // so there is no separate per-tick "session close" scan. The old
    // sessions_needing_close() path is retired here — it fired on the same
    // idle condition as episodes_needing_close() and, since session rows no
    // longer flip to 'complete', would have re-rolled every idle session on
    // every tick (harmless to the DB thanks to in-place upsert, but wasteful
    // inference). `sess_n` stays 0.
    (void)sess_n;

    // Episode close (EPISODE_PLAN Phase 1): every session whose open episode
    // is idle past episode_idle_minutes. Each close cascades into the session
    // and project rollups (Phase 2).
    size_t ep_n = 0;
    const int idle_min = config().episode_idle_minutes;
    for (const auto& guid : backend->episodes_needing_close(idle_min)) {
        Job j{JobKind::EpisodeClose, -1, -1,
              {}, {}, {}, guid, {}};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++ep_n;
    }

    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZER_START, turn_n, draft_n, sess_n));
    if (ep_n)
        Diskerror::logger::info(std::format(
            "[summarizer] catch-up queued {} episode close(s)", ep_n));
    cv_.notify_all();
}

void SummarizerService::worker_loop() {
    while (running_.load()) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [this] { return !running_.load() || !queue_.empty(); });
            if (!running_.load()) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }

        try {
            switch (job.kind) {
                case JobKind::L2Turn:           handle_l2(job);         break;
                case JobKind::L3UpdateSession:  handle_l3_update(job);  break;
                case JobKind::L3CloseSession:   handle_l3_close(job);   break;
                case JobKind::L4UpdateProject:  handle_l4_update(job);   break;
                case JobKind::EpisodeClose:     handle_episode_close(job); break;
                case JobKind::DraftRetry:       handle_draft(job);      break;
            }
        } catch (const std::exception& e) {
            if (job.kind == JobKind::DraftRetry)
                Diskerror::logger::warn(std::format(
                    lang::WARN_SUMMARIZER_DRAFT, job.summary_id, e.what()));
            else
                Diskerror::logger::warn(std::format(
                    lang::WARN_SUMMARIZER_L2, job.turn_id, e.what()));
        }
    }
}

void SummarizerService::pause_timer_loop() {
    // The pause-check cadence is the smaller of 60s and idle_minutes/2,
    // capped at 10s minimum so we don't spin tightly on misconfiguration.
    // Phase 2.4: timing flows from episode_idle_minutes (summary_pause_minutes
    // is a deprecated alias resolved into it at config load).
    auto poll_interval = [] {
        int pm = config().episode_idle_minutes;
        if (pm <= 0) return std::chrono::seconds(60);
        int secs = std::min(60, std::max(10, (pm * 60) / 2));
        return std::chrono::seconds(secs);
    };

    while (running_.load()) {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait_for(lk, poll_interval(), [this] { return !running_.load(); });
        if (!running_.load()) return;
        lk.unlock();

        try {
            // Mop-up scan: anything the L2 worker hasn't picked up (a turn
            // captured during inference-down should still be queued for
            // retry), plus session closures, plus draft rewrites once
            // inference is back.
            enqueue_catch_up();
        } catch (const std::exception& e) {
            Diskerror::logger::warn(std::format(
                lang::WARN_SUMMARIZER_L3, std::string("(catch-up)"), e.what()));
        }
    }
}

// --------------------------------------------------------------------
// L2: summarize one turn into a row that inherits the turn's timestamp.
// On inference failure, write a heuristic `draft` row instead — the
// chronology is preserved and the next successful pass can overwrite.
// --------------------------------------------------------------------
bool SummarizerService::handle_l2(const Job& j) {
    if (j.assistant_text.empty()) return true;  // partial turn

    // System-injected turns (model-switch notes, compaction handoffs,
    // interruption notices, background-process reports) carry no *user*
    // content — but the assistant side may still be a real, substantive
    // response (e.g. an interrupted turn the agent finished answering).
    // Blank the user side and summarize the assistant alone, rather than
    // skipping the turn: skipping would drop a good answer, and feeding the
    // system note to the summarizer produced degenerate output like
    // "The assistant " / "System". The raw turn row is untouched.
    std::string eff_user = is_system_injected_turn(j.user_text)
                               ? std::string{} : j.user_text;

    // Trivial-turn filter: skip inference (and skip writing any summary row)
    // when the turn is too short to contain real information. "test | test",
    // one-word pings, and similar noise don't belong in the summary store.
    // The raw turn row is still kept for audit purposes; we just don't waste
    // an inference call or pollute the search index with near-empty rows.
    // Threshold: combined user+assistant text under 80 chars, OR assistant
    // text alone under 20 chars. Both are tuned to catch single-word/phrase
    // exchanges while keeping real single-sentence answers. Uses the
    // effective (system-note-stripped) user text so an interrupted turn with
    // a real answer still qualifies.
    {
        const size_t combined = eff_user.size() + j.assistant_text.size();
        if (combined < 80 || j.assistant_text.size() < 20) {
            // Trivial turn: don't spend an inference call. The raw placeholder
            // row (written at capture) IS the summary for a turn this short.
            // Stamp it done with the summarizer model so it leaves the
            // unsummarized queue instead of being re-enqueued every tick.
            const std::string summarizer_model =
                inference_ ? inference_->memory_model : "";
            if (!summarizer_model.empty())
                backend_.mark_turn_summarized(j.session_guid, j.source_timestamp,
                                              summarizer_model);
            Diskerror::logger::info(std::format(
                "[summarizer] skipping trivial turn (session={}, ts={}, "
                "combined_chars={})",
                j.session_guid.empty() ? "-" : j.session_guid,
                j.source_timestamp, combined));
            return true;
        }
    }

    // Idempotency guard. The live enqueue (capture path) and the pause-timer
    // catch-up scan can both queue the same turn. turn_summary_exists() now
    // only counts *summarized* rows (model_id NOT NULL) — the raw placeholder
    // does not block its own summarization. The worker is single-threaded, so
    // by the time a duplicate job runs the first has already promoted the row.
    if (backend_.turn_summary_exists(j.session_guid, j.source_timestamp))
        return true;

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_) {
        summary = summarize_transcript(
            *inference_,
            {{ eff_user, j.assistant_text }});
    }

    const std::string& guid = j.session_guid;
    const std::string& ts   = j.source_timestamp;
    const std::string  display_guid = guid.empty() ? "-" : guid;

    if (summary.empty()) {
        // Inference unreachable. Leave the raw NULL-model placeholder in
        // place — it already makes the turn searchable, and the next catch-up
        // pass retries summarization. No draft row needed.
        Diskerror::logger::info(std::format(
            lang::MSG_SUMMARIZER_DRAFT, display_guid, ts));
        return false;
    }

    // Promote the placeholder in place: real summary text + summarizer model.
    backend_.finalize_turn_summary(guid, ts, summary, summarizer_model);
    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZER_L2, display_guid, ts));

    // Phase 2: no per-turn running-L3 update. The session rollup is rebuilt
    // only on boundaries (episode close / session end), driven from
    // handle_episode_close — this is what kills the running-L3 churn that was
    // snapshotting the same summary dozens of times.
    return true;
}

// --------------------------------------------------------------------
// Session rollup (Phase 2): rebuild the running session summary from the
// session's episodes plus any tail L2 turns not yet folded into an episode.
// INSERT-OR-UPDATE IN PLACE keyed on (session_id, level='session') IGNORING
// status — this is the bloat-bug fix. The old code matched status='current'
// and INSERTed a fresh row whenever it found none (e.g. after a close flipped
// the row to 'complete'), stacking dozens of near-duplicate rows. Now there is
// exactly one 'session' row per session_id, forever.
// --------------------------------------------------------------------
bool SummarizerService::handle_l3_update(const Job& j) {
    if (j.session_guid.empty()) return true;

    // Corpus = episodes (oldest-first) + tail L2 turns past the last episode.
    auto texts = backend_.episode_texts(j.session_guid);
    const std::string since = backend_.last_episode_end(j.session_guid);
    for (const auto& r : backend_.l2_summaries_since(j.session_guid, since))
        if (!r.text.empty()) texts.push_back(r.text);

    // Fallback for sessions with no episodes yet (e.g. very short session
    // ended before an episode closed): roll up straight from all L2 summaries.
    if (texts.empty())
        texts = backend_.l2_summary_texts(j.session_guid);
    if (texts.empty()) return true;

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_)
        summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;  // retry next pass

    // Match ignoring status so we never insert a duplicate.
    auto existing = backend_.session_summary_row(j.session_guid);
    if (existing) {
        backend_.update_summary_text(existing->first, summary, summarizer_model);
        backend_.set_summary_updated_at(existing->first);
    } else {
        backend_.store_summary(summary, "session", "current",
                               summarizer_model, j.session_guid, "", "");
    }
    Diskerror::logger::debug(std::format(
        "[summarizer] session rollup updated for {}", j.session_guid));
    return true;
}

// --------------------------------------------------------------------
// Session close (Phase 2): a session idle past the threshold gets a FINAL
// rollup so any tail turns not yet in an episode are folded in, then the
// project rollup cascades. We no longer flip status current→complete as a
// "close" mechanism — that flip was what made the old find-existing query
// (which matched status='current') miss the row and INSERT a duplicate. The
// session row stays a single running row keyed by (session_id, level).
// --------------------------------------------------------------------
bool SummarizerService::handle_l3_close(const Job& j) {
    if (j.session_guid.empty()) return true;

    // Final rollup (insert-or-update in place, ignoring status).
    if (!handle_l3_update(j)) return false;

    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZED_SESSION, j.session_guid, 0));

    // Refresh the L4 project summary.
    Job l4j{JobKind::L4UpdateProject, -1, -1, {}, {}, {}, {}, {}};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(l4j));
        cv_.notify_one();
    }
    return true;
}

// --------------------------------------------------------------------
// L4 update: rebuild the running project summary. Phase 2: session rows now
// stay running (never flipped to 'complete'), so the corpus is the newest
// 'session' summary per session_id. Upserts a single project row.
// --------------------------------------------------------------------
bool SummarizerService::handle_l4_update(const Job& /*j*/) {
    auto texts = backend_.latest_session_summary_texts();
    if (texts.empty()) return true;

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_)
        summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;

    auto existing = backend_.current_project_summary();
    if (existing) {
        backend_.update_summary_text(existing->first, summary, summarizer_model);
        backend_.set_summary_updated_at(existing->first);
    } else {
        backend_.store_summary(summary, "project", "current",
                               summarizer_model, "", "", "");
    }
    Diskerror::logger::debug("[summarizer] L4 project summary updated");
    return true;
}

// --------------------------------------------------------------------
// Episode close (EPISODE_PLAN Phase 1): summarize the run of L2 turn
// summaries accumulated since this session's previous episode end into a
// single immutable level='episode' row, spanning first→last turn. On
// inference failure, leave the turns unclosed (retry next catch-up) —
// same pattern as L2 draft handling.
// --------------------------------------------------------------------
bool SummarizerService::handle_episode_close(const Job& j) {
    if (j.session_guid.empty()) return true;

    const std::string since = backend_.last_episode_end(j.session_guid);
    auto rows = backend_.l2_summaries_since(j.session_guid, since);
    if (rows.empty()) return true;  // nothing new to close

    std::vector<std::string> texts;
    texts.reserve(rows.size());
    for (const auto& r : rows)
        if (!r.text.empty()) texts.push_back(r.text);
    if (texts.empty()) return true;

    const std::string first_ts = rows.front().timestamp;
    const std::string last_ts  = rows.back().timestamp;

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_)
        summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;  // inference down — retry next pass

    backend_.store_episode(summary, summarizer_model, j.session_guid,
                           first_ts, last_ts);
    Diskerror::logger::info(std::format(
        "[summarizer] episode closed for session {} ({} turns, {} → {})",
        j.session_guid, texts.size(), first_ts, last_ts));

    // Phase 2: an episode boundary is when the session rollup regenerates.
    // Queue a session rollup (which folds this new episode in), then the
    // project rollup cascades off it.
    {
        Job sj{JobKind::L3UpdateSession, -1, -1, {}, {}, {}, j.session_guid, {}};
        Job pj{JobKind::L4UpdateProject, -1, -1, {}, {}, {}, {}, {}};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(sj));
        queue_.push_back(std::move(pj));
        cv_.notify_one();
    }
    return true;
}

// --------------------------------------------------------------------
// Draft retry: re-summarize a draft-tagged L2 row using inference. On
// success, the row's text/embedding/tags are rewritten in place
// (timestamp preserved — the chronology is what makes the link work).
// On failure, the draft row stays; next catch-up will pick it up again.
// --------------------------------------------------------------------
bool SummarizerService::handle_draft(const Job& j) {
    if (j.summary_id <= 0 || !inference_) return false;
    auto* backend = &backend_;
    if (!backend) return false;

    // To re-summarize, we need the source turn. The draft row's
    // (session_id, timestamp) pair points right at it.
    const auto turns = backend->turns_by_session_desc(j.session_guid, /*limit=*/0);
    auto it = std::find_if(turns.begin(), turns.end(),
        [&](const TurnRecord& t) { return t.timestamp == j.source_timestamp; });
    if (it == turns.end()) return false;  // turn vanished — drop the draft

    // Apply the same strip as in handle_l2 to remove system-injected turns
    std::string eff_user = is_system_injected_turn(it->user_text) ? std::string{} : it->user_text;

    auto fresh = summarize_transcript(
        *inference_, {{ eff_user, it->assistant_text }});
    if (fresh.empty()) return false;

    // update_summary_text re-embeds + records the model. We then clear
    // the `draft` tag so this row doesn't get re-enqueued forever.
    if (!backend->update_summary_text(j.summary_id, fresh, it->model_name)) {
        return false;
    }
    backend->set_summary_tags(j.summary_id, "");
    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZER_REDRAFT, j.summary_id));
    return true;
}

std::string SummarizerService::heuristic_draft(const std::string& user_text,
                                               const std::string& assistant_text) {
    // Assistant-only turns (system-injected user side stripped upstream) emit
    // just the assistant half — no empty "User:" prefix.
    if (user_text.empty())
        return "Assistant: " + trim_to(assistant_text, kDraftSideCap);
    return "User: " + trim_to(user_text, kDraftSideCap) +
           " | Assistant: " + trim_to(assistant_text, kDraftSideCap);
}

} // namespace ragger
