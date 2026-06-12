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

    size_t turn_n = 0, draft_n = 0, sess_n = 0;

    // L2 catch-up: every unsummarized turn (oldest first).
    // `session_guid` is empty for anonymous turns (turns.session_id NULL);
    // we still summarize them — the L2 row just lands without a session.
    for (const auto& t : backend->unsummarized_turns()) {
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

    // Draft retry: every draft-tagged row (housekeeping retry).
    for (const auto& d : backend->draft_summaries()) {
        Job j{JobKind::DraftRetry, -1, d.summary_id,
              {}, {}, {}, d.session_guid, d.timestamp};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++draft_n;
    }

    // L3 close: every session ready for finalization.
    const int pause_min = config().summary_pause_minutes;
    for (const auto& guid : backend->sessions_needing_close(pause_min)) {
        Job j{JobKind::L3CloseSession, -1, -1,
              {}, {}, {}, guid, {}};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++sess_n;
    }

    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZER_START, turn_n, draft_n, sess_n));
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
                case JobKind::L4UpdateProject:  handle_l4_update(job);  break;
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
    // The pause-check cadence is the smaller of 60s and pause_minutes/2,
    // capped at 10s minimum so we don't spin tightly on misconfiguration.
    auto poll_interval = [] {
        int pm = config().summary_pause_minutes;
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
            Diskerror::logger::info(std::format(
                "[summarizer] skipping trivial turn (session={}, ts={}, "
                "combined_chars={})",
                j.session_guid.empty() ? "-" : j.session_guid,
                j.source_timestamp, combined));
            return true;
        }
    }

    // Idempotency guard. The live enqueue (capture path) and the pause-timer
    // catch-up scan can both queue the same turn: the scan reads
    // unsummarized_turns() before the worker has written this turn's L2 row,
    // so it re-queues it. Without this check that produced two summaries per
    // turn (two inference calls, two rows at the same timestamp). The worker
    // is single-threaded, so by the time the duplicate job runs the first has
    // already committed — skip it. A pre-existing 'draft' row also counts:
    // the DraftRetry path upgrades it in place rather than adding a fresh L2.
    if (backend_.turn_summary_exists(j.session_guid, j.source_timestamp))
        return true;

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
        // Fallback: heuristic draft, tagged for later rewrite.
        const auto draft = heuristic_draft(eff_user, j.assistant_text);
        backend_.store_summary(draft, "turn", "complete",
                              j.model_name, guid, ts, "draft");
        Diskerror::logger::info(std::format(
            lang::MSG_SUMMARIZER_DRAFT, display_guid, ts));
        return false;
    }

    backend_.store_summary(summary, "turn", "complete",
                          j.model_name, guid, ts, "");
    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZER_L2, display_guid, ts));

    // After each successful L2, kick an L3 update for this session so
    // the running session summary always reflects the latest turn.
    if (!guid.empty()) {
        Job l3j{JobKind::L3UpdateSession, -1, -1,
                {}, {}, {}, guid, {}};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(l3j));
        cv_.notify_one();
    }
    return true;
}

// --------------------------------------------------------------------
// L3 update: rebuild the running session summary from all L2 summaries.
// Called after every successful L2 write. Upserts a single status='current'
// row (create on first call, update in-place on subsequent ones).
// --------------------------------------------------------------------
bool SummarizerService::handle_l3_update(const Job& j) {
    if (j.session_guid.empty()) return true;

    auto texts = backend_.l2_summary_texts(j.session_guid);
    if (texts.empty()) return true;

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_)
        summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;  // retry next pass

    auto existing = backend_.current_session_summary(j.session_guid);
    if (existing) {
        backend_.update_summary_text(existing->first, summary, summarizer_model);
    } else {
        backend_.store_summary(summary, "session", "current",
                               summarizer_model, j.session_guid, "", "");
    }
    Diskerror::logger::debug(std::format(
        "[summarizer] L3 updated for session {}", j.session_guid));
    return true;
}

// --------------------------------------------------------------------
// L3 close: finalize the session — mark its running 'current' L3
// summary as 'complete', then trigger an L4 update.
// Called by the pause timer when a session has been idle long enough.
// --------------------------------------------------------------------
bool SummarizerService::handle_l3_close(const Job& j) {
    if (j.session_guid.empty()) return true;

    auto existing = backend_.current_session_summary(j.session_guid);
    if (!existing) {
        // No current L3 yet — try an update first then close.
        if (!handle_l3_update(j)) return false;
        existing = backend_.current_session_summary(j.session_guid);
        if (!existing) return false;
    }

    backend_.set_summary_status(existing->first, "complete");
    Diskerror::logger::info(std::format(
        lang::MSG_SUMMARIZED_SESSION, j.session_guid, 0));

    // Immediately update the L4 project summary.
    Job l4j{JobKind::L4UpdateProject, -1, -1, {}, {}, {}, {}, {}};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(l4j));
        cv_.notify_one();
    }
    return true;
}

// --------------------------------------------------------------------
// L4 update: rebuild the running project summary from all complete L3s.
// Upserts a single status='current' project row (session-unscoped).
// --------------------------------------------------------------------
bool SummarizerService::handle_l4_update(const Job& /*j*/) {
    auto texts = backend_.complete_l3_summary_texts();
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
    } else {
        backend_.store_summary(summary, "project", "current",
                               summarizer_model, "", "", "");
    }
    Diskerror::logger::debug("[summarizer] L4 project summary updated");
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

    auto fresh = summarize_transcript(
        *inference_, {{ it->user_text, it->assistant_text }});
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
