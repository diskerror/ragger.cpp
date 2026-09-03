/**
 * SummarizerService implementation. See include/ragger/summarizer_service.h.
 */

#include "summarizer_service.h"

#include "config.h"
#include "inference.h"
#include "lang.h"
#include "storage_backend.h"
#include "summarizer.h"
#include "Logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

/// Cosine similarity between two float vectors of equal length.
/// Returns 0 if either vector has zero magnitude.
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f, ma = 0.0f, mb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        ma  += a[i] * a[i];
        mb  += b[i] * b[i];
    }
    float denom = std::sqrt(ma) * std::sqrt(mb);
    return (denom > 0.0f) ? dot / denom : 0.0f;
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

    Diskerror::Logger::info(lang::MSG_SUMMARIZER_STOP);
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
    // drains. Configurable via the catch_up_batch_size setting
    // (default 10) — small on purpose: the main use case
    // beyond normal live-turn trickle is a deliberate manual resummarize
    // (nulling model_id on a batch of `summaries` rows via direct SQL),
    // which should redo a handful of inference calls per tick, not
    // hundreds at once.
    const int catch_up_cap = std::max(1, config().catch_up_batch_size);

    size_t turn_n = 0, draft_n = 0;

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

    // Poison-abandon retry: delete "bad"-tagged turn_summaries rows so the
    // underlying turns fall back into unsummarized_turns()'s candidate set.
    // Whatever caused max_turn_failures consecutive failures (inference
    // outage, bad config) may since be fixed; this is the only path back
    // for a turn that was permanently abandoned. Reset on this tick, picked
    // up as ordinary L2 catch-up on the NEXT tick (unsummarized_turns() was
    // already queried above this tick, before the reset took effect).
    size_t reset_n = static_cast<size_t>(
        backend->reset_abandoned_turn_summaries(catch_up_cap));

    // Session boundary closes: housekeeping-only trigger (no fast-path from
    // store_turn -- Reid's explicit design: summarization work spawns from
    // turns but housekeeping is what catches/retries it reliably).
    size_t sess_close_n = 0;
    for (const auto& run : backend->sessions_needing_close_boundary()) {
        Job j{JobKind::SessionClose, run.last_turn_id, -1, {}, {}, {}, run.session_guid, {}};
        j.first_ts = run.first_ts;
        j.last_ts  = run.last_ts;
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++sess_close_n;
    }

    // Project boundary closes: same housekeeping-only trigger.
    size_t proj_close_n = 0;
    for (const auto& run : backend->projects_needing_close_boundary(config().project_gap_days)) {
        Job j{JobKind::ProjectClose, run.last_turn_id, -1, {}, {}, {}, {}, {}};
        j.first_ts = run.first_ts;
        j.last_ts  = run.last_ts;
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++proj_close_n;
    }

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

    Diskerror::Logger::info(std::format(
        lang::MSG_SUMMARIZER_START, turn_n, draft_n, sess_close_n));
    if (reset_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] housekeeping reset {} poison-abandoned turn_summaries "
            "row(s) for retry", reset_n));
    if (proj_close_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] catch-up queued {} project close(s)", proj_close_n));
    if (ep_n)
        Diskerror::Logger::info(std::format(
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
                case JobKind::SessionClose:     handle_session_close(job); break;
                case JobKind::ProjectClose:     handle_project_close(job); break;
                case JobKind::EpisodeClose:     handle_episode_close(job); break;
                case JobKind::DraftRetry:       handle_draft(job);      break;
            }
        } catch (const std::exception& e) {
            if (job.kind == JobKind::DraftRetry)
                Diskerror::Logger::warn(std::format(
                    lang::WARN_SUMMARIZER_DRAFT, job.summary_id, e.what()));
            else
                Diskerror::Logger::warn(std::format(
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
            Diskerror::Logger::warn(std::format(
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
            // Trivial turn: don't spend an inference call. Mark it done via
            // mark_turn_summarized so it leaves the unsummarized queue
            const std::string summarizer_model =
                inference_ ? inference_->memory_model : "";
            if (!summarizer_model.empty())
                backend_.mark_turn_summarized(j.turn_id, summarizer_model);
            Diskerror::Logger::info(std::format(
                "[summarizer] skipping trivial turn (turn_id={}, session={}, ts={}, "
                "combined_chars={})",
                j.turn_id, j.session_guid.empty() ? "-" : j.session_guid,
                j.source_timestamp, combined));
            return true;
        }
    }

    // Idempotency guard. The live enqueue (capture path) and the pause-timer
    // catch-up scan can both queue the same turn. turn_summary_exists() is a
    // plain row-existence check — no placeholder mechanism exists anymore, so
    // any turn_summaries row for this turn_id means it's already done (either
    // a real summary, a trivial-skip marker, or a poison-abandon marker). The
    // worker is single-threaded, so by the time a duplicate job runs, the
    // first has already inserted the row.
    if (backend_.turn_summary_exists(j.turn_id))
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
        // Inference unreachable or returned garbage. Track failures so one
        // poison turn can't block the queue forever.
        const int max_fails = config().max_turn_failures;
        if (max_fails > 0) {
            int n;
            {
                std::lock_guard<std::mutex> lk(mutex_);
                n = ++turn_failures_[j.turn_id];
            }
            if (n >= max_fails) {
                // Abandon: stamp with model "bad" (model_id 1) to remove
                // from unsummarized_turns() permanently.
                backend_.mark_turn_summarized(j.turn_id, "bad");
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    turn_failures_.erase(j.turn_id);
                }
                Diskerror::Logger::warn(std::format(
                    "[summarizer] abandoned turn after {} failures "
                    "(turn_id={}, session={}, ts={})", n, j.turn_id, display_guid, ts));
                return true;  // consumed — won't retry
            }
        }
        // No turn_summaries row is created on inference failure — the turn
        // stays in the unsummarized queue and the next catch-up pass retries.
        Diskerror::Logger::info(std::format(
            lang::MSG_SUMMARIZER_DRAFT, display_guid, ts));
        return false;
    }

    // Insert the turn_summaries row: real summary text + summarizer model.
    bool ok = backend_.finalize_turn_summary(j.turn_id, summary, summarizer_model);
    if (!ok) {
        Diskerror::Logger::warn(std::format(
            "[summarizer] finalize_turn_summary failed for turn_id={} "
            "(session={}, ts={})", j.turn_id, display_guid, ts));
        return false;
    }
    {
        // Clear any prior failure count for this turn.
        std::lock_guard<std::mutex> lk(mutex_);
        turn_failures_.erase(j.turn_id);
    }
    Diskerror::Logger::info(std::format(
        lang::MSG_SUMMARIZER_L2, display_guid, ts));

    // Phase 2: no per-turn running-L3 update. The session rollup is rebuilt
    // only on boundaries (episode close / session end), driven from
    // handle_episode_close — this is what kills the running-L3 churn that was
    // snapshotting the same summary dozens of times.
    return true;
}

// --------------------------------------------------------------------
// Session boundary close (housekeeping-only trigger): a maximal run of
// consecutive turns sharing one session_id, edge-detected the instant a
// different session_id's turn arrives after it. Immutable INSERT of one
// level='session' row per closed run -- no running-upsert, no status flip.
// The watermark only advances on success (insert or nothing-to-summarize);
// on failure the run is retried by the next housekeeping tick.
// --------------------------------------------------------------------
bool SummarizerService::handle_session_close(const Job& j) {
    if (j.session_guid.empty()) return true;  // shouldn't happen, defensive

    auto texts = backend_.bounded_session_rollup_texts(j.session_guid, j.first_ts, j.last_ts);
    if (texts.empty()) {
        // Nothing to summarize (e.g. a run with only trivial/skipped turns) --
        // still advance the watermark so this run isn't rescanned forever.
        backend_.advance_session_boundary_watermark(j.turn_id /* = last_turn_id */);
        return true;
    }

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_) summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;  // retry next housekeeping tick -- watermark NOT advanced

    int id = backend_.store_session_summary(summary, summarizer_model, j.session_guid,
                                            j.first_ts, j.last_ts);
    if (id <= 0) {
        Diskerror::Logger::warn(std::format(
            "[summarizer] store_session_summary failed for session {} ({} -> {})",
            j.session_guid, j.first_ts, j.last_ts));
        return false;  // retry -- watermark NOT advanced
    }

    backend_.advance_session_boundary_watermark(j.turn_id /* = last_turn_id */);
    Diskerror::Logger::info(std::format(
        lang::MSG_SUMMARIZER_L3, j.session_guid));
    return true;
}

// --------------------------------------------------------------------
// Project boundary close (housekeeping-only trigger): a maximal run of
// turns (any session) separated from the next by a >= project_gap_days
// gap. Corpus = every level='session' row whose span falls entirely
// within the closed run's [first_ts, last_ts]. Immutable INSERT of one
// level='project' row per closed run, same watermark-on-success discipline
// as handle_session_close.
// --------------------------------------------------------------------
bool SummarizerService::handle_project_close(const Job& j) {
    auto texts = backend_.bounded_project_rollup_texts(j.first_ts, j.last_ts);
    if (texts.empty()) {
        backend_.advance_project_boundary_watermark(j.turn_id /* = last_turn_id */);
        return true;
    }

    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    std::string summary;
    if (inference_) summary = summarize_texts(*inference_, texts);
    if (summary.empty()) return false;  // retry next housekeeping tick -- watermark NOT advanced

    int id = backend_.store_project_summary(summary, summarizer_model, j.first_ts, j.last_ts);
    if (id <= 0) {
        Diskerror::Logger::warn(std::format(
            "[summarizer] store_project_summary failed ({} -> {})",
            j.first_ts, j.last_ts));
        return false;  // retry -- watermark NOT advanced
    }

    backend_.advance_project_boundary_watermark(j.turn_id /* = last_turn_id */);
    Diskerror::Logger::info(std::format(
        "[summarizer] project boundary closed ({} -> {})", j.first_ts, j.last_ts));
    return true;
}

// --------------------------------------------------------------------
// Episode close (EPISODE_PLAN Phase 1): close the open episode for a
// session, splitting it into multiple episodes where the sliding
// average-similarity threshold detects a topic break.
//
// Algorithm: walk the candidate turns in order. For each turn (index > 0)
// compute the average of its turn-embedding cosine similarity and its
// summary-embedding cosine similarity vs the previous turn. If that
// average is <= a time-scaled threshold, a new episode starts here.
//
// Threshold = min(cap, base + step * int(gap_minutes / step_minutes))
// where gap_minutes is the idle time between consecutive turns. Longer
// gaps make the threshold easier to trigger, because a gap is itself
// evidence of a topic break.
//
// Each segment (span of turns between boundaries) gets its own
// level='episode' row in the DB. On inference failure for ANY segment,
// the whole close is aborted (retried next pass) — partial writes within
// one session's close are avoided.
// --------------------------------------------------------------------
bool SummarizerService::handle_episode_close(const Job& j) {
    if (j.session_guid.empty()) return true;

    const std::string since = backend_.last_episode_end(j.session_guid);
    auto candidates = backend_.episode_candidate_turns(j.session_guid, since);
    if (candidates.empty()) return true;  // nothing new to close

    // Config for the sliding threshold.
    const auto& cfg = config();
    const float th_base = cfg.episode_threshold_base;
    const float th_cap  = cfg.episode_threshold_cap;
    const float th_step = cfg.episode_threshold_step;
    const float step_min = cfg.episode_step_minutes;

    // Find episode boundaries via sliding average-similarity threshold.
    // boundary_indices[k] means candidate[k] starts a new episode.
    // Index 0 is always an implicit boundary (start of unclosed turns).
    std::vector<size_t> boundary_indices;
    boundary_indices.push_back(0);

    for (size_t i = 1; i < candidates.size(); ++i) {
        const auto& prev = candidates[i - 1];
        const auto& curr = candidates[i];

        float turn_sim = cosine_similarity(curr.turn_embedding, prev.turn_embedding);
        float summ_sim = cosine_similarity(curr.summary_embedding, prev.summary_embedding);
        float avg_sim  = (turn_sim + summ_sim) / 2.0f;

        // Compute time-scaled threshold.
        float gap_minutes = static_cast<float>(curr.epoch - prev.epoch) / 60.0f;
        float threshold = th_base;
        if (step_min > 0.0f)
            threshold += th_step * static_cast<float>(
                static_cast<int>(gap_minutes / step_min));
        threshold = std::min(threshold, th_cap);

        if (avg_sim <= threshold)
            boundary_indices.push_back(i);
    }

    // Summarize each segment.
    const std::string summarizer_model =
        inference_ ? inference_->memory_model : "";

    struct EpisodeSegment {
        std::string summary;
        std::string first_ts;
        std::string last_ts;
        size_t      turn_count;
    };
    std::vector<EpisodeSegment> segments;
    segments.reserve(boundary_indices.size());

    for (size_t b = 0; b < boundary_indices.size(); ++b) {
        size_t start = boundary_indices[b];
        size_t end   = (b + 1 < boundary_indices.size())
                     ? boundary_indices[b + 1]
                     : candidates.size();

        std::vector<std::string> texts;
        texts.reserve(end - start);
        for (size_t k = start; k < end; ++k)
            if (!candidates[k].summary_text.empty())
                texts.push_back(candidates[k].summary_text);
        if (texts.empty()) continue;

        std::string summary;
        if (inference_)
            summary = summarize_texts(*inference_, texts);
        if (summary.empty()) return false;  // inference down — retry ALL next pass

        segments.push_back({
            std::move(summary),
            candidates[start].timestamp,
            candidates[end - 1].timestamp,
            texts.size()
        });
    }

    if (segments.empty()) return true;

    // Commit all segments — inference succeeded for each.
    for (const auto& seg : segments) {
        backend_.store_episode(seg.summary, summarizer_model, j.session_guid,
                               seg.first_ts, seg.last_ts);
        Diskerror::Logger::info(std::format(
            "[summarizer] episode closed for session {} ({} turns, {} → {})",
            j.session_guid, seg.turn_count, seg.first_ts, seg.last_ts));
    }

    if (segments.size() > 1)
        Diskerror::Logger::info(std::format(
            "[summarizer] session {} split into {} episodes by similarity threshold",
            j.session_guid, segments.size()));

    // Session/project rollups no longer cascade off episode close -- they
    // close on their own boundary logic (session_id edge / time gap),
    // scanned independently by the housekeeping tick (enqueue_catch_up).
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
    Diskerror::Logger::info(std::format(
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
