/**
 * SummarizerService implementation. See include/ragger/summarizer_service.h.
 */

#include "summarizer_service.h"

#include "config.h"
#include "episode_boundaries.h"
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

SummarizerService::SummarizerService(
    StorageBackend& backend,
    std::function<std::shared_ptr<InferenceClient>()> inference_provider)
    : backend_(backend), inference_provider_(std::move(inference_provider)) {}

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
    Job j{JobKind::L2Turn, turn_id,
          user_text, assistant_text, model_name, session_guid, source_timestamp};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        // PHASE6 2.3: bounded queue. An unbounded queue behind a dead
        // inference server is a memory leak; dropping the live job is safe
        // because the periodic tick re-finds the turn via unsummarized_turns.
        if (queue_.size() >= 1000) {
            static std::atomic<bool> warned_full{false};
            if (!warned_full.exchange(true))
                Diskerror::Logger::warn(
                    "[summarizer] live queue full (1000) — dropping live jobs; "
                    "the periodic tick will pick turns up (one-time warning)");
            return;
        }
        queue_.push_back(std::move(j));
    }
    cv_.notify_one();
}

size_t SummarizerService::queue_size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return queue_.size();
}

void SummarizerService::enqueue_catch_up() {
    auto* backend = &backend_;

    // PHASE6 5: one scan at a time (pause timer + housekeeping can race).
    std::unique_lock scan_lk(scan_mu_, std::try_to_lock);
    if (!scan_lk.owns_lock()) return;

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

    size_t turn_n = 0;

    // L2 catch-up: unsummarized turns (newest first), capped per tick.
    // `session_guid` is empty for anonymous turns (turns.session_id NULL);
    // we still summarize them — the L2 row just lands without a session.
    for (const auto& t : backend->unsummarized_turns(catch_up_cap)) {
        if (t.assistant_text.empty()) continue;  // partial turn, skip
        Job j{JobKind::L2Turn, t.turn_id,
              t.user_text, t.assistant_text, t.model_name,
              t.session_guid, t.timestamp};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            queue_.push_back(std::move(j));
        }
        ++turn_n;
    }

    // Poison-abandon retry: delete "bad"-tagged turn_summaries rows so the
    // underlying turns fall back into unsummarized_turns()'s candidate set.
    // Whatever caused max_turn_failures consecutive failures (inference
    // outage, bad config) may since be fixed; this is the only path back
    // for a turn that was permanently abandoned. Reset on this tick, picked
    // up as ordinary L2 catch-up on the NEXT tick (unsummarized_turns() was
    // already queried above this tick, before the reset took effect).
    // PHASE6 4: gated by exponential back-off. While inference is down,
    // resetting abandoned turns every tick just re-arms the poison loop;
    // wait 5 min after a reset that actually reset rows, doubling to 6 h.
    // A successful summary anywhere calls reset_backoff_.succeeded() so the
    // reset runs again on the next tick once inference is back.
    size_t reset_n = 0;
    {
        const auto now = std::chrono::steady_clock::now();
        bool do_reset;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            do_reset = reset_backoff_.due(now);
        }
        if (do_reset) {
            reset_n = static_cast<size_t>(
                backend->reset_abandoned_turn_summaries(catch_up_cap));
            if (reset_n > 0) {
                std::lock_guard<std::mutex> lk(mutex_);
                reset_backoff_.failed(now);
            }
        }
    }

    // Episode closes: every session whose open episode is idle past
    // episode_idle_minutes (the only rollup layer left at 0.17).
    size_t ep_n = 0;
    const int idle_min = config().episode_idle_minutes;
    for (const auto& guid : backend->episodes_needing_close(idle_min)) {
        Job j{JobKind::EpisodeClose, -1,
              {}, {}, {}, guid, {}};
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(j));
        ++ep_n;
    }

    Diskerror::Logger::info(std::format(
        lang::MSG_SUMMARIZER_START, turn_n, ep_n));
    if (reset_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] housekeeping reset {} poison-abandoned turn_summaries "
            "row(s) for retry", reset_n));
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
                case JobKind::L2Turn:       handle_l2(job);            break;
                case JobKind::EpisodeClose: handle_episode_close(job); break;
            }
        } catch (const std::exception& e) {
            Diskerror::Logger::warn(std::format(
                "[summarizer] job failed (kind={}, turn_id={}, session={}): {}",
                static_cast<int>(job.kind), job.turn_id,
                job.session_guid.empty() ? "-" : job.session_guid, e.what()));
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
            // retry), plus idle episode closes.
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

    // Resolve the inference client ONCE for the whole job (C §1.3): a
    // SIGHUP reload mid-job replaces the daemon's shared_ptr but cannot
    // free the object this job holds.
    auto inf = inference_provider_ ? inference_provider_() : nullptr;

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
            // mark_turn_summarized so it leaves the unsummarized queue —
            // PHASE6 3: unconditionally, even with inference down. The old
            // gate on a non-empty model name meant trivial turns were never
            // marked while inference was down and got re-queued every tick
            // forever. Model name: the real summarizer model when known,
            // else the pseudo-model "(skipped)" — never "bad", which
            // reset_abandoned_turn_summaries() deletes.
            const std::string summarizer_model =
                inf && !inf->memory_model.empty() ? inf->memory_model
                                                  : std::string("(skipped)");
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
        inf ? inf->memory_model : "";

    std::string summary;
    if (inf) {
        summary = summarize_transcript(
            *inf,
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
            "[summarizer] L2 deferred (inference unreachable) for session {} at {}",
            display_guid, ts));
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
        // Clear any prior failure count for this turn, and open the
        // abandoned-turn reset window again: inference just worked, so
        // poison turns abandoned during the outage should retry now.
        std::lock_guard<std::mutex> lk(mutex_);
        turn_failures_.erase(j.turn_id);
        reset_backoff_.succeeded();
    }
    Diskerror::Logger::info(std::format(
        lang::MSG_SUMMARIZER_L2, display_guid, ts));

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

    auto inf = inference_provider_ ? inference_provider_() : nullptr;  // once per job

    const std::string since = backend_.last_episode_end(j.session_guid);
    auto candidates = backend_.episode_candidate_turns(j.session_guid, since);
    if (candidates.empty()) return true;  // nothing new to close

    // Config for the sliding threshold.
    const auto& cfg = config();
    const float th_base = cfg.episode_threshold_base;
    const float th_cap  = cfg.episode_threshold_cap;
    const float th_step = cfg.episode_threshold_step;
    const float step_min = cfg.episode_step_minutes;

    // Find episode boundaries via the sliding average-similarity threshold.
    // PHASE6 7: the algorithm lives in episode_boundaries.cpp as a pure
    // function (moved VERBATIM — behavior is bit-for-bit what Phase 5 ran);
    // boundary_indices[k] means candidate[k] starts a new episode, and
    // index 0 is always an implicit boundary (start of unclosed turns).
    std::vector<size_t> boundary_indices = find_episode_boundaries(
        candidates, EpisodeBoundaryParams{th_base, th_cap, th_step, step_min});

    // Summarize each segment.
    const std::string summarizer_model =
        inf ? inf->memory_model : "";

    struct EpisodeSegment {
        std::string summary;
        size_t      start, end;   // candidate index range [start, end)
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
        if (inf)
            summary = summarize_texts(*inf, texts);
        if (summary.empty()) return false;  // inference down — retry ALL next pass

        segments.push_back({std::move(summary), start, end, texts.size()});
    }

    if (segments.empty()) return true;

    // Commit all segments in order (PHASE5 D.2): each close records the
    // episode AND its membership atomically (store_episode_close). A 0
    // return means the span was already closed (unique key) — log and move
    // on, do not retry. An exception aborts the remaining segments for this
    // session; the next tick re-derives candidates from the new
    // last_episode_end and retries only what is left. Retry cost after a
    // mid-loop failure is one more model call per uncommitted segment —
    // accepted: inference already succeeded once, and each segment commits
    // on its own so partial progress is durable.
    for (const auto& seg : segments) {
        EpisodeClose ec;
        ec.text         = seg.summary;
        ec.model_name   = summarizer_model;
        ec.session_guid = j.session_guid;
        ec.span_start   = candidates[seg.start].epoch;
        ec.span_end     = candidates[seg.end - 1].epoch;
        for (size_t k = seg.start; k < seg.end; ++k)
            ec.member_turn_summary_ids.push_back(candidates[k].turn_summary_id);

        const int id = backend_.store_episode_close(ec);
        if (id == 0) {
            Diskerror::Logger::info(std::format(
                "[summarizer] episode span already closed for session {} "
                "({} -> {}) — skipped", j.session_guid,
                ec.span_start, ec.span_end));
            continue;
        }
        Diskerror::Logger::info(std::format(
            "[summarizer] episode closed for session {} ({} turns, {} -> {})",
            j.session_guid, seg.turn_count, ec.span_start, ec.span_end));
    }

    if (segments.size() > 1)
        Diskerror::Logger::info(std::format(
            "[summarizer] session {} split into {} episodes by similarity threshold",
            j.session_guid, segments.size()));

    return true;
}

} // namespace ragger
