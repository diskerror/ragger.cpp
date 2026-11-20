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
#include <map>
#include <set>

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
    Job j{JobKind::TurnSummary, turn_id,
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
    // notify_ALL: two threads wait on cv_ (worker_loop and pause_timer_loop);
    // notify_one can be stolen by the pause timer (predicate false -> back to
    // sleep), leaving the job queued and the worker asleep forever.
    cv_.notify_all();
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

    // Turn-summary catch-up: unsummarized turns (newest first), capped per tick.
    // `session_guid` is empty for anonymous turns (turns.session_id NULL);
    // we still summarize them — the turn-summary row just lands without a session.
    for (const auto& t : backend->unsummarized_turns(catch_up_cap)) {
        if (t.assistant_text.empty()) continue;  // partial turn, skip
        Job j{JobKind::TurnSummary, t.turn_id,
              t.user_text, t.assistant_text, t.model_name,
              t.session_guid, t.timestamp};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            queue_.push_back(std::move(j));
        }
        ++turn_n;
    }

    // X22: stale turn-summary refresh catch-up. A turn merged after its turn summary was
    // summarized carries stale > 0 (the persistent counter store_turn's merge
    // branch bumps), so this pass retries reliably across restarts, MCP
    // writes and a full queue. Same per-tick cap as the turn-summary pass above. Only
    // rows NOT in an episode: refreshing an in-episode turn summary has to rebuild the
    // episode summary with it, which is X23's episode-regeneration job —
    // those candidates are returned by the same query and skipped here (they
    // become trigger (a) of that pass below).
    size_t refresh_n = 0;
    std::set<int> regen_stale_episodes;
    for (const auto& c : backend->stale_turn_summary_candidates(catch_up_cap)) {
        if (c.episode_id != 0) {          // X23 refreshes in-episode rows
            regen_stale_episodes.insert(c.episode_id);
            continue;
        }
        Job j{JobKind::TurnSummaryRefresh, c.turn_id,
              c.user_text, c.assistant_text, c.turn_model_name,
              c.session_guid, {}, c.turn_summary_id, c.stale};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            // Bounded queue, one-time warning, same style as enqueue_turn:
            // with a dead endpoint the stale counters persist in the DB, so
            // dropping the live job loses nothing.
            if (queue_.size() >= 1000) {
                static std::atomic<bool> warned_refresh_full{false};
                if (!warned_refresh_full.exchange(true))
                    Diskerror::Logger::warn(
                        "[summarizer] catch-up queue full (1000) — dropping stale turn-summary "
                        "refresh jobs; the periodic tick will pick them up "
                        "(one-time warning)");
                break;
            }
            queue_.push_back(std::move(j));
        }
        ++refresh_n;
    }

    // X23: episode-regeneration catch-up, after the X22 pass and
    // under the same queue-empty / per-tick-cap / bounded-queue rules. Two
    // triggers: (a) every DISTINCT episode among the stale candidates above
    // that this pass skipped (episode_id != 0) — its member text changed, so
    // the episode has to be rebuilt with it; (b) every stranded candidate
    // (a late turn summary inside an already-closed span of its session) — attach it
    // and fix the episode. Grouping by episode_id IS the pass's dedupe of
    // (episode_id, attach id): a stale member and a late row in the same
    // episode are one job, never two.
    // X31: neither trigger can name a LOCKED episode — both selections
    // (stale_turn_summary_candidates, stranded_turn_summary_candidates)
    // exclude locked episodes, so the scan never queues a regen for one.
    size_t regen_n = 0;
    {
        std::map<int, std::vector<int>> regen_jobs;   // episode_id -> late rows
        for (const int ep : regen_stale_episodes)
            regen_jobs.try_emplace(ep);
        int stranded_ties = 0;   // counted for the CLI sweep, not reported here
        for (const auto& s : backend->stranded_turn_summary_candidates(stranded_ties))
            regen_jobs[s.episode_id].push_back(s.turn_summary_id);

        for (auto& [ep_id, attach] : regen_jobs) {
            if (regen_n >= static_cast<size_t>(catch_up_cap)) break;
            Job j{JobKind::EpisodeRegen, .episode_id = ep_id,
                  .attach_ids = std::move(attach)};
            {
                std::lock_guard<std::mutex> lk(mutex_);
                if (queue_.size() >= 1000) {
                    static std::atomic<bool> warned_regen_full{false};
                    if (!warned_regen_full.exchange(true))
                        Diskerror::Logger::warn(
                            "[summarizer] catch-up queue full (1000) — dropping "
                            "episode-regeneration jobs; the periodic tick will pick "
                            "them up (one-time warning)");
                    break;
                }
                queue_.push_back(std::move(j));
            }
            ++regen_n;
        }
    }

    // Poison-abandon retry: delete "bad"-tagged turn_summaries rows so the
    // underlying turns fall back into unsummarized_turns()'s candidate set.
    // Whatever caused max_turn_failures consecutive failures (inference
    // outage, bad config) may since be fixed; this is the only path back
    // for a turn that was permanently abandoned. Reset on this tick, picked
    // up as ordinary turn-summary catch-up on the NEXT tick (unsummarized_turns() was
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

    // S25 (C m11): report only when the scan actually queued work — this
    // runs every catch-up tick; a 0/0 line every poll is noise. The separate
    // episode-close line is folded into this one message.
    if (turn_n + ep_n > 0)
        Diskerror::Logger::info(std::format(
            lang::MSG_SUMMARIZER_START, turn_n, ep_n));
    if (refresh_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] queued {} stale turn-summary refresh job(s)", refresh_n));
    if (regen_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] queued {} episode-regeneration job(s)", regen_n));
    if (reset_n)
        Diskerror::Logger::info(std::format(
            "[summarizer] housekeeping reset {} poison-abandoned turn_summaries "
            "row(s) for retry", reset_n));
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
                case JobKind::TurnSummary:       handle_turn_summary(job);            break;
                case JobKind::TurnSummaryRefresh:    handle_turn_summary_refresh(job);    break;
                case JobKind::EpisodeClose: handle_episode_close(job); break;
                case JobKind::EpisodeRegen: handle_episode_regen(job); break;
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
            // Mop-up scan: anything the turn-summary worker hasn't picked up (a turn
            // captured during inference-down should still be queued for
            // retry), plus idle episode closes.
            enqueue_catch_up();
        } catch (const std::exception& e) {
            // S25: the "(catch-up)" placeholder argument was dead — the
            // reworded message names the stage itself.
            Diskerror::Logger::warn(std::format(
                lang::WARN_SUMMARIZER_L3, e.what()));
        }
    }
}

// --------------------------------------------------------------------
// Turn summary: summarize one turn into a row that inherits the turn's timestamp.
// On inference failure NO row is written: the turn stays in the
// unsummarized queue and the next catch-up pass retries it. Only the
// poison-abandon path below (after max_turn_failures) writes a row, and
// that one is the '' marker with model 'bad' — the old heuristic `draft`
// row path is gone.
// --------------------------------------------------------------------

// X22: the ONE {user, assistant} -> {summary, model} block, moved
// VERBATIM out of handle_turn_summary (system-injected blanking, the `user | assistant`
// source, the LR S62 floor copy with no model call, summarize_transcript +
// choose_summary). handle_turn_summary's behavior is bit-for-bit unchanged; the extra
// `source`/`floor_copy` fields exist only so handle_turn_summary keeps its exact
// floor-copy log and its poison-abandon accounting around the call.
SummarizerService::TurnSummaryText SummarizerService::make_turn_summary_text(
        InferenceClient* inf, const std::string& user,
        const std::string& assistant, const std::string& turn_model_name) {
    // System-injected turns (model-switch notes, compaction handoffs,
    // interruption notices, background-process reports) carry no *user*
    // content — but the assistant side may still be a real, substantive
    // response (e.g. an interrupted turn the agent finished answering).
    // Blank the user side and summarize the assistant alone, rather than
    // skipping the turn: skipping would drop a good answer, and feeding the
    // system note to the summarizer produced degenerate output like
    // "The assistant " / "System". The raw turn row is untouched.
    const std::string eff_user = is_system_injected_turn(user)
                                   ? std::string{} : user;

    // LR S62: the exact text the summarizer would see for this turn (the
    // system-injected user side already blanked). Doubles as the copy text
    // for the floor and the shorter-text rule below.
    const std::string source = eff_user.empty()
        ? assistant
        : eff_user + " | " + assistant;

    // LR S62 floor: a source at/below summarizer_min_words words is copied
    // verbatim with NO model call — this path needs no endpoint at all, so
    // short turns still leave the unsummarized queue with inference down
    // (the PHASE6 3 invariant, now carried by the floor instead of the
    // trivial-skip marker). Provenance of a copy is the TURN's model
    // (addendum 3; NULL = unknown allowed).
    if (count_words(source) <= static_cast<size_t>(std::max(0, config().summarizer_min_words)))
        return {source, turn_model_name, source, true};

    if (!inf) return {};  // not configured: nothing to summarize against

    const std::string summary = summarize_transcript(
        *inf,
        {{ eff_user, assistant }});
    if (summary.empty()) return {};  // inference unreachable or garbage

    // LR S62 (shorter-text rule): a summary that measures >= its source
    // (ties included) is rejected in favor of a verbatim copy — and a copy
    // records the TURN's model, not the summarizer's (addendum 3).
    const auto choice = choose_summary(source, summary);
    return {choice.text, choice.copied ? turn_model_name : inf->memory_model,
            source, false};
}

void SummarizerService::handle_turn_summary(const Job& j) {
    if (j.assistant_text.empty()) return;  // partial turn

    // Resolve the inference client ONCE for the whole job (C §1.3): a
    // SIGHUP reload mid-job replaces the daemon's shared_ptr but cannot
    // free the object this job holds.
    auto inf = inference_provider_ ? inference_provider_() : nullptr;

    // LR S63: the old "trivial turn" filter (combined < 80 chars or
    // assistant < 20 chars -> mark_turn_summarized with the "(skipped)"
    // pseudo-model) is GONE. Short exchanges now ride the normal path: the
    // S62 floor copies their source into a REAL turn_summaries row (text,
    // embedding, terms, the turn's model) without an inference call. The
    // only remaining ''-marker writer is poison-abandon below (D-5).

    // Idempotency guard. The live enqueue (capture path) and the pause-timer
    // catch-up scan can both queue the same turn. turn_summary_exists() is a
    // plain row-existence check — no placeholder mechanism exists anymore, so
    // any turn_summaries row for this turn_id means it's already done (a
    // real summary/copy, or a poison-abandon marker; legacy DBs may also
    // hold old trivial-skip markers). The
    // worker is single-threaded, so by the time a duplicate job runs, the
    // first has already inserted the row.
    if (backend_.turn_summary_exists(j.turn_id))
        return;

    const std::string& guid = j.session_guid;
    const std::string& ts   = j.source_timestamp;
    const std::string  display_guid = guid.empty() ? "-" : guid;

    const auto made = make_turn_summary_text(inf.get(), j.user_text, j.assistant_text,
                                   j.model_name);

    // LR S62 floor path: verbatim copy, no model call, no endpoint needed.
    if (made.floor_copy) {
        if (!backend_.finalize_turn_summary(j.turn_id, made.text, made.model)) {
            Diskerror::Logger::warn(std::format(
                "[summarizer] floor-copy finalize_turn_summary failed for turn_id={} "
                "(session={}, ts={})", j.turn_id,
                display_guid, ts));
            return;
        }
        Diskerror::Logger::info(std::format(
            "[summarizer] turn-summary floor copy ({} words, no model call) for session {}",
            count_words(made.source), display_guid));
        return;
    }

    if (made.text.empty()) {
        // S25 (C M10): "not configured" is not "unreachable". With no inference
        // endpoint there is nothing to retry against: counting failures here
        // stamped turns 'bad' after max_turn_failures ticks, the back-off job
        // reset them, and the cycle repeated forever on a fresh install. Leave
        // the turn unsummarized (an endpoint appearing later resumes it) and
        // say so once per service lifetime.
        if (!inf) {
            std::call_once(no_inference_warned_, [] {
                Diskerror::Logger::info(lang::MSG_SUMMARIZER_NO_ENDPOINT);
            });
            return;
        }

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
                return;  // consumed — won't retry
            }
        }
        // No turn_summaries row is created on inference failure — the turn
        // stays in the unsummarized queue and the next catch-up pass retries.
        Diskerror::Logger::info(std::format(
            "[summarizer] turn summary deferred (inference unreachable) for session {} at {}",
            display_guid, ts));
        return;
    }

    // Insert the turn_summaries row (text + provenance already chosen by
    // make_turn_summary_text: the summarizer's model, or the turn's for a copy).
    bool ok = backend_.finalize_turn_summary(j.turn_id, made.text, made.model);
    if (!ok) {
        Diskerror::Logger::warn(std::format(
            "[summarizer] finalize_turn_summary failed for turn_id={} "
            "(session={}, ts={})", j.turn_id, display_guid, ts));
        return;
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
        lang::MSG_SUMMARIZER_TURN_SUMMARY, display_guid, ts));

    return;
}

// --------------------------------------------------------------------
// X22: refresh the turn-summary row of a turn that was merged after it was
// summarized (store_turn's merge branch bumped its `stale` counter). The
// row stays stale — retried by the next tick — whenever nothing is written:
// no inference configured, or an empty summary. Nothing here counts toward
// poison-abandon: the row already holds a real summary, and the persistent
// counter is this job's retry mechanism.
// --------------------------------------------------------------------
void SummarizerService::handle_turn_summary_refresh(const Job& j) {
    auto inf = inference_provider_ ? inference_provider_() : nullptr;  // once per job
    if (!inf) {
        std::call_once(no_inference_warned_, [] {
            Diskerror::Logger::info(lang::MSG_SUMMARIZER_NO_ENDPOINT);
        });
        return;  // the row stays stale; the next tick retries
    }

    const auto made = make_turn_summary_text(inf.get(), j.user_text, j.assistant_text,
                                   j.model_name);
    if (made.text.empty()) {
        Diskerror::Logger::info(std::format(
            "[summarizer] turn-summary refresh deferred (inference unreachable) for "
            "turn_summary_id={} (session={})", j.turn_summary_id,
            j.session_guid.empty() ? "-" : j.session_guid));
        return;  // no write; the row stays stale for the next tick
    }

    // clear_stale=true with the observed counter: a merge that landed while
    // this summary was being built keeps the row stale for the next tick.
    if (!backend_.refresh_turn_summary(j.turn_summary_id, made.text, made.model,
                                       j.observed_stale, /*clear_stale=*/true)) {
        Diskerror::Logger::warn(std::format(
            "[summarizer] refresh_turn_summary found no row for turn_summary_id={}",
            j.turn_summary_id));
        return;
    }
    {
        // Inference just worked: reopen the abandoned-reset window (same
        // signal handle_turn_summary gives on a successful summary).
        std::lock_guard<std::mutex> lk(mutex_);
        reset_backoff_.succeeded();
    }
    Diskerror::Logger::info(std::format(
        "[summarizer] turn-summary refreshed for turn_summary_id={} (session={})",
        j.turn_summary_id, j.session_guid.empty() ? "-" : j.session_guid));
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
void SummarizerService::handle_episode_close(const Job& j) {
    if (j.session_guid.empty()) return;

    auto inf = inference_provider_ ? inference_provider_() : nullptr;  // once per job

    const std::string since = backend_.last_episode_end(j.session_guid);
    auto candidates = backend_.episode_candidate_turns(j.session_guid, since);
    if (candidates.empty()) return;  // nothing new to close

    // S25 (C M10): same "not configured" guard as handle_turn_summary — an episode
    // cannot be closed without inference; leave it open for a later pass.
    if (!inf) {
        std::call_once(no_inference_warned_, [] {
            Diskerror::Logger::info(lang::MSG_SUMMARIZER_NO_ENDPOINT);
        });
        return;
    }

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
    struct EpisodeSegment {
        std::string summary;
        std::string model;      // provenance written to episodes.model_id
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
        std::string single_model;
        for (size_t k = start; k < end; ++k)
            if (!candidates[k].summary_text.empty()) {
                texts.push_back(candidates[k].summary_text);
                single_model = candidates[k].summary_model_name;
            }
        if (texts.empty()) continue;

        // LR S62 (shorter-text rule, addenda 2/3), episode side: the ONE-
        // member copy, the joined-text floor and the summarize_texts +
        // choose_summary path. X23 moved that block VERBATIM into
        // summarize_member_texts so the episode-regeneration job and the
        // `backfill stranded` sweep rebuild an episode exactly the way a
        // close writes one; behavior here is bit-for-bit unchanged (an empty
        // summary is the inference-failure signal that aborts the whole
        // close for a retry next pass).
        auto ms = summarize_member_texts(*inf, texts, single_model);
        if (ms.text.empty()) return;  // inference failed — retry ALL next pass

        segments.push_back({std::move(ms.text), std::move(ms.model),
                            start, end, texts.size()});
    }

    if (segments.empty()) return;

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
        ec.model_name   = seg.model;
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

    return;
}

// --------------------------------------------------------------------
// X23: episode regeneration. Two reasons to rebuild an episode:
// one of its members' turn-summary rows went stale (a merge of the source turn —
// X22's counter), and/or a late turn-summary row appeared inside its closed span
// (a stranded row). Reid, 2026-11-14: the episode text is DERIVED, so it is
// replaced in place (no history, no change-of-meaning detection), and a late
// summary is attached to the episode it belongs to.
//
// Order matters: stale members are refreshed FIRST (their new text is what
// the episode summarizes), with clear_stale=false — the counters are cleared
// inside the episode's own transaction, so an episode step that fails leaves
// everything stale and the whole thing is retried by the next tick. Nothing
// here counts toward poison-abandon: the persistent `stale` counter and the
// stranded selection are this job's retry mechanism.
// --------------------------------------------------------------------
void SummarizerService::handle_episode_regen(const Job& j) {
    if (j.episode_id <= 0) return;

    // X31 (defence in depth): a locked episode is never rebuilt. The catch-up
    // selections already hide locked episodes, so reaching here means the
    // episode was locked after this job was queued — drop it without a write;
    // nothing re-queues it (the selections keep hiding it).
    if (backend_.episode_is_locked(j.episode_id)) {
        Diskerror::Logger::debug(std::format(
            "[summarizer] episode regen skipped for locked episode {}", j.episode_id));
        return;
    }

    auto inf = inference_provider_ ? inference_provider_() : nullptr;  // once per job

    // (1) No inference: same once-only notice as the other handlers; the
    // rows stay stale/stranded and the next tick retries.
    if (!inf) {
        std::call_once(no_inference_warned_, [] {
            Diskerror::Logger::info(lang::MSG_SUMMARIZER_NO_ENDPOINT);
        });
        return;
    }

    using Member = StorageBackend::EpisodeMember;
    std::vector<Member> members = backend_.episode_members(j.episode_id);

    // (2) Slot each late row into the member list at its turn_datetime
    // (de-duplicated by turn_summary_id). A row that is no longer
    // unattached/non-empty is dropped — for that row there is nothing to do.
    std::vector<int> attach_ids;
    for (const int ts_id : j.attach_ids) {
        const auto lm = backend_.turn_summary_member_by_id(ts_id);
        if (!lm || lm->episode_id != 0 || lm->text.empty()) continue;
        if (std::any_of(members.begin(), members.end(),
                        [ts_id](const Member& m) { return m.turn_summary_id == ts_id; }))
            continue;
        const auto at = std::lower_bound(
            members.begin(), members.end(), *lm,
            [](const Member& a, const Member& b) {
                return a.turn_datetime < b.turn_datetime ||
                       (a.turn_datetime == b.turn_datetime &&
                        a.turn_summary_id < b.turn_summary_id);
            });
        members.insert(at, *lm);
        attach_ids.push_back(ts_id);
    }
    if (members.empty()) return;   // nothing to build an episode summary from

    // (3) Refresh every stale member BEFORE the episode text is built.
    std::vector<std::pair<int,int>> cleared;   // {turn_summary_id, observed stale}
    for (auto& m : members) {
        if (m.stale <= 0) continue;
        if (m.turn_id == 0) {
            // The source turn is gone (turn_id NULL via ON DELETE SET NULL):
            // there is nothing left to re-summarize. Clear the counter with
            // the episode instead — otherwise this row would re-trigger a
            // regeneration (and a model call) every tick forever. Its last
            // real text stays as the member text.
            Diskerror::Logger::warn(std::format(
                "[summarizer] episode regen: stale turn_summary_id={} has no live "
                "turn; clearing its counter without refreshing (episode {})",
                m.turn_summary_id, j.episode_id));
            cleared.emplace_back(m.turn_summary_id, m.stale);
            continue;
        }
        const auto made = make_turn_summary_text(inf.get(), m.user_text, m.assistant_text,
                                       m.turn_model_name);
        if (made.text.empty()) {
            // Inference unreachable: NO episode write at all — the member
            // keeps its counter and the next tick retries the whole unit.
            Diskerror::Logger::info(std::format(
                "[summarizer] episode regen deferred (inference unreachable) for "
                "turn_summary_id={} (episode {})", m.turn_summary_id, j.episode_id));
            return;
        }
        if (!backend_.refresh_turn_summary(m.turn_summary_id, made.text, made.model,
                                           m.stale, /*clear_stale=*/false)) {
            Diskerror::Logger::warn(std::format(
                "[summarizer] episode regen: refresh_turn_summary found no row for "
                "turn_summary_id={}", m.turn_summary_id));
            return;
        }
        m.text = made.text;   // the episode summarizes the NEW text
        cleared.emplace_back(m.turn_summary_id, m.stale);
    }

    // (4) The episode summary: the SAME rules as a segment of
    // handle_episode_close (shared helper), over the members in
    // turn_datetime order.
    std::vector<std::string> texts;
    texts.reserve(members.size());
    std::string single_model;
    for (const auto& m : members) {
        texts.push_back(m.text);
        single_model = m.model_name;
    }
    const auto ms = summarize_member_texts(*inf, texts, single_model);

    // (5) An empty summary is inference failure: no episode write, retried
    // next tick (the refreshed members keep their counters).
    if (ms.text.empty()) {
        Diskerror::Logger::info(std::format(
            "[summarizer] episode regen deferred (inference unreachable) for "
            "episode {}", j.episode_id));
        return;
    }

    if (!backend_.replace_episode_text(j.episode_id, ms.text, ms.model, cleared,
                                       attach_ids)) {
        // Lost the attach race (another writer attached the late row first)
        // or the episode is gone: replace_episode_text rolled the whole unit
        // back. No retry — the winning writer's state is already correct.
        Diskerror::Logger::info(std::format(
            "[summarizer] episode regen skipped for episode {}: already attached "
            "(or the episode/late row is gone)", j.episode_id));
        return;
    }
    {
        // Inference just worked: reopen the abandoned-reset window (same
        // signal handle_turn_summary_refresh gives on a successful refresh).
        std::lock_guard<std::mutex> lk(mutex_);
        reset_backoff_.succeeded();
    }
    Diskerror::Logger::info(std::format(
        "[summarizer] episode {} regenerated ({} member(s), {} late row(s) attached)",
        j.episode_id, members.size(), attach_ids.size()));
}

} // namespace ragger
