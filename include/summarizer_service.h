/**
 * SummarizerService — daemon-resident L2 turn-summary + episode-close worker.
 *
 * Lifecycle is owned by the HTTP server. When the daemon is up, this
 * service keeps the inference + embedder warm; turns captured over HTTP or
 * MCP that lack an L2 summary get one shortly after capture, and sessions
 * whose latest turn is older than `episode_idle_minutes` get their open
 * episode closed. When the daemon is *not* running, MCP-mode capture still
 * writes the raw turn (see `ragger::capture_turn`) and this service catches
 * up the moment the daemon next starts.
 *
 * Linkage rule: an L2 row inherits the source turn's timestamp, so the
 * (session_id, timestamp) pair is the (turn → summary) join — no FK column
 * is needed, and rebuilding embeddings under a new model preserves the
 * link. See `StorageBackend::unsummarized_turns()`.
 *
 * Fallback: when the inference endpoint is unreachable, the turn simply
 * stays unsummarized and the next catch-up pass retries it (db 0.17: the
 * old `draft`-tagged heuristic path is gone — nothing ever wrote drafts
 * anymore, and the rollup layers they fed were removed).
 */
#pragma once

#include "backoff.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ragger {

class StorageBackend;
class InferenceClient;

class SummarizerService {
public:
    /// inference_provider is called ONCE per job (never cached across
    /// jobs): the daemon may replace the inference client on SIGHUP, and
    /// a job must hold the client it started with (shared_ptr) for its
    /// whole duration so a mid-job reload cannot free it. A null return
    /// means "no inference", same semantics as the old raw pointer.
    SummarizerService(StorageBackend& backend,
                      std::function<std::shared_ptr<InferenceClient>()> inference_provider);
    ~SummarizerService();

    /// Start worker + pause-timer threads. Enqueues the catch-up scan
    /// (un-summarized turns + ready-to-close episodes) before returning.
    /// Safe to call once.
    void start();

    /// Signal shutdown, wake worker, join threads. Safe to call multiple
    /// times; idempotent.
    void stop();

    /// Enqueue an L2 summarization for a turn that just landed. Called
    /// from the capture path after the row commits. Non-blocking; safe to
    /// call from request handlers.
    void enqueue_turn(int turn_id,
                      const std::string& user_text,
                      const std::string& assistant_text,
                      const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& source_timestamp);

    /// One-shot scan: enqueue every un-summarized turn (newest first) and
    /// any session with an idle open episode. Called by start(), by the
    /// pause timer, and by the housekeeping pass (SIGUSR1).
    void enqueue_catch_up();

    /// Current queue depth (for tests).
    size_t queue_size() const;

private:
    // db 0.17: the session/project rollup and draft job kinds are gone with
    // session/project rollup layers (PHASE5 D.1). X22 (Q2/L2): L2Refresh
    // re-summarizes a turn whose L2 row was flagged stale by a merge. X23
    // (Q5/L4): EpisodeRegen rebuilds an episode — for its stale members, for
    // late L2 rows that must be attached to it, or both.
    enum class JobKind { L2Turn, L2Refresh, EpisodeClose, EpisodeRegen };

    struct Job {
        JobKind     kind;
        int         turn_id      = -1;
        std::string user_text;
        std::string assistant_text;
        std::string model_name;
        std::string session_guid;
        std::string source_timestamp;
        // X22: L2Refresh carries one stale candidate (the row to rewrite
        // and the counter observed at selection time).
        int         turn_summary_id = -1;
        int         observed_stale  = 0;
        // X23 (Q5/L4): EpisodeRegen carries the episode to rebuild plus the
        // late (stranded) L2 rows to attach to it. One job per episode per
        // pass: the attach ids are grouped, which is also the pass's dedupe
        // of (episode_id, attach id) — a stale member and a late row in the
        // same episode are ONE job, not two.
        int                    episode_id = -1;
        std::vector<int>       attach_ids;
    };

    void worker_loop();
    void pause_timer_loop();

    /// Handlers — report through logs/DB state only (S25: void).
    void handle_l2(const Job& j);
    void handle_l2_refresh(const Job& j); // X22: rewrite a merged turn's L2 in place
    void handle_episode_close(const Job& j); // close open episode(s) -> immutable episodes rows
    /// X23 (Q5/L4): rebuild one episode — refresh its stale members first,
    /// attach its late L2 rows, then replace its DERIVED text with the same
    /// rules a segment close uses. Nothing is written when inference fails.
    void handle_episode_regen(const Job& j);

    /// X22 (Q2/L2): the ONE block that turns {user, assistant} into an L2
    /// summary — system-injected blanking, the `user | assistant` source, the
    /// summarizer_min_words floor copy (no model call), summarize_transcript +
    /// choose_summary. Shared by handle_l2 and handle_l2_refresh so both
    /// produce byte-identical text and provenance. Empty text = inference
    /// failed or not configured (the caller decides what that means; poison
    /// accounting stays in handle_l2). `floor_copy` marks the verbatim path
    /// so handle_l2 keeps its exact pre-X22 logging.
    struct L2Text {
        std::string text;
        std::string model;
        std::string source;     // the joined source text (for the floor log)
        bool        floor_copy = false;
    };
    L2Text make_l2_text(InferenceClient* inf,
                        const std::string& user,
                        const std::string& assistant,
                        const std::string& turn_model_name);

    StorageBackend&  backend_;
    std::function<std::shared_ptr<InferenceClient>()> inference_provider_;

    std::deque<Job>            queue_;
    mutable std::mutex         mutex_;
    // PHASE6 5: only one catch-up scan at a time. enqueue_catch_up is called
    // from the pause timer AND the server housekeeping thread; the queue_
    // guard alone releases the lock before the scans, so both could scan and
    // double-queue. A try_lock miss means the other caller is already doing
    // the work — returning is correct.
    std::mutex                 scan_mu_;
    std::condition_variable    cv_;
    std::atomic<bool>          running_{false};
    std::thread                worker_thread_;
    std::thread                pause_timer_thread_;

    // PHASE6 4: exponential back-off for the poison-abandon reset inside
    // enqueue_catch_up (5 min base, 6 h cap). Guarded by mutex_.
    Backoff reset_backoff_{std::chrono::seconds(300), std::chrono::seconds(21600)};

    // Per-turn failure counter keyed by turn_id (real FK, replaces the old
    // "session_guid\x1F timestamp" compound-string key). Guarded by mutex_.
    // Entries are removed on success or abandonment. NOT persisted (PHASE6 8):
    // a daemon restart gives every failing turn a fresh set of attempts —
    // bounded by reset_backoff_'s wait, not by max_turn_failures alone.
    std::unordered_map<int, int> turn_failures_;

    // S25 (C M10): "no inference endpoint configured" is logged once per
    // service lifetime, not once per skipped job.
    std::once_flag no_inference_warned_;
};

} // namespace ragger
