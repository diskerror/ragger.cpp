/**
 * SummarizerService — daemon-resident L2/L3 summarization worker.
 *
 * Lifecycle is owned by the HTTP server. When the daemon is up, this
 * service keeps the inference + embedder warm; turns captured over HTTP or
 * MCP that lack an L2 summary get one shortly after capture, and sessions
 * whose latest turn is older than `summary_pause_minutes` get a finalized
 * L3 summary. When the daemon is *not* running, MCP-mode capture still
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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

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

private:
    // db 0.17: the session/project rollup and draft job kinds are gone with
    // session/project rollup layers (PHASE5 D.1).
    enum class JobKind { L2Turn, EpisodeClose };

    struct Job {
        JobKind     kind;
        int         turn_id      = -1;
        std::string user_text;
        std::string assistant_text;
        std::string model_name;
        std::string session_guid;
        std::string source_timestamp;
    };

    void worker_loop();
    void pause_timer_loop();

    /// Handlers — each returns true on success, false on retryable error.
    bool handle_l2(const Job& j);
    bool handle_episode_close(const Job& j); // close open episode(s) -> immutable episodes rows

    StorageBackend&  backend_;
    std::function<std::shared_ptr<InferenceClient>()> inference_provider_;

    std::deque<Job>            queue_;
    std::mutex                 mutex_;
    std::condition_variable    cv_;
    std::atomic<bool>          running_{false};
    std::thread                worker_thread_;
    std::thread                pause_timer_thread_;

    // Per-turn failure counter keyed by turn_id (real FK, replaces the old
    // "session_guid\x1F timestamp" compound-string key). Guarded by mutex_.
    // Entries are removed on success or abandonment.
    std::unordered_map<int, int> turn_failures_;
};

} // namespace ragger
