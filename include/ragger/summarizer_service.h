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
 * Fallback: when the inference endpoint is unreachable, the worker writes
 * a `draft`-tagged heuristic summary instead of dropping the turn. A
 * housekeeping pass (`rerun_drafts`) re-summarizes drafts whenever
 * inference is back.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ragger {

class StorageBackend;
class InferenceClient;

class SummarizerService {
public:
    SummarizerService(StorageBackend& backend, InferenceClient* inference);
    ~SummarizerService();

    /// Start worker + pause-timer threads. Enqueues the catch-up scan
    /// (un-summarized turns + draft rewrites + ready-to-close sessions)
    /// before returning. Safe to call once.
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

    /// One-shot scan: enqueue every un-summarized turn (oldest first),
    /// every draft row, and any session ready to close. Called by start(),
    /// by the pause timer, and by the housekeeping pass (SIGUSR1).
    void enqueue_catch_up();

private:
    enum class JobKind { L2Turn, L3CloseSession, DraftRetry, L3UpdateSession, L4UpdateProject };

    struct Job {
        JobKind     kind;
        int         turn_id      = -1;
        int         summary_id   = -1;
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
    bool handle_l3_update(const Job& j);   // upsert running L3 from L2 summaries
    bool handle_l3_close(const Job& j);    // finalize running L3 → complete
    bool handle_l4_update(const Job& j);   // upsert running L4 from complete L3s
    bool handle_draft(const Job& j);

    /// Build a heuristic "draft" summary when inference is unreachable.
    /// Concatenates trimmed user + assistant so the row carries something
    /// retrievable; tagged `draft` for housekeeping to overwrite later.
    static std::string heuristic_draft(const std::string& user_text,
                                       const std::string& assistant_text);

    StorageBackend&  backend_;
    InferenceClient* inference_;  // borrowed; lifetime owned by Server

    std::deque<Job>            queue_;
    std::mutex                 mutex_;
    std::condition_variable    cv_;
    std::atomic<bool>          running_{false};
    std::thread                worker_thread_;
    std::thread                pause_timer_thread_;
};

} // namespace ragger
