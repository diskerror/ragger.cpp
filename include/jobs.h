/**
 * LR S91 (D4, D-6) — single-flight maintenance job runner in the daemon.
 *
 * Jobs are the S90 verbs by name: `reindex`, `housekeeping`, and
 * `backfill:<kind>` (the kinds of S66/S67). There is deliberately NO
 * `re-embed` job: Reid runs re-embed himself (rule 17), and the existing
 * `/embedding/update` flow is untouched. The routes live in server.cpp
 * behind the admin token; these jobs are NEVER added to the MCP tools or
 * the plugin schemas.
 *
 * Single-flight: at most one job runs at a time; a second start is refused
 * (the route maps that to 409). The job body runs on one worker thread and
 * uses the S90 contract underneath: ProgressFn for progress (the library
 * never prints), a CancelFlag checked between batches, cancelled_error to
 * mark a cancel, and committed work staying committed — the DB is always
 * consistent and a rerun finishes the rest.
 *
 * States: queued | running | done | failed | cancelled. `progress` is
 * {done, total} of the CURRENT unit of work (for multi-table jobs, the
 * table being walked); `message` carries the result counts or the failure.
 *
 * Daemon shutdown (jobs.h contract, S91): shutdown() sets the cancel flag
 * and joins the worker, so the process never leaves a half-run detached
 * thread; the DB is left consistent by the same S90 (c) guarantee.
 */
#pragma once

#include "text_index.h"   // ProgressFn, CancelFlag, cancelled_error

#include "nlohmann_json.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ragger {

class RaggerMemory;
class InferenceClient;

class JobRunner {
public:
    /// `inference` snapshots the daemon's current inference client (may be
    /// null: a job that needs it then fails with a clear message; dry-run
    /// passes and no-inference jobs work regardless). `housekeeping_pass`
    /// runs one real housekeeping pass (the daemon's own, so the cache
    /// invalidation and signal semantics match the timer path).
    JobRunner(RaggerMemory& memory,
              std::function<std::shared_ptr<InferenceClient>()> inference,
              std::function<void()> housekeeping_pass);
    ~JobRunner();

    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;

    enum class StartResult { Started, Busy, Unknown };

    /// Start a job. `dry_run` selects only (S90 (a)) — nothing is written.
    StartResult start(const std::string& name, bool dry_run, std::string& error);

    /// Ask the running job to stop at its next batch boundary (no-op if
    /// none). The job ends in state `cancelled`.
    void cancel();

    /// {"current": <job>|null, "last": <job>|null}. `current` is non-null
    /// only while queued/running; `last` keeps the finished job's final
    /// state, progress and message.
    nlohmann::json status_json();

    /// Monotonic counter bumped on every state or progress change. SSE
    /// subscribers keep their own last-seen value (the S17 per-subscriber
    /// offset pattern), so a `job` frame never disturbs log tailing.
    uint64_t event_seq() const { return seq_.load(std::memory_order_relaxed); }

    /// Cancel and join. Called from the daemon's shutdown path.
    void shutdown();

    /// Job names accepted by start(): reindex, housekeeping, backfill:<kind>.
    static bool known_job(const std::string& name);

private:
    struct State {
        std::string name;
        std::string state = "queued";   // queued|running|done|failed|cancelled
        int done  = 0;
        int total = 0;
        std::string message;
    };

    void run(const std::string& name, bool dry_run);

    /// Apply fn to the current state and bump the event sequence (one lock).
    void touch(const std::function<void(State&)>& fn);

    RaggerMemory& memory_;
    std::function<std::shared_ptr<InferenceClient>()> inference_;
    std::function<void()> housekeeping_pass_;

    mutable std::mutex mu_;
    State current_;
    bool    has_current_ = false;
    State   last_;
    bool    has_last_ = false;
    std::atomic<uint64_t> seq_{0};
    std::shared_ptr<CancelFlag> cancel_;   // per-run flag; cancel() sets it
    std::unique_ptr<std::thread> worker_;
};

} // namespace ragger
