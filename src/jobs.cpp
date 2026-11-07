/**
 * LR S91 — JobRunner implementation. See include/jobs.h for the contract.
 */
#include "jobs.h"

#include "backfill.h"
#include "config.h"
#include "inference.h"
#include "memory.h"
#include "sqlite/backend.h"
#include "storage_backend.h"

#include <logger.h>

#include <algorithm>
#include <format>

namespace ragger {

namespace {
const std::vector<std::string>& backfill_kinds() {
    static const std::vector<std::string> kKinds = {
        "l2-null", "l2-empty", "l2-regenerate",
        "triggers", "stranded", "session-names"};
    return kKinds;
}

std::string join_counts(const std::vector<std::pair<std::string, int>>& v) {
    std::string out;
    for (const auto& [name, n] : v)
        out += (out.empty() ? "" : ", ") + name + "=" + std::to_string(n);
    return out;
}
} // namespace

JobRunner::JobRunner(RaggerMemory& memory,
                     std::function<std::shared_ptr<InferenceClient>()> inference,
                     std::function<void()> housekeeping_pass)
    : memory_(memory),
      inference_(std::move(inference)),
      housekeeping_pass_(std::move(housekeeping_pass)) {}

JobRunner::~JobRunner() {
    shutdown();
}

bool JobRunner::known_job(const std::string& name) {
    if (name == "reindex" || name == "housekeeping") return true;
    const std::string prefix = "backfill:";
    if (name.rfind(prefix, 0) != 0) return false;
    const auto& kinds = backfill_kinds();
    return std::find(kinds.begin(), kinds.end(), name.substr(prefix.size())) !=
           kinds.end();
}

JobRunner::StartResult JobRunner::start(const std::string& name, bool dry_run,
                                        std::string& error) {
    if (!known_job(name)) {
        error = "unknown job '" + name +
                "' (reindex, housekeeping, backfill:<kind>)";
        return StartResult::Unknown;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (has_current_) {   // only set while queued/running
        error = "a job is already running (" + current_.name + ", " +
                current_.state + ")";
        return StartResult::Busy;
    }
    // Reap the previous worker (its state is terminal and already moved to
    // `last_`) before spawning a new one.
    if (worker_ && worker_->joinable()) worker_->join();
    current_ = State{name, "queued", 0, 0, {}};
    has_current_ = true;
    cancel_ = std::make_shared<CancelFlag>(false);
    ++seq_;
    worker_ = std::make_unique<std::thread>([this, name, dry_run] {
        run(name, dry_run);
    });
    return StartResult::Started;
}

void JobRunner::cancel() {
    std::lock_guard<std::mutex> lk(mu_);
    if (cancel_) cancel_->store(true);
}

void JobRunner::shutdown() {
    std::unique_ptr<std::thread> worker;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (cancel_) cancel_->store(true);
        worker = std::move(worker_);
    }
    if (worker && worker->joinable()) worker->join();
}

void JobRunner::touch(const std::function<void(State&)>& fn) {
    std::lock_guard<std::mutex> lk(mu_);
    if (has_current_) fn(current_);
    ++seq_;
}

nlohmann::json JobRunner::status_json() {
    std::lock_guard<std::mutex> lk(mu_);
    auto job_json = [](const State& s) {
        return nlohmann::json{{"name", s.name},
                              {"state", s.state},
                              {"done", s.done},
                              {"total", s.total},
                              {"message", s.message}};
    };
    return nlohmann::json{
        {"current", has_current_ ? job_json(current_) : nlohmann::json(nullptr)},
        {"last", has_last_ ? job_json(last_) : nlohmann::json(nullptr)}};
}

void JobRunner::run(const std::string& name, bool dry_run) {
    // The worker owns a shared copy so cancel() can reach it even if the
    // runner is destroyed while the job is in flight.
    std::shared_ptr<CancelFlag> cancel;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancel = cancel_;
    }
    touch([](State& s) { s.state = "running"; });

    // Progress: map the S90 ProgressFn onto {done, total} of the current
    // unit (per table for reindex; per row for backfill). done < 0 closes a
    // table's line in the CLI printer; the job keeps the last counts.
    ProgressFn progress = [this](const char* /*table*/, int done, int total) {
        if (done < 0) return;
        touch([done, total](State& s) { s.done = done, s.total = total; });
    };

    auto finish = [this](const std::string& state, std::string message) {
        std::lock_guard<std::mutex> lk(mu_);
        if (has_current_) {
            current_.state = state;
            current_.message = std::move(message);
            last_ = current_;
            has_last_ = true;
            current_ = {};
            has_current_ = false;
        }
        ++seq_;
    };

    try {
        auto* backend = memory_.backend();
        if (name == "reindex") {
            if (dry_run) {
                const auto plan = backend->reindex_plan("all");
                int total = 0;
                for (const auto& [t, n] : plan) total += n;
                finish("done", "would reindex " + std::to_string(total) +
                                   " row(s): " + join_counts(plan));
                return;
            }
            const auto counts = backend->reindex_all(progress, cancel.get());
            int total = 0;
            for (const auto& [t, n] : counts) total += n;
            finish("done", "reindexed " + std::to_string(total) +
                               " row(s): " + join_counts(counts));
            return;
        }
        if (name == "housekeeping") {
            if (dry_run) {
                const auto plan =
                    backend->housekeeping_plan(config().cleanup_max_age_hours);
                finish("done",
                       std::format("would purge {} turn(s), {} unsummarized, "
                                   "{} embed missing, {} embed stale",
                                   plan.purge_turns, plan.unsummarized,
                                   plan.embed_missing, plan.embed_stale));
                return;
            }
            housekeeping_pass_();
            finish("done", "housekeeping pass complete");
            return;
        }

        // backfill:<kind>
        const std::string kind = name.substr(std::string("backfill:").size());
        // The daemon may have no inference configured; the S66 functions
        // refuse with "inference not configured" when one is needed (dry
        // runs never need it). An empty client stands in for "none".
        auto shared_inf = inference_ ? inference_() : nullptr;
        InferenceClient empty_client{std::vector<Endpoint>{}};
        InferenceClient& inf = shared_inf ? *shared_inf : empty_client;

        BackfillResult res;
        if (kind == "l2-null" || kind == "l2-empty" || kind == "l2-regenerate")
            res = backfill_l2(*backend, inf, kind.substr(3), dry_run, {}, {},
                              progress, cancel.get());
        else if (kind == "triggers")
            res = backfill_triggers(*backend, inf, dry_run, progress, cancel.get());
        else if (kind == "stranded")
            res = backfill_stranded(*backend, dry_run, progress, cancel.get());
        else
            res = backfill_session_names(*backend, inf, dry_run, progress,
                                         cancel.get());

        const std::string counts =
            std::format("{} candidate(s), {} applied, {} skipped, {} failed",
                        res.candidates, res.applied, res.skipped, res.failed);
        finish(res.failed > 0 ? "failed" : "done", counts);
    } catch (const cancelled_error& e) {
        finish("cancelled", e.what());
    } catch (const std::exception& e) {
        Diskerror::Logger::warn(std::string("job failed: ") + e.what());
        finish("failed", e.what());
    }
}

} // namespace ragger
