/**
 * backoff.h — minimal exponential back-off with injected time (PHASE6 4).
 *
 * Purpose: the summarizer's abandoned-turn reset (reset_abandoned_turn_summaries)
 * must not run on every tick while inference is down — each run re-arms the
 * poison turns and guarantees another steady write loop until the outage ends.
 * The service waits `base` after the first reset, doubling up to `max`, and
 * resets to "always due" the moment a summary succeeds (inference is back).
 *
 * Time is injected (steady_clock::time_point parameter) so tests need no
 * sleeping.
 */
#pragma once

#include <chrono>
#include <cstdint>

namespace ragger {

class Backoff {
public:
    Backoff(std::chrono::seconds base, std::chrono::seconds max)
        : base_(base), max_(max) {}

    /// True when `now` has reached the scheduled retry time. Initially due.
    bool due(std::chrono::steady_clock::time_point now) const {
        return now >= next_;
    }

    /// Record a failure at `now`: delay = min(max, delay*2) (first failure:
    /// base), next retry at now + delay.
    void failed(std::chrono::steady_clock::time_point now) {
        delay_ = (delay_.count() == 0) ? base_
                                       : std::min(max_, delay_ * 2);
        next_  = now + delay_;
    }

    /// Record success: forget the schedule — always due again.
    void succeeded() {
        delay_ = std::chrono::seconds(0);
        next_  = std::chrono::steady_clock::time_point{};
    }

private:
    std::chrono::seconds base_;
    std::chrono::seconds max_;
    std::chrono::seconds delay_{0};
    std::chrono::steady_clock::time_point next_{};   // epoch: due from the start
};

} // namespace ragger
