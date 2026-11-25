/**
 * episode_boundaries.h — pure episode-cut decision (PHASE6 7).
 *
 * Extracted VERBATIM from SummarizerService::handle_episode_close so the
 * algorithm can be unit-tested without a DB or an inference server. The
 * rules are unchanged (TODO Q6 default); Phase 6 deliberately does NOT
 * alter the similarity-vs-gap cut rule.
 */
#pragma once

#include "storage_types.h"

#include <cstddef>
#include <vector>

namespace ragger {

/// Sliding-threshold parameters (from config: episode_threshold_base/cap/
/// step, episode_step_minutes).
struct EpisodeBoundaryParams {
    float base;
    float cap;
    float step;
    float step_minutes;
};

/// Indices at which a new episode starts. Empty input -> empty. Otherwise
/// index 0 is always present. For i >= 1: the average of the turn-embedding
/// and summary-embedding cosines vs candidate i-1 is compared against
/// threshold = min(cap, base + step * int(gap_minutes / step_minutes));
/// a boundary is recorded when avg <= threshold (note `<=`).
/// The gap (X32) is begin(curr) - prev.epoch, where begin(curr) is
/// curr.started_epoch when known (> 0) and curr.epoch otherwise; the result
/// is clamped at 0 (a started_on before the previous turn's end adds no idle
/// time). step_minutes <= 0 disables the time scaling.
std::vector<size_t> find_episode_boundaries(const std::vector<EpisodeCandidateTurn>& c,
                                            const EpisodeBoundaryParams& p);

} // namespace ragger
