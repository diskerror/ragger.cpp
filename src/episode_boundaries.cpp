/**
 * episode_boundaries.cpp — see episode_boundaries.h. The loop body is the
 * one SummarizerService used inline through Phase 5, moved unchanged.
 */
#include "episode_boundaries.h"

#include <algorithm>
#include <cmath>

namespace ragger {

namespace {

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

std::vector<size_t> find_episode_boundaries(const std::vector<EpisodeCandidateTurn>& candidates,
                                            const EpisodeBoundaryParams& p) {
    std::vector<size_t> boundary_indices;
    if (candidates.empty()) return boundary_indices;
    boundary_indices.push_back(0);

    for (size_t i = 1; i < candidates.size(); ++i) {
        const auto& prev = candidates[i - 1];
        const auto& curr = candidates[i];

        float turn_sim = cosine_similarity(curr.turn_embedding, prev.turn_embedding);
        float summ_sim = cosine_similarity(curr.summary_embedding, prev.summary_embedding);
        float avg_sim  = (turn_sim + summ_sim) / 2.0f;

        // Compute time-scaled threshold. X32: the gap runs from the END of
        // the previous turn to the START of the current one (an assistant
        // reply can take a long time; the idle time between exchanges is
        // what matters). begin(curr) falls back to curr.epoch when
        // started_epoch is unknown (0), and the gap is clamped at 0.
        const int64_t begin_curr = curr.started_epoch > 0 ? curr.started_epoch : curr.epoch;
        int64_t gap_seconds = begin_curr - prev.epoch;
        if (gap_seconds < 0) gap_seconds = 0;
        float gap_minutes = static_cast<float>(gap_seconds) / 60.0f;
        float threshold   = p.base;
        if (p.step_minutes > 0.0f)
            threshold += p.step * static_cast<float>(
                static_cast<int>(gap_minutes / p.step_minutes));
        threshold = std::min(threshold, p.cap);

        if (avg_sim <= threshold)
            boundary_indices.push_back(i);
    }
    return boundary_indices;
}

} // namespace ragger
