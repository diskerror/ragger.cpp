/**
 * LR S66/S67 — back-fill passes behind `ragger backfill <kind> [--dry-run]`.
 *
 * Reid runs every data operation (rule 17); this is the code he runs. Each
 * pass: select candidates (pure SELECT), summarize/fill per row, write in
 * place under the step's column rules, report progress through ProgressFn
 * (the library never prints), and invalidate the caches ONCE at the end.
 *
 * In-place rules (logic-review step doc, rule 18/19):
 *  - turn_summaries: only text, embedding, embedding_version, terms and
 *    summary_model_id change. turn_datetime, turn_id, session_id,
 *    episode_id never change; summarized_on is untouched (D-8).
 *  - knowledge (triggers): only trigger + embedding/embedding_version +
 *    terms change; text, status, supersedes never change (fill-NULL).
 *  - S62 applies everywhere (floor copy, shorter-text rule); an empty
 *    model result leaves the row unchanged and counts as a failure — the
 *    abandon marker mechanism itself stays in the live summarizer path
 *    (D-5), the back-fill never stamps 'bad'.
 *  - Every pass is idempotent/resumable: a second run selects nothing it
 *    already fixed.
 */
#pragma once

#include <string>
#include <vector>

#include "text_index.h"   // ragger::ProgressFn

namespace ragger {

class StorageBackend;
class InferenceClient;

/// Back-fill result counters. `failed` > 0 => the CLI exits non-zero and
/// reports each failure; the pass continues past a bad row.
struct BackfillResult {
    int candidates = 0;
    int applied    = 0;
    int skipped    = 0;   // guard rejected (already fixed concurrently)
    int failed     = 0;   // inference empty or write threw (row unchanged)
    int ties       = 0;   // stranded kind: rows covered by >1 episode span
    int open_tail  = 0;   // stranded kind: rows after the last span (left)
};

/// Kinds l2-null / l2-empty / l2-regenerate over `turn_summaries`.
/// `regenerate` REQUIRES a filter (model_name or ids) — addendum 5: the
/// default selection is empty and running with no filter refuses.
/// Throws std::invalid_argument on an unknown kind or an unfiltered
/// regenerate. `dry_run` selects and counts only — nothing is written.
BackfillResult backfill_l2(StorageBackend& backend, InferenceClient& inference,
                           const std::string& kind, bool dry_run,
                           const std::string& model_name = {},
                           const std::vector<int>& ids = {},
                           const ProgressFn& progress = {});

/// Kind triggers: decision|insight rows with no trigger get one drafted by
/// the summarizer and written into the NULL/empty column only. The row is
/// re-embedded + re-indexed in the same call (knowledge embeds
/// trigger + '\n' + text — the vector and terms MUST follow the fill).
BackfillResult backfill_triggers(StorageBackend& backend,
                                 InferenceClient& inference, bool dry_run,
                                 const ProgressFn& progress = {});

/// Kind stranded (S67): membership-only attach of L2 rows that lie inside
/// an already-closed episode span of their session. Rows after the last
/// span of their session are the open tail: counted, never touched.
BackfillResult backfill_stranded(StorageBackend& backend, bool dry_run,
                                 const ProgressFn& progress = {});

/// Kind session-names (S67, D5): sessions with no name whose L2 texts form
/// the title corpus; the summarizer drafts one line, written with
/// name_source = 'summarizer'. A named session is never touched.
BackfillResult backfill_session_names(StorageBackend& backend,
                                      InferenceClient& inference, bool dry_run,
                                      const ProgressFn& progress = {});

} // namespace ragger
