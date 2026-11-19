-- rebuild-check.sql — read-only counts for the v0.17 rebuild (LR S111).
--
-- One row per check as (check_name, count). SELECT statements only —
-- nothing here writes, and the test suite greps for that. Run it in
-- semqlite against a live database:
--
--   semqlite ~/.ragger/memories.db ".read scripts/rebuild-check.sql"
--
-- What a non-zero count means per check is stated in its comment. Checks
-- mirror the rules the code enforces (turn times, episode times, S67
-- stranded/open-tail, back-fill targets); after a clean rebuild most are
-- expected to read 0 — the ones marked [informational] count legitimate
-- states and are there to be read, not to be zeroed.

WITH cur AS (
    SELECT CAST(COALESCE(
        (SELECT value FROM settings WHERE key = 'embedding_version'), '1'
    ) AS INTEGER) AS v
)

-- A captured turn that never reached the summarizer queue or whose
-- summary insert failed. Expected 0 outside an inference outage.
SELECT 'turns_without_summary' AS check_name,
       (SELECT COUNT(*) FROM turns t
        WHERE NOT EXISTS (SELECT 1 FROM turn_summaries ts
                          WHERE ts.turn_id = t.turn_id)) AS count

UNION ALL
-- L2 rows with NULL text — the normalization S63 removed; must stay 0.
SELECT 'l2_null_text',
       (SELECT COUNT(*) FROM turn_summaries WHERE text IS NULL)

UNION ALL
-- Empty L2 rows that are NOT the poison marker: an unexplained hole.
SELECT 'l2_empty_text_not_bad',
       (SELECT COUNT(*) FROM turn_summaries ts
        LEFT JOIN models m ON m.model_id = ts.summary_model_id
        WHERE ts.text = '' AND COALESCE(m.name, '') != 'bad')

UNION ALL
-- [informational] Poison-abandoned turns (D-5): failed max_turn_failures
-- times, kept as empty rows on purpose.
SELECT 'l2_bad_rows',
       (SELECT COUNT(*) FROM turn_summaries ts
        JOIN models m ON m.model_id = ts.summary_model_id
        WHERE m.name = 'bad')

UNION ALL
-- L2 time must equal its turn's time (rule 18). Expected 0 after S56 +
-- the l2-null/l2-regenerate back-fill.
SELECT 'l2_time_mismatch',
       (SELECT COUNT(*) FROM turn_summaries ts
        JOIN turns t ON t.turn_id = ts.turn_id
        WHERE ts.turn_datetime != t.created_at)

UNION ALL
-- An episode's modified_on is the time its text was last written, so it can
-- never precede the end of the span it summarizes. Expected 0.
SELECT 'episodes_modified_before_span_end',
       (SELECT COUNT(*) FROM episodes WHERE modified_on < span_end)

UNION ALL
-- Episodes with no member summaries — a closed span nothing points at.
SELECT 'episodes_without_members',
       (SELECT COUNT(*) FROM episodes e
        WHERE NOT EXISTS (SELECT 1 FROM turn_summaries ts
                          WHERE ts.episode_id = e.episode_id))

UNION ALL
-- Stranded L2 (S67 rule): unattached, non-empty, and covered by an
-- episode's span. Expected 0 after `ragger backfill stranded`.
SELECT 'l2_stranded',
       (SELECT COUNT(*) FROM turn_summaries ts
        WHERE ts.episode_id IS NULL AND ts.text != ''
          AND ts.turn_datetime IS NOT NULL
          AND EXISTS (SELECT 1 FROM episodes e
                      WHERE e.session_id = ts.session_id
                        AND ts.turn_datetime >= e.span_start
                        AND ts.turn_datetime <= e.span_end))

UNION ALL
-- [informational] Open tail: unattached L2 past every episode of their
-- session — not yet closed, legitimately waiting for housekeeping.
SELECT 'l2_open_tail',
       (SELECT COUNT(*) FROM turn_summaries ts
        WHERE ts.episode_id IS NULL AND ts.text != ''
          AND ts.turn_datetime IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM episodes e
                          WHERE e.session_id = ts.session_id
                            AND ts.turn_datetime >= e.span_start
                            AND ts.turn_datetime <= e.span_end))

UNION ALL
-- Decisions missing their trigger (S66 target). Expected 0 after
-- `ragger backfill triggers` (or a dry-run count of what it would draft).
SELECT 'decisions_missing_trigger',
       (SELECT COUNT(*) FROM knowledge
        WHERE kind = 'decision'
          AND ("trigger" IS NULL OR "trigger" = ''))

UNION ALL
-- Sessions with no name that have real summaries (S67 session-names
-- target). Expected 0 after `ragger backfill session-names`.
SELECT 'unnamed_sessions_with_l2',
       (SELECT COUNT(*) FROM sessions s
        WHERE s.name IS NULL
          AND EXISTS (SELECT 1 FROM turn_summaries ts
                      WHERE ts.session_id = s.session_id AND ts.text != ''))

UNION ALL
-- Embedding work remaining per table (missing blob, or version behind
-- settings.embedding_version): exactly what `ragger re-embed all`
-- consumes. Expected 0 only after a full re-embed.
SELECT 'embeddings_missing_or_stale_turns',
       (SELECT COUNT(*) FROM turns, cur
        WHERE embedding IS NULL
           OR embedding_version IS NULL
           OR embedding_version != cur.v)

UNION ALL
SELECT 'embeddings_missing_or_stale_turn_summaries',
       (SELECT COUNT(*) FROM turn_summaries, cur
        WHERE embedding IS NULL
           OR embedding_version IS NULL
           OR embedding_version != cur.v)

UNION ALL
SELECT 'embeddings_missing_or_stale_documents',
       (SELECT COUNT(*) FROM documents, cur
        WHERE embedding IS NULL
           OR embedding_version IS NULL
           OR embedding_version != cur.v)

UNION ALL
SELECT 'embeddings_missing_or_stale_knowledge',
       (SELECT COUNT(*) FROM knowledge, cur
        WHERE embedding IS NULL
           OR embedding_version IS NULL
           OR embedding_version != cur.v)

UNION ALL
SELECT 'embeddings_missing_or_stale_episodes',
       (SELECT COUNT(*) FROM episodes, cur
        WHERE embedding IS NULL
           OR embedding_version IS NULL
           OR embedding_version != cur.v)

UNION ALL
-- Evidence pointing at an episode with empty text — a source that cannot
-- support the knowledge row pointing at it. Expected 0.
SELECT 'evidence_empty_episode_text',
       (SELECT COUNT(*) FROM knowledge_episode_evidence kee
        JOIN episodes e ON e.episode_id = kee.episode_id
        WHERE e.text = '');
