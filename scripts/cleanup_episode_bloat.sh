#!/bin/bash
# cleanup_episode_bloat.sh — EPISODE_PLAN Phase 3 one-time cleanup.
#
# Collapses the legacy running-L3 duplicate `level='session'` rows that the
# pre-episode summarizer stacked up (a session's running summary snapshotted
# many times, e.g. summaries 1956-1977: 22 near-duplicate rows for one
# session; session 72 had ~192). Keeps the NEWEST session row per session_id,
# deletes the older intermediates.
#
# Scope note (IMPORTANT): only `session_id IS NOT NULL` rows are touched. The
# `session_id IS NULL` session rows are NOT running-L3 bloat — they are legacy
# global `complete` summaries plus `import`-tagged rows (Role-1 store() and
# summary imports both land as level='session', session_id NULL). Collapsing
# those by a naive PARTITION BY session_id (all NULLs = one partition) would
# destroy legitimate data, so they are preserved untouched.
#
# These are STRUCTURAL duplicates (same session, running-summary snapshots),
# not semantic near-dups — no cosine/threshold work is needed here (that is
# Phase 4 topic detection).
#
# Safety:
#   * DRY-RUN by default. Pass --apply to actually delete.
#   * Refuses to run while the daemon is up (holds the DB).
#   * Backs the DB up before deleting.
#   * External-content FTS (summaries_fts + summaries_phon_fts) stay in sync
#     via the base table's AFTER DELETE triggers; we still checkpoint and run
#     integrity_check afterward.
#
# Usage:
#   scripts/cleanup_episode_bloat.sh [--db PATH] [--apply]
#     --db PATH   target DB (default: $HOME/.ragger/memories.db)
#     --apply     perform the delete (omit for a dry-run report)

set -euo pipefail

DB="$HOME/.ragger/memories.db"
APPLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --db)    DB="${2:?--db needs a path}"; shift 2 ;;
        --db=*)  DB="${1#--db=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$DB" ]]; then
    echo "DB not found: $DB" >&2
    exit 1
fi

# The ranking CTE, shared by dry-run and apply. Only session_id IS NOT NULL.
read -r -d '' RANKED_CTE <<'SQL' || true
WITH ranked AS (
  SELECT summary_id,
         ROW_NUMBER() OVER (PARTITION BY session_id
           ORDER BY timestamp DESC, summary_id DESC) AS rn
  FROM summaries
  WHERE level='session' AND session_id IS NOT NULL)
SQL

echo "== EPISODE_PLAN Phase 3 cleanup =="
echo "DB: $DB"

# Refuse to run against the LIVE DB while its daemon is up (best-effort). A
# running daemon holding the connection can cause BUSY/LOCKED and leave the FTS
# half-updated. Only relevant when targeting the default DB — a --db copy is
# not what the daemon has open.
if [[ "$DB" == "$HOME/.ragger/memories.db" ]] && command -v ragger >/dev/null 2>&1; then
    if ragger status 2>/dev/null | grep -qi "is running"; then
        echo "ERROR: ragger daemon appears to be running. Stop it first:" >&2
        echo "  ragger stop   # and confirm no listener on :8432" >&2
        exit 1
    fi
fi

total=$(sqlite3 "$DB" "SELECT COUNT(*) FROM summaries WHERE level='session';")
nonnull=$(sqlite3 "$DB" "SELECT COUNT(*) FROM summaries WHERE level='session' AND session_id IS NOT NULL;")
nullrows=$(sqlite3 "$DB" "SELECT COUNT(*) FROM summaries WHERE level='session' AND session_id IS NULL;")
keep=$(sqlite3 "$DB" "SELECT COUNT(DISTINCT session_id) FROM summaries WHERE level='session' AND session_id IS NOT NULL;")
todelete=$(sqlite3 "$DB" "$RANKED_CTE SELECT COUNT(*) FROM ranked WHERE rn>1;")

echo "  level='session' rows total : $total"
echo "  session_id IS NOT NULL     : $nonnull  (kept: $keep, delete: $todelete)"
echo "  session_id IS NULL (kept)  : $nullrows  (legacy global + import; preserved)"

if [[ "$todelete" -eq 0 ]]; then
    echo "Nothing to clean up. Done."
    exit 0
fi

if [[ "$APPLY" -ne 1 ]]; then
    echo
    echo "DRY-RUN. Would delete $todelete duplicate session rows."
    echo "Re-run with --apply to perform the deletion."
    exit 0
fi

# --- apply ---
BACKUP="${DB}.pre-episode-cleanup.$(date +%Y%m%d_%H%M%S)"
cp "$DB" "$BACKUP"
echo "Backup: $BACKUP"

deleted=$(sqlite3 "$DB" <<SQL
$RANKED_CTE
DELETE FROM summaries WHERE summary_id IN (SELECT summary_id FROM ranked WHERE rn>1);
SELECT changes();
SQL
)
echo "Deleted: $deleted rows"

echo "Checkpoint + integrity checks..."
sqlite3 "$DB" "PRAGMA wal_checkpoint(TRUNCATE);" >/dev/null
integ=$(sqlite3 "$DB" "PRAGMA integrity_check;")
echo "  integrity_check: $integ"
# FTS integrity: the special 'integrity-check' command validates the index.
fts1=$(sqlite3 "$DB" "INSERT INTO summaries_fts(summaries_fts) VALUES('integrity-check'); SELECT 'summaries_fts ok';" 2>&1 || echo "summaries_fts FAILED")
fts2=$(sqlite3 "$DB" "INSERT INTO summaries_phon_fts(summaries_phon_fts) VALUES('integrity-check'); SELECT 'summaries_phon_fts ok';" 2>&1 || echo "summaries_phon_fts FAILED")
echo "  $fts1"
echo "  $fts2"

remaining_dups=$(sqlite3 "$DB" "SELECT COUNT(*) FROM (SELECT session_id FROM summaries WHERE level='session' AND session_id IS NOT NULL GROUP BY session_id HAVING COUNT(*)>1);")
echo "  sessions still holding >1 row: $remaining_dups (want 0)"

if [[ "$integ" == "ok" && "$remaining_dups" -eq 0 ]] \
   && echo "$fts1" | grep -q ok && echo "$fts2" | grep -q ok; then
    echo "Cleanup OK. If anything looks wrong, restore: cp \"$BACKUP\" \"$DB\""
else
    echo "WARNING: post-checks did not all pass. Restore from backup and investigate:" >&2
    echo "  cp \"$BACKUP\" \"$DB\"" >&2
    exit 1
fi
