#!/usr/bin/env bash
# migrate_to_db0.12.sh
#
# Migrates a Ragger memories.db from db_version '4' to db_version '0.12':
#   1. Splits summaries.level='turn' rows into a new turn_summaries table
#      with a real turn_id FK (ON DELETE SET NULL), resolved via the OLD
#      fragile (session_id, created_at) match ONE LAST TIME (this script
#      is the final place that match ever needs to happen again).
#   2. Converts every TEXT timestamp column to INTEGER Unix-epoch seconds,
#      except documents.imported_at (stays TEXT 'YYYYMMDD' by design).
#      Conversion uses unixepoch(col, 'utc') because the stored strings
#      are LOCAL wall-clock time, not UTC — plain unixepoch(col) silently
#      assumes UTC input and shifts every timestamp by the local offset.
#   3. Renames users.created/users.modified to users.created_at/updated_at.
#   4. Adds has_embedding/has_phon + datetime()-rendering _view per table.
#   5. Stamps settings.db_version = '0.12'.
#
# WORKFLOW (always operates on the given/default DB file, in place):
#   1. Stop the ragger daemon if it's running against this DB.
#   2. Build the new schema at "${NAME}_NEW0.12.db" alongside the original
#      (never touches the original file itself at this stage).
#   3. Copy + translate all data into the new file.
#   4. Build/rebuild indexes (including FTS5) in the new file, verify row
#      counts + integrity there.
#   5. Only if all checks pass: rename the original to
#      "${NAME}_BACKUP_<timestamp>.db" and rename the new file to the
#      original's name. Restart the daemon if it was stopped.
#   On any failure before step 5, the original file is untouched and the
#   half-built "${NAME}_NEW0.12.db" is left on disk for inspection (not
#   auto-deleted, so you can see what went wrong).
#
# USAGE
#   scripts/migrate_to_db0.12.sh [--db PATH] [--yes]
#
#   --db PATH   DB file to migrate. Default: $HOME/.ragger/memories.db
#                (or $RAGGER_BASE/memories.db if RAGGER_BASE is set).
#   --yes       Skip the confirmation prompt (for scripting/CI use).
#
# EXIT CODES
#   0  success (including "already migrated, nothing to do")
#   1  migration aborted — see stderr. The original DB is guaranteed
#      untouched at this point; the new file (if partially built) is left
#      at "${NAME}_NEW0.12.db" for inspection, not deleted.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
RAGGER_BASE_DEFAULT="${RAGGER_BASE:-$HOME/.ragger}"
DB="${RAGGER_BASE_DEFAULT}/memories.db"
ASSUME_YES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --db)     DB="$2"; shift 2 ;;
        --yes)    ASSUME_YES=1; shift ;;
        -h|--help)
            sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -d "$DB" ]; then
    DB="${DB%/}/memories.db"
    echo "Note: --db was a directory; using $DB"
fi

DB_DIR="$(cd "$(dirname "$DB")" && pwd)"
DB_BASE="$(basename "$DB")"
DB="${DB_DIR}/${DB_BASE}"          # normalize to absolute path for reliable comparisons
DB_NAME="${DB_BASE%.db}"          # strip trailing .db if present, else unchanged

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_SQL="${SCRIPT_DIR}/schema_db0.12.sql"
OLD_VERSION="4"
NEW_VERSION="0.12"
NEW_DB="${DB_DIR}/${DB_NAME}_NEW0.12.db"
BACKUP="${DB_DIR}/${DB_NAME}_BACKUP_$(date +%Y%m%d-%H%M%S).db"

echo "== Ragger DB migration: db_version ${OLD_VERSION} -> ${NEW_VERSION} =="
echo "DB:      $DB"
echo "New:     $NEW_DB"
echo "Backup:  $BACKUP  (original renamed here once verified)"
echo

if [ ! -f "$DB" ]; then
    echo "ERROR: DB not found: $DB" >&2
    exit 1
fi

if [ ! -f "$SCHEMA_SQL" ]; then
    echo "ERROR: schema file not found: $SCHEMA_SQL" >&2
    exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "ERROR: sqlite3 CLI not found on PATH." >&2
    exit 1
fi

if [ -e "$NEW_DB" ]; then
    echo "ERROR: $NEW_DB already exists — remove/inspect it before re-running." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Early guard: already migrated? Make re-runs a clean no-op rather than a
# silent double-migration or a confusing failure.
# ---------------------------------------------------------------------------
CURRENT_VERSION="$(sqlite3 "$DB" "SELECT value FROM settings WHERE key='db_version';" 2>/dev/null || echo "")"
if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
    echo "Database at $DB is already at db_version ${NEW_VERSION}. Nothing to do."
    exit 0
fi
if [ -n "$CURRENT_VERSION" ] && [ "$CURRENT_VERSION" != "$OLD_VERSION" ]; then
    echo "ERROR: unexpected db_version '$CURRENT_VERSION' (expected '$OLD_VERSION' or absent)." >&2
    echo "This script only migrates FROM db_version '$OLD_VERSION'. Aborting." >&2
    exit 1
fi
echo "Current db_version: ${CURRENT_VERSION:-<none, pre-v4-versioning>}"
echo

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Proceed with migration? [y/N] " REPLY
    case "$REPLY" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# ---------------------------------------------------------------------------
# 1. Stop the daemon ONLY if we're operating on the live default DB path —
#    a copy sitting on the Desktop or /tmp has nothing to do with whatever
#    daemon happens to be running against the real file. Restart on exit
#    (success or failure) via trap, so a mid-script abort never leaves
#    ragger stopped.
# ---------------------------------------------------------------------------
LIVE_DB="$(cd "$RAGGER_BASE_DEFAULT" 2>/dev/null && pwd)/memories.db"
DAEMON_STOPPED=0
if [ "$DB" = "$LIVE_DB" ] && command -v ragger >/dev/null 2>&1; then
    if ragger status >/dev/null 2>&1; then
        echo "Stopping ragger daemon (operating on live DB)..."
        ragger stop || true
        DAEMON_STOPPED=1
    fi
else
    echo "Not the live DB path — leaving ragger daemon alone."
fi

restart_daemon_if_stopped() {
    if [ "$DAEMON_STOPPED" -eq 1 ]; then
        echo "Restarting ragger daemon..."
        ragger start || true
    fi
}
trap restart_daemon_if_stopped EXIT

# ---------------------------------------------------------------------------
# 2. Check for other holders of the DB file before touching anything.
# ---------------------------------------------------------------------------
if command -v lsof >/dev/null 2>&1; then
    HOLDERS="$(lsof "$DB" 2>/dev/null || true)"
    if [ -n "$HOLDERS" ]; then
        echo "ERROR: another process still holds $DB open. Aborting." >&2
        echo "$HOLDERS" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Build the new schema at NEW_DB, alongside the original. The original
#    is never touched until the final rename in step 8.
# ---------------------------------------------------------------------------
sqlite3 "$NEW_DB" < "$SCHEMA_SQL"
# The schema file's own INSERT OR IGNORE stamps db_version=0.12 already;
# clear it here so the row-count/verify steps below start from a known
# empty state and the real stamp happens explicitly in step 7.
sqlite3 "$NEW_DB" "DELETE FROM settings WHERE key='db_version';"

echo "New schema built at: $NEW_DB"

# ---------------------------------------------------------------------------
# 4. Copy all tables across via ATTACH — read from the OLD db, write into
#    NEW_DB, all within one sqlite3 session so no second physical copy of
#    the whole old DB is needed just to read from it.
# ---------------------------------------------------------------------------
echo "Copying data..."

sqlite3 "$NEW_DB" <<SQL
ATTACH DATABASE '${DB}' AS old;

-- models, sessions, decisions: straight copy, TEXT timestamp -> epoch.
-- 'utc' modifier: stored strings are LOCAL time, not UTC (see header note).
INSERT INTO models (model_id, name, created_at)
SELECT model_id, name, unixepoch(created_at, 'utc') FROM old.models;

INSERT INTO sessions (session_id, guid, created_at)
SELECT session_id, guid, unixepoch(created_at, 'utc') FROM old.sessions;

INSERT INTO decisions (decision_id, text, status, tags, created_at, embedding, phon)
SELECT decision_id, text, status, tags, unixepoch(created_at, 'utc'), embedding, phon
FROM old.decisions;

-- documents: straight copy, EXCEPT imported_at is cleaned up here — it's
-- meant to be a date-only batch key ('YYYYMMDD') but the capture side has
-- been writing a full timestamp ('YYYY-MM-DD HH:MM:SS'). Strip the time
-- portion and the dashes so it lands as the intended 'YYYYMMDD' TEXT
-- (stays TEXT by design — see plan notes — just fixed to the right shape).
INSERT INTO documents (document_id, text, path, title, tags, year, chunk_index,
                        imported_at, embedding, phon)
SELECT document_id, text, path, title, tags, year, chunk_index,
       replace(substr(imported_at, 1, 10), '-', ''), embedding, phon
FROM old.documents;

-- users: rename created/modified -> created_at/updated_at IN THE SAME
-- INSERT as the epoch conversion — no intermediate state with old names
-- and new types or vice versa.
INSERT INTO users (id, username, token_hash, password_hash, created_at, updated_at)
SELECT id, username, token_hash, password_hash,
       unixepoch(created, 'utc'), unixepoch(modified, 'utc')
FROM old.users;

-- turns: straight copy, epoch conversion, turn_id PRESERVED EXACTLY (it is
-- the FK turn_summaries.turn_id will point to — do NOT renumber here).
INSERT INTO turns (turn_id, user_text, assistant_text, model_id, session_id,
                    created_at, embedding, phon)
SELECT turn_id, user_text, assistant_text, model_id, session_id,
       unixepoch(created_at, 'utc'), embedding, phon
FROM old.turns;

-- summaries: episode/session/project rows only (level='turn' rows go to
-- turn_summaries below instead).
INSERT INTO summaries (summary_id, text, level, status, tags, session_id,
                        model_id, created_at, updated_at, embedding, phon)
SELECT summary_id, text, level, status, tags, session_id,
       model_id, unixepoch(created_at, 'utc'), unixepoch(updated_at, 'utc'), embedding, phon
FROM old.summaries
WHERE level != 'turn';

-- turn_summaries: populated from the OLD summaries' level='turn' rows.
-- turn_id is resolved via the OLD fragile (session_id, created_at) match —
-- this is the LAST time that match is ever needed; every future write
-- goes through the real turn_id FK. A row whose (session_id, created_at)
-- doesn't match any turn (shouldn't happen post-hotfix, but possible for
-- stale/orphaned rows) still gets copied with turn_id NULL rather than
-- being silently dropped — nothing from the old DB is discarded by this
-- migration, per "everything is derived data but nothing is thrown away
-- carelessly."
--
-- Tie-break for same-second collisions: a handful of (session_id,
-- created_at) pairs in turns have TWO rows (a pre-existing capture-side
-- dup bug, out of scope for this migration to fix) — the LEFT JOIN below
-- picks the LOWEST turn_id (earliest insert, most likely canonical) via
-- the MIN(turn_id) subquery, so exactly one turn_summaries row is produced
-- per old summary row regardless. See the row-count verification step
-- below, which would catch a regression here.
INSERT INTO turn_summaries (turn_id, session_id, turn_model_id, summary_model_id,
                             turn_datetime, summarized_on, text, embedding, phon)
SELECT
    (SELECT MIN(t2.turn_id) FROM old.turns t2
     WHERE t2.created_at = s.created_at AND t2.session_id IS s.session_id),
    s.session_id,
    (SELECT t3.model_id FROM old.turns t3
     WHERE t3.turn_id = (SELECT MIN(t2.turn_id) FROM old.turns t2
                          WHERE t2.created_at = s.created_at AND t2.session_id IS s.session_id)),
    s.model_id,
    unixepoch(s.created_at, 'utc'),
    unixepoch(s.updated_at, 'utc'),
    s.text,
    s.embedding,
    s.phon
FROM old.summaries s
WHERE s.level = 'turn';

DETACH DATABASE old;
SQL

echo "Data copy complete."

# ---------------------------------------------------------------------------
# 5. Verify row counts: old vs new, per table. Abort loudly on any
#    mismatch — do NOT proceed to the swap. NEW_DB is left on disk for
#    inspection rather than deleted.
# ---------------------------------------------------------------------------
echo "Verifying row counts..."

verify_count() {
    local label="$1" old_sql="$2" new_sql="$3"
    local old_count new_count
    old_count="$(sqlite3 "$DB" "$old_sql")"
    new_count="$(sqlite3 "$NEW_DB" "$new_sql")"
    if [ "$old_count" != "$new_count" ]; then
        echo "ERROR: row count mismatch for $label — old=$old_count new=$new_count" >&2
        echo "New (unverified) file left at: $NEW_DB" >&2
        exit 1
    fi
    echo "  $label: $old_count == $new_count  OK"
}

verify_count "models"     "SELECT count(*) FROM models;"     "SELECT count(*) FROM models;"
verify_count "sessions"   "SELECT count(*) FROM sessions;"   "SELECT count(*) FROM sessions;"
verify_count "decisions"  "SELECT count(*) FROM decisions;"  "SELECT count(*) FROM decisions;"
verify_count "documents"  "SELECT count(*) FROM documents;"  "SELECT count(*) FROM documents;"
verify_count "users"      "SELECT count(*) FROM users;"      "SELECT count(*) FROM users;"
verify_count "turns"      "SELECT count(*) FROM turns;"      "SELECT count(*) FROM turns;"
verify_count "summaries (non-turn)" \
    "SELECT count(*) FROM summaries WHERE level != 'turn';" \
    "SELECT count(*) FROM summaries;"
verify_count "turn_summaries (from old level='turn')" \
    "SELECT count(*) FROM summaries WHERE level = 'turn';" \
    "SELECT count(*) FROM turn_summaries;"

# Extra check: how many turn_summaries rows failed to resolve a turn_id via
# the old fuzzy match — informational, not a hard failure (a handful of
# unresolvable historical rows is plausible; a LARGE number would indicate
# something wrong with the match logic above, worth a human look before
# proceeding to the swap).
UNRESOLVED="$(sqlite3 "$NEW_DB" "SELECT count(*) FROM turn_summaries WHERE turn_id IS NULL;")"
echo "  turn_summaries with unresolved turn_id: $UNRESOLVED (informational)"

echo "Row counts verified."

# ---------------------------------------------------------------------------
# 6. Rebuild all FTS5 tables from scratch and verify docsize counts match
#    base table counts (the reliable desync probe — a plain count(*) on an
#    external-content FTS reads through to the base and always "matches"
#    even when the index itself is empty).
# ---------------------------------------------------------------------------
echo "Building FTS5 indexes..."

sqlite3 "$NEW_DB" <<'SQL'
INSERT INTO turns_fts(turns_fts) VALUES('rebuild');
INSERT INTO turns_phon_fts(turns_phon_fts) VALUES('rebuild');
INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild');
INSERT INTO summaries_phon_fts(summaries_phon_fts) VALUES('rebuild');
INSERT INTO turn_summaries_fts(turn_summaries_fts) VALUES('rebuild');
INSERT INTO turn_summaries_phon_fts(turn_summaries_phon_fts) VALUES('rebuild');
INSERT INTO decisions_fts(decisions_fts) VALUES('rebuild');
INSERT INTO decisions_phon_fts(decisions_phon_fts) VALUES('rebuild');
INSERT INTO documents_fts(documents_fts) VALUES('rebuild');
INSERT INTO documents_phon_fts(documents_phon_fts) VALUES('rebuild');
SQL

verify_fts_docsize() {
    local base_table="$1" fts_table="$2"
    local base_count fts_count
    base_count="$(sqlite3 "$NEW_DB" "SELECT count(*) FROM ${base_table};")"
    fts_count="$(sqlite3 "$NEW_DB" "SELECT count(*) FROM ${fts_table}_docsize;")"
    if [ "$base_count" != "$fts_count" ]; then
        echo "ERROR: FTS docsize mismatch for $fts_table — base=$base_count fts=$fts_count" >&2
        echo "New (unverified) file left at: $NEW_DB" >&2
        exit 1
    fi
    echo "  $fts_table: $fts_count  OK"
}

verify_fts_docsize "turns"          "turns_fts"
verify_fts_docsize "turns"          "turns_phon_fts"
verify_fts_docsize "summaries"      "summaries_fts"
verify_fts_docsize "summaries"      "summaries_phon_fts"
verify_fts_docsize "turn_summaries" "turn_summaries_fts"
verify_fts_docsize "turn_summaries" "turn_summaries_phon_fts"
verify_fts_docsize "decisions"      "decisions_fts"
verify_fts_docsize "decisions"      "decisions_phon_fts"
verify_fts_docsize "documents"      "documents_fts"
verify_fts_docsize "documents"      "documents_phon_fts"

echo "FTS5 indexes verified."

# ---------------------------------------------------------------------------
# 7. Integrity check + stamp db_version. Stamping is what makes a
#    v0.12.0+ binary agree to run against this file.
# ---------------------------------------------------------------------------
INTEGRITY="$(sqlite3 "$NEW_DB" "PRAGMA integrity_check;")"
if [ "$INTEGRITY" != "ok" ]; then
    echo "ERROR: PRAGMA integrity_check failed:" >&2
    echo "$INTEGRITY" >&2
    echo "New (unverified) file left at: $NEW_DB" >&2
    exit 1
fi
echo "Integrity check: ok"

sqlite3 "$NEW_DB" "INSERT OR REPLACE INTO settings (key, value) VALUES ('db_version', '${NEW_VERSION}');"
echo "Stamped db_version = ${NEW_VERSION}"

# ---------------------------------------------------------------------------
# 8. Everything checked out. Move the original aside as a timestamped
#    backup, then move the new file into the original's place. Two
#    renames, same filesystem — each individually atomic; original is
#    fully preserved under BACKUP either way.
# ---------------------------------------------------------------------------
mv "$DB" "$BACKUP"
mv "$NEW_DB" "$DB"

echo
echo "== Migration complete. =="
echo "  Live DB:      $DB  (db_version ${NEW_VERSION})"
echo "  Original kept at: $BACKUP"

# restart_daemon_if_stopped runs automatically via the EXIT trap.
