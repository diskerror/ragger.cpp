#!/usr/bin/env bash
# migrate_to_db0.15.sh
#
# Migrates a Ragger memories.db from db_version '0.12' to db_version '0.15':
#   1. Adds an `embedding_version INTEGER` column to every embedded table
#      (turns, turn_summaries, summaries, decisions, documents), placed
#      immediately before `embedding`.
#   2. For every row with a non-NULL `embedding`, extracts the version tag
#      that used to be packed as the FIRST BYTE of the blob (see c_lib's
#      EmbeddingCodec.h) into the new `embedding_version` column, and
#      rewrites `embedding` to hold only the payload that follows that byte
#      (unchanged for rows where `embedding IS NULL` — both columns stay
#      NULL together).
#   3. Stamps settings.db_version = '0.15'.
#
# After this migration, "does this row need re-embedding" is a plain
# `embedding_version IS NULL OR embedding_version != <current>` integer
# comparison instead of a per-row blob-decode-and-compare.
#
# WORKFLOW (always operates on the given/default DB file, in place):
#   1. Stop the ragger daemon if it's running against this DB.
#   2. Build the new schema at "${NAME}_NEW0.15.db" alongside the original
#      (never touches the original file itself at this stage).
#   3. Copy + translate all data into the new file, splitting each
#      embedding blob's version byte out into embedding_version.
#   4. Build/rebuild indexes (including FTS5) in the new file, verify row
#      counts + integrity there, plus a spot-check that decoded version
#      bytes and stripped payload sizes look sane.
#   5. Only if all checks pass: rename the original to
#      "${NAME}_BACKUP_<timestamp>.db" and rename the new file to the
#      original's name. Restart the daemon if it was stopped.
#   On any failure before step 5, the original file is untouched and the
#   half-built "${NAME}_NEW0.15.db" is left on disk for inspection (not
#   auto-deleted, so you can see what went wrong).
#
# USAGE
#   scripts/migrate_to_db0.15.sh [--db PATH] [--yes]
#
#   --db PATH   DB file to migrate. Default: $HOME/.ragger/memories.db
#                (or $RAGGER_BASE/memories.db if RAGGER_BASE is set).
#   --yes       Skip the confirmation prompt (for scripting/CI use).
#
# EXIT CODES
#   0  success (including "already migrated, nothing to do")
#   1  migration aborted — see stderr. The original DB is guaranteed
#      untouched at this point; the new file (if partially built) is left
#      at "${NAME}_NEW0.15.db" for inspection, not deleted.

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
SCHEMA_SQL="${SCRIPT_DIR}/schema_db0.15.sql"
OLD_VERSION="0.12"
NEW_VERSION="0.15"
NEW_DB="${DB_DIR}/${DB_NAME}_NEW0.15.db"
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
if [ "$CURRENT_VERSION" != "$OLD_VERSION" ]; then
    echo "ERROR: unexpected db_version '$CURRENT_VERSION' (expected '$OLD_VERSION')." >&2
    echo "This script only migrates FROM db_version '$OLD_VERSION'. Aborting." >&2
    exit 1
fi
echo "Current db_version: ${CURRENT_VERSION}"
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
LIVE_DB="$(cd "$RAGGER_BASE_DEFAULT" 2>/dev/null && pwd || true)/memories.db"
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
# The schema file's own INSERT OR IGNORE stamps db_version=0.15 already;
# clear it here so the row-count/verify steps below start from a known
# empty state and the real stamp happens explicitly in step 7.
sqlite3 "$NEW_DB" "DELETE FROM settings WHERE key='db_version';"

echo "New schema built at: $NEW_DB"

# ---------------------------------------------------------------------------
# 4. Copy all tables across via ATTACH — read from the OLD db, write into
#    NEW_DB, all within one sqlite3 session so no second physical copy of
#    the whole old DB is needed just to read from it.
#
#    For each embedded table: embedding_version is decoded from the first
#    byte of the OLD blob (byte value 0-255 read via the hex()/instr()
#    trick below — pure SQL, no extension needed), and the NEW embedding
#    column gets everything AFTER that first byte (substr(b, 2), which is
#    NULL when b is NULL — no separate CASE needed).
# ---------------------------------------------------------------------------
echo "Copying data..."

# byte0(b) expression, inlined at each embedding column below:
#   CASE WHEN b IS NULL THEN NULL ELSE
#     (instr('0123456789ABCDEF', upper(substr(hex(b),1,1))) - 1) * 16
#     + (instr('0123456789ABCDEF', upper(substr(hex(b),2,1))) - 1)
#   END
# The explicit CASE is required: hex(NULL) is '' (empty string), NOT NULL,
# so the arithmetic below it would silently produce 0 instead of NULL for
# a NULL blob without this guard.

sqlite3 "$NEW_DB" <<SQL
ATTACH DATABASE '${DB}' AS old;

-- models, sessions: no embedding column, straight copy.
INSERT INTO models (model_id, name, created_at)
SELECT model_id, name, created_at FROM old.models;

INSERT INTO sessions (session_id, guid, name, name_source, created_at)
SELECT session_id, guid, name, name_source, created_at FROM old.sessions;

-- decisions
INSERT INTO decisions (decision_id, text, status, tags, created_at,
                        embedding_version, embedding, phon)
SELECT decision_id, text, status, tags, created_at,
       CASE WHEN embedding IS NULL THEN NULL ELSE
         (instr('0123456789ABCDEF', upper(substr(hex(embedding),1,1))) - 1) * 16
         + (instr('0123456789ABCDEF', upper(substr(hex(embedding),2,1))) - 1)
       END,
       substr(embedding, 2),
       phon
FROM old.decisions;

-- documents
INSERT INTO documents (document_id, text, path, title, tags, year, chunk_index,
                        imported_at, embedding_version, embedding, phon)
SELECT document_id, text, path, title, tags, year, chunk_index, imported_at,
       CASE WHEN embedding IS NULL THEN NULL ELSE
         (instr('0123456789ABCDEF', upper(substr(hex(embedding),1,1))) - 1) * 16
         + (instr('0123456789ABCDEF', upper(substr(hex(embedding),2,1))) - 1)
       END,
       substr(embedding, 2),
       phon
FROM old.documents;

-- users: straight copy.
INSERT INTO users (id, username, token_hash, password_hash, created_at, updated_at)
SELECT id, username, token_hash, password_hash, created_at, updated_at
FROM old.users;

-- turns
INSERT INTO turns (turn_id, user_text, assistant_text, model_id, session_id,
                    created_at, embedding_version, embedding, phon)
SELECT turn_id, user_text, assistant_text, model_id, session_id, created_at,
       CASE WHEN embedding IS NULL THEN NULL ELSE
         (instr('0123456789ABCDEF', upper(substr(hex(embedding),1,1))) - 1) * 16
         + (instr('0123456789ABCDEF', upper(substr(hex(embedding),2,1))) - 1)
       END,
       substr(embedding, 2),
       phon
FROM old.turns;

-- summaries (episode/session/project rows only, as of 0.12)
INSERT INTO summaries (summary_id, text, level, tags, session_id,
                        model_id, created_at, updated_at,
                        embedding_version, embedding, phon)
SELECT summary_id, text, level, tags, session_id, model_id, created_at, updated_at,
       CASE WHEN embedding IS NULL THEN NULL ELSE
         (instr('0123456789ABCDEF', upper(substr(hex(embedding),1,1))) - 1) * 16
         + (instr('0123456789ABCDEF', upper(substr(hex(embedding),2,1))) - 1)
       END,
       substr(embedding, 2),
       phon
FROM old.summaries;

-- turn_summaries
INSERT INTO turn_summaries (turn_summary_id, text, turn_id, session_id,
                             turn_model_id, summary_model_id,
                             turn_datetime, summarized_on,
                             embedding_version, embedding, phon)
SELECT turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
       turn_datetime, summarized_on,
       CASE WHEN embedding IS NULL THEN NULL ELSE
         (instr('0123456789ABCDEF', upper(substr(hex(embedding),1,1))) - 1) * 16
         + (instr('0123456789ABCDEF', upper(substr(hex(embedding),2,1))) - 1)
       END,
       substr(embedding, 2),
       phon
FROM old.turn_summaries;

-- settings: carry over everything except db_version (stamped explicitly
-- in step 7) and the embedding_version bookkeeping key, whose MEANING
-- changes with this migration (it's still "the current version to compare
-- rows against", but the per-row storage rides in a column now, not a
-- blob byte -- the setting's value itself is unaffected and copies fine,
-- listed here only to make the exclusion intentional/documented).
INSERT INTO settings (key, value)
SELECT key, value FROM old.settings WHERE key != 'db_version';

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

verify_count "models"          "SELECT count(*) FROM models;"          "SELECT count(*) FROM models;"
verify_count "sessions"        "SELECT count(*) FROM sessions;"        "SELECT count(*) FROM sessions;"
verify_count "decisions"       "SELECT count(*) FROM decisions;"       "SELECT count(*) FROM decisions;"
verify_count "documents"       "SELECT count(*) FROM documents;"       "SELECT count(*) FROM documents;"
verify_count "users"           "SELECT count(*) FROM users;"           "SELECT count(*) FROM users;"
verify_count "turns"           "SELECT count(*) FROM turns;"           "SELECT count(*) FROM turns;"
verify_count "summaries"       "SELECT count(*) FROM summaries;"       "SELECT count(*) FROM summaries;"
verify_count "turn_summaries"  "SELECT count(*) FROM turn_summaries;"  "SELECT count(*) FROM turn_summaries;"
verify_count "settings (excl. db_version)" \
    "SELECT count(*) FROM settings WHERE key != 'db_version';" \
    "SELECT count(*) FROM settings WHERE key != 'db_version';"

echo "Row counts verified."

# ---------------------------------------------------------------------------
# 6. Verify the embedding split: every row that had a non-NULL embedding in
#    the OLD db must now have a non-NULL embedding_version in the NEW db
#    (and vice versa — NULL/non-NULL must move together), and the NEW
#    embedding blob must be exactly 1 byte shorter than the OLD one.
# ---------------------------------------------------------------------------
echo "Verifying embedding_version split..."

verify_split() {
    local table="$1" pk="$2"
    local mismatches
    mismatches="$(sqlite3 "$NEW_DB" <<SQL
ATTACH DATABASE '${DB}' AS old;
SELECT count(*) FROM ${table} n
JOIN old.${table} o ON o.${pk} = n.${pk}
WHERE (o.embedding IS NULL) != (n.embedding_version IS NULL)
   OR (o.embedding IS NOT NULL AND length(o.embedding) != length(n.embedding) + 1);
SQL
)"
    if [ "$mismatches" != "0" ]; then
        echo "ERROR: embedding_version/embedding split mismatch in $table: $mismatches row(s)" >&2
        echo "New (unverified) file left at: $NEW_DB" >&2
        exit 1
    fi
    echo "  $table: embedding split OK"
}

verify_split "turns"          "turn_id"
verify_split "turn_summaries" "turn_summary_id"
verify_split "summaries"      "summary_id"
verify_split "decisions"      "decision_id"
verify_split "documents"      "document_id"

echo "Embedding split verified."

# ---------------------------------------------------------------------------
# 7. Rebuild all FTS5 tables from scratch and verify docsize counts match
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
# 8. Integrity check + stamp db_version. Stamping is what makes a
#    v0.15+ binary agree to run against this file.
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
# 9. Everything checked out. Move the original aside as a timestamped
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
