#!/usr/bin/env bash
# refresh_views.sh
#
# ONE-TIME (and safely repeatable) refresh of the convenience _view objects
# in a Ragger memories.db.
#
# WHY THIS EXISTS
#   create_views() used to say "CREATE VIEW IF NOT EXISTS", which meant an
#   existing database kept whatever view definitions it was first created
#   with -- FOREVER. When the v0.16 work retired the `phon` column, added
#   unigram_count/bigram_count, and introduced the terms index, existing DBs
#   were left with stale views that still referenced the old shape.
#
#   The binary now DROPs and recreates its views on every open, so once you
#   are running a build newer than commit b984207 this happens automatically.
#   This script is for refreshing a DB WITHOUT waiting for / running that
#   binary -- e.g. an archived copy, a backup you want to browse, or simply
#   to see the new views right now.
#
# WHAT IT DOES
#   Drops and recreates all 14 views:
#     - 9 table views: epoch ints rendered as local datetime, embedding BLOB
#       collapsed to has_embedding (0/1), and (new) unigram_count/bigram_count
#       surfaced on the five text tables.
#     - 5 junction views (NEW): <t>_terms_view resolves term_id -> the actual
#       term string, ordered most-frequent-first within each record.
#
#   Views hold NO DATA. Dropping and recreating them cannot lose anything;
#   it only rewrites definitions. Base tables are never touched.
#
# WHAT IT DOES *NOT* DO
#   This is NOT the 0.15 -> 0.16 migration. It does not rebuild tables, drop
#   `phon`, or populate the terms index. If the DB is still on 0.15, run the
#   ragger binary against it to migrate first; this script will tell you so
#   and refuse to create the junction views (their tables won't exist yet).
#
# USAGE
#   ./scripts/refresh_views.sh [path/to/memories.db]
#   (defaults to ~/.ragger/memories.db)
#
#   Stop the daemon first if it is running against this DB:
#     ragger stop
set -euo pipefail

DB="${1:-$HOME/.ragger/memories.db}"

if [[ ! -f "$DB" ]]; then
    echo "error: no such database: $DB" >&2
    exit 1
fi

SQLITE="${SQLITE:-sqlite3}"
command -v "$SQLITE" >/dev/null 2>&1 || { echo "error: $SQLITE not found" >&2; exit 1; }

echo "Database: $DB"

# Refuse to run against a DB a daemon is actively writing. A view swap is
# fast, but a busy DB means someone else owns this file right now.
if ! "$SQLITE" "$DB" "PRAGMA quick_check;" >/dev/null 2>&1; then
    echo "error: cannot open $DB (locked by a running daemon? try 'ragger stop')" >&2
    exit 1
fi

DBVER="$("$SQLITE" "$DB" "SELECT value FROM settings WHERE key='db_version';" 2>/dev/null || echo "")"
echo "db_version: ${DBVER:-<unknown>}"

# The junction views require the v0.16 terms index to exist.
HAVE_TERMS="$("$SQLITE" "$DB" \
    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='terms';")"

if [[ "$HAVE_TERMS" != "1" ]]; then
    cat >&2 <<EOF

warning: this DB has no 'terms' table, so it has not been migrated to 0.16 yet.
         Table views will be refreshed, but the 5 junction views will be SKIPPED.
         To migrate, run the ragger binary against this DB first.

EOF
fi

# Sanity: the text-table views below select unigram_count/bigram_count, which
# only exist post-0.16. Bail rather than create views that error on use.
HAVE_COUNTS="$("$SQLITE" "$DB" \
    "SELECT count(*) FROM pragma_table_info('turns') WHERE name='unigram_count';")"
if [[ "$HAVE_COUNTS" != "1" ]]; then
    echo "error: turns.unigram_count missing -- this DB predates v0.16." >&2
    echo "       Run the ragger binary against it to migrate first." >&2
    exit 1
fi

echo "Refreshing views..."

"$SQLITE" "$DB" <<'SQL'
BEGIN;

DROP VIEW IF EXISTS users_view;
CREATE VIEW users_view AS
SELECT id, username, token_hash, password_hash,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
FROM users;

DROP VIEW IF EXISTS models_view;
CREATE VIEW models_view AS
SELECT model_id, name,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM models;

DROP VIEW IF EXISTS sessions_view;
CREATE VIEW sessions_view AS
SELECT session_id, guid, name, name_source,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM sessions;

DROP VIEW IF EXISTS turns_view;
CREATE VIEW turns_view AS
SELECT turn_id, user_text, assistant_text, model_id, session_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM turns;

DROP VIEW IF EXISTS turn_summaries_view;
CREATE VIEW turn_summaries_view AS
SELECT turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
       datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
       datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM turn_summaries
WHERE text != '';

DROP VIEW IF EXISTS summaries_view;
CREATE VIEW summaries_view AS
SELECT summary_id, text, level, tags, session_id, model_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM summaries;

DROP VIEW IF EXISTS decisions_view;
CREATE VIEW decisions_view AS
SELECT decision_id, text, status, tags,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM decisions;

DROP VIEW IF EXISTS document_sources_view;
CREATE VIEW document_sources_view AS
SELECT document_source_id, title, path, year, tags,
       datetime(imported_at, 'unixepoch', 'localtime') AS imported_at
FROM document_sources;

DROP VIEW IF EXISTS documents_view;
CREATE VIEW documents_view AS
SELECT document_id, text, tags, chunk_index, document_source_id,
       datetime(modified_on, 'unixepoch', 'localtime') AS modified_on,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM documents;

COMMIT;
SQL

if [[ "$HAVE_TERMS" == "1" ]]; then
"$SQLITE" "$DB" <<'SQL'
BEGIN;

DROP VIEW IF EXISTS turns_terms_view;
CREATE VIEW turns_terms_view AS
SELECT j.turn_id AS turn_id, t.term, j.count
FROM turns_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.turn_id, j.count DESC, t.term;

DROP VIEW IF EXISTS turn_summaries_terms_view;
CREATE VIEW turn_summaries_terms_view AS
SELECT j.turn_summary_id AS turn_summary_id, t.term, j.count
FROM turn_summaries_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.turn_summary_id, j.count DESC, t.term;

DROP VIEW IF EXISTS summaries_terms_view;
CREATE VIEW summaries_terms_view AS
SELECT j.summary_id AS summary_id, t.term, j.count
FROM summaries_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.summary_id, j.count DESC, t.term;

DROP VIEW IF EXISTS documents_terms_view;
CREATE VIEW documents_terms_view AS
SELECT j.document_id AS document_id, t.term, j.count
FROM documents_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.document_id, j.count DESC, t.term;

DROP VIEW IF EXISTS decisions_terms_view;
CREATE VIEW decisions_terms_view AS
SELECT j.decision_id AS decision_id, t.term, j.count
FROM decisions_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.decision_id, j.count DESC, t.term;

COMMIT;
SQL
fi

# Verify every view actually RUNS, not merely that it was created. A view with
# a bad column reference is accepted at CREATE time and only fails on SELECT.
echo "Verifying..."
FAILED=0
while read -r v; do
    [[ -z "$v" ]] && continue
    if "$SQLITE" "$DB" "SELECT * FROM \"$v\" LIMIT 1;" >/dev/null 2>&1; then
        printf '  %-28s ok\n' "$v"
    else
        printf '  %-28s FAILED\n' "$v" >&2
        FAILED=1
    fi
done < <("$SQLITE" "$DB" "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;")

if [[ "$FAILED" != "0" ]]; then
    echo >&2
    echo "error: one or more views do not execute. See above." >&2
    exit 1
fi

COUNT="$("$SQLITE" "$DB" "SELECT count(*) FROM sqlite_master WHERE type='view';")"
echo
echo "Done. $COUNT views refreshed and verified."
echo
echo "Try:  $SQLITE \"$DB\" \"SELECT * FROM decisions_terms_view WHERE decision_id=1 LIMIT 10;\""
