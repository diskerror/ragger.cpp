#!/bin/bash
# verify-schema-sql.sh — prove scripts/schema_db0.17.sql still describes
# EXACTLY what the built binary creates (PHASE9).
#
#   1. Build a FRESH DB with the built binary into a temp base (never ~/.ragger).
#   2. Load scripts/schema_db0.17.sql into a second, empty DB with semqlite.
#   3. Dump both (schema objects whitespace-normalized, sorted, plus the
#      db_version seed value) and diff. Exit 1 on any difference.
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/build/ragger"
SQL="$SCRIPT_DIR/schema_db0.17.sql"

[ -x "$BIN" ] || { echo "verify-schema-sql: build/ragger not found -- run scripts/build.sh first" >&2; exit 1; }
[ -f "$SQL" ] || { echo "verify-schema-sql: $SQL missing" >&2; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

"$BIN" --ragger-base "$T/binary" --port 59996 count > "$T/binary.log" 2>&1 \
    || { echo "verify-schema-sql: fresh-DB creation failed:"; cat "$T/binary.log"; exit 1; }

semqlite "$T/sqlfile.db" ".read $SQL" > /dev/null

dump() {
    python3 - "$1" <<'PY'
import sqlite3, sys, re
db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "SELECT type||'|'||name||'|'||COALESCE(sql,'') FROM sqlite_master "
    "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
    "AND name NOT IN ('semqlite_config','semext_config') "
    "ORDER BY type, name").fetchall()
for (r,) in rows:
    print(re.sub(r'\s+', ' ', r).strip())
print("db_version=" + db.execute(
    "SELECT value FROM settings WHERE key='db_version'").fetchone()[0])
db.close()
PY
}

dump "$T/binary/memories.db" > "$T/binary.dump"
dump "$T/sqlfile.db"        > "$T/sqlfile.dump"

if diff -u "$T/binary.dump" "$T/sqlfile.dump"; then
    echo "verify-schema-sql: OK ($(wc -l < "$T/binary.dump" | tr -d ' ') schema objects match)"
else
    echo "verify-schema-sql: SCHEMA DRIFT -- scripts/schema_db0.17.sql differs from the binary" >&2
    exit 1
fi
