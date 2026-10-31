#!/bin/bash
# test.sh — Build and run all tests.
#
# Usage: ./scripts/test.sh [--with-stats]
#
# Default: one clean RAGGER_STATS=OFF build + ctest + schema check. Policy:
# the stats flag is instrumentation only, so the stats-ON build is exercised
# only when coding for stats or when building/testing for the live install.
#   --with-stats  also run a clean RAGGER_STATS=ON pass first (build-stats/)
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

WITH_STATS=0
case "${1:-}" in
    "") ;;
    --with-stats) WITH_STATS=1 ;;
    *) echo "usage: $0 [--with-stats]" >&2; exit 2 ;;
esac

"$SCRIPT_DIR/check-source.sh"

if [ "$WITH_STATS" -eq 1 ]; then
    echo "=== Test pass: RAGGER_STATS=ON ==="
    "$SCRIPT_DIR/build.sh" clean --stats
    cd build-stats
    ctest --output-on-failure
    cd "$SCRIPT_DIR/.."
    echo ""
fi

echo "=== Test pass: RAGGER_STATS=OFF ==="
"$SCRIPT_DIR/build.sh" clean --no-stats
cd build
ctest --output-on-failure
cd "$SCRIPT_DIR/.."
# schema verification needs build/ragger; the schema does not depend on the
# stats flag.
"$SCRIPT_DIR/verify-schema-sql.sh"

echo ""
if [ "$WITH_STATS" -eq 1 ]; then
    echo "✓ All tests passed (stats ON and OFF)"
else
    echo "✓ All tests passed (stats OFF; use --with-stats for the stats build)"
fi
