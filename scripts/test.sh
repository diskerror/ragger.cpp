#!/bin/bash
# test.sh — Build and run all tests (with and without RAGGER_STATS)
#
# Per project policy: compile-time flags like RAGGER_STATS must be tested
# both enabled and disabled on every release.
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Test pass 1/2: RAGGER_STATS=ON ==="
"$SCRIPT_DIR/build.sh" clean --stats
cd build
ctest --output-on-failure
cd "$SCRIPT_DIR/.."

echo ""
echo "=== Test pass 2/2: RAGGER_STATS=OFF ==="
"$SCRIPT_DIR/build.sh" clean --no-stats
cd build
ctest --output-on-failure
cd "$SCRIPT_DIR/.."

echo ""
echo "✓ All tests passed (with and without RAGGER_STATS)"
