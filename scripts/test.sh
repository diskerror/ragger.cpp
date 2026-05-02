#!/bin/bash
# test.sh — Build and run all tests
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
"$SCRIPT_DIR/build.sh"
cd build
ctest --output-on-failure
