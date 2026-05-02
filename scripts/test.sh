#!/bin/bash
# test.sh — Build and run all tests
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin:$PATH"
./build.sh
cd build
ctest --output-on-failure
