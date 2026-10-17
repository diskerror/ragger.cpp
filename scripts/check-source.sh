#!/bin/bash
# check-source.sh — release gate: no scrubbed "*** " literals in the tree.
#
# Why: commit 340ab92 committed "Authorization: *** " (real bytes) where the
# Bearer scheme word belonged — a tool's output redaction was pasted back as
# source, silently breaking every keyed HTTP request. Some tool output shows
# "***" as DISPLAY redaction; this script checks the real bytes.
#
# Exits 1 and prints file:line for every match of "*** " (three asterisks +
# space) in the source trees. Excludes this script itself (its grep pattern
# would otherwise match its own text).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

matches=$(grep -rnE '\*\*\* ' src include scripts hermes-plugin openclaw-plugin \
    --exclude check-source.sh 2>/dev/null)

if [ -n "$matches" ]; then
    echo "check-source.sh: scrubbed '*** ' literal(s) found:" >&2
    echo "$matches" >&2
    exit 1
fi
exit 0
