#!/bin/bash
# pin-deps.sh — freeze fetched dependencies at the commits your last
# DEPS_LATEST build used. Run just before tagging a release, then commit
# cmake/pins.cmake so the tag carries the exact versions it was tested with.
#
#   cmake -B build -DDEPS_LATEST=ON && cmake --build build && ctest   # test against latest
#   scripts/pin-deps.sh [build-dir]                                   # default: build
#   git add cmake/pins.cmake && git commit -m "Pin dependencies"
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${1:-build}"
RESOLVED="$BUILD_DIR/deps-resolved.cmake"
PINS="cmake/pins.cmake"

if [[ ! -f "$RESOLVED" ]]; then
    echo "No $RESOLVED — configure with -DDEPS_LATEST=ON first." >&2
    exit 1
fi

changed=0
total=0
while read -r key sha; do
    old=$(sed -nE "s/^set\(${key}[[:space:]]+([0-9a-f]{40}).*/\1/p" "$PINS")
    if [[ -z "$old" ]]; then
        echo "warning: $key not in $PINS — skipped" >&2
        continue
    fi
    total=$((total + 1))
    if [[ "$old" == "$sha" ]]; then
        echo "  $key: unchanged (${sha:0:12})"
    else
        sed -E -i.bak "s|^(set\(${key}[[:space:]]+)[0-9a-f]{40}|\1${sha}|" "$PINS"
        echo "  $key: ${old:0:12} -> ${sha:0:12}"
        changed=$((changed + 1))
    fi
done < <(sed -nE 's/^set\(([A-Z0-9_]+_SHA) ([0-9a-f]{40})\)$/\1 \2/p' "$RESOLVED")
rm -f "$PINS.bak"

echo "$changed of $total pin(s) changed in $PINS"
[[ $changed -gt 0 ]] && echo "Review, then: git add $PINS && git commit -m \"Pin dependencies\"" || true
