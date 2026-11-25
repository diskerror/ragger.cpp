#!/usr/bin/env bash
# summarizer-compare.sh — run summarizer-lab.sh on the SAME rows for several
# models and merge the results into one report, grouped by record.
#
# summarizer-lab.sh tries one model per run. This calls it once per model with
# identical item arguments, then merges the per-model text files so each record
# shows every model's verdict, timing, size and result side by side.
# Nothing is written to the database (the lab opens it read-only).
#
# Requires: bash, curl, jq, awk (plus what summarizer-lab.sh requires).
#
# Usage:
#   scripts/summarizer-compare.sh --models NAME[,NAME...|all] ITEM-ARGS [options]
#
#   --models LIST    comma-separated model names exactly as the endpoint
#                    serves them, or "all" for everything at /v1/models.
#   --outdir DIR     where to write (default:
#                    ~/ownCloud/Ragger/summarizer-lab/<timestamp>)
#   --no-warmup      skip the small warm-up call per model; without warm-up
#                    each model's first timed call includes llama-swap's
#                    cold-load time.
#   --unload-after   after each model's run, eject it from RAM with the
#                    LM Studio CLI (lms unload). LM Studio keeps loaded models
#                    resident, so without this several test models pile up and
#                    later loads fail on resources.
#
# Item and endpoint args are passed through to summarizer-lab.sh unchanged:
#   --text "raw"  --turns 1,2  --episodes 2086  --titles 4  --triggers 36
#   --host H  --port P  --api PATH  --db DB  --prompts settings|builtin
#
# Output in <outdir>:
#   <model-slug>.txt   one per model, exactly summarizer-lab.sh's format
#   compare.txt        merged report: summary table, then every record with
#                      the source once and each model's verdict/result under it
#
# Merge assumption: a section line "=== table id ===" in a lab output file
# starts a record; source/result text lines are not expected to match that
# pattern or the "--- source" / "--- result" markers.
#
# WARNING: if the endpoint also serves your own inference (e.g. an agent
# running on a model behind the same one-at-a-time proxy), loading other
# models here evicts that instance and the agent's current turn dies when
# the run finishes — and the eviction may be unrecoverable if the proxy's
# memory admission rejects reloading the original preset. Use a separate
# server/port for lab runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB="$SCRIPT_DIR/summarizer-lab.sh"

HOST=127.0.0.1
PORT=1234
MODELS=""
OUTDIR=""
WARMUP=1
UNLOAD_AFTER=0
PASS=()

die() { echo "summarizer-compare: $*" >&2; exit 1; }
usage() { sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ -x "$LAB" ] || die "lab script not found or not executable: $LAB"

while [ $# -gt 0 ]; do
    case "$1" in
        --models)    MODELS="$2"; shift 2 ;;
        --outdir)    OUTDIR="$2"; shift 2 ;;
        --no-warmup) WARMUP=0; shift ;;
        --unload-after) UNLOAD_AFTER=1; shift ;;
        --host)      HOST="$2";  PASS+=("$1" "$2"); shift 2 ;;
        --port)      PORT="$2";  PASS+=("$1" "$2"); shift 2 ;;
        --api|--db|--prompts|--text|--turns|--episodes|--titles|--triggers)
                     PASS+=("$1" "$2"); shift 2 ;;
        -h|--help)   usage 0 ;;
        *)           die "unknown option: $1 (see --help)" ;;
    esac
done

for t in curl jq awk; do
    command -v "$t" >/dev/null || die "$t is required"
done

[ -n "$MODELS" ] || die "--models is required (comma-separated names, or 'all')"

has_item=0
for a in ${PASS[@]+"${PASS[@]}"}; do
    case "$a" in
        --text|--turns|--episodes|--titles|--triggers) has_item=1 ;;
    esac
done
[ "$has_item" = 1 ] || die "give at least one item arg: --text, --turns, --episodes, --titles or --triggers"

BASE="http://$HOST:$PORT"

# Resolve "all" to every model the endpoint serves.
if [ "$MODELS" = all ]; then
    MODELS="$(curl -sS -m 10 "$BASE/v1/models" | jq -r '.data[].id' | paste -sd, -)" \
        || die "could not list models at $BASE/v1/models"
    [ -n "$MODELS" ] || die "endpoint serves no models"
    echo "models (all): $MODELS" >&2
fi

if [ -z "$OUTDIR" ]; then
    OUTDIR="$HOME/ownCloud/Ragger/summarizer-lab/$(date +%Y-%m-%d-%H%M%S)"
fi
mkdir -p "$OUTDIR" || die "cannot create $OUTDIR"

warmup() {   # one tiny call so llama-swap loads the model before timing
    local m="$1" body r
    body="$(jq -n --arg m "$m" \
        '{model:$m, max_tokens:8, stream:false, messages:[{role:"user",content:"ping"}]}')"
    r="$(curl -sS -m 600 -o /dev/null -w '%{http_code} %{time_total}' \
        -H 'Content-Type: application/json' --data-binary "$body" "$BASE$API_PATH" 2>&1)" \
        || { echo "  warm-up failed: $r (continuing)" >&2; return; }
    if [ "${r%% *}" = 200 ]; then
        echo "  warm-up: ${r#* } s" >&2
    else
        echo "  warm-up: HTTP ${r%% *} (model may not load; continuing)" >&2
    fi
}

unload() {   # eject a model from RAM (LM Studio keeps loaded models resident)
    local m="$1" r lms_bin loaded
    lms_bin="$(command -v lms || true)"
    [ -n "$lms_bin" ] || [ ! -x "$HOME/.lmstudio/bin/lms" ] || lms_bin="$HOME/.lmstudio/bin/lms"
    if [ -z "$lms_bin" ]; then
        echo "  unload: lms CLI not found; model left loaded" >&2
        return
    fi
    # LM Studio may register the instance as "<name>@<quant>" once more than
    # one quantization of the model exists on disk, so match both forms.
    loaded="$(curl -sS -m 5 "$BASE/api/v0/models" 2>/dev/null \
        | jq -r --arg m "$m" '.data[]
            | select(.state != "not-loaded")
            | select(.id == $m or (.id | startswith($m + "@")))
            | .id')"
    if [ -z "$loaded" ]; then
        echo "  unload: $m not reported loaded" >&2
        return
    fi
    while IFS= read -r id; do
        [ -z "$id" ] && continue
        r="$("$lms_bin" unload "$id" 2>&1)" && echo "  unload: $r" >&2 \
            || echo "  unload failed ($id): $r (continuing)" >&2
    done <<< "$loaded"
}

# --api default matches the lab's; only needed for the warm-up URL.
API_PATH=/v1/chat/completions
prev=""
for a in ${PASS[@]+"${PASS[@]}"}; do
    [ "$prev" = "--api" ] && API_PATH="$a"
    prev="$a"
done

FILES=()
NAMES=()
IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
for m in "${MODEL_ARR[@]}"; do
    m="${m// /}"
    [ -z "$m" ] && continue
    case "$m" in
        *"|"*) die "model name may not contain '|': $m" ;;
    esac
    slug="$(printf '%s' "$m" | tr '/ ' '__')"
    echo "==> $m" >&2
    if [ "$WARMUP" = 1 ]; then warmup "$m"; fi
    if ! "$LAB" --model "$m" --output "$OUTDIR/$slug.txt" ${PASS[@]+"${PASS[@]}"}; then
        echo "  summarizer-lab.sh exited non-zero for $m; continuing" >&2
    fi
    if [ "$UNLOAD_AFTER" = 1 ]; then unload "$m"; fi
    [ -f "$OUTDIR/$slug.txt" ] || { echo "  no output file for $m; excluded from merge" >&2; continue; }
    FILES+=("$OUTDIR/$slug.txt")
    NAMES+=("$m")
done

[ "${#FILES[@]}" -gt 0 ] || die "no model produced any output"

MERGED="$OUTDIR/compare.txt"

{
    echo "summarizer-compare  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "endpoint:   $BASE$API_PATH"
    echo "models:     ${NAMES[*]}"
    echo "item args:  ${PASS[*]}"
    echo
    awk -v models="$(printf '%s|' "${NAMES[@]}" | sed 's/|$//')" \
        -v base="$BASE$API_PATH" '
    function save() {
        if (hdr == "") return
        if (!(hdr in seenh)) { seenh[hdr] = 1; keys[++nk] = hdr; source[hdr] = csrc }
        meta[hdr SUBSEP mname[fi]] = cmeta
        result[hdr SUBSEP mname[fi]] = cres
    }
    function cell(m,   lines, nl, i, v, t, w, sv) {
        nl = split(m, lines, "\n")
        v = "?"; t = "?"; w = "?"
        for (i = 1; i <= nl; i++) {
            if (lines[i] ~ /^verdict:/)      { v = lines[i]; sub(/^verdict:[ ]*/, "", v) }
            else if (lines[i] ~ /^time:/)    { t = lines[i]; sub(/^time:[ ]*/, "", t); sub(/ s$/, "", t) }
            else if (lines[i] ~ /^size:/ && match(lines[i], /result [0-9]+ chars \/ [0-9]+ words/)) {
                w = substr(lines[i], RSTART, RLENGTH)
                sub(/^result [0-9]+ chars \/ /, "", w); sub(/ words$/, "", w)
            }
        }
        sv = "ok"
        if      (v ~ /^cut off/)     sv = "cut"
        else if (v ~ /^not shorter/) sv = "long"
        else if (v ~ /^error/)       sv = "err"
        return sv " " t "s " w "w"
    }
    function report(   i, j, key, line, mm) {
        print "== summary =="
        print "endpoint: " base
        mm = ""
        for (j = 1; j <= nm; j++) mm = mm (j > 1 ? ", " : "") mname[j]
        print "models:   " mm
        print ""
        line = sprintf("%-16s", "record")
        for (j = 1; j <= nm; j++) line = line sprintf("%-20s", substr(mname[j], 1, 19))
        print line
        for (i = 1; i <= nk; i++) {
            key = keys[i]
            line = sprintf("%-16s", key)
            for (j = 1; j <= nm; j++) {
                if ((key SUBSEP mname[j]) in meta)
                    line = line sprintf("%-20s", cell(meta[key SUBSEP mname[j]]))
                else
                    line = line sprintf("%-20s", "(none)")
            }
            print line
        }
        print ""
        print "== details =="
        for (i = 1; i <= nk; i++) {
            key = keys[i]
            print ""
            print "################ " key " ################"
            print "--- source"
            printf "%s", source[key]
            for (j = 1; j <= nm; j++) {
                print "--- " mname[j]
                if ((key SUBSEP mname[j]) in meta) {
                    printf "%s", meta[key SUBSEP mname[j]]
                    print "--- result"
                    printf "%s", result[key SUBSEP mname[j]]
                } else
                    print "(no result for this model)"
            }
        }
    }
    BEGIN { nm = split(models, mname, "|"); fi = 0; lastfile = "" }
    FILENAME != lastfile { save(); fi++; lastfile = FILENAME; hdr = "" }
    /^=== .* ===$/ {
        save()
        hdr = substr($0, 5, length($0) - 8)
        state = "meta"; cmeta = ""; csrc = ""; cres = ""
        next
    }
    hdr == "" { next }
    state == "meta" && /^--- source$/ { state = "source"; next }
    state == "meta"    { cmeta = cmeta $0 "\n"; next }
    state == "source" && /^--- result$/ { state = "result"; next }
    state == "source"  { csrc = csrc $0 "\n"; next }
                         { cres = cres $0 "\n" }
    END { save(); report() }
    ' "${FILES[@]}"
} > "$MERGED"

echo >&2
echo "summary:" >&2
awk '/^== summary ==/{f=1} /^$/{if(f&&++b>2)exit} f' "$MERGED" >&2
echo >&2
echo "merged report: $MERGED" >&2
echo "per-model files: $OUTDIR" >&2
