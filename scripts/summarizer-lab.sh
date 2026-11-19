#!/usr/bin/env bash
# summarizer-lab.sh — try a summarizer model on real Ragger rows, read-only.
#
# Sends the same prompts and input Ragger would send for a turn summary, a
# session title or a knowledge trigger to an OpenAI-compatible endpoint
# (LM Studio, llama-server, ...) and writes source, result, finish reason
# and time to a text file, so several models can be compared side by side.
# Nothing is written to the database (it is opened read-only).
#
# Requires: bash, curl, jq, sqlite3.
#
# Usage:
#   scripts/summarizer-lab.sh --model NAME --output FILE \
#       [--text "raw text"] [--turns 1,2,42] [--titles 4,9] [--triggers 36,5] \
#       [--host 127.0.0.1] [--port 1234] [--api /v1/chat/completions] \
#       [--db ~/.ragger/memories] [--prompts settings|builtin]
#
#   --model omitted or empty: list the endpoint's model names and exit.
#   --prompts settings (default): use the prompts saved in the database's
#     settings table where set, else the built-ins — what the daemon uses.
#     builtin: always the built-in defaults from include/lang/en.h.
#   --titles takes session_id values; --triggers takes knowledge_id values.
#
# Differences from the daemon, on purpose: every item is sent to the model,
# including short turns Ragger would copy without a call (marked "floor" in
# the output), and the result is shown even when Ragger would reject it
# (longer than the source, or cut off at the token limit).

set -euo pipefail
export LC_ALL=C   # byte lengths, like the C++ std::string::size()

HOST=127.0.0.1
PORT=1234
API=/v1/chat/completions
DB="$HOME/.ragger/memories"
MODEL=""
OUTPUT=""
TEXT=""
TURNS=""
TITLES=""
TRIGGERS=""
PROMPTS=settings

die() { echo "summarizer-lab: $*" >&2; exit 1; }
usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --host)     HOST="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --api)      API="$2"; shift 2 ;;
        --db)       DB="$2"; shift 2 ;;
        --model)    MODEL="${2-}"; shift; [ $# -gt 0 ] && shift ;;
        --output)   OUTPUT="$2"; shift 2 ;;
        --text)     TEXT="$2"; shift 2 ;;
        --turns)    TURNS="$2"; shift 2 ;;
        --titles)   TITLES="$2"; shift 2 ;;
        --triggers) TRIGGERS="$2"; shift 2 ;;
        --prompts)  PROMPTS="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          die "unknown option: $1 (see --help)" ;;
    esac
done

for t in curl jq sqlite3; do
    command -v "$t" >/dev/null || die "$t is required"
done

BASE="http://$HOST:$PORT"

# No model: list what the endpoint serves.
if [ -z "$MODEL" ]; then
    curl -sS -m 10 "$BASE/v1/models" | jq -r '.data[].id' \
        || die "could not list models at $BASE/v1/models"
    exit 0
fi

[ -n "$OUTPUT" ] || die "--output is required"
[ -n "$TEXT$TURNS$TITLES$TRIGGERS" ] || die "give at least one of --text, --turns, --titles, --triggers"
case "$PROMPTS" in settings|builtin) ;; *) die "--prompts must be settings or builtin" ;; esac

# The database: accept the name with or without .db.
DB="${DB/#\~/$HOME}"
[ -f "$DB" ] || [ ! -f "$DB.db" ] || DB="$DB.db"
if [ -n "$TURNS$TITLES$TRIGGERS" ] || [ "$PROMPTS" = settings ]; then
    [ -f "$DB" ] || die "database not found: $DB"
fi

sql() { sqlite3 -readonly -batch -noheader "$DB" "$1"; }

ids() {   # validate a comma list of integers, print one per line
    local list="$1" id
    IFS=',' read -r -a arr <<< "$list"
    for id in "${arr[@]}"; do
        id="${id// /}"
        [ -z "$id" ] && continue
        [[ "$id" =~ ^[0-9]+$ ]] || die "not a row id: $id"
        echo "$id"
    done
}

# ---- settings and prompts -------------------------------------------------

setting() {   # setting KEY -> value or "" (no database = "")
    [ -f "$DB" ] || { echo ""; return; }
    sql "SELECT value FROM settings WHERE key = '$1';"
}

EN_H="$(cd "$(dirname "$0")/.." && pwd)/include/lang/en.h"

builtin_prompt() {   # builtin_prompt PROMPT_NAME -> text from en.h
    [ -f "$EN_H" ] || die "built-in prompts not found: $EN_H"
    sed -n "/inline constexpr std::string_view $1 =/,/;[[:space:]]*\$/p" "$EN_H" \
        | sed '1d' \
        | sed -E 's/^[^"]*"//; s/"[;]?[[:space:]]*$//' \
        | tr -d '\n' \
        | sed -e 's/\\"/"/g' -e 's/\\\\/\\/g'
}

prompt_for() {   # prompt_for SETTING_KEY PROMPT_NAME
    local v=""
    if [ "$PROMPTS" = settings ]; then v="$(setting "$1")"; fi
    if [ -z "${v//[[:space:]]/}" ]; then builtin_prompt "$2"; else printf '%s' "$v"; fi
}

TURN_PROMPT="$(prompt_for summarizer_prompt PROMPT_TURN_SUMMARY)"
TITLE_PROMPT="$(prompt_for summarizer_title_prompt PROMPT_SESSION_TITLE)"
TRIGGER_PROMPT="$(prompt_for summarizer_trigger_prompt PROMPT_KNOWLEDGE_TRIGGER)"

num_setting() {   # num_setting KEY DEFAULT -> positive int or DEFAULT
    local v; v="$(setting "$1")"
    if [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ]; then echo "$v"; else echo "$2"; fi
}
TARGET_PCT="$(num_setting summarizer_target_pct 30)"
MAX_PCT="$(num_setting summarizer_max_pct 60)"
MAX_TOKENS="$(num_setting summarizer_max_tokens 1024)"
MIN_WORDS="$(setting summarizer_min_words)"; [[ "$MIN_WORDS" =~ ^[0-9]+$ ]] || MIN_WORDS=20

# ---- helpers ---------------------------------------------------------------

bytes() { printf '%s' "$1" | wc -c | tr -d ' '; }
words() { printf '%s' "$1" | wc -w | tr -d ' '; }

# Same markers as kSystemMarkers in src/summarizer.cpp.
system_injected() {
    local body="${1#"${1%%[![:space:]]*}"}"
    case "$body" in
        "[System note:"*|"[Note: model was just switched"*|"[CONTEXT COMPACTION"*|"[IMPORTANT: Background process"*) return 0 ;;
    esac
    return 1
}

turn_system_prompt() {   # turn_system_prompt SOURCE_BYTES -> prompt with sizes filled
    local n="$1" t c p
    t=$(( n * TARGET_PCT / 100 )); [ "$t" -lt 40 ] && t=40
    c=$(( n * MAX_PCT / 100 ));    [ "$c" -lt "$t" ] && c="$t"
    p="$TURN_PROMPT"
    p="${p/\{\}/$t}"
    p="${p/\{\}/$c}"
    printf '%s' "$p"
}

# call SYSTEM USER -> sets REPLY_TEXT REPLY_FINISH REPLY_TIME REPLY_ERROR
call() {
    local body resp http
    body="$(jq -n --arg m "$MODEL" --arg s "$1" --arg u "$2" --argjson mt "$MAX_TOKENS" \
        '{model:$m, max_tokens:$mt, stream:false,
          messages:([if $s == "" then empty else {role:"system",content:$s} end] + [{role:"user",content:$u}])}')"
    resp="$(mktemp "${TMPDIR:-/tmp}/sumlab.XXXXXX")"
    REPLY_ERROR=""
    http="$(curl -sS -m 600 -o "$resp" -w '%{http_code} %{time_total}' \
        -H 'Content-Type: application/json' --data-binary "$body" "$BASE$API" 2>&1)" \
        || { REPLY_ERROR="curl: $http"; REPLY_TEXT=""; REPLY_FINISH=""; REPLY_TIME=""; rm -f "$resp"; return; }
    REPLY_TIME="${http#* }"
    if [ "${http%% *}" != 200 ]; then
        REPLY_ERROR="HTTP ${http%% *}: $(head -c 300 "$resp")"
        REPLY_TEXT=""; REPLY_FINISH=""
    else
        REPLY_TEXT="$(jq -r '(.choices[0].message.content // "")
            | gsub("<think>[\\s\\S]*?</think>"; "")
            | sub("^\\s+"; "") | sub("\\s+$"; "")' "$resp")"
        REPLY_FINISH="$(jq -r '.choices[0].finish_reason // ""' "$resp")"
    fi
    rm -f "$resp"
}

COUNT=0
emit() {   # emit TABLE ID SOURCE NOTE
    local src="$3" note="$4" verdict=""
    COUNT=$((COUNT + 1))
    if [ -n "$REPLY_ERROR" ]; then
        verdict="error"
    elif [ "$REPLY_FINISH" = length ]; then
        verdict="cut off at max_tokens: Ragger would not store it"
    elif [ "$(words "$REPLY_TEXT")" -ge "$(words "$src")" ]; then
        verdict="not shorter than the source: Ragger would keep the source"
    else
        verdict="ok"
    fi
    {
        echo "=== $1 $2 ==="
        [ -n "$note" ] && echo "note:      $note"
        echo "time:      ${REPLY_TIME:-?} s"
        echo "finish:    ${REPLY_FINISH:-?}"
        echo "size:      source $(bytes "$src") chars / $(words "$src") words; result $(bytes "$REPLY_TEXT") chars / $(words "$REPLY_TEXT") words"
        echo "verdict:   $verdict"
        [ -n "$REPLY_ERROR" ] && echo "error:     $REPLY_ERROR"
        echo "--- source"
        printf '%s\n' "$src"
        echo "--- result"
        printf '%s\n' "$REPLY_TEXT"
        echo
    } >> "$OUTPUT"
    echo "  $1 $2: $verdict (${REPLY_TIME:-?} s)" >&2
}

# ---- run -------------------------------------------------------------------

{
    echo "summarizer-lab  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "model:      $MODEL"
    echo "endpoint:   $BASE$API"
    echo "database:   $DB (read-only)"
    echo "prompts:    $PROMPTS"
    echo "max_tokens: $MAX_TOKENS   target/max %: $TARGET_PCT/$MAX_PCT   min words floor: $MIN_WORDS"
    echo
} > "$OUTPUT"

if [ -n "$TEXT" ]; then
    call "$(turn_system_prompt "$(bytes "$TEXT")")" "$TEXT"
    emit text - "$TEXT" "sent as the user message with the turn prompt"
fi

for id in $(ids "$TURNS"); do
    [ "$(sql "SELECT count(*) FROM turns WHERE turn_id = $id;")" = 1 ] || { echo "  turns $id: not found" >&2; continue; }
    u="$(sql "SELECT COALESCE(user_text,'') FROM turns WHERE turn_id = $id;")"
    a="$(sql "SELECT COALESCE(assistant_text,'') FROM turns WHERE turn_id = $id;")"
    system_injected "$u" && u=""
    # The source as Ragger measures it, and the conversation as it sends it.
    if [ -z "$u" ]; then src="$a"; conv="**Assistant:** $a"$'\n\n'
    else src="$u | $a"; conv="**User:** $u"$'\n\n'"**Assistant:** $a"$'\n\n'; fi
    note=""
    n=$(( $(bytes "$u") + $(bytes "$a") ))
    if [ "$(words "$src")" -le "$MIN_WORDS" ] || [ "$n" -lt 120 ]; then
        note="floor: Ragger would copy this turn without calling the model"
    fi
    call "$(turn_system_prompt "$n")" "$conv"
    emit turns "$id" "$src" "$note"
done

for id in $(ids "$TITLES"); do
    corpus="$(sql "SELECT group_concat(text, char(10)||char(10)) FROM (
                     SELECT text FROM turn_summaries
                     WHERE session_id = $id AND text IS NOT NULL AND text <> ''
                     ORDER BY turn_datetime, turn_summary_id);")"
    if [ -z "$corpus" ]; then echo "  sessions $id: no turn summaries" >&2; continue; fi
    name="$(sql "SELECT COALESCE(name,'') || ' [' || COALESCE(name_source,'') || ']' FROM sessions WHERE session_id = $id;")"
    call "$TITLE_PROMPT" "$corpus"
    emit sessions "$id" "$corpus" "current name: $name"
done

for id in $(ids "$TRIGGERS"); do
    text="$(sql "SELECT COALESCE(text,'') FROM knowledge WHERE knowledge_id = $id;")"
    if [ -z "$text" ]; then echo "  knowledge $id: not found or empty" >&2; continue; fi
    meta="$(sql "SELECT kind || ', ' || status || ', trigger: ' || COALESCE(NULLIF(trigger,''),'(none)') FROM knowledge WHERE knowledge_id = $id;")"
    call "$TRIGGER_PROMPT" "$text"
    emit knowledge "$id" "$text" "$meta"
done

echo "$COUNT item(s) written to $OUTPUT" >&2
