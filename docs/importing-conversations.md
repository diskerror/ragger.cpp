# Importing Past Conversations

Ragger can ingest your existing Claude conversation history — either from
Claude Code sessions stored under `~/.claude/projects/`, or from a claude.ai
"Export Data" archive — so past exchanges become searchable alongside new
ones. Original turn timestamps are preserved, so memory dates reflect when
the conversation actually happened rather than the import time.

The entry point is `scripts/import-claude-conversations.py` (Python 3.9+).

## Prerequisites

- Ragger daemon running (`ragger start`) if importing — not needed for
  file-output mode
- `~/.ragger/token` present (the installer bootstraps this automatically)
- `pip install requests` (only needed for `--import` mode)

## Source formats

| Format  | Flag             | What it parses                                                      |
|---------|------------------|---------------------------------------------------------------------|
| `code`  | `--format code`  | Claude Code JSONL files — single file or directory of them          |
| `web`   | `--format web`   | `conversations.json` from a claude.ai Settings → Export Data archive |

Both parsers pair each user turn with the next assistant text reply into a
single memory entry. Tool-use chains, thinking blocks, and other non-
conversational events are skipped so semantic search isn't polluted by
internal agent plumbing.

## Two modes

### Compile to a text file

Readable archive — useful for review, grep, or loading somewhere else:

~~~bash
./scripts/import-claude-conversations.py \
    --format code \
    --path ~/.claude/projects/-Volumes-WDBlack2-CLionProjects-Ragger \
    --output ~/claude-history.txt
~~~

Output is one block per exchange, chronological:

~~~
=== 2026-04-21T20:10:10.422Z  [claude-code:25ec068a] ===

User: There are two items in the Kanban priority board...
Assistant: I need some context before writing the prompts...
~~~

### Import into Ragger

Post every exchange to the running daemon. The daemon's embedder runs on
each text, so this may take a while for large histories:

~~~bash
./scripts/import-claude-conversations.py \
    --format code \
    --path ~/.claude/projects/-Volumes-WDBlack2-CLionProjects-Ragger \
    --import
~~~

By default it hits `http://localhost:8432` and reads the bearer token from
`~/.ragger/token`. Override with `--server` and `--token` if needed.

Under the hood the script POSTs to `/store` with a `metadata.timestamp`
field containing the original ISO-8601 timestamp of the user turn. The
backend honors that override and records the memory at the historical
time, so memory age-based features (recency weighting, fading memory
decay, cleanup windows) treat the imported turns correctly.

## Filters

Narrow down what gets imported or exported:

| Flag                      | Purpose                                       |
|---------------------------|-----------------------------------------------|
| `--session UUID`          | Keep only a single Claude Code session ID     |
| `--since YYYY-MM-DD`      | Only turns on or after this UTC date          |
| `--until YYYY-MM-DD`      | Only turns strictly before this UTC date      |

For `--format code` the session ID is the JSONL filename (a UUID). You can
get the list with `ls ~/.claude/projects/<slug>/`.

## Metadata written

Each imported memory lands with:

- `collection`: `memory`
- `category`: `conversation`
- `source`: `claude-code` or `claude-web`
- `session_id`: the original conversation UUID
- `timestamp`: the original turn time (replaces "now" at the DB layer)

This matches the metadata pattern used by Ragger Chat's own turn capture,
so imported history and new chat turns are interchangeable from a search
perspective.

## Caveats

- **Tool output is dropped.** Imports keep only the text of user questions
  and the first text block of each assistant reply. Bash output, file
  reads, and thinking blocks are not embedded — they would dominate the
  vector space without adding conversational signal.
- **The daemon must be rebuilt first** to honor `metadata.timestamp`.
  Older binaries ignore the override and record the import time instead.
  A `ragger version` check after rebuild is enough to confirm.
- **Pacing.** The script doesn't throttle. If the embedder falls behind,
  reduce load by importing in batches with `--since` / `--until`.
