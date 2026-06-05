# Importing Past Conversations

Ragger can ingest your existing Claude conversation history — either from
Claude Code sessions stored under `~/.claude/projects/`, or from a claude.ai
"Export Data" archive — so past exchanges become searchable alongside new
ones. Original turn timestamps are preserved, so memory dates reflect when
the conversation actually happened rather than the import time.

The entry point is `ragger import conversations`, a built-in subcommand of
the daemon binary. No Python or external dependencies are required.

## Prerequisites

- A working Ragger install (`~/.ragger/memories.db` initialised)
- The daemon does **not** need to be running — the importer writes
  directly to the same database

## Source formats

| Format  | Flag             | What it parses                                                       |
|---------|------------------|----------------------------------------------------------------------|
| `code`  | `--format=code`  | Claude Code JSONL files — single file or directory of them           |
| `web`   | `--format=web`   | `conversations.json` from a claude.ai Settings → Export Data archive |

Both parsers pair each user turn with the next assistant text reply into a
single memory entry. Tool-use chains, thinking blocks, and other
non-conversational events are skipped so semantic search isn't polluted by
internal agent plumbing.

## Basic use

~~~bash
ragger import conversations --format=code \
    ~/.claude/projects/-Volumes-WDBlack2-CLionProjects-Ragger

ragger import conversations --format=web \
    ~/Downloads/claude-export/conversations.json
~~~

Each exchange becomes one row in the `memories` table, embedded so it shows
up in semantic search. The metadata recorded on every imported row:

- `collection`: `memory`
- `category`: `conversation`
- `source`: `claude-code` or `claude-web`
- `session_id`: the original conversation UUID
- `timestamp`: the original turn time (replaces "now" at the DB layer)

This mirrors the metadata pattern used by Ragger's `capture_turn` pipeline,
so imported history and live-captured turns are interchangeable from a
search perspective.

## Filters

Narrow down what gets imported:

| Flag                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `--session=UUID`           | Keep only a single Claude Code session ID     |
| `--since=YYYY-MM-DD`       | Only turns on or after this date              |
| `--until=YYYY-MM-DD`       | Only turns strictly before this date          |

For `--format=code` the session ID is the JSONL filename (a UUID). You can
get the list with `ls ~/.claude/projects/<slug>/`.

## Caveats

- **Tool output is dropped.** Imports keep only the text of user questions
  and the first text block of each assistant reply. Bash output, file
  reads, and thinking blocks are not embedded — they would dominate the
  vector space without adding conversational signal.
- **Pacing.** The importer doesn't throttle. If the embedder falls behind
  on a huge archive, import in batches with `--since` / `--until`.

## See also

- [`ragger import summaries`](search-and-rag.md) — bulk-load hand-authored
  L4 project notes (a similar one-shot ingest path, but the rows go into
  `summaries` so they survive the raw-turn retention window).
