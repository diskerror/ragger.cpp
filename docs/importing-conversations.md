# Importing Past Conversations

Ragger can ingest your existing conversation history — Claude Code sessions,
a claude.ai "Export Data" archive, or a Telegram chat export — so past
exchanges become searchable alongside new ones. Original timestamps are
preserved wherever the source provides them, so memory dates reflect when
the conversation actually happened rather than the import time.

The entry point is `ragger import-conversations`, a built-in subcommand of
the daemon binary. No Python or external dependencies are required.

> **Deprecated:** the old `ragger import conversations --format=...` spelling
> and the general-purpose `ragger import <file>` are both gone as
> catch-all verbs. Document import is now `ragger import-docs`; conversation
> import is `ragger import-conversations`. A bare `ragger import <file>` is
> kept working as a deprecated alias for `import-docs` (it prints a warning).

## Prerequisites

- A working Ragger install (`~/.ragger/memories.db` initialised)
- The daemon does **not** need to be running — the importer writes
  directly to the same database (it takes out its own write lock with a
  busy-timeout, so it's safe to run alongside a live daemon, just slower)

## Source formats — auto-detected

You normally don't need to specify a format. `import-conversations` sniffs
the input file's JSON structure and picks the right parser:

| Format     | Detected from                                                          |
|------------|-------------------------------------------------------------------------|
| `web`      | Claude.ai `conversations.json` — array/`{"conversations":[...]}` of objects with `chat_messages`/`messages` + `uuid` |
| `code`     | Claude Code JSONL — lines shaped `{"type":"user"\|"assistant","message":{...}}` |
| `telegram` | Telegram export JSON — top-level `messages[]`, or `chats.list[]` for a full multi-chat export |

`--format=code|web|telegram` is available as an override for the rare
ambiguous file, but is not required for any of the exports above.

A file that looks like a Claude `memories.json` (a narrative blob with
`conversations_memory` + `account_uuid`, no per-message timestamps) is
**rejected** if handed to `import-conversations` on its own without a date
flag — see [Memories](#memories-decisions) below.

Both `code` and `web` parsers pair each user turn with the next assistant
text reply into a single memory entry. Tool-use chains, thinking blocks, and
other non-conversational events are skipped so semantic search isn't
polluted by internal agent plumbing. The `telegram` parser merges
consecutive same-sender bubbles into one turn and splits sessions at each
`/new` command the user sent to the live agent (the `/new` message itself is
dropped); `--self="Your Display Name"` is required so it knows which side
of the chat is you.

## Basic use

~~~bash
# Claude Code JSONL (single file or a directory of them)
ragger import-conversations ~/.claude/projects/-my-project

# claude.ai web export — just the conversations file
ragger import-conversations ~/Downloads/claude-export/conversations.json

# Telegram export (single-chat result.json or full multi-chat export)
ragger import-conversations tg_export.json --format=telegram --self="Jane Doe"
~~~

Each exchange becomes one row in the `turns` table (L1), with an
auto-generated L2 placeholder summary — mirroring exactly what live capture
writes, so imported history and live-captured turns are interchangeable
from a search perspective. Recorded per row:

- `model_name`: `<source>-import` (e.g. `web-import`, `telegram-import`)
- `session_id`: the original conversation UUID (code/web) or a synthetic
  `/new`-boundary GUID (telegram)
- `created_at`: the original turn time (not the import time)

Re-running an import is idempotent: a ±30s fuzzy match on `(user_text,
timestamp)` skips exchanges already present, whether from a previous import
run or from live capture.

### Cross-source session-id reconciliation

Telegram exports carry only synthetic session GUIDs (there's no real
session concept in a chat export). If you've imported the same
conversation from *both* Telegram (pasted manually) and a Claude/Code
export, `import-conversations` reconciles them automatically, regardless of
which one you import first:

- Importing **Claude/Code data** after a Telegram import: any exchange that
  text-matches an existing Telegram-sourced row gets that row's session id
  upgraded in place to the real, authoritative one — no duplicate row.
- Importing **Telegram data** after (or before) a Claude/Code import: at the
  end of the run, Telegram's own newly-inserted rows are checked against the
  rest of the table and adopt a matching non-synthetic session id if found.

Only synthetic session ids (empty, `telegram-import*`, or anything
containing `test`/`debug`/`synthetic`) are ever overwritten this way — a
real session id an agent assigned is never touched.

## Whole-export import: `--all`

A claude.ai "Export Data" download is a directory containing
`conversations.json`, optionally `memories.json`, and a `projects/`
directory. `--all` treats that whole set as **one** import:

~~~bash
ragger import-conversations ~/Downloads/claude-export --all
~~~

`--all` is required any time multiple files are being pulled into one
import — this is deliberate, there's no implicit default:

- **Directory input always requires `--all`.** Pointing at a bare directory
  without it is an error telling you to add the flag.
- **A single `conversations.json` file that has export siblings sitting next
  to it** (a `memories.json` and/or a `projects/` folder in the same
  directory) also requires `--all` if you want those siblings pulled in.
  Passing just the file without `--all` imports **only** the conversations
  and errors instead of silently ignoring the siblings.
- Passing a `conversations.json` **alone in its own directory** (no
  siblings present) needs no flag at all.

With `--all`, in addition to the turns described above:

- Each conversation's own `summary` field (Claude's own per-conversation
  digest) is stored as one L3 `summaries` row (`level=session`,
  `tags=session`), dated by that conversation's real `created_at` — no
  re-summarization needed, it's already good.
- Each file under `projects/*.json` contributes one L4 `summaries` row
  (`level=project`, `tags=claude-project`) built from that project's `name`
  + `description`, dated by the project's own `created_at`.
- `memories.json`, if present, is imported as described next.

## Memories (decisions)

Claude's `memories.json` is a single narrative blob — Claude's own rolling
summary of who you are and what you've been working on — with **no
per-message timestamps of its own**. `import-conversations` handles it two
ways:

**As part of `--all`:** the file's date needs no flag — it's still under
active development to derive per-section dates from the accompanying
`conversations.json` (topically matching narrative chunks to the
conversations they describe). For now the whole blob is stored as a single
`decisions` row (L5/L6, `tags=claude-memory`), dated by the file's mtime.
`--fdate`/`--date=` are **ignored** in this path (the file's own mtime is
used automatically) — they only apply to the standalone case below.

**As a standalone file** (memories.json passed by itself, not via `--all`):
there's no conversations.json alongside it to derive a date from at all, so
one of these is **required**:

~~~bash
# Use the file's own last-modified time
ragger import-conversations ~/Downloads/memories.json --fdate

# Or specify an explicit date
ragger import-conversations ~/Downloads/memories.json --date=2026-06-08
~~~

Without one of these two flags, a standalone `memories.json` is rejected
with an explanatory error rather than guessed at. Re-running either form is
idempotent — the same (text, timestamp) pair is not inserted twice.

If you'd rather not import memories.json as a `decisions` row at all,
`ragger import-docs` will happily chunk it as a plain document instead
(see [Importing documents](#) — but note `import-docs` does **not**
special-case JSON; convert it to Markdown first with a script or a tool
like [docling](https://github.com/docling-project/docling) and inspect the
result before importing, since guessing which JSON strings are prose is
exactly the kind of implicit heuristic that can silently mangle data).

## Filters

Narrow down what gets imported:

| Flag                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `--session=ID`             | Keep only a single session id                 |
| `--since=YYYY-MM-DD`       | Only turns on or after this date              |
| `--until=YYYY-MM-DD`       | Only turns strictly before this date          |

For `--format=code` the session id is the JSONL filename (a UUID). You can
get the list with `ls ~/.claude/projects/<slug>/`.

## Caveats

- **Tool output is dropped.** Imports keep only the text of user questions
  and the first text block of each assistant reply. Bash output, file
  reads, and thinking blocks are not embedded — they would dominate the
  vector space without adding conversational signal.
- **Pacing.** The importer doesn't throttle. If the embedder falls behind
  on a huge archive, import in batches with `--since`/`--until`, or stop the
  daemon first (`ragger stop`) so the importer has the DB to itself.
- **memories.json splitting is a work in progress.** Today it lands as one
  undifferentiated `decisions` row per file. A finer per-section split
  (with per-chunk dates derived from topically-matched conversations) is
  planned but not yet wired into the live import path.

## See also

- `ragger import-docs` — general markdown/text import (paragraph-aware
  chunking) into L5 documents. Not conversation-aware; use
  `import-conversations` for chat history.
- `ragger import-conversations summaries <file> [file...]` /
  `--jsonl=FILE` — bulk-load hand-authored L4 project notes directly (a
  simpler one-shot ingest path for notes you wrote yourself, as opposed to
  conversation exports).
