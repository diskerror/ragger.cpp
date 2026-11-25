# What capture records

Ragger's memory grows from two streams: **live capture** (your agent's
conversation turns, pushed by the Hermes plugin) and **imports** (documents
and conversation exports you bring in yourself). This page states exactly
what enters the database and what does not — read from the code, not from
aspiration.

## Live capture (Hermes plugin)

While the `capture_turns` setting is on, the plugin sends **one record per
completed turn**: the user text, the final assistant text, the model id and
the session id (`sync_turn` → `POST /turn`). A hook that knows when the user
prompt arrived may also send `started_at` (epoch seconds or a
`YYYY-MM-DD HH:MM:SS` timestamp); it is stored as `turns.started_on` and
episode timing then measures the idle gap from the end of one turn to the
start of the next instead of end-to-end (a long reply no longer counts as
idle time). Without it the column stays NULL and today's end-to-end rule
applies. That is the whole payload.

Three special cases:

- **Interrupted turns.** A turn you stop before the reply finishes is
  captured by the `post_turn_finalized` hook. That hook does not exist
  upstream — it comes from the core patch `scripts/install-hermes.sh`
  applies (re-run it after every `hermes update`, or interrupted turns stop
  being captured).
- **Stopped before any response was generated.** The user text alone is
  stored via `/store`, tagged `[incomplete prompt — no response generated]`,
  so it stays searchable and identifiable.
- **The built-in `memory` tool.** A `post_tool_call` hook forwards its
  add/replace calls to `/store`. This is the *only* tool traffic Ragger
  ever sees.

**Never sent:** thinking/reasoning blocks, discarded draft replies, tool
calls and their results (except the `memory` tool above). Ragger stores the
final text of each exchange — the reasoning that produced it stays in the
agent's own session log.

## The same-exchange rule (`store_turn`)

Hooks can fire twice for one exchange, and a multi-part reply arrives as
several posts. Before inserting, `store_turn` probes the **newest turn of
the same session with the same user text within 5 minutes** and merges
instead of duplicating:

| situation | result |
|---|---|
| the stored reply is empty (partial row) | filled in place |
| the new text is identical to, or contained in, the stored reply | nothing happens |
| the new text **starts with** the stored reply | replaces it (a superset — the reply grew) |
| anything else | **appended** after a blank line, to the same row |

The last row is a real limitation, stated plainly: **Ragger cannot tell a
rewritten draft from a multi-part reply.** If an agent re-answers the same
prompt with different wording inside the window, both texts end up in one
turn, joined by a blank line. Turn rows keep only the final assistant text
as far as the merge can know it — never a queue of drafts.

## Imports

Imported exchanges carry an explicit `source_timestamp`, and a timestamped
insert **never runs the same-exchange probe** (it reasons about "now", which
is meaningless for old data). Every imported exchange is therefore its own
row, even if two look identical. Documents (`import-docs`) are chunked and
indexed separately from conversation turns.

## What summarization adds

Captured turns are queued to the summarizer: turn summaries are
written per turn (short exchanges are copied verbatim rather than sent to a
model), and episodes group consecutive summaries of one conversation span.
When a captured turn later grows — a same-exchange continuation merges into
it — its turn summary is marked with a counter of the merges its text does
not reflect yet, and the daemon's periodic catch-up rewrites the summary in
place from the whole reply (a summary that already belongs to an episode is
left for the episode pass). See `docs/search-and-rag.md` for how the layers
are searched.
