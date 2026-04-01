# Agent Instructions

Best practices for AI agents using Ragger as long-term memory.
Applies to OpenClaw, Claude, or any agent framework.

## Read Project Docs First

Before working on any project, the agent should read its documentation
(README, ROADMAP, CLAUDE.md, etc.) — not rely solely on memory search.
Memory stores summaries; project files have the authoritative details.

## Store Decisions Separately

Don't bury technical decisions inside session summaries. Library choices,
architecture decisions, and design rationale should be stored as their own
memory entries with specific tags so they're findable later:

```bash
# Bad: buried in a session summary
"Session 2026-03-17: discussed architecture, chose cpp-httplib for HTTP..."

# Good: standalone decision
"Ragger C++ HTTP server: migrated from Crow to cpp-httplib (2026-03-31)
for native SSE streaming. Rationale: Crow lacked chunked transfer support,
cpp-httplib provides set_chunked_content_provider() with identical routing API.
Single-header, no Boost.Asio dependency."
```

## Reference the Source

When storing a decision, include where the details live (file paths,
commit hashes). The memory entry is a pointer; the project file is
the source of truth.

## Usage Scenarios

**Solo developer + AI assistant (local):**
- One Ragger instance at `~/.ragger/memories.db`
- Agent stores conversation context, decisions, and lessons learned
- Import reference docs into named collections (`--collection docs`)
- Agent searches `memory` by default, reaches into `docs`/`reference`
  when the question calls for it

**Solo developer + multiple AI tools:**
- Same Ragger server (port 8432) shared across tools
- OpenClaw, CLI scripts, editor plugins all use the same HTTP API
- Collections separate concerns: `memory` for agent notes, `docs` for
  reference material, `work` for project-specific context

**Team / shared server:**
- Ragger runs as a system service with multi-user support
- Per-user memory via auth token → user isolation
- Shared reference collections available to all users
- Private memories stay private

**Offline / air-gapped:**
- Everything runs locally — no network calls
- Download the embedding model once, then disconnect
- SQLite database is a single file — easy to backup, move, or encrypt

**Development + production split:**
- Dev instance for experimentation, separate prod database
- Export/import via CLI for promoting curated memories
- Same binary, different `--db` paths

## Collection Strategy

Start simple and add collections as needs emerge:

| Stage | Collections |
|-------|-------------|
| Getting started | Just `memory` (the default) |
| Adding reference docs | `memory` + `docs` |
| Multiple doc sources | `memory` + `sibelius` + `orchestration` + ... |
| Team use | Per-user `memory` + shared `reference` |

The agent should know what collections exist and when to search them.
Store this knowledge as a memory entry so it persists across sessions.

## Conversation Memory Lifecycle

AI agents lose context between sessions and during compaction (context
window compression). Ragger can serve as persistent memory, but *how*
conversations get captured matters as much as *that* they're captured.

### The Problem

Raw conversation transcripts are verbose, full of false starts and
filler. Storing them verbatim dilutes search quality. But waiting too
long to summarize risks losing the conversation entirely (session
timeout, compaction, crash).

### A Practical Pattern

1. **Buffer** — Store conversation substance into a `conversation`
   collection as it happens. Keep it lightweight — decisions, questions,
   answers, not "Great question! Let me think about that."

2. **Summarize on pause** — After a period of inactivity (20 minutes
   works well), summarize the buffered conversation. Extract decisions,
   facts, lessons, and action items into proper memory entries in the
   `memory` collection with appropriate categories. Delete the raw
   conversation chunks.

3. **Summarize before compaction** — If the agent's context window is
   getting full, proactively summarize and store before the runtime
   compresses it. Compaction is lossy — it keeps what the summarizer
   thinks matters, which may not be what you care about later.

4. **Store decisions immediately** — Don't wait for summarization.
   Library choices, architecture decisions, design rationale — store
   these as they happen with specific tags. They're the most valuable
   and most easily lost.

### Open Questions

The right approach depends on your use case and is worth experimenting
with:

- **What's the right pause interval?** Too short and you're
  summarizing mid-thought. Too long and compaction gets there first.
  20 minutes is a starting point.

- **What level of detail to keep?** A summary of "we discussed the
  database schema" is useless. "Chose SQLite over Postgres because
  single-file deployment matters more than concurrent writes" is
  valuable. The summarizer needs enough context to know what matters.

- **Who summarizes?** The same agent that had the conversation has
  the best context. A separate summarization job has less context but
  can use a cheaper/faster model. Trade-offs either way.

- **How to handle multi-topic conversations?** A single conversation
  might cover three unrelated projects. Each topic should become its
  own memory entry with its own tags, not one monolithic summary.

- **Conversation collection search policy?** Should `conversation`
  be searched by default, or only explicitly? Raw buffer chunks are
  noisy — excluding them from default search keeps results clean
  while the data is still available when needed.

These are active design questions. If you find patterns that work well,
consider contributing them back.
