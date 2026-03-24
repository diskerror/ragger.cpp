# Design Decisions

Key design choices and the reasoning behind them. Updated as the
project evolves.

## Philosophy: Better, Not Different

Ragger mirrors how human memory and thinking work — but with better
speed and reliability. We don't invent new paradigms; we replicate
familiar patterns (remembering, forgetting, searching, summarizing)
and make them faster and more dependable.

## Search Defaults to All Collections

`collections=None` searches everything. The original design filtered
to a single collection by default, which silently excluded reference
docs from results. Users import docs to find them later — hiding
them behind a filter defeats the purpose.

Both personal memories and reference docs are queried equally. No
artificial weighting — relevance scores do the filtering organically.

## Collections vs Categories

- **Collections** partition the search space (which databases to query)
- **Categories** classify within a collection (what kind of memory)

Collections are structural. Categories are semantic. You search
*across* collections but filter *by* category.

## Config Layering

Two config files, always:

1. **System config** (`/etc/ragger.ini` or `--config=`): Infrastructure
   settings. Loaded first.
2. **User config** (`~/.ragger/ragger.ini`): Personal preferences.
   Always read on top.

**SERVER_LOCKED keys** (host, port, db_path, log_dir, embedding model,
system ceilings) always come from the system config. Everything else:
user wins if set.

This means an admin can lock down infrastructure while letting users
customize their search weights, chat settings, and logging toggles.

### Auto-Bootstrap

First run creates `~/.ragger/` and writes a default config embedded
in the binary. No dependency on an example config file at runtime.
New users get a working setup without reading docs first.

## Log Directory: Always Server-Locked

One server instance = one log location. Even in no-share mode with
multiple users connecting, all server logs go to the same directory.
Tracking errors across multiple log files is painful — centralizing
them is worth the trade-off.

CLI commands log to `~/.ragger/` (the user's own space). Only the
daemon uses the system log dir.

## Weight Ratio Pattern

All config weight pairs use unnormalized values. Specify A and B
independently; runtime normalizes:

```
real_A = A / (A + B)
```

So `bm25_weight = 3` and `vector_weight = 7` means 30%/70%. This is
more intuitive than requiring values that sum to 1.0, and survives
one value changing without recalculating the other.

## System Ceilings

The `_clamp_to_ceiling()` pattern lets admins set maximum values that
user config can't exceed. A ceiling of 0 means "no limit imposed."

If a user sets a value of 0 (meaning "unlimited"), the ceiling still
applies — 0 becomes the ceiling value. This prevents users from
accidentally bypassing limits.

## "No-Share" vs "Shared" Mode

The `single_user` flag (better described as "no-share") controls
whether a common/shared memory database exists. It does **not** limit
the number of users — multiple users can each run their own instance
in no-share mode.

- **No-share** (`single_user = true`): Each user has their own
  `~/.ragger/memories.db`. No system-level shared database.
- **Shared** (`single_user = false`): A common database exists at
  the system level. Users have personal databases that are searched
  alongside the common one.

## "Common" Not "System" for Shared Memory

The shared database is called "common memory," not "system memory."
"System" sounds like hardware or OS internals. "Common" correctly
implies shared human knowledge.

## Bad Memories Are Marked, Not Deleted

When a memory is identified as wrong or harmful, it gets a `bad: true`
metadata flag rather than being deleted. This preserves institutional
memory of anti-patterns — knowing what *not* to do is as valuable as
knowing what to do.

## The "Keep" Tag

Memories with `"keep": true` in metadata are protected from deletion
(both single and batch). This is auto-set when memories are moved to
the common database, preventing accidental loss of curated shared
knowledge. Users can also set it manually for important personal memories.

## Token-Based Auth, Not Passwords (for API)

API clients authenticate with bearer tokens stored in `~/.ragger/token`.
Tokens are auto-generated on first run. This is simpler and more
secure than username/password for programmatic access.

Browser auth (for future web UI) uses hashed passwords + session
cookies — a separate mechanism for a different access pattern.

## Chat Scope: Memory-Aware Conversation Only

`ragger chat` stores turns, summarizes on pause/quit, and searches
memory for context. It is **not** an agent framework — no file editing,
no tool calling, no autonomous actions. That's the job of tools like
OpenClaw that use Ragger as their memory backend.

Clear separation: Ragger is the memory service. Agent frameworks are
the tool-wielders.

## Turn Cleanup via Summarization, Not Cron

Expired raw conversation turns are deleted when summaries are created
(on pause or quit), not by a separate cleanup job. This keeps the
lifecycle simple: turns exist until they're summarized, then they're
gone. No orphaned cleanup processes, no timing races.

## Persona Files Are Static Config

SOUL.md, USER.md, AGENTS.md, TOOLS.md are loaded at chat start and
not modified at runtime. They're configuration, not state. If they
need updating, that's done outside the chat session.

Priority order: SOUL > USER > AGENTS > TOOLS > MEMORY.

## Schema-Driven API Formats

Inference proxy format definitions live in JSON files (`formats/`),
not hardcoded logic. OpenAI format is the hardcoded fallback. This
makes adding new providers a data change, not a code change.

## Inference Proxy as Service-Level Feature

The inference proxy runs on a separate port (:8433) from the memory
API (:8432). This keeps concerns separate and allows them to be
independently secured, monitored, and scaled.

## Dynamic Context Sizing

Each inference endpoint can specify `max_context` (token limit).
Below `PERSONA_SIZING_THRESHOLD` (32768 tokens), persona content is
proportionally sized using `persona_pct`. This prevents persona
content from overwhelming small context windows while using full
persona on large ones.
