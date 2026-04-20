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

## Config: One File, One Owner

There's a single config file: `~/.ragger/settings.ini`. No layering,
no system vs. user split, no SERVER_LOCKED overlay. The person who
installs Ragger owns the daemon and owns the config.

This was deliberately simplified. The old two-file design assumed a
root-installed system daemon serving OS users, which no longer matches
how Ragger is deployed — it now installs per-user under `~/.ragger/`
with no sudo. With one install, one owner, and one config, layering
became ceremony without benefit.

Sub-users (provisioned via `ragger useradd` and authenticating over
HTTP with tokens) don't have their own config files. They're clients
of the daemon, not co-owners of it. The daemon owner sets ceilings in
`settings.ini` to cap what those clients can request — see "System
Ceilings" below.

### Auto-Bootstrap

First run copies `example-settings.ini` to `~/.ragger/settings.ini`.
New users get a working setup without reading docs first.

## Log Directory

All daemon logs go to `~/.ragger/logs/`. CLI commands log to the same
place. One install, one log directory — no split between "system" and
"user" logs.

## Weight Ratio Pattern

All config weight pairs use unnormalized values. Specify A and B
independently; runtime normalizes:

```
real_A = A / (A + B)
```

So `bm25_weight = 3` and `vector_weight = 7` means 30%/70%. This is
more intuitive than requiring values that sum to 1.0, and survives
one value changing without recalculating the other.

## Ceilings

The `_clamp_to_ceiling()` pattern lets the daemon owner cap values
that HTTP clients (sub-users) request. A ceiling of 0 means "no limit
imposed."

If a client passes a value of 0 (meaning "unlimited"), the ceiling
still applies — 0 is clamped to the ceiling value. This prevents a
client from accidentally bypassing limits by omitting the parameter.

## One Database Per Install

Every install has exactly one database: `~/.ragger/memories.db` under
the daemon owner's home directory. When sub-users are provisioned with
tokens, they read and write the same database — the daemon enforces
per-user isolation at the API layer rather than at the filesystem
layer.

This replaced an older model that split "common" (shared) and "user"
(private) databases into separate files. That split introduced
cross-database search merging, file-permission puzzles, and a
`single_user` flag to pick between them. Dropping it removed a lot of
complexity for a feature nobody was asking for.

If two people genuinely want separate stores, they each run their own
Ragger install under their own account — no shared daemon involved.

## Bad Memories Are Marked, Not Deleted

When a memory is identified as wrong or harmful, it gets a `bad: true`
metadata flag rather than being deleted. This preserves institutional
memory of anti-patterns — knowing what *not* to do is as valuable as
knowing what to do.

## Boolean Flags Are Tags, Not Key-Value Pairs

`keep`, `bad`, `important`, and similar flags are stored as tags in the
comma-separated `tags` column — not as `"keep": true` in JSON metadata.
This simplifies the schema: tags are a flat list of labels, not a mix
of strings and booleans scattered across two storage locations.

The `keep` tag prevents deletion (both single and batch). It's auto-set
when memories are stored to the common database, preventing accidental
loss of curated shared knowledge. The `bad` tag marks wrong or harmful
memories — preserved as institutional memory of anti-patterns.

## Dual Authentication: Tokens + Passwords

Ragger uses two authentication mechanisms for different access patterns:

**Tokens** — Machine auth for API and MCP clients.
- Auto-generated at provisioning, stored in `~/.ragger/token`
- SHA-256 hash stored in the database
- Used by OpenClaw, Claude Desktop, and other programmatic clients
- Auto-rotate on a configurable interval (default: 24 hours)

**Passwords** — Human auth for the web UI.
- User-chosen, set during installation or via `ragger passwd`
- PBKDF2-SHA256 hash stored in the database (600k iterations)
- Only the user knows the password — even root can't recover it, only reset it
- Optional: blank password disables web UI access

### Why Both?

Tokens live on disk (`~/.ragger/token`) and are readable by any user
with sudo privileges. That's acceptable for machine clients but not
for human authentication — a compromised token file shouldn't grant
persistent human-level access.

### Why Tokens Auto-Rotate

The primary motivation is **backup exposure**, not live access.

macOS Time Machine (and similar backup systems) makes local snapshots
of the user's computer before transferring them to remote storage.
Token rotation ensures the token has likely expired before the snapshot
reaches the backup drive or server. This prevents a valid token from
sitting indefinitely in backup storage where it might be extracted.

The sudo-user window (where another admin could read the token file)
is a secondary concern — token rotation limits that exposure too, but
backup safety is the driving design constraint.

### Network Scope

Currently the HTTP API binds to `127.0.0.1` (localhost only). Token
auth is sufficient for local access since all users are already on
the machine. Authentication for remote connections (binding to
`0.0.0.0` or a specific interface) will be implemented soon — this
will require TLS and stricter token/session validation.

### Two Access Levels

| Level          | Capabilities                                                     |
|----------------|------------------------------------------------------------------|
| **Sub-user**   | Use ragger over HTTP with a token; change own password           |
| **Daemon owner** | All of the above, plus `ragger useradd`/`userdel`, edit `~/.ragger/settings.ini`, direct SQLite access, reset any sub-user's password |

There is no root/sudo tier. The daemon runs as the owning user — no
elevated privileges, no system-wide resources.

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

## Raw vs Digested Conversation Storage

For semantic memory retrieval, summarized/digested turns are almost
always more useful than raw transcripts. When you search "what did we
decide about logging," you want the decision, not 15 back-and-forth
messages that led to it.

However, some environments legally require raw conversation records
(compliance, legal discovery, customer support, audit trails). The
`permanent` storage mode addresses this: raw turns are kept forever
with `keep: true` auto-set, while summaries are still created
alongside for efficient search. Both exist in the database — raw for
the record, summarized for retrieval.

Default recommendation: store raw turns briefly for active context,
then summarize and discard. Use `permanent` only when retention
requirements demand it.

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

## Chat Streaming

The Python server streams chat responses via Server-Sent Events (SSE),
sending tokens to the client as they arrive from the inference backend.

The C++ server currently buffers the full response and sends it at once.
SSE streaming is planned but requires either replacing the HTTP library
(Crow 1.3 has no native streaming support) or maintaining a fork.
