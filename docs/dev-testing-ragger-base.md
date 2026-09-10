# Developer Testing Against a DB Copy — `--ragger-base`

## What This Is

`--ragger-base <dir>` is an undocumented, testing-only CLI flag that relocates
Ragger's *entire* on-disk footprint (DB, logs, models, recipes, formats,
socket, token, stats.db, agent-memory-instructions.md) to `<dir>` instead of
`~/.ragger`. It's the standard way to exercise migrations, reindexing, or any
destructive change against a real copy of your data without touching the live
DB. See `src/config.cpp` (path helpers) and `src/util/fs.cpp`
(`ragger_base_dir()` / `set_ragger_base_override()`) for the implementation.

**Never touch `~/.ragger/memories.db` directly for testing.** Always copy it
first.

## Standard Setup

```bash
mkdir -p /tmp/ragtest
cp ~/.ragger/memories.db /tmp/ragtest/memories.db

# REQUIRED: bump the port so this copy can never cross-talk with a live daemon
semqlite /tmp/ragtest/memories.db \
  "INSERT OR REPLACE INTO settings(key,value) VALUES('port','<some unused port>');"

ragger --ragger-base /tmp/ragtest <command>
```

Use Reid's `semqlite` (not plain `sqlite3`) when hand-inspecting a copy.

## Why the Port Bump Is Mandatory

`--ragger-base` correctly relocates the config/DB footprint: `init_config()`
overlays the `settings` table from the COPY's DB (via `resolved_db_path()`),
so `cfg.port` reflects whatever the copy's `settings` table says.

The catch: many CLI commands (`count`, etc.) first call
`client.is_available()` against `cfg.bind_address:cfg.port` to see if a daemon
is already serving that DB, and route through it if so. A straight file copy
of `~/.ragger/memories.db` still has the LIVE daemon's port in its `settings`
row. Result: `is_available()` succeeds (it finds the real live daemon, which
is in fact running and healthy) and the command silently talks to your LIVE
daemon/DB instead of the throwaway copy — no error, no warning, just silently
wrong results, because from the client's point of view a real daemon
answered.

Bumping the copy's `port` setting to an unused value guarantees
`is_available()` fails for the copy, so every command falls through to
opening the copy's DB directly (`RaggerMemory memory(db_path)`) — the actual
isolated behavior a test needs.

This is **usage discipline, not a code bug** — `--ragger-base` itself works
correctly; forgetting the port bump is what causes cross-talk.

## Quick Sanity Check

Confirm you're actually isolated before trusting any other test output:

```bash
# Pick a value obviously not your real row count
ragger --ragger-base /tmp/ragtest count
```

If this returns your live count instead of the copy's, the port bump didn't
take — check `settings.port` in the copy again.

## Commands That Always Bypass the Daemon

Some subcommands (e.g. `ragger decision list`) construct `RaggerMemory`
directly and never consult `client.is_available()` at all — they're safe to
use for verification even without the port bump, but bumping the port is
still the standard/required setup so any command works consistently.

## Idempotency

Re-run the same command/migration against the same copy a second time and
confirm no error and no double-application (row counts don't change,
duplicate columns aren't added, etc.) before considering a step verified.
