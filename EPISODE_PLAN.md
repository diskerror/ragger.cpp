# EPISODE_PLAN.md — Episode layer + boundary-triggered summaries

> **TEMPORARY working spec.** Scratch document for the current redesign, not
> permanent documentation. Delete once Phases 1–3 land. Phase 4 lives in GitHub
> issue #72.

## Context / motivation

The summarizer currently keeps a **running L3 "session" summary** that it
rewrites after *every turn* via `handle_l3_update`. Two defects:

1. **Bloat bug.** On session close it flips the row to `status='complete'` and
   leaves it. If the session gets more turns later, `handle_l3_update` finds no
   `current` row and **INSERTs a new one** instead of reopening the old — so
   inference outages + the 60s housekeeping catch-up stacked up dozens of
   near-duplicate `complete` rows (e.g. summaries 1956–1977: 22 rows, one
   running summary snapshotted 22×; session 72 had ~192).
2. **Conflated concepts.** One `level='session'` row was doing two jobs at once:
   "the running state of this session" *and* "a distinct episode of work."

### The fix in one sentence

Split the two jobs into two levels, and **trigger summaries on boundaries, not
on every turn**. Running summaries stay running (insert-or-update in place) —
the bug was *inserting* when it should have *updated*, not the running concept
itself.

## Locked decisions

- **Level hierarchy** (DB `level` strings in **bold**):
  | level | scope | write pattern |
  |-------|-------|---------------|
  | **`turn`** (L2) | one exchange | insert once (existing) |
  | **`episode`** (L3a) | a bounded stretch of work within a session | insert once on close, **immutable** |
  | **`session`** (L3b) | 1:1 with the agent's session id | **running**: insert-or-update in place, one row per session_id |
  | **`project`** (L4) | global rollup | **running**: insert-or-update in place, one row |

- **"Session" == whatever the agent calls a session.** No reinterpretation, no
  switch/return special-casing. Episode ⊂ session (an episode never spans two
  sessions). Any user-felt "I came back to this later" nuance is carried by
  **episodes**, not by session bookkeeping.

- **Episode boundaries (v1, time-only):** an episode closes on
  1. **session end**, or
  2. **idle gap** ≥ `episode_idle_minutes` since the last turn in that session.
  Topic-shift boundaries are **Phase 4 / issue #72** (deferred).

- **`episode_idle_minutes`**: any positive integer, **default 15**. No range
  clamp (ridiculous values are the user's business — consistent with Ragger's
  retention philosophy). Short by design: models the walk-away-and-return rhythm
  (5–30 min).

- **Running summaries keep timestamps for both insert and update times** (see
  schema change below) so we can tell when a running row was first created vs
  last regenerated.

- **No more per-turn running updates.** `session`/`project` regenerate on
  **boundaries** (episode close / session change / extended break), not per turn.

- Work happens **in-branch on `master`** (solo project). Standard discipline:
  flag-off risky bits, verify against a **copy** of `~/.ragger/memories.db`,
  version bump lands in the same commit as its tag.

---

## Current-state reference (verified)

- **Schema** (`summaries`): `summary_id, model_id, text, embedding, level,
  status, tags, timestamp, session_id, phon`. Indices on level/status/timestamp/
  session. FTS triggers (`summaries_ai/ad/au`) + phon-FTS triggers.
- **Config** (`include/ragger/config.h`):
  - `housekeeping_interval = 60` (sec; 0=off, <10 clamped to 10)
  - `summary_pause_minutes = 20` (idle gap that closes a session's running L3)
- **JobKind** (`include/ragger/summarizer_service.h`):
  `{ L2Turn, L3CloseSession, DraftRetry, L3UpdateSession, L4UpdateProject }`
- **Backend ops** (`storage_backend.h`): `store_summary`,
  `current_session_summary`, `current_project_summary`, `update_summary_text`,
  `set_summary_status`.
- **Housekeeping** (`run_housekeeping`): purge old turns → `enqueue_catch_up()`
  (unsummarized turns/drafts + session closes) → `backfill_embeddings()`. Every
  60s.
- **Summarizer flow today**: L2 per turn → after each L2 queues an
  `L3UpdateSession` (running rewrite) → pause timer queues `L3CloseSession`
  (flip current→complete, queue L4) → `L4UpdateProject`.

---

## PHASE 1 — Episode layer

**Goal:** introduce `level='episode'`, written once when an episode closes.
Leave the existing running-L3 path alone for now (removed in Phase 2) so Phase 1
is additive and independently testable.

### 1.1 Config
`include/ragger/config.h`
- Add `int episode_idle_minutes = 15;`
- Keep `summary_pause_minutes` for now (Phase 2 migrates it). Document that
  `episode_idle_minutes` supersedes it.

`src/config.cpp`
- Parse `episode_idle_minutes` (any positive int; reject ≤0 → fall back to 15).
- Accept `summary_pause_minutes` as a **deprecated alias**: if
  `episode_idle_minutes` is unset but `summary_pause_minutes` is present, use it.
- Add to the `RELOAD(...)` list.

`~/.ragger/settings.ini` (docs/example): add `episode_idle_minutes = 15` under
`[summarizer]`, comment the old key as deprecated alias.

### 1.2 Schema / migration
No column changes required for episodes themselves — reuse `summaries` with
`level='episode'`. But add running-summary timestamps (needed Phase 2, land now):

- `ALTER TABLE summaries ADD COLUMN updated_at TEXT;` (NULL for existing rows;
  set = `timestamp` on first update). `timestamp` remains the **insert/create**
  time; `updated_at` is the **last regenerate** time.
- Migration guard: additive `ADD COLUMN` after the table exists; no FTS impact
  (not an indexed column). Follow the phon-column migration ordering rules in
  the ragger-memory-pipeline skill (ALTERs after all CREATEs).

### 1.3 Backend ops (5-file chain per skill)
Add:
- `open_episode_exists(session_guid) -> optional<int>` — is there an unclosed
  episode for this session? (An episode is "open" until summarized. Represent
  the open episode implicitly: turns since the last `episode` row for the
  session, OR track via a marker — see 1.4.)
- `store_episode(text, model, session_guid, first_ts, last_ts)` — insert one
  immutable `level='episode'` row; `timestamp=first_ts`, tags may carry the
  span. Mirror `store_summary` (embed, phon, invalidate cache).
- `episode_turn_texts(session_guid, since_ts) -> vector<string>` — the L2
  summaries composing the closing episode (turns since the previous episode's
  end within this session).

### 1.4 Defining "the open episode"
An episode = the run of turns in a session **after** the previous episode's last
turn, up to the boundary. Detection without new state:
- The previous episode's `last_ts` (stored in tags or derived as MAX timestamp
  of `level='episode'` rows for the session).
- "Turns needing an episode" = L2 turns for the session with
  `timestamp > last_episode_ts`.
- **Close trigger:** `MAX(turn.timestamp) for session < now - episode_idle_minutes`
  (idle) **OR** session flagged ended. Reuses the existing
  `sessions_needing_close` query shape, swapping `summary_pause_minutes` →
  `episode_idle_minutes` and the "no complete L3" test for "has turns past the
  last episode."

### 1.5 Summarizer
`summarizer_service.{h,cpp}`
- Add `JobKind::EpisodeClose`.
- `enqueue_catch_up()`: for each session with turns past its last episode AND
  idle ≥ threshold (or ended), queue an `EpisodeClose`.
- `handle_episode_close(job)`: gather `episode_turn_texts`, summarize once,
  `store_episode(...)`. On inference failure, leave turns unclosed (retry next
  catch-up) — same pattern as L2 draft handling.

### 1.6 Server / MCP
No new HTTP surface required for Phase 1 (episodes are internal). Optionally
extend `/search` merge to include `level='episode'` (it already merges all
context-table rows, so episodes are searchable for free once written).

### 1.7 Verification
- Copy DB: `cp ~/.ragger/memories.db /tmp/ep-test.db`.
- Point a test daemon at the copy; set `episode_idle_minutes=1` for fast cycles.
- Feed a few turns, wait > threshold, confirm exactly **one** `episode` row
  appears spanning them, embedding non-NULL, searchable via `/search`.
- Confirm no episode row appears while turns are still arriving inside the
  window.

---

## PHASE 2 — session/project become boundary-triggered running rollups

**Goal:** kill the per-turn running-L3 churn (the bloat source); make `session`
and `project` regenerate **in place** on boundaries, one row per key.

### 2.1 Remove per-turn running update
- Delete the "after each L2, queue `L3UpdateSession`" hook in `handle_l2`.
- `session` is now rebuilt only when an **episode closes** (Phase 1 trigger)
  and when the **session ends**.

### 2.2 session rollup — insert-or-update, never insert-twice
- Rework `handle_l3_update` (or replace with `handle_session_rollup`) to:
  1. `current_session_summary(guid)` — find the one session row.
  2. Rebuild text from the session's **episodes** (+ any tail turns not yet in
     an episode).
  3. If a row exists → `update_summary_text` + set `updated_at=now`.
     Else → `store_summary(level='session', status='current', ...)`.
- **Invariant: exactly one `level='session'` row per session_id.** The bloat bug
  cannot recur because we never insert when a row exists — regardless of
  `status`. Fix the original defect: match on `(session_id, level='session')`
  **ignoring status**, not on `status='current'`.
- Drop the current→complete status flip as the "close" mechanism, or keep
  `status` purely informational (`current` while session live, `complete` after
  end) — but **never** let status gate the find-existing query.

### 2.3 project rollup
- `handle_l4_update` already upserts one project row via
  `current_project_summary` — keep, but trigger it on **session close /
  extended break**, not on every L3 close cascade. Set `updated_at` on rewrite.

### 2.4 Retire `summary_pause_minutes`
- All timing now flows from `episode_idle_minutes`. Keep the alias parse for one
  release, then remove.

### 2.5 Verification
- On the DB copy, run a multi-episode session; assert:
  - N `episode` rows (immutable, one per episode),
  - exactly **1** `session` row (its `updated_at` advances, `timestamp` fixed),
  - exactly **1** `project` row.
- Simulate an inference outage across a housekeeping tick; confirm **no**
  duplicate `session` rows appear (the original 1956–1977 reproduction must come
  back clean).

---

## PHASE 3 — one-time cleanup of existing bloat

**Goal:** collapse the legacy running-L3 duplicates already in the live DB. Done
**last**, after new logic is live, so we clean once.

### 3.1 Plan
- **Backup first:** `cp ~/.ragger/memories.db ~/.ragger/memories.db.pre-episode-cleanup`.
- Legacy rows are all `level='session'`. Keep the **newest per session_id**
  (`MAX(timestamp)`), delete the older intermediates:
  ```sql
  WITH ranked AS (
    SELECT summary_id,
           ROW_NUMBER() OVER (PARTITION BY session_id
             ORDER BY timestamp DESC, summary_id DESC) rn
    FROM summaries WHERE level='session')
  DELETE FROM summaries WHERE summary_id IN (SELECT summary_id FROM ranked WHERE rn>1);
  ```
  (Dry-run measured earlier: 726 session rows → 74 kept, 652 removed.)
- **No cosine / thresholds needed** — these are structural duplicates (same
  session, same level, running-summary snapshots), not semantic near-dups. The
  vecsql cosine work is **not** used here (it's for Phase 4 topic detection).

### 3.2 Execution discipline
- Stop daemon (`ragger stop` + confirm no listener on :8432).
- Run delete on the backed-up-then-live DB via the single connection.
- `PRAGMA wal_checkpoint(TRUNCATE); PRAGMA integrity_check;` — watch the
  external-content FTS (`summaries_fts`, `summaries_phon_fts`); if "malformed",
  rebuild per the skill's FTS-desync fix.
- Restart, verify one `session` row per session_id, search still works.

### 3.3 Verification
- `SELECT session_id, COUNT(*) FROM summaries WHERE level='session'
   GROUP BY session_id HAVING COUNT(*)>1;` → **zero rows**.
- Spot-check that the kept row per session is the last/most-complete one.

---

## PHASE 4 — topic-shift episode boundaries (DEFERRED)

Tracked in **GitHub issue #72**. Close an episode on semantic topic shift
(embedding distance between consecutive L2 turn summaries below a threshold),
with anti-thrash guards. Reuses the `~/PyCharmProjects/vecsql` cosine tooling to
calibrate the threshold offline against captured sessions. Build only after
Phases 1–3 prove out.

---

## Files touched (summary)

| file | phases |
|------|--------|
| `include/ragger/config.h` | 1 (add field), 2 (retire alias) |
| `src/config.cpp` | 1 (parse + alias), 2 |
| `include/ragger/storage_backend.h` + `sqlite_backend.h` | 1 (new ops) |
| `src/sqlite_backend.cpp` | 1 (schema migration, episode ops, updated_at), 2 (rollup upsert fix) |
| `include/ragger/memory.h` + `src/memory.cpp` | 1 (facade forwarders) |
| `include/ragger/summarizer_service.h` + `src/summarizer_service.cpp` | 1 (EpisodeClose), 2 (drop per-turn update, rollup triggers) |
| `src/server.cpp` | 1 (housekeeping trigger wiring), 2 |
| `~/.ragger/settings.ini` (example/docs) | 1 |
| live DB | 3 (cleanup) |

## Ordering rationale

Episode layer first (additive, safe) → then flip session/project to
boundary rollups and remove the churn (behavior change, needs the episode layer
to feed it) → then clean legacy bloat once (after logic is correct) → topic
shifts last (needs stable episodes + threshold calibration).
