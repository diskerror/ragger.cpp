# Chat Persistence

When running `ragger chat`, conversation turns can be stored for context
across sessions. This system is designed to balance continuity with
performance and database size.

## Turn Storage Modes

Set via the `store_turns` config key in `[chat]`:

| Mode             | Value       | Behavior                                                 | Use Case                                         |
|------------------|-------------|----------------------------------------------------------|--------------------------------------------------|
| **Always**       | `true`      | Store every turn, summarize, delete raw after expiration | Long-term projects, reference conversations      |
| **Permanent**    | `permanent` | Store every turn, summarize, but keep raw turns forever  | Compliance, legal, audit trails, support records |
| **Session-only** | `session`   | Keep turns in memory during the session, discard on exit | Privacy-sensitive chats, ephemeral work          |
| **Off**          | `false`     | No turn storage at all                                   | Stateless use, testing                           |

**Pros and cons:**

- **Always (`true`):** Full history survives crashes and provides rich context for future sessions, but grows the
  database over time. Requires periodic cleanup via expiration settings.
- **Permanent (`permanent`):** Like `true`, but raw turns are never deleted — only summarized alongside.
  Raw turns automatically get `keep: true` metadata, protecting them from both expiration and
  manual deletion. The database grows indefinitely; plan storage accordingly. Essential for
  regulated environments where conversation records must be retained.
- **Session (`session`):** No database growth, no privacy concerns, but context is lost if the process crashes or exits
  unexpectedly.
- **Off (`false`):** Fastest and smallest, but the chat has no memory of prior turns within the same session.

## Summary Timing

Summaries extract decisions, facts, and lessons from raw conversation turns
and store them as structured memories. Raw turns are deleted after
summarization to keep the database lean.

| Setting              | Default | Description                                        |
|----------------------|---------|----------------------------------------------------|
| `summarize_on_pause` | `true`  | Summarize buffered turns after inactivity          |
| `summarize_on_quit`  | `true`  | Summarize buffered turns on graceful exit          |
| `pause_minutes`      | `10`    | Inactivity threshold for pause-based summarization |

**Best practice:** Enable both. Pause-based summarization happens
automatically during long pauses (e.g., lunch break, end of workday).
Quit-based summarization ensures nothing is lost on normal exit.

## "Keep" Tag

Turns tagged with `keep` in metadata are never deleted, even after
summarization or expiration. This is useful for:

- Reference conversations (e.g., API design discussions)
- Training examples
- High-value turns you want to preserve verbatim

**Auto-tagging:** Queries against the `common` database (shared reference
material) automatically set the `keep` tag, since those turns are typically
reference lookups rather than ephemeral chat.

Set manually when storing:

```python
memory.store(
    text="Important decision about API design...",
    metadata={"category": "decision", "keep": True}
)
```

## Launch-time Orphan Recovery

If `ragger chat` crashes or is killed (SIGKILL), buffered turns may be
left in the database without summarization. On the next launch, Ragger
checks for orphaned turns (turns stored since the last summary timestamp)
and offers to:

1. Summarize them now
2. Keep them as-is
3. Delete them

This ensures no context is permanently lost due to unexpected termination.

## Turn Expiration

To prevent unbounded database growth, old turns can be automatically
deleted based on age or count.

| Setting                      | Default | Description                                              |
|------------------------------|---------|----------------------------------------------------------|
| `max_turn_retention_minutes` | `60`    | Delete turns older than this (0 = no age limit)          |
| `max_turns_stored`           | `100`   | Keep at most this many recent turns (0 = no count limit) |

Turns with the `keep` tag are always exempt from expiration.

**System ceilings:** System config can set hard limits (e.g.,
`max_turn_retention_minutes = 60`) that users cannot exceed, preventing
runaway database growth in multi-user deployments.

## File Size and Performance Impact

**Rough numbers:**

- Each turn: ~500 bytes of text + 1.5 KB embedding = ~2 KB total
- 100 turns/day × 30 days = 3,000 turns = ~6 MB
- 100 turns/day × 365 days = 36,500 turns = ~73 MB

Search performance degrades linearly with document count. At 50K documents,
search takes ~10-50ms on Apple Silicon. At 100K+, expect slower results
unless you enable collection filtering or add a vector index.

**Mitigation strategies:**

1. **Summarize aggressively:** Short retention (e.g., 60 minutes) with
   frequent summarization keeps raw turn count low.
2. **Use collections:** Store turns in a separate `conversation` collection
   and exclude it from routine searches.
3. **Periodic cleanup:** Delete old turns with a cron job or manual
   `ragger rebuild-bm25` after bulk deletions.

## Config Examples

### Aggressive Cleanup (minimal database, fast search)

```ini
[chat]
store_turns = true
max_turn_retention_minutes = 30
max_turns_stored = 50
summarize_on_pause = true
summarize_on_quit = true
pause_minutes = 10
```

### Long Retention (full history, reference use)

```ini
[chat]
store_turns = true
max_turn_retention_minutes = 10080  # 7 days
max_turns_stored = 1000
summarize_on_pause = true
summarize_on_quit = true
pause_minutes = 30
```

### Permanent Record (compliance, audit)

```ini
[chat]
store_turns = permanent
summarize_on_pause = true
summarize_on_quit = true
pause_minutes = 10
```

Raw turns are kept forever with `keep: true`. Summaries are still
created for efficient search. Expiration settings are ignored in this
mode since raw turns are never deleted.

### Session-only (no persistence)

```ini
[chat]
store_turns = session
summarize_on_quit = false
```

### Privacy-focused (no storage at all)

```ini
[chat]
store_turns = false
```

## Related

- [Configuration](configuration.md) — Full config reference
- [Search & RAG](search-and-rag.md) — Performance characteristics
