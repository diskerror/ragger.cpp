# Configuration

Ragger reads one INI file. There is no system-level overlay; the file
you see is the file the daemon reads.

## Location

`~/.ragger/settings.ini`. `install.sh` copies `example-settings.ini` from
the source tree the first time, then leaves it alone on re-installs.
`example-settings.ini` is the single source of truth for the full key
list — it's also compiled directly into the binary as the built-in
default (via a CMake build step), so the shipped template and the
fallback config can never drift apart.

Every other on-disk path Ragger uses (database, models, recipes, formats,
log, token, stats, socket) is hardcoded relative to a single base
directory, `~/.ragger` by default. None of that is independently
configurable in `settings.ini`. The only way to relocate the whole tree
is the hidden `--ragger-base <path>` CLI flag — undocumented in
`--help`, meant for tests, not for normal use. If you need Ragger's data
on a different disk, symlink `~/.ragger` there instead.

To run against a different settings file (testing, side-by-side configs):

```bash
ragger serve --config /custom/path/settings.ini
```

## Quick map of the sections

| Section          | Purpose                                                        |
|------------------|-----------------------------------------------------------------|
| `[server]`       | Socket/bind/port, TLS cert/key, turn-capture/build-context toggles, recipes |
| `[embedding]`    | Model, vector dimensions, on-disk precision                    |
| `[search]`       | Default limit, score floor, BM25/vector/phonetic blend, `inject_data` |
| `[summarizer]`   | Inference endpoint used by the L2/L3 background worker         |
| `[embed]`        | Embedding subprocess pool — timeouts, retries, concurrency     |
| `[logging]`      | Log level (one unified log file — see below)                  |
| `[housekeeping]` | Retention, pass cadence, episode-idle minutes                  |
| `[import]`       | Minimum chunk size for markdown imports                        |
| `[paths]`        | Path normalization options                                     |

The tables below cover every key currently recognized by the parser.

## `[server]`

| Key              | Default         | Description |
|------------------|-----------------|-------------|
| `socket_enable`  | `true`          | AF_UNIX listener, always at `~/.ragger/ragger.sock` (path not configurable). Legacy key `socket = <path>` is still accepted for back-compat — any non-empty value enables the listener. |
| `bind`           | `(empty)`       | TCP bind address. Empty = unix socket only. Must be a valid IPv4/IPv6 literal or hostname — a malformed value fails at startup. |
| `port`           | `8432`          | TCP port (only meaningful when `bind` is set). |
| `server_name`    | `(empty)`       | Hostname used by cpp-httplib (e.g. `ragger.local`). |
| `cert`           | `(empty)`       | Path to a TLS certificate chain (PEM). Both `cert` and `key` must be set to enable native TLS. Native TLS isn't wired up yet (see [TLS setup](tls-setup.md)) — use a reverse proxy for now. |
| `key`            | `(empty)`       | Path to a TLS private key (PEM). |
| `capture_turns`  | `true`          | **Write side.** When `true`, `capture_turn` (MCP) / `POST /turn` ingests agent-pushed turns into the `turns` table. Set `false` to make the call a no-op. |
| `build_context`  | `false`         | **Read side.** When `true`, `build_context` / `GET /session/<id>` assembles a recipe-shaped payload. Only meaningful when `capture_turns` is also on. Agent-driven `search`/`store` work either way. Not currently documented in `example-settings.ini` — code and default (`false`) remain intact if you want to opt back in by hand. |
| `default_recipe` | `natural_fading` | Recipe used when the caller doesn't name one. See [Recipes](#recipes) below. Also undocumented in the shipped template for the same reason as `build_context`; both keys are still parsed. |

Both listeners can run at the same time. If `socket_enable` is false and
`bind` is empty, the daemon falls back to `bind = 127.0.0.1` so there's
always at least one listener.

`[tls]` / `[ssl]` (legacy standalone sections) are still silently
accepted with `cert`/`key` keys for back-compat with older configs —
new configs should set `cert`/`key` under `[server]` instead.

## `[embedding]`

`model + dimensions + vector_type` define the DB's vector identity and
are recorded in the `settings` table. Changing any of them once the DB
holds data requires `ragger rebuild-embeddings` to re-encode everything;
mismatches abort startup with a clear error.

| Key          | Default            | Description                                                  |
|--------------|--------------------|--------------------------------------------------------------|
| `model`      | `all-MiniLM-L6-v2` | Sentence-transformer model name; resolved under `~/.ragger/models/<model-name>/`. |
| `dimensions` | `384`              | Vector size; must match the model.                           |
| `vector_type`| `f16`              | On-disk precision. `f16` halves blob size; `f32` is full precision. In-memory math is always f32. |

Model files always live at `~/.ragger/models/<model-name>/` — not
independently configurable; hardcoded like every other Ragger path.

## `[search]`

| Key                 | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `default_limit`     | `5`     | Default result count.                                                       |
| `default_min_score` | `0.4`   | Cosine floor for returned results.                                          |
| `bm25_enabled`      | `true`  | Blend BM25 keyword scoring into the rank.                                   |
| `bm25_weight`       | `4`     | Weight for the BM25 signal (ratio, not percentage; integers preferred).     |
| `vector_weight`     | `8`     | Weight for the cosine signal.                                               |
| `phon_weight`       | `1`     | Weight for the phonetic "sounds-like" (Double Metaphone) signal. `0` off.   |
| `inject_data`       | `false` | Reserved — parsed and hot-reloadable, not yet wired to any search behavior. |
| `max_search_limit`  | `0`     | Ceiling on client-requested `limit` for sub-users. `0` disables the cap.    |

Weights are ratios; they're normalized internally. The default blend is
`8` vector / `4` bm25 / `1` phon — meaning-first, with keyword and a gentle
phonetic nudge. `phon_weight = 0` disables the sounds-like signal entirely.

## `[summarizer]` — the only inference endpoint

Ragger only calls inference for L2/L3 summarization. There is no chat
surface. (Older versions had a separate `[inference]` block for a
general-purpose endpoint plus `[inference.<name>]` multi-endpoint
routing — both are gone from the documented config; `[summarizer]` is
the sole inference section going forward.)

| Key          | Default                       | Description |
|--------------|--------------------------------|-------------|
| `model`      | `qwen3-4b-instruct-2507`       | Model name the summarizer asks for. |
| `api_url`    | `http://localhost:1234/v1`    | OpenAI-compatible base URL. Default targets LM Studio's localhost convention. |
| `api_key`    | *(empty)*                     | Bearer key for the endpoint, if it needs one.   |
| `max_tokens` | `1024`                         | Cap for a single summarization call.            |
| `target_pct` | `40`                           | Target summary length as % of the raw turn. `0` = built-in default of 30%. |
| `max_pct`    | `60`                           | Hard cap as % of the raw turn. `0` = built-in default of 60%. |
| `prompt`     | *(built-in default)*           | System prompt sent to the summarizer. Quote it to protect `#` from being read as a comment. Two `{}` placeholders (target chars, hard cap) are required if you override it. Set to a single space `" "` to suppress the system prompt entirely. |

**Choosing a model.** Summarization is a short, low-complexity task —
model quality matters far less than model speed.

- **Best choice: a small non-thinking MLX model** (e.g. `qwen3-4b`,
  `gemma-4-e2b-it-mlx`). MLX models run on Apple Silicon's Neural
  Engine / GPU, load instantly, and average **~1 second per summary**
  with no context-window issues.
- **Avoid thinking/reasoning models** (DeepSeek-R1, Gemma-4-E4B, Qwen3
  with reasoning enabled, etc.). They spend 200–300 tokens on internal
  chain-of-thought before writing a single output token — summaries
  take 30–60× longer for no quality gain on this task.
- **Keep `max_tokens` at 1024 or below.** The summarizer targets a
  fraction of the source length and caps well under it. 1024 tokens is
  generous for any single turn. Requesting more wastes KV cache and
  will crash GGUF models loaded in LM Studio with `parallel > 1`
  (each slot's budget is `context_length / parallel`; requesting more
  than that causes LM Studio to drop the connection with no error body).
- **If using a GGUF model, set `parallel = 1` in LM Studio.** Ragger's
  summarizer never runs concurrent requests, so there is no benefit to
  parallel slots — and each slot you add divides the KV budget.

**Fallback behaviour.** When the endpoint is unreachable, the worker
writes a heuristic *draft* L2 (a trimmed `User: … | Assistant: …`
concatenation, tagged `draft`) so the chronology stays intact. A
housekeeping pass re-summarizes drafts the next time inference is
available — no manual catch-up needed.

## `[housekeeping]`

| Key                     | Default | Description                                                                        |
|-------------------------|---------|------------------------------------------------------------------------------------|
| `cleanup_max_age_hours` | `0`     | Raw-turn lifetime. `0` = keep forever (default). Set to hours (e.g. `336` = 2 weeks) to auto-purge old turns. |
| `housekeeping_interval` | `60`    | Seconds between background passes. `0` disables; values under 10 are clamped to 10.|
| `episode_idle_minutes`  | `15`    | Idle gap (minutes) that closes an episode of work within a session, and drives session/project rollups. Any positive integer. |
| `summary_pause_minutes` | `20`    | Deprecated alias for `episode_idle_minutes` — honored only when `episode_idle_minutes` is unset. |

The same housekeeping loop catches turns that landed without a
summary (e.g. captured via MCP while the daemon was down) and rewrites
draft L2 rows when inference comes back.

## `[embed]`

The `ragger embed` subprocess pool that actually invokes the embedding
model.

| Key           | Default | Description                                |
|---------------|---------|--------------------------------------------|
| `timeout_ms`  | `10000` | Per-subprocess timeout.                    |
| `retries`     | `1`     | Retries on a failed or timed-out subprocess. |
| `max_workers` | `8`     | Cap on concurrent embed subprocesses.      |

## `[logging]`

| Key         | Default                  | Description                                                              |
|-------------|--------------------------|----------------------------------------------------------------------------|
| `log_level` | `warn`                   | One of `trace`, `debug`, `info`, `warn`, `error`, `critical`.            |

Everything — queries, HTTP, MCP, general daemon activity — goes to one
log file, hardcoded at `~/.ragger/activity.log`. There are no separate
per-stream log files or toggles (older versions had `log_dir`,
`query_log`, `http_log`, `mcp_log`, `debug_log` — all removed; a single
unified log replaces them).

## `[paths]`

| Key              | Default | Description                                         |
|------------------|---------|------------------------------------------------------|
| `normalize_home` | `true`  | Replace `/Users/<you>` (or `/home/<you>`) with `~` in stored text. |

## `[import]`

| Key                   | Default | Description                            |
|-----------------------|---------|-----------------------------------------|
| `minimum_chunk_size`  | `300`   | Minimum characters per chunk when importing markdown/text. |

## Recipes

A recipe is a JSON file in `~/.ragger/recipes/` that describes how
`build_context` should assemble the payload for an agent. Each recipe is
a list of layers walked back from the latest turn:

```json
{
  "name": "natural_fading",
  "description": "Default: recent raw → turn summaries → session → project → decisions",
  "layers": [
    { "kind": "raw_turn",        "limit": 2 },
    { "kind": "turn_summary",    "limit": 5 },
    { "kind": "session_summary", "limit": 1 },
    { "kind": "project_summary", "limit": 1 },
    { "kind": "decisions",       "limit": 3 }
  ],
  "max_tokens": 8000,
  "chars_per_token": 4.0
}
```

Available layer kinds: `raw_turn`, `turn_summary`, `session_summary`,
`project_summary`, `decisions`. `limit` is per-layer (`0` = unlimited
for that layer). The legacy field name `"count"` is still accepted for
backward compatibility but deprecated — use `"limit"` in new recipes.
`max_tokens` is a ceiling; assembly trims from the
oldest end if it would be exceeded. See `recipes/natural_fading.json`
in the source tree for the canonical example, and the four others
(`reconnect`, `deep_recall`, `tldr`, `raw_only`) for different
intents.

**Resolution order for the active recipe** (each step falls through
to the next on miss):

1. The `recipe` argument on the `build_context` MCP tool or the
   `?recipe=` query param on `GET /session/<id>`.
2. DB `settings.recipe` (set by `ragger recipe`). The sentinel value
   `default` means "track `[server] default_recipe` in settings.ini" —
   pick it to revert.
3. `[server] default_recipe` in `settings.ini`.
4. The first built-in if all of the above miss.

The DB lookup happens once per `build_context` call; changing your
choice with `ragger recipe` takes effect on the next request, no
daemon restart needed.

Note: `build_context`/`default_recipe` are still functional (code and
recipes are fully intact) but no longer documented in the shipped
`example-settings.ini` template — set them by hand if you want the
feature; the default remains off (`build_context = false`).

## Ceilings for sub-users

When the daemon serves additional users over HTTP (each with their own
bearer token), the owner can cap the parameters those clients send.
Today the only ceiling is `max_search_limit` (in `[search]`). A value
of `0` disables it.

For a personal install with no sub-users, leave ceilings at `0`.

## Reloading without a restart

`ragger reload` sends `SIGHUP` to the daemon, which re-reads the file
and applies hot-reloadable keys in place. Keys that require a restart
(e.g. `port`, `socket_enable`, `bind`, `tls_cert`, `tls_key`,
`embedding_model`, `embedding_dimensions`, `embedding_vector_type`) log
a warning instead of silently being ignored.

## Build-time instrumentation: retrieval stats (`RAGGER_STATS`)

This one is **not** a settings.ini key — it's a *compile-time* flag, off by
default. It exists for studying how retrieval behaves (focused vs. associative
hits, how results shift as the corpus grows), not for normal operation.

```bash
# Normal build — flag off, zero overhead, byte-for-byte unaffected:
cmake -S . -B build
cmake --build build --target ragger -j4

# Instrumented build — flag on:
cmake -S . -B build-stats -DRAGGER_STATS=ON
cmake --build build-stats --target ragger -j4
```

When **off** (the default) the logging code compiles to nothing; a normal
binary has no stats behavior at all. When **on**, every search appends a row
to a **separate, discardable** database at `~/.ragger/stats.db`. It is never
touched by — and never touches — `memories.db`. Logging is best-effort: if the
stats write fails for any reason, the search itself is unaffected. To stop
collecting, rebuild without the flag and (optionally) delete `stats.db`; to
start fresh, just delete it and it's recreated on the next run.

### Reading the stats

There is intentionally **no built-in viewer or report.** `stats.db` is a plain
SQLite database with two tables — `lookups` (one row per search) and `hits`
(the top-ranked results for each lookup, linked by `lookup_id`). Getting useful
information out requires **basic SQL skills**; anyone comfortable with a
`SELECT ... JOIN ... ORDER BY` can shape it however they like. A starting
point:

```sql
-- The top hits for the most recent searches, with the per-signal breakdown:
SELECT l.id, l.query, h.rank,
       ROUND(h.score,3)      AS cos,    -- raw cosine (the reported score)
       ROUND(h.vec_score,3)  AS vec,    -- normalized vector contribution
       ROUND(h.bm25_score,3) AS bm25,   -- normalized keyword contribution
       ROUND(h.phon_score,3) AS phon,   -- normalized phonetic contribution
       ROUND(h.blended,2)    AS blend,  -- final weighted ranking score
       h.collection, h.snippet
FROM lookups l
JOIN hits h ON h.lookup_id = l.id
ORDER BY l.id DESC, h.rank
LIMIT 30;
```

### Per-signal ranking breakdown

Each `hits` row records not just the final rank but **how each search signal
contributed**, so you can judge whether a weight (`vector_weight`,
`bm25_weight`, `phon_weight`) needs tuning — or whether a signal earns its keep
at all:

| column | meaning |
|--------|---------|
| `score` | Raw vector cosine similarity — the absolute "how close in meaning" number, also what the API reports. |
| `vec_score` | The **min-max-normalized** cosine that actually fed the blend (how it competed *within this result set*). |
| `bm25_score` | Normalized BM25 keyword contribution, or **NULL** when `bm25_enabled = false`. |
| `phon_score` | Normalized phonetic ("sounds-like", Double Metaphone) contribution, or **NULL** when `phon_weight = 0`. |
| `blended` | The final weighted ranking value: `vector_weight·vec + bm25_weight·bm25 + phon_weight·phon`. This is what actually ordered the results. |

A **NULL** in `bm25_score`/`phon_score` means that signal was switched off for
the search (so you can `AVG()`/filter without a sentinel skewing the numbers); a
real **0.0** means the signal was active but contributed nothing for that row.
Because each signal is normalized to `[0,1]` before weighting, the columns are
directly comparable — e.g. a hit with a low `vec` but a high `phon` is a
"sounds-like pulled this in" case, and one with `bm25` near 1.0 won on exact
keywords. Example dissections:

```sql
-- Hits where the phonetic signal was the strongest contributor:
SELECT l.query, h.rank, h.vec_score, h.bm25_score, h.phon_score
FROM lookups l JOIN hits h ON h.lookup_id = l.id
WHERE h.phon_score IS NOT NULL
  AND h.phon_score > h.vec_score AND h.phon_score > h.bm25_score
ORDER BY l.id DESC;

-- Average contribution of each signal across all logged hits:
SELECT ROUND(AVG(vec_score),3)  AS avg_vec,
       ROUND(AVG(bm25_score),3) AS avg_bm25,
       ROUND(AVG(phon_score),3) AS avg_phon
FROM hits;
```

The schema is deliberately small and obvious — open it with `sqlite3
~/.ragger/stats.db` and `.schema` to see every column, then slice it to answer
whatever question you're chasing.

> **Upgrading an existing `stats.db`:** the four breakdown columns
> (`vec_score`, `bm25_score`, `phon_score`, `blended`) are added automatically
> via in-place `ALTER TABLE` the next time an instrumented binary opens the
> database — rows logged before the upgrade simply carry `NULL` for them. No
> manual migration or reset needed.

## Related

- [Getting Started](getting-started.md) — Install + first run
- [Deployment](deployment.md) — Daemon lifecycle, sub-users
- [HTTP API](http-api.md) — Endpoints, including `/turn` and `/session/<id>`
- [Agent integration](agent-integration.md) — How agents use capture and `build_context`
