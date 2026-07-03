# Configuration

Ragger reads one INI file. There is no system-level overlay; the file
you see is the file the daemon reads.

## Location

`~/.ragger/settings.ini`. `install.sh` copies `example-settings.ini` from
the source tree the first time, then leaves it alone on re-installs.

To run against a different file (testing, side-by-side configs):

```bash
ragger serve --config /custom/path/settings.ini
```

## Quick map of the sections

| Section          | Purpose                                                        |
|------------------|----------------------------------------------------------------|
| `[server]`       | Bind address, port, turn-capture/build-context toggles, recipes |
| `[storage]`      | Where the SQLite database lives                                |
| `[embedding]`    | Model, vector dimensions, on-disk precision                    |
| `[search]`       | Default limit, score floor, BM25/vector blend                  |
| `[inference]`    | Summarization endpoint (used by the L2/L3 worker)              |
| `[embed]`        | Embedding subprocess pool — timeouts, retries, concurrency     |
| `[tls]`          | Cert/key paths if you terminate TLS in-process                 |
| `[logging]`      | Log directory, level, per-stream toggles                       |
| `[housekeeping]` | Retention, pass cadence, summary-pause minutes                 |
| `[import]`       | Minimum chunk size for markdown imports                        |
| `[paths]`        | Path normalization options                                     |
| `[models]`       | Aliases for inference model routing                            |

The shipped `example-settings.ini` is the single source of truth for the
full key list. The tables below cover the keys you're most likely to
touch.

## `[server]`

| Key              | Default         | Description |
|------------------|-----------------|-------------|
| `socket`         | `~/.ragger/ragger.sock` | Unix socket; the primary transport. Empty disables it. |
| `bind`           | `(empty)`       | TCP bind address. Empty = unix socket only. |
| `port`           | `8432`          | TCP port (only meaningful when `bind` is set). |
| `server_name`    | `(empty)`       | Hostname used by cpp-httplib (e.g. `ragger.local`). |
| `capture_turns`  | `true`          | **Write side.** When `true`, `capture_turn` (MCP) / `POST /turn` ingests agent-pushed turns into the `turns` table. Set `false` to make the call a no-op. |
| `build_context`  | `false`         | **Read side.** When `true`, `build_context` / `GET /session/<id>` assembles a recipe-shaped payload. Only meaningful when `capture_turns` is also on. Agent-driven `search`/`store` work either way. |
| `default_recipe` | `natural_fading` | Recipe used when the caller doesn't name one. See [Recipes](#recipes) below. |
| `recipes_dir`    | `~/.ragger/recipes` | Where the daemon loads recipe JSON from. Built-ins kick in only if this directory is missing or empty. |

## `[storage]`

| Key            | Default                  | Description                  |
|----------------|--------------------------|------------------------------|
| `db_path`      | `~/.ragger/memories.db`  | SQLite database file.        |
| `formats_dir`  | `~/.ragger/formats`      | Inference API format JSON.   |

## `[embedding]`

`model + dimensions + vector_type` define the DB's vector identity and
are recorded in the `settings` table. Changing any of them once the DB
holds data requires `ragger rebuild-embeddings` to re-encode everything;
mismatches abort startup with a clear error.

| Key          | Default            | Description                                                  |
|--------------|--------------------|--------------------------------------------------------------|
| `model`      | `all-MiniLM-L6-v2` | Sentence-transformer model name; resolved under `model_dir`. |
| `dimensions` | `384`              | Vector size; must match the model.                           |
| `vector_type`| `f16`              | On-disk precision. `f16` halves blob size; `f32` is full precision. In-memory math is always f32. |
| `model_dir`  | `~/.ragger/models` | Where ONNX model files live.                                 |

## `[search]`

| Key                 | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `default_limit`     | `5`     | Default result count.                                                       |
| `default_min_score` | `0.4`   | Cosine floor for returned results.                                          |
| `bm25_enabled`      | `true`  | Blend BM25 keyword scoring into the rank.                                   |
| `bm25_weight`       | `4`     | Weight for the BM25 signal (ratio, not percentage; integers preferred).     |
| `vector_weight`     | `8`     | Weight for the cosine signal.                                               |
| `phon_weight`       | `1`     | Weight for the phonetic "sounds-like" (Double Metaphone) signal. `0` off.   |
| `max_search_limit`  | `0`     | Ceiling on client-requested `limit` for sub-users. `0` disables the cap.    |

Weights are ratios; they're normalized internally. The default blend is
`8` vector / `4` bm25 / `1` phon — meaning-first, with keyword and a gentle
phonetic nudge. `phon_weight = 0` disables the sounds-like signal entirely.

## `[inference]` — summarization endpoint

Ragger only calls inference for L2/L3 summarization. There is no chat
surface.

| Key          | Default                       | Description |
|--------------|-------------------------------|-------------|
| `model`      | *(empty)*                     | Model name the summarizer asks for. Empty disables real summarization (drafts only). |
| `api_url`    | `http://localhost:1234/v1`    | OpenAI-compatible base URL. Default targets LM Studio's localhost convention. |
| `api_key`    | *(empty)*                     | Bearer key for the endpoint, if it needs one.   |
| `max_tokens` | `4096`                        | Cap for a single summarization call.            |

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
- **Keep `max_tokens` at 1024 or below.** The summarizer targets ¼ of
  the source length and caps at the source length. 1024 tokens is
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

For multiple endpoints (e.g. a fast local model plus a remote
fallback), declare `[inference.<name>]` sections; the client routes by
fnmatch globs in each section's `models` key. See
`example-settings.ini` for a worked example.

## `[housekeeping]`

| Key                     | Default | Description                                                                        |
|-------------------------|---------|------------------------------------------------------------------------------------|
| `cleanup_max_age_hours` | `0`     | Raw-turn lifetime. `0` = keep forever (default). Set to hours (e.g. `336` = 2 weeks) to auto-purge old turns. |
| `housekeeping_interval` | `60`    | Seconds between background passes. `0` disables; values under 10 are clamped to 10.|
| `summary_pause_minutes` | `20`    | Idle gap that closes a session's running L3 summary.                               |

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
|-------------|--------------------------|--------------------------------------------------------------------------|
| `log_file`  | `~/.ragger/activity.log` | Primary log file.                                                        |
| `log_level` | `warn`                   | One of `trace`, `debug`, `info`, `warn`, `error`, `critical`.            |
| `log_dir`   | `~/.ragger`              | Directory for per-stream logs (query, http, mcp).                        |
| `query_log` | `true`                   | Log every search query as single-line JSON to `query.log`.               |
| `http_log`  | `true`                   | Log HTTP requests.                                                       |
| `mcp_log`   | `true`                   | Log MCP requests.                                                        |
| `debug_log` | `false`                  | Opt-in per-chunk / per-embedding tracing.                                |

## Recipes

A recipe is a JSON file in `recipes_dir` (default
`~/.ragger/recipes/`) that describes how `build_context` should assemble
the payload for an agent. Each recipe is a list of layers walked back
from the latest turn:

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
   `default` means "track [server] default_recipe in settings.ini" —
   pick it to revert.
3. `[server] default_recipe` in `settings.ini`.
4. The first built-in if all of the above miss.

The DB lookup happens once per `build_context` call; changing your
choice with `ragger recipe` takes effect on the next request, no
daemon restart needed.

## Ceilings for sub-users

When the daemon serves additional users over HTTP (each with their own
bearer token), the owner can cap the parameters those clients send.
Today the only ceiling is `max_search_limit` (in `[search]`). A value
of `0` disables it.

For a personal install with no sub-users, leave ceilings at `0`.

## Reloading without a restart

`ragger reload` sends `SIGHUP` to the daemon, which re-reads the file
and applies hot-reloadable keys in place. Keys that require a restart
log a warning instead of silently being ignored. The reload also
re-scans `recipes_dir`.

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
-- The top hits for the most recent searches:
SELECT l.id, l.ts, l.query, h.rank, h.score, h.collection, h.snippet
FROM lookups l
JOIN hits h ON h.lookup_id = l.id
ORDER BY l.id DESC, h.rank
LIMIT 30;
```

The schema is deliberately small and obvious — open it with `sqlite3
~/.ragger/stats.db` and `.schema` to see every column, then slice it to answer
whatever question you're chasing (which queries return weak top scores, how
often a given collection surfaces, how the score spread changes, and so on).

## Related

- [Getting Started](getting-started.md) — Install + first run
- [Deployment](deployment.md) — Daemon lifecycle, sub-users
- [HTTP API](http-api.md) — Endpoints, including `/turn` and `/session/<id>`
- [Agent integration](agent-integration.md) — How agents use capture and `build_context`
