# Configuration

Ragger uses a single INI config file.

## Config File Location

`~/.ragger/settings.ini` — one file, owned by the daemon owner.
There is no system-level config and no overlay; the file you see
is the file the daemon reads.

**Command-line override** (for testing alternate configs):

```bash
ragger serve --config /custom/path/settings.ini
```

## Auto-bootstrap

If `~/.ragger/settings.ini` is missing on first run, `install.sh`
copies `example-settings.ini` from the source tree. Fresh installs
work out of the box with sensible defaults.

## Ceilings

When the daemon serves additional sub-users over HTTP (each with a
token), the daemon owner can cap the parameters those clients send.
Ceilings are keys with the `_limit` suffix:

| Client-requested value       | Ceiling key                | Default      |
|------------------------------|----------------------------|--------------|
| `default_limit`              | `max_search_limit`         | 0 (no limit) |
| `max_persona_chars`          | `max_persona_chars_limit`  | 0 (no limit) |
| `max_memory_results`         | `max_memory_results_limit` | 0 (no limit) |
| `max_turn_retention_minutes` | (hard-set in settings.ini) | —            |
| `max_turns_stored`           | (hard-set in settings.ini) | —            |

**How it works:**

- If a client requests `default_limit = 50` but the ceiling is
  `max_search_limit = 20`, the effective limit is 20.
- A ceiling of `0` means no ceiling — any value is accepted.

**Why ceilings?**

When the daemon serves multiple sub-users over HTTP, ceilings prevent
one client from overwhelming shared resources (e.g., searching 10,000
results per query, loading 50 MB of persona files into every chat
session).

For a purely personal install (no sub-users), ceilings are rarely
needed — leave them at `0`.

## Full Config Reference

Below is a complete reference with defaults and descriptions. See also
`example-settings.ini` in the project root — a ready-to-edit template
that install.sh copies to `~/.ragger/settings.ini` on first run.

### `[server]`

| Key          | Default     | Description                           |
|--------------|-------------|---------------------------------------|
| `host`       | `127.0.0.1` | Bind address for HTTP server          |
| `port`       | `8432`      | Port for HTTP server                  |
| `auth_token` | (none)      | Bearer token for HTTP auth (optional) |

### `[storage]`

| Key                  | Default                 | Description                         |
|----------------------|-------------------------|-------------------------------------|
| `db_path`            | `~/.ragger/memories.db` | SQLite database path                |

### `[embedding]`

| Key          | Default            | Description                                 |
|--------------|--------------------|---------------------------------------------|
| `model`      | `all-MiniLM-L6-v2` | HuggingFace model name                      |
| `dimensions` | `384`              | Embedding vector size                       |
| `device`     | `cpu`              | Device for inference (`cpu`, `cuda`, `mps`) |

### `[search]`

| Key                 | Default | Description                                       |
|---------------------|---------|---------------------------------------------------|
| `default_limit`     | `5`     | Default number of results                         |
| `default_min_score` | `0.4`   | Minimum cosine similarity score                   |
| `bm25_enabled`      | `true`  | Enable BM25 hybrid search                         |
| `bm25_weight`       | `3`     | BM25 score weight (ratio, not percentage)         |
| `vector_weight`     | `7`     | Vector score weight (ratio, not percentage)       |
| `bm25_k1`           | `1.5`   | BM25 term frequency saturation                    |
| `bm25_b`            | `0.75`  | BM25 document length normalization                |
| `query_log`         | `true`  | Enable query logging to `~/.ragger/query.log`     |
| `max_search_limit`  | `0`     | Ceiling for user `default_limit` (0 = no ceiling) |

**Note on weights:** `bm25_weight = 3` and `vector_weight = 7` means
30% BM25, 70% vector (the ratio is normalized). Using integers avoids
floating-point config parsing issues.

### `[chat]`

| Key                          | Default | Description                                              |
|------------------------------|---------|----------------------------------------------------------|
| `store_turns`                | `true`  | Turn storage mode (`true`, `session`, `false`)           |
| `summarize_on_pause`         | `true`  | Summarize buffered turns after inactivity                |
| `summarize_on_quit`          | `true`  | Summarize buffered turns on exit                         |
| `pause_minutes`              | `10`    | Inactivity threshold for pause summarization             |
| `max_turn_retention_minutes` | `60`    | Delete turns older than this (0 = no limit)              |
| `max_turns_stored`           | `100`   | Keep at most this many recent turns (0 = no limit)       |
| `max_persona_chars`          | `0`     | Limit persona file size (0 = no limit)                   |
| `max_memory_results`         | `0`     | Limit memory search results in chat (0 = no limit)       |
| `max_persona_chars_limit`    | `0`     | System ceiling for `max_persona_chars` (0 = no ceiling)  |
| `max_memory_results_limit`   | `0`     | System ceiling for `max_memory_results` (0 = no ceiling) |

### `[inference]`

| Key        | Default | Description                         |
|------------|---------|-------------------------------------|
| `model`    | (none)  | Default LLM model for `ragger chat` |
| `api_base` | (none)  | OpenAI-compatible API base URL      |
| `api_key`  | (none)  | API key for inference endpoint      |

## Example

See `example-settings.ini` in the source tree — it's the file
`install.sh` copies to `~/.ragger/settings.ini` on first run, with
every section and sensible defaults. Edit your copy there; the
example in the source tree is just the template.

Condensed sample:

```ini
[server]
host = 127.0.0.1
port = 8432
auth_token = your-secret-token-here

[storage]
db_path = ~/.ragger/memories.db

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 3
vector_weight = 7
# Uncomment when serving sub-users and you want to cap their requests:
# max_search_limit = 50

[inference]
model = qwen/qwen2.5-coder-14b
api_base = http://localhost:1234/v1

[chat]
max_persona_chars = 4000
max_memory_results = 2
pause_minutes = 15
```

## Related

- [Getting Started](getting-started.md) — Installation and setup
- [Chat Persistence](chat-persistence.md) — Turn storage and summarization
- [Deployment](deployment.md) — Daemon lifecycle and sub-user provisioning
