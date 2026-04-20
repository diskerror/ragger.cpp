# Configuration

Ragger uses layered INI config files for flexible deployment across
single-user and multi-user scenarios.

## Config File Locations

Ragger reads config in this order (later values override earlier ones):

1. **System config:** `/etc/ragger.ini` (or path specified via `--config`)
2. **User config:** `~/.ragger/settings.ini` (always read if it exists)

**Command-line override:**

```bash
ragger serve --config /custom/path/settings.ini
```

This replaces the system config location. User config (`~/.ragger/settings.ini`)
is still read on top if present.

## Auto-bootstrap

If no config exists on first run, Ragger auto-creates `~/.ragger/settings.ini`
with default settings. This makes single-user installs work out of the box.

## SERVER_LOCKED Keys

Keys marked as `SERVER_LOCKED` in the system config cannot be overridden
by user config. This prevents users from changing infrastructure settings
like host, port, or database path in multi-user deployments.

Example:

```ini
# System config (/etc/ragger.ini)
[server]
host = 0.0.0.0            # SERVER_LOCKED
port = 8432               # SERVER_LOCKED
```

If a user sets `port = 9000` in `~/.ragger/settings.ini`, the value is
ignored. The server still binds to port 8432.

**Currently locked keys:**

- `[server] host`
- `[server] port`
- `[storage] db_path`
- `[embedding] model`
- `[embedding] dimensions`

For single-user setups, `SERVER_LOCKED` has no effect (no system config).

## System Ceilings

System config can set hard limits on user-configurable values. Ceilings
are named with a `_limit` suffix:

| User Setting                 | System Ceiling                | Default Ceiling |
|------------------------------|-------------------------------|-----------------|
| `default_limit`              | `max_search_limit`            | 0 (no limit)    |
| `max_persona_chars`          | `max_persona_chars_limit`     | 0 (no limit)    |
| `max_memory_results`         | `max_memory_results_limit`    | 0 (no limit)    |
| `max_turn_retention_minutes` | — (hard-set in system config) | —               |
| `max_turns_stored`           | — (hard-set in system config) | —               |

**How it works:**

- If the user sets `default_limit = 50` but the system ceiling is
  `max_search_limit = 20`, the effective limit is 20.
- A ceiling of `0` means no ceiling — the user can set any value.

**Why ceilings?**

In multi-user deployments, ceilings prevent one user from overwhelming
shared resources (e.g., searching 10,000 results per query, loading
50 MB of persona files into every chat session).

For single-user setups, ceilings are rarely needed.

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
| `default_collection` | `memory`                | Default collection for new memories |

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

## Example Configs

### Example System Config (`/etc/ragger.ini`)

```ini
# System infrastructure settings (multi-user deployment)
[server]
host = 0.0.0.0
port = 8432
auth_token = your-secret-token-here

[storage]
db_path = /var/ragger/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 3
vector_weight = 7
max_search_limit = 50  # Users can't request >50 results

[chat]
# Hard limits for multi-user
max_turn_retention_minutes = 60
max_turns_stored = 100
max_persona_chars_limit = 10000
max_memory_results_limit = 10
```

### Example User Config (`~/.ragger/settings.ini`)

```ini
# Personal preferences
[search]
default_limit = 10
default_min_score = 0.3

[inference]
model = qwen/qwen2.5-coder-14b
api_base = http://localhost:1234/v1

[chat]
max_persona_chars = 4000  # Limit context for local models
max_memory_results = 2
pause_minutes = 15
```

## Related

- [Getting Started](getting-started.md) — Installation and setup
- [Chat Persistence](chat-persistence.md) — Turn storage and summarization
- [Deployment](deployment.md) — Multi-user setup
