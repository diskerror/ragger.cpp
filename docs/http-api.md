# HTTP API

Ragger runs as an HTTP server for tool integration with AI agents,
IDEs, and custom applications.

## Starting the Server

Install the user daemon once (`./install-bin.sh` writes the LaunchAgent /
systemd-user unit), then control it with:

```bash
ragger start        # bring the daemon up
ragger status       # check whether it's running
ragger restart      # bounce it after changing restart-required config
ragger stop         # take it down
```

Host, port, and bind interface come from the config (compiled-in
defaults overlaid by the DB `settings` table); change them with the
dashboard or `ragger config set` — all three are restart-required.

`ragger serve` is the foreground entry the daemon process itself
invokes — you won't normally run it directly. Use it for debugging:

```bash
ragger serve                # foreground, 127.0.0.1:8432 from config
ragger serve --port 9000    # override port
ragger serve --host 0.0.0.0 # bind all interfaces
```

See [Deployment](deployment.md) for the full lifecycle.

## Authentication

On first start, the daemon auto-generates a random token and writes it to
`~/.ragger/token`. Run `ragger add-self` once to register that token with
the database, then include it in the `Authorization` header:

```bash
# Register your token (one-time, after first daemon start)
ragger add-self

# Read the token
cat ~/.ragger/token

# Use it in requests
curl -H "Authorization: Bearer $(cat ~/.ragger/token)" \
  http://localhost:8432/health
```

Requests arriving on the unix socket (`~/.ragger/ragger.sock`) or from
localhost (`127.0.0.1` / `::1`) are pre-authenticated as the default user,
so token headers are not required from the local machine.

## Endpoints

### Health Check

**GET** `/health`

Returns server status and memory count.

**Response:**

```json
{
  "status": "ok",
  "memories": 42
}
```

**Example:**

```bash
curl http://localhost:8432/health
```

---

### Count Memories

**GET** `/count`

Returns the total number of stored memories.

**Response:**

```json
{
  "count": 42
}
```

**Example:**

```bash
curl http://localhost:8432/count
```

---

### Store Memory

**POST** `/store`

Store a new memory with optional metadata.

**Request:**

```json
{
  "text": "The deploy script requires Node 18+",
  "metadata": {
    "tags": "docs,deployment",
    "source": "deployment-notes.md"
  }
}
```

**Fields:**

- `text` (required) — Memory content
- `metadata` (optional) — JSON object with arbitrary fields

**Common metadata fields:**

- `tags` — Comma-separated labels (e.g. `"docs,deployment"`). Use `keep` to exempt from turn expiration.
- `source` — Where this memory came from
- `keep` — Boolean shorthand; equivalent to including `"keep"` in tags

**Response:**

```json
{
  "id": "123",
  "status": "stored"
}
```

**Example:**

```bash
curl -X POST http://localhost:8432/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploy to staging every Friday", "metadata": {"category": "preference"}}'
```

---

### Search Memories

**POST** `/search`

Search memories using hybrid vector + BM25 search.

**Request:**

```json
{
  "query": "deployment requirements",
  "limit": 5,
  "min_score": 0.4
}
```

**Fields:**

- `query` (required) — Search query
- `limit` (optional) — Maximum results (default: from config, usually 5)
- `min_score` (optional) — Minimum cosine similarity score (default: from config, usually 0.4)

> **Note:** The `collections` parameter is accepted by the API but has no effect in the current schema — there is no collection column. Use `tags` when storing to group memories; filter by `tags` via `/search_by_metadata`.

**Response:**

```json
{
  "results": [
    {
      "id": "123",
      "text": "The deploy script requires Node 18+",
      "score": 0.823,
      "metadata": {
        "tags": "docs,deployment",
        "source": "deployment-notes.md"
      },
      "timestamp": "2024-03-20T15:23:45"
    }
  ],
  "timing": {
    "elapsed_ms": 12.3,
    "total_docs": 10614
  }
}
```

**Score:** Raw cosine similarity (0.0 to 1.0). Higher is better.

**Example:**

```bash
curl -X POST http://localhost:8432/search \
  -H "Content-Type: application/json" \
  -d '{"query": "API authentication", "limit": 3}'
```

---

**POST** `/turn`

Ingest one completed conversation turn (user + assistant) for background
summarization. Called from an agent's turn hook. The same five-field payload is
accepted by the `capture_turn` MCP tool. Requires `[server] capture_turns = true`;
otherwise the call is accepted but a no-op (`status: "disabled"`).

Body fields: `user` (required), `assistant`, `model`, `session_id` (groups turns
for session summaries). Response: `{ "status": "captured" | "disabled", "turn_id": <int> }`.

```bash
curl -X POST http://localhost:8432/turn \
  -H "Content-Type: application/json" \
  -d '{"user": "how do I rebuild?", "assistant": "run build.sh clean", "model": "devstral", "session_id": "sess-abc-123"}'
```

---

**GET** `/session/<session_id>[?recipe=<name>]`

Assemble a session's context as recipe-shaped chunks (oldest first) — the
read-side counterpart to `/turn`, and the same payload returned by the
`build_context` MCP tool. Requires `[server] capture_turns` **and**
`build_context`; otherwise returns `{ "status": "disabled" }`.

`recipe` is optional. Resolution: explicit query arg → DB
`settings.recipe` (set by `ragger recipe`) → `[server] default_recipe`
from config → first built-in.

Response shape:

```json
{
  "status":     "ok",
  "session_id": "sess-abc-123",
  "recipe":     "natural_fading",
  "chunks": [
    {
      "kind":      "session_summary",
      "text":      "Discussed deployment pipeline; chose Node 18.",
      "timestamp": "2026-03-15 14:02:11"
    },
    {
      "kind":      "raw_turn",
      "text":      "User: how do I rebuild?\n\nAssistant: run build.sh clean",
      "timestamp": "2026-03-15 14:21:07"
    }
  ]
}
```

`kind` is one of `raw_turn`, `turn_summary`, `session_summary`,
`project_summary`, `decision`. `timestamp` is the source-row timestamp
(empty for cross-session items like project summaries).

```bash
curl http://localhost:8432/session/sess-abc-123
curl http://localhost:8432/session/sess-abc-123?recipe=reconnect
```

---

## MCP Server

Ragger implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
for integration with AI agents. Protocol version: `2024-11-05`.

**When to use MCP vs HTTP:** MCP is ideal for purely local use where
the agent fork+execs `ragger mcp` directly against the same
`~/.ragger/memories.db` — no daemon, no auth overhead. When the
daemon serves additional sub-users over HTTP (with tokens), prefer
HTTP for those clients — it already handles auth, routing, and
concurrent access.

### Starting the MCP Server

```bash
ragger mcp
```

The MCP server runs over stdin/stdout using JSON-RPC 2.0. It implements the
standard MCP handshake and tool discovery:

1. **`initialize`** — Returns server info (`ragger-memory`) and capabilities
2. **`notifications/initialized`** — Client acknowledgment (no response)
3. **`tools/list`** — Returns available tools with JSON Schema definitions
4. **`tools/call`** — Executes a tool and returns results

### Tools

| Tool            | Description                                                                  | Required params       |
|-----------------|------------------------------------------------------------------------------|-----------------------|
| `store`         | Store a memory for later retrieval                                           | `text` (string)       |
| `search`        | Hybrid semantic + keyword search                                             | `query` (string)      |
| `capture_turn`  | Hand a completed user→assistant turn to the daemon for summarization         | `user` (string)       |
| `build_context` | Return a recipe-shaped context payload for a session                         | `session_id` (string) |

**Optional `store` params:** `metadata` (object — tags, source).

**Optional `search` params:** `limit` (integer), `min_score` (number).
The `collections` parameter is accepted but is a no-op in the current schema.

**Optional `capture_turn` params:** `assistant`, `model`, `session_id`,
`metadata`. Requires `[server] capture_turns = true`; otherwise the
call returns `{"status":"disabled"}`.

**Optional `build_context` params:** `recipe` (string — name from
`~/.ragger/recipes/`). Requires `[server] capture_turns` **and**
`build_context`. Same response shape as the HTTP
`GET /session/<id>` endpoint above.

### Example Session

```json
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"ragger-memory","version":"0.7.0"}}}

→ {"jsonrpc":"2.0","method":"notifications/initialized"}

→ {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
← {"jsonrpc":"2.0","id":2,"result":{"tools":[...]}}

→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search","arguments":{"query":"deployment requirements","limit":5}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"[...]"}]}}
```

### Plain Text Mode

The MCP server also accepts plain text queries for interactive use.
Non-JSON lines are treated as search queries with human-readable output:

```
> deployment requirements
1. [score: 0.823] (deployment-notes.md) [memory]
   The deploy script requires Node 18+

Timing: 12.3ms (10614 chunks)
```

JSON-RPC and plain text can be interleaved freely in the same session.

```bash
echo "API authentication" | ragger mcp
```

See the [MCP specification](https://modelcontextprotocol.io/docs/spec/)
for full protocol details.

---

## Error Responses

All endpoints return JSON error responses for failures:

```json
{
  "error": "Invalid request: missing 'text' field"
}
```

**HTTP status codes:**

- `200 OK` — Success
- `400 Bad Request` — Invalid request (missing fields, malformed JSON)
- `401 Unauthorized` — Missing or invalid bearer token
- `500 Internal Server Error` — Server-side error

**Example:**

```bash
curl -X POST http://localhost:8432/store \
  -H "Content-Type: application/json" \
  -d '{}'
# {"error": "Invalid request: missing 'text' field"}
```

---

## Related

- [Getting Started](getting-started.md) — Running the server
- [Configuration](configuration.md) — Setting host, port, capture/build flags, recipes
- [Deployment](deployment.md) — Service units, sub-users, reverse proxy
- [Agent integration](agent-integration.md) — When agents call which tool
