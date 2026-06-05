# HTTP API

Ragger runs as an HTTP server for tool integration with AI agents,
IDEs, and custom applications.

## Starting the Server

Install the user daemon once (`./install.sh` writes the LaunchAgent /
systemd-user unit), then control it with:

```bash
ragger start        # bring the daemon up
ragger status       # check whether it's running
ragger restart      # bounce it after editing settings.ini
ragger stop         # take it down
```

Host, port, and bind interface come from `~/.ragger/settings.ini`.

`ragger serve` is the foreground entry the daemon process itself
invokes — you won't normally run it directly. Use it for debugging:

```bash
ragger serve                # foreground, 127.0.0.1:8432 from config
ragger serve --port 9000    # override port
ragger serve --host 0.0.0.0 # bind all interfaces
```

See [Deployment](deployment.md) for the full lifecycle.

## Authentication

Bearer token authentication is optional. Set via config:

```ini
[server]
auth_token = your-secret-token-here
```

Include the token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer your-secret-token-here" \
  http://localhost:8432/health
```

If no `auth_token` is set, the server allows unauthenticated access
(suitable for localhost-only deployments).

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
    "category": "fact",
    "source": "deployment-notes.md",
    "collection": "memory"
  }
}
```

**Fields:**

- `text` (required) — Memory content
- `metadata` (optional) — JSON object with arbitrary fields

**Common metadata fields:**

- `collection` — Collection name (defaults to `memory`)
- `category` — Memory type (fact, decision, preference, lesson, session-summary)
- `source` — Where this memory came from
- `keep` — Boolean, exempt from turn expiration (chat persistence)

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
  "min_score": 0.4,
  "collections": [
    "memory",
    "docs"
  ]
}
```

**Fields:**

- `query` (required) — Search query
- `limit` (optional) — Maximum results (default: from config, usually 5)
- `min_score` (optional) — Minimum cosine similarity score (default: from config, usually 0.4)
- `collections` (optional) — Collections to search (default: all collections)

**Collections:**

- Omit `collections` to search all collections
- Pass `["*"]` to explicitly search everything
- Pass specific names to filter: `["memory", "docs"]`

**Response:**

```json
{
  "results": [
    {
      "id": "123",
      "text": "The deploy script requires Node 18+",
      "score": 0.823,
      "metadata": {
        "category": "fact",
        "source": "deployment-notes.md",
        "collection": "memory"
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
  -d '{"query": "API authentication", "limit": 3, "collections": ["docs"]}'
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
in `settings.ini` → first built-in.

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

**Optional `store` params:** `metadata` (object — category, tags,
source, collection).

**Optional `search` params:** `limit` (integer), `min_score` (number),
`collections` (string array).

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
