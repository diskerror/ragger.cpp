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

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `store` | Store a memory for later retrieval | `text` (string) |
| `search` | Search memories by semantic similarity | `query` (string) |

**Optional `store` params:** `metadata` (object — category, tags, source, collection)

**Optional `search` params:** `limit` (integer), `min_score` (number), `collections` (string array)

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
- [Configuration](configuration.md) — Setting host, port, auth_token
- [Deployment](deployment.md) — Production setup with process managers
- [Python API](python-api.md) — Using RaggerMemory as a library
