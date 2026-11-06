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

The daemon listens on TCP (`127.0.0.1:8432` by default) and on the unix
socket `<ragger-base>/ragger.sock` at the same time.

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
  http://localhost:8432/count
```

Requests arriving on the unix socket or from loopback (`127.0.0.1` /
`::1`) are pre-authenticated as the default user, so token headers are
not required from the local machine.

Browsers can't set headers on `EventSource` or plain navigation, so the
token is also accepted as a query parameter (`?token=...`) — same trust
model: holding the token = authorized. The dashboard and its SSE stream
use this.

After auth, a cross-origin write guard runs: a browser request carrying an
`Origin` header whose authority doesn't match `Host` is refused with
`403` (`text/plain`). GET/HEAD/OPTIONS and requests with no `Origin`
(curl, CLI, plugins, MCP) pass through.

Every route below except **GET /health** requires auth.

## Error Envelopes

There are two shapes in use today (see Known Inconsistencies):

- **`text/plain`** — emitted by the auth wrapper and several field checks:
  `401 Unauthorized`, `403 ERROR: cross-origin request refused`,
  `400 Missing 'text' field`, `400 JSON error: <detail>`,
  `400 ERROR: <validation message>`, `500 ERROR: <detail>`.
- **`application/json`** — `{"error": "<message>"}` returned directly by
  handlers (config, models, `/turn` missing-field, …).

Validation failures (`std::invalid_argument` — bad ids, bad ports,
unknown statuses, illegal transitions) are always `400`, never `500`.

## Endpoints

34 routes are registered; each `###` heading below is one route.

### GET /health

No auth. Liveness + build identity + row counts.

**Response:**

```json
{
  "status": "ok",
  "version": "0.17.0",
  "commit": "<git describe or 'unknown'>",
  "built": "<build date>",
  "memories": 42,
  "embeddable_rows": 137,
  "degraded": false
}
```

`memories` counts knowledge rows; `embeddable_rows` counts all rows the
embedding rebuild would re-encode. `degraded` is true when stored
embeddings don't match the configured model (drift) and search has
fallen back to text-only scoring.

```bash
curl http://localhost:8432/health
```

---

### GET /count

Returns the total number of knowledge rows.

**Response:** `{"count": 42}`

---

### POST /store

Store a memory. The server decides the kind itself (same function as the
MCP `store` tool): `metadata.category == "decision"` stores a knowledge
**decision**, anything else a knowledge **note**.

**Request:**

```json
{
  "text": "The deploy script requires Node 18+",
  "metadata": { "tags": "docs,deployment", "source": "deployment-notes.md" }
}
```

- `text` (required) — memory content.
- `metadata` (optional) — recognized fields are promoted to columns
  (`tags`, `status`, `timestamp`, …); unknown fields are dropped.
  `source` defaults to the caller's username.
- `tags` — comma-separated labels; `keep` exempts from turn expiration.

**Response:** `{"id": "123", "status": "stored", "kind": "note"}`

**Status codes:** `400` missing `text` (text/plain), bad JSON, or a
rejected `status` value.

---

### POST /decision

Store a structured decision (knowledge `kind='decision'`).

**Request fields:** `text` (required), `trigger` (required — "Use when
…" phrasing), `rationale`, `scope`, `tags` (string `"a,b"` or array),
`status` (default `current`; must be creatable:
`current | provisional | pending`), `supersedes` (knowledge id this
replaces), `session` (GUID), `evidence` (array of episode ids or `"episode:<id>"`
handles; LR S80: evidence points at episodes, `turn_summary:`/`turn:` handles
are rejected with a pointer).

**Response:**

```json
{
  "id": "knowledge:5",
  "kind": "decision",
  "status": "stored",
  "superseded": 3,
  "possible_supersedes": [{"id": 4, "score": 0.81}]
}
```

`superseded` is the id of the row this one replaced (null if none).
`possible_supersedes` is a hint list of current decisions this row may
contradict — nothing is changed by it.

**Status codes:** `400` on any validation failure (missing trigger,
non-creatable status, unknown supersedes target).

---

### POST /search

Hybrid search (vector cosine + custom text index) over the four corpora.

**Request:**

```json
{ "query": "deployment requirements", "limit": 5, "min_score": 0.4 }
```

- `query` (required)
- `limit` (optional; default: the `default_search_limit` setting)
- `min_score` (optional; default: the `default_min_score` setting)
- `collections` — accepted, currently ignored (no collection column).

**Response:**

```json
{
  "results": [
    {
      "id": "knowledge:123",
      "text": "The deploy script requires Node 18+",
      "score": 0.823,
      "metadata": { "tags": "docs,deployment", "source": "knowledge" },
      "timestamp": "2026-03-20 15:23:45"
    }
  ],
  "timing": { "elapsed_ms": 12.3, "total_docs": 10614, "total_ms": 13 }
}
```

`metadata.source` labels the corpus: `turn_summary`, `episode`,
`knowledge`, or `document`.

---

### POST /memory/search

Layered agent search across turn summaries, episodes, and knowledge.
Documents are **not** searched here (since v0.17, S76) — use
`POST /memory/documents`.

**Request fields:** `query` (required), `layers` (subset of
`turn | turn_summary | episode | knowledge | session`; `turn` is raw exchange
text and `session` matches session titles (lexical only, never vector
scored) — both searched only when named explicitly; naming `document` is a
`400` pointing to `/memory/documents`; the input aliases `decision` /
`insight` restrict knowledge to that kind; empty = all),
`kinds` (`decision | insight | note` filter), `since` / `until`
(`YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, or relative `30d` / `12h`),
`status` (knowledge only: `active` (default = current+provisional) |
`any` | one exact status), `session`, `exclude_session`, `limit`
(default 10), `max_tokens` (0 = no budget), `mode` (`hybrid` (default) |
`semantic` (vector only; 400 without a ready embedder) | `keyword`
(text index only)), `tags` (array or comma string; knowledge rows must
carry ALL listed tags — AND rule, trimmed, case-insensitive; layers
without tags are excluded).

**Response:** `{"results": [AgentHit]}` where each hit is
`{"id": "<layer>:<rowid>", "layer", "score", "matched_by":
"vector|lexical|both", "created", "snippet", "extra"}` (`extra` is
layer-specific: knowledge → `{status, trigger}`, turn_summary →
`{turn_id, session}`, session → `{started_on, ended_on}` with
`id = session:<guid>`, …).

**Status codes:** `400` on unknown layer/kind or bad time format.

---

### POST /memory/documents

Search imported documents — their own verb (v0.17, S76). Same hybrid scoring
and hit shape as `/memory/search`, restricted to the documents corpus.

**Request fields:** `query` (required), `source` (only documents whose stored
source path contains this substring, case-insensitive), `limit` (default 10,
max 50), `mode` (`hybrid` | `semantic` | `keyword`, as for `/memory/search`).

**Response:** `{"results": [AgentHit]}` with `id: "document:<rowid>"` and
`extra: {title, source_path, chunk_index}`.

**Status codes:** `400` when `query` is missing.

---

### POST /memory/get

Full rows by typed handle, or one session's conversation history. Exactly
one mode per call.

**Request fields:** `ids` (array of typed handles — mode 1) XOR `session`
(GUID — mode 2, with `after_turn`, `before_turn`, `limit` clamped
[1,100], default 20); `expand`: `none | context | sources | chain`.

**Response:** `{"items": [...]}` — full row objects per handle (or
`{"id", "error": "not found"}` entries); mode 2 returns the session's
turns, each carrying `turn_summary_id` and `episode_id` (`null` when the
turn has none) so a reader can fill `evidence[]` handles.

**Status codes:** `400` if both or neither mode is set, or a handle is
malformed.

---

### POST /memory/note

Store a note (knowledge `kind='note'`), or an insight when `evidence` is
supplied.

**Request fields:** `text` (required), `tags` (optional), `evidence`
(optional array of episode ids or `"episode:<id>"` handles — when present
the row is stored as `kind='insight'` linked to its sources).

**Response (note):** `{"id": "123", "status": "stored",
"possible_supersedes": [...]}`
**Response (insight):** `{"id": "knowledge:5", "kind": "insight",
"status": "stored", "superseded": null, "possible_supersedes": [...]}`

**Status codes:** `400` missing `text`, bad evidence ids.

---

### POST /memory/timeline

Deterministic newest-first range scan (no embeddings).

**Request fields:** `session` (GUID; empty = all), `since` / `until`
(same formats as `/memory/search`), `granularity`:
`turn | episode | session | knowledge` (default `episode`), `limit`
(default 20, clamped [1,100]), `cursor` (opaque
`"<created_epoch>:<rowid>"` from the previous page; empty = newest first
page).

**Response:** `{"items": [...], "next_cursor": "<...>" | null}`

---

### POST /memory/update

Metadata-only curation of a knowledge row. Text is never edited in place.

**Request fields:** `id` (typed handle `knowledge:<n>`), `status`
(must be agent-settable and a legal transition from the row's current
status; `superseded` is never settable here), `tags` (array or
comma-separated string; replaces the whole tag set), `text` (present ⇒
always rejected).

**Response:** the updated row (same shape as `/memory/get` items).

**Status codes:** `400` for legacy handles (`decision:<n>`,
`insight:<n>`, `summary:<n>` → "use knowledge:<id>"), immutable layers
(episodes, turns, turn summaries, documents), unknown status, illegal
transition, or `text` supplied.

---

### POST /memory/review

The pending-review queue.

**Request fields:** `action`: `list | approve | reject`; `ids` (required
for approve/reject; `knowledge:<n>` handles or bare integers).
Approve moves each row `pending → current`, reject `pending → retired`;
a non-pending id fails only that id.

**Response (list):** `{"pending": [{"id", "trigger", "snippet", ...}]}`
**Response (approve/reject):** `{"done": [...], "failed": [{"id", "error"}]}`

---

### POST /turn

Ingest one completed conversation turn (user + assistant) for background
summarization. Called from an agent's turn hook. The same payload is
accepted by the `capture_turn` MCP tool. Requires `[server] capture_turns
= true`; otherwise the call is accepted but a no-op (`status:
"disabled"`).

**Body fields:** `user` (required), `assistant`, `model`, `session_id`
(groups turns into a session), `session_name`, `name_source`.

**Response:** `{ "status": "captured" | "disabled", "turn_id": <int> }`

When captured with a non-empty `assistant`, the turn is handed straight
to the daemon's summarizer queue (the request never waits for inference).

**Status codes:** `400 {"error":"missing required field: user"}` (JSON
envelope).

```bash
curl -X POST http://localhost:8432/turn \
  -H "Content-Type: application/json" \
  -d '{"user": "how do I rebuild?", "assistant": "run build.sh clean", "model": "devstral", "session_id": "sess-abc-123"}'
```

---

### GET /session/&lt;session_id&gt;

Assemble a session's context as recipe-shaped chunks (oldest first) — the
read-side counterpart to `/turn`, and the same payload returned by the
`build_context` MCP tool. Requires `[server] capture_turns` **and**
`build_context`; otherwise returns `{ "status": "disabled" }`.

Optional query parameter: `?recipe=<name>`. Resolution: explicit query
arg → DB
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
      "kind":      "episode",
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

`kind` is one of `raw_turn`, `turn_summary`, `episode`, `decision`,
`general_search`. `timestamp` is the source-row timestamp (empty for
cross-session items).

```bash
curl http://localhost:8432/session/sess-abc-123
curl http://localhost:8432/session/sess-abc-123?recipe=reconnect
```

---

### POST /search_by_metadata

Metadata-only scan (no scoring).

**Request:** `{"metadata": {...}, "limit": 0, "after": "", "before": ""}`
— `metadata` object required; `limit` 0 = unlimited; `after`/`before` are
timestamp bounds.

**Response:** `{"results": [same shape as /search results], "count": N}`

---

### GET /user/token

Return the caller's own bootstrap token, read from `~/.ragger/token` in
the authenticated system user's home directory.

**Response:** `{"token": "<token>", "username": "<name>"}`; `404`
(JSON) when the system user or the token file is missing.

---

### GET /dashboard

The single-page control panel (embedded HTML/JS, `text/html`). From
non-loopback it needs `?token=...`; the page then talks to `/config`,
`/stats`, and `/events` with the same token.

---

### GET /config

All settings as an array of entries in dashboard-tab order; each entry
carries its current value plus schema metadata (type, default, edit
policy, enum options) so a UI can render the right control. Secret
fields surface as `"<set>"`, never their real value.

---

### GET /config/&lt;key&gt;

One setting (same entry shape as `GET /config`). `404`
`{"error":"unknown key"}` for keys not in the schema.

---

### PUT /config/&lt;key&gt;

Set one setting. **Body:** `{"value": "<string>"}` — an empty value
reverts the key to its schema default. Writing a secret field back as
`"<set>"` is a no-op (the mask never overwrites the real credential).

**Response:** `{"key", "value", "restart_required", "rebuild_required"}`
(value is the resolved post-default one).

**Status codes:** `404` unknown key, `403` locked key, `400` invalid
value (all JSON envelope).

The write updates the daemon's live config snapshot immediately (this
route runs inside the daemon) and persists to the DB `settings` table.
The CLI `ragger config set` shares the same path and additionally signals
the running daemon (SIGHUP) to reload.

---

### GET /stats

One-shot status snapshot: `{"status": "running", "version", "memories",
"tables": {"turns": N, "turn_summaries": N, "episodes": N, "knowledge":
N, "sessions": N, "documents": N, "models": N}}`.

---

### GET /models

ONNX embedding models available on disk under `<ragger-base>/models/`.
Only the two-level `provider/model` layout is offered as a new choice
(legacy flat dirs are kept for compatibility but not listed). A directory
counts as a model when it has `tokenizer.json` and
(`model.onnx` or `onnx/model.onnx`).

**Response:** `{"models": [{"name": "<display>", "path":
"<provider>/<model>"}]}` — `path` is the canonical key for the config;
colliding display names get the provider prefix.

---

### GET /models/info?path=&lt;provider/model&gt;

Read `config.json` from a local model dir.

**Response:** `{"dimensions": N, "model_type": "...", "hidden_size": N}`;
`400` missing `path`, `404` no `config.json`, `500` unparseable.

---

### GET /models/external?host=X&amp;port=Y

Query a remote embedding server's `/v1/models`. The API key is **not** a
parameter — it comes from config (`embedding_external_api_key`); passing
one in the URL would leak a credential.

**Response:** `{"models": ["...", ...]}`; `400` missing host/port or
non-numeric port.

---

### POST /models/external/probe

Send a test embedding request to a remote endpoint and report the vector
dimensionality. **Body:** `{"host": "...", "port": N, "model": "..."}`
(key from config, not body).

**Response:** `{"dimensions": N, "served_model": "...",
"model_mismatch": bool}`; `400` missing fields, `502` probe failed or
implausible embedding (likely not an embedding model).

---

### GET /models/huggingface?q=&lt;search&gt;

Search HuggingFace for ONNX embedding models (both
`sentence-similarity` and `feature-extraction` pipeline tags, merged,
deduped, sorted by downloads, top 20 each).

**Response:** `{"models": [{"id": "<repo>", "downloads": N}]}`

---

### POST /models/download

Download an ONNX model from HuggingFace into
`<ragger-base>/models/<provider>/<model>/`.

**Body:** `{"repo": "provider/model"}` — strictly validated (no path
traversal).

**Response:** `{"status": "downloaded" | "already_present", "path":
"<repo>"}`; `400` bad/missing repo, `502` model.onnx download failed.

---

### GET /models/summarizer

List models on the summarizer endpoint (`summarizer_api_url`, falling
back to `inference_api_url`; empty `{"models":[]}` when neither is set).
Prefers LM Studio's authoritative type classification (drops
`embeddings`); otherwise filters names containing `embed`/`minilm`.

**Response:** `{"models": ["...", ...]}`

---

### GET /models/embedding-external

List embedding-capable models on the local inference endpoint (the
inverse of `/models/summarizer`). Prefers LM Studio's typed listing
(keeps `embeddings` only); otherwise keeps names matching embedding hints
(`embed`, `minilm`, `bge`, `gte`, `e5-`, `nomic`, `gist`,
`sentence-t5`, `sentence-transformer`, `arctic-embed`, `uae-`).

**Response:** `{"models": ["...", ...]}`

---

### POST /models/embedding-external/probe

Send a test embedding to the configured inference/summarizer endpoint.
**Body:** `{"model": "<name>"}`.

**Response:** `{"dimensions": N, "served_model": "...",
"model_mismatch": bool}`; `400` missing model or no endpoint configured,
`502` probe failed or implausible embedding.

---

### GET /embedding/status

Current-vs-desired embedding identity and flags:

```json
{
  "current": {"model": "...", "vector_type": "...", "dimensions": 384, "engine": "..."},
  "desired": {"model": "...", "vector_type": "...", "dimensions": 384, "engine": "..."},
  "needs_update": false,
  "reembedding": false,
  "repair_pending": false,
  "degraded": false
}
```

---

### POST /embedding/update

Run the staged re-embed now (no body). Runs in a background thread — the
request returns immediately; progress shows in `/embedding/status`
(`reembedding`) and the SSE stream.

**Response:** `{"status": "started", "reembedding": true}` or
`{"status": "up-to-date", "reembedding": false}`; `409` when a re-embed
is already running.

---

### GET /restart/status

Bound (current) vs configured (desired) listen parameters:

```json
{
  "bound_port": 8432, "configured_port": 8432,
  "bound_bind": "127.0.0.1", "configured_bind": "127.0.0.1",
  "needs_restart": false,
  "scheme": "http"
}
```

`configured_port` is `desired_port` when set, else the committed `port`.

---

### POST /restart

Re-exec the daemon in place. Returns immediately with the URL the
dashboard should reconnect to; then the listeners tear down and the
supervisor re-execs.

**Response:** `{"status": "restarting", "reconnect_url":
"http://127.0.0.1:8432/dashboard", "port": 8432}`

---

### GET /events

Server-Sent Events stream (one-directional; the dashboard subscribes with
`EventSource`, `?token=...` from non-loopback). Every ~2 s it pushes:

- `event: stats` — the `GET /stats` payload.
- `event: activity` — `{"line": "..."}` for each new activity-log line
  since the last drain (first read starts at EOF, so only new lines
  stream).

Config writes go through `PUT /config`, not this stream.

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

| Tool              | Description                                                                | Required params                 |
|-------------------|----------------------------------------------------------------------------|---------------------------------|
| `store`           | Store a memory (decision when `metadata.category="decision"`, else note)   | `text` (string)                 |
| `search`          | Hybrid semantic + keyword search                                           | `query` (string)                |
| `status`          | Server status (version, counts)                                            | —                               |
| `capture_turn`    | Hand a completed user→assistant turn to the daemon for summarization       | `user` (string)                 |
| `build_context`   | Return a recipe-shaped context payload for a session                       | `session_id` (string)           |
| `get_config`      | Read live configuration (all settings, or one `key`)                       | — (`key` optional)              |
| `memory_decision` | Store a structured decision                                                | `text`, `trigger`              |
| `memory_note`     | Store a note, or an insight with `evidence`                                | `text`                          |
| `memory_search`   | Layered search across all memory layers                                    | `query` (string)                |
| `memory_get`      | Full rows by typed handle, or session history paging                       | — (exactly one mode per call)   |
| `memory_timeline` | Deterministic newest-first range scan                                      | —                               |
| `memory_update`   | Metadata-only curation of a knowledge row                                  | `id` (handle)                   |

**Optional `search` params:** `limit` (integer; default: the
`default_search_limit` setting), `min_score` (number). The `collections`
parameter is accepted but is a no-op in the current schema.

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
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"ragger-memory","version":"<build version>"}}}

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

## Known Inconsistencies

Documented as-is; no fixes promised here:

- **Two error envelopes.** The auth wrapper and several field checks
  return `text/plain` (`Unauthorized`, `Missing 'text' field`,
  `ERROR: ...`, `JSON error: ...`), while config/models/`/turn` handlers
  return `application/json` `{"error": ...}`. Clients must handle both.
- **`collections` is inert** on `/search` and MCP `search` (accepted,
  ignored — the lean schema has no collection column).
- **`/health` is unauthenticated** and exposes `commit`/`built` to any
  peer that can reach the port (trimming it for non-local peers is an
  open question).

---

## Related

- [Getting Started](getting-started.md) — Running the server
- [Configuration](configuration.md) — Setting host, port, capture/build flags, recipes
- [Deployment](deployment.md) — Service units, sub-users, reverse proxy
- [Agent integration](agent-integration.md) — When agents call which tool
