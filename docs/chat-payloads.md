# Chat payload shapes

Concrete JSON for every wire format Ragger speaks during a chat turn.
Sample content is taken from this very conversation so the shapes are
realistic rather than toy.

Three hops per chat turn:

```
┌────────────┐  A. POST /chat        ┌──────────┐  B. POST /v1/chat/completions  ┌──────────────┐
│ web/MCP    │ ─────────────────────▶│  Ragger  │ ──────────────────────────────▶│ LM Studio /  │
│  client    │ ◀─── SSE tokens ──────│  Server  │ ◀────── SSE chunks ────────────│ Ollama / etc │
└────────────┘                       └──────────┘                                └──────────────┘
```

- **Hop A**: client ↔ Ragger's own `/chat` endpoint
- **Hop B**: Ragger ↔ upstream inference engine (format depends on endpoint)
- **Hop C**: optional `lm_proxy_url` passthrough for `/v1/*` routes (no rewrite)

Each section below shows the exact JSON for each hop.

---

## A. Client ↔ Ragger `/chat`

### A.1 Request: `POST /chat`

```http
POST /chat HTTP/1.1
Host: ragger.local:8432
Authorization: Bearer <user-token>
Content-Type: application/json

{
  "message": "Can ragger use an external engine for embedding?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "claude-opus-4-7"
}
```

| Field        | Type   | Required | Notes                                                        |
|--------------|--------|----------|--------------------------------------------------------------|
| `message`    | string | yes      | User turn input                                              |
| `session_id` | string | no       | Omit on first turn — server generates & returns a new UUID   |
| `model`      | string | no       | Override user's preferred / config default                   |

### A.2 Response: `text/event-stream`

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Session-Id: 550e8400-e29b-41d4-a716-446655440000

data: {"token":"Technically"}

data: {"token":" straightforward"}

data: {"token":" — LM Studio exposes"}

data: {"token":" `/v1/embeddings`..."}

data: {"done":true,"session_id":"550e8400-e29b-41d4-a716-446655440000"}
```

Error form (e.g. upstream model unavailable):

```
data: {"error":"endpoint 'lmstudio' unreachable at http://localhost:1234"}

data: {"done":true}
```

---

## B. Ragger ↔ upstream inference engine

Ragger looks up the endpoint by glob-matching the `model` field against
each configured endpoint's `models` pattern, loads that endpoint's
`format` (e.g. `openai`, `anthropic`) from
`formats/<name>.json`, and builds the request with the format's
`request_transform` rules.

### B.1 OpenAI format (LM Studio, Ollama `/v1`, OpenAI itself)

**Request → `POST <api_url>/chat/completions`:**

```http
POST /v1/chat/completions HTTP/1.1
Host: localhost:1234
Authorization: Bearer sk-local-xxxxxxxx
Content-Type: application/json

{
  "model": "qwen/qwen3-coder-30b",
  "max_tokens": 4096,
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "# SOUL.md\nYou are Reid's coding partner...\n\n# USER.md\nReid — solo dev, C++/Python...\n\n## Relevant memories:\n\nall-MiniLM-L6-v2 is the default embedding model, 384 dims.\n\n---\n\nReid prefers concise answers with field-level precision over prose summaries."
    },
    {
      "role": "user",
      "content": "Can ragger use an external engine for embedding?"
    },
    {
      "role": "assistant",
      "content": "Technically straightforward — LM Studio exposes /v1/embeddings (OpenAI-compatible)..."
    },
    {
      "role": "user",
      "content": "Still thinking future: how about just adding the ability to point to a different embedding model besides setting the size?"
    }
  ]
}
```

Notes:
- `system` goes in as `messages[0]` with `role=system`.
- `stream` is `true` for `/chat` (Ragger pipes tokens out); `false` for
  background summarisation.
- `max_tokens` comes from the endpoint's `max_tokens` (non-zero
  overrides the global `inference_max_tokens`).

**Streaming response chunks:**

```
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Technically"}}]}

data: {"choices":[{"delta":{"content":" straightforward"}}]}

data: {"choices":[{"delta":{"content":" — LM Studio"}}]}

data: {"choices":[{"finish_reason":"stop","delta":{}}]}

data: [DONE]
```

Ragger extracts `choices[0].delta.content` from each chunk. Empty
deltas (first and last) are silently skipped. Stream terminates on the
literal line `data: [DONE]`.

**Non-streaming response** (e.g. background summarisation):

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "qwen/qwen3-coder-30b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Technically straightforward — LM Studio exposes /v1/embeddings..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 412, "completion_tokens": 87, "total_tokens": 499}
}
```

Ragger extracts `choices[0].message.content`. Everything else is ignored.

### B.2 Anthropic format

**Request → `POST <api_url>/messages`:**

```http
POST /v1/messages HTTP/1.1
Host: api.anthropic.com
x-api-key: sk-ant-xxxxxxxx
anthropic-version: 2023-06-01
Content-Type: application/json

{
  "model": "claude-opus-4-7",
  "max_tokens": 4096,
  "stream": true,
  "system": "# SOUL.md\nYou are Reid's coding partner...\n\n# USER.md\nReid — solo dev...\n\n## Relevant memories:\n\nall-MiniLM-L6-v2 is the default embedding model, 384 dims.\n\n---\n\nReid prefers concise answers with field-level precision over prose summaries.",
  "messages": [
    {
      "role": "user",
      "content": "Can ragger use an external engine for embedding?"
    },
    {
      "role": "assistant",
      "content": "Technically straightforward — LM Studio exposes /v1/embeddings..."
    },
    {
      "role": "user",
      "content": "Still thinking future: how about just adding the ability to point to a different embedding model besides setting the size?"
    }
  ]
}
```

Key differences from OpenAI:
- `system` is its own **top-level field**, not a message.
- `messages` contains **only** `user` and `assistant` turns.
- Auth header is `x-api-key:` (not `Authorization: Bearer`); also
  `anthropic-version` is sent (from `auth_extra` in the format file).
- Endpoint path is `/v1/messages` (not `/v1/chat/completions`).

**Streaming response chunks** (SSE; events are typed):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01","role":"assistant","content":[]}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Technically"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" straightforward"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}
```

Ragger filters on `type == "content_block_delta"` before extracting
`delta.text`. All other chunk types are dropped. Stream terminates on
`type == "message_stop"`.

**Non-streaming response:**

```json
{
  "id": "msg_01ABC",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-7",
  "content": [
    {
      "type": "text",
      "text": "Technically straightforward — LM Studio exposes /v1/embeddings..."
    }
  ],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 412, "output_tokens": 87}
}
```

Ragger extracts `content[0].text`.

---

## C. `lm_proxy_url` passthrough

If `lm_proxy_url` is set in `settings.ini`, Ragger exposes
`/v1/chat/completions` and `/v1/completions` routes that forward
verbatim — **no rewriting** of method, headers (other than standard
proxying), path, or body. Whatever the client sends is what the
upstream sees. Whatever the upstream replies is what the client sees.

This exists so tools that only speak OpenAI-compat (e.g. IDE plugins)
can point at Ragger and share the same local LM Studio as the chat
REPL, without having to know Ragger's own `/chat` shape.

---

## Appendix — message assembly

For hop B, Ragger's `ChatSession::build_messages(system_prompt,
memory_context, max_turns)` produces the message array like this:

```
messages[0] = { role: "system",    content: workspace_files + "\n\n## Relevant memories:\n\n" + memory_context }
messages[1] = { role: "user",      content: <turn 1 user text>     }
messages[2] = { role: "assistant", content: <turn 1 assistant text>}
messages[3] = { role: "user",      content: <turn 2 user text>     }
...
messages[N] = { role: "user",      content: <current user input>   }
```

Where:

- **workspace_files**: concatenation of `~/.ragger/SOUL.md`,
  `USER.md`, `AGENTS.md`, `TOOLS.md`, truncated by priority to fit
  `chat_persona_pct` of the context window.

- **memory_context**: top-K hits from
  `RaggerMemory::search(current_input, chat_max_memory_results, 0.3)`,
  joined by `"\n\n---\n\n"`. Empty string if no hits above threshold.

- **history bound**: `max_turns * 2` prior messages; older turns are
  dropped (the auto-summariser condenses them into a single memory
  before they fall off).

For the **Anthropic** format, Ragger peels `messages[0]` off and
moves its `content` into the top-level `system` field; the rest of
the array is sent as-is.

---

## Appendix — format file contract

Format definitions live at (search order):

1. `~/.ragger/formats/<name>.json`  (user override)
2. `/var/ragger/formats/<name>.json`  (system)
3. `<install>/formats/<name>.json`  (shipped)

Minimum schema:

```json
{
  "path": "/v1/chat/completions",
  "auth": "bearer",
  "auth_header": "Authorization",
  "auth_prefix": "Bearer ",
  "auth_extra": {},
  "request_transform": {
    "system_location": "message",
    "model_field":     "model",
    "messages_field":  "messages",
    "max_tokens_field":"max_tokens",
    "stream_field":    "stream"
  },
  "response_content":    "choices[0].message.content",
  "stream_content":      "choices[0].delta.content",
  "stream_type_field":   "",
  "stream_type_value":   "",
  "stream_stop":         "data: [DONE]"
}
```

For Anthropic, the deltas change to:

```json
{
  "path": "/v1/messages",
  "auth": "header",
  "auth_header": "x-api-key",
  "auth_prefix": "",
  "auth_extra": {"anthropic-version": "2023-06-01"},
  "request_transform": {
    "system_location": "top_level",
    ...
  },
  "response_content":  "content[0].text",
  "stream_content":    "delta.text",
  "stream_type_field": "type",
  "stream_type_value": "content_block_delta",
  "stream_stop":       ""
}
```

Drop a new JSON file matching this shape to add a custom provider —
no Ragger rebuild required.
