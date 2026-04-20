# OpenClaw Integration

Ragger integrates with [OpenClaw](https://github.com/openclaw/openclaw) as a
memory plugin via HTTP (daemon) or MCP (per-user stdio).

## 1. Install the OpenClaw plugin

Copy the `memory-ragger` plugin directory into `~/.openclaw/extensions/`:

```
~/.openclaw/extensions/memory-ragger/
├── openclaw.plugin.json
└── index.ts
```

## 2. Configure OpenClaw

### Local-only (MCP, recommended for personal use)

For purely local use, MCP mode requires no daemon — just the `ragger`
binary. OC fork+execs it on demand against your own `~/.ragger/`:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-ragger"
    },
    "entries": {
      "memory-ragger": {
        "enabled": true,
        "config": {
          "transport": "mcp",
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

This spawns `ragger mcp` on demand. No daemon, no ports, no auth tokens.

### Daemon (HTTP, required for remote or multi-user access)

If the daemon is running (`ragger start` — see [Deployment](deployment.md))
and you want to hit it over HTTP with a token, use HTTP transport:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-ragger"
    },
    "entries": {
      "memory-ragger": {
        "enabled": true,
        "config": {
          "transport": "http",
          "serverUrl": "http://localhost:8432",
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

### Auto Transport (Fallback)

Use `"transport": "auto"` to try HTTP first, fall back to MCP:

```json
{
  "config": {
    "transport": "auto",
    "serverUrl": "http://localhost:8432",
    "autoRecall": true,
    "autoCapture": true
  }
}
```

Good for development — connects to daemon when running, spawns MCP when not.

### Transport Comparison

- **mcp** — No daemon, no auth. OC spawns its own `ragger mcp` process
  against your `~/.ragger/memories.db`. Simplest for personal use.
  Each spawn loads the embedding model, so this is wasteful when the
  daemon is already running.
- **http** — Connects to the running daemon (`ragger start`) with a
  bearer token. Required when the daemon needs to serve multiple
  users (each with their own token) or when OC runs on a different
  machine than Ragger.
- **auto** — Tries HTTP first, falls back to MCP. Uses token for
  HTTP, none for MCP fallback. Good for development.

**Recommendation:** Use `mcp` when you're the only user and don't
need a daemon. Use `http` when the daemon is already running, or
when sub-users need to connect with tokens.

## Agent Tools

This gives the agent three tools:
- **memory_search** — Semantic search over stored memories
- **memory_store** — Save new memories with optional metadata
- **memory_get** — Get the count of stored memories

With `autoRecall` enabled, relevant memories are automatically injected
into context before each agent turn. With `autoCapture`, important user
messages are stored automatically after conversations.
