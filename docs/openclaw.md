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

### Single-user Setup (Recommended)

For personal use, MCP mode requires no daemon — just the `ragger` binary:

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

### Multi-user Setup (Daemon)

If you're running a system daemon (see [Deployment](deployment.md)), use HTTP transport:

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

- **mcp** — Single or multi-user. No daemon, no auth. Each user spawns their own instance. Simple but uses more resources in multi-user setups (each instance loads the embedding model).
- **http** — Multi-user only. Requires `ragger serve` running as `ragger`/`_ragger` system user with user table and token auth. One daemon serves all users efficiently.
- **auto** — Development/mixed. Tries HTTP first, falls back to MCP. Uses token for HTTP, none for MCP fallback.

**Recommendation:** Use `mcp` for personal use. Use `http` for managed multi-user deployments where `ragger serve` is running.

## Agent Tools

This gives the agent three tools:
- **memory_search** — Semantic search over stored memories
- **memory_store** — Save new memories with optional metadata
- **memory_get** — Get the count of stored memories

With `autoRecall` enabled, relevant memories are automatically injected
into context before each agent turn. With `autoCapture`, important user
messages are stored automatically after conversations.
