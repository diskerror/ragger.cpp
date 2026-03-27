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

In `~/.openclaw/openclaw.json`, set the memory plugin slot:

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
          "serverUrl": "http://localhost:8432",
          "transport": "auto",
          "autoRecall": true,
          "autoCapture": true,
          "serverCommand": "/usr/local/bin/ragger",
          "serverArgs": ["serve"],
          "mcpCommand": "ragger"
        }
      }
    }
  }
}
```

### Transport Options

- **`transport: "http"`** — Use HTTP daemon only (multi-user, auth via token)
- **`transport: "mcp"`** — Use MCP stdio only (spawns `ragger mcp` per-user)
- **`transport: "auto"`** (default) — Try HTTP, fall back to MCP if daemon unavailable

**HTTP mode:**
- Connects to Ragger daemon (LaunchDaemon, systemd, or manual)
- Multi-user with token auth (`~/.ragger/token`)
- `serverCommand` spawns the daemon if not running (optional)

**MCP mode:**
- Spawns `ragger mcp` as the current user (no daemon needed)
- Uses stdio for JSON-RPC communication
- Single-user, no authentication required
- `mcpCommand` defaults to `ragger` from PATH

**Auto mode:**
- Tries HTTP first
- Falls back to MCP if HTTP server unavailable
- Best for mixed environments (daemon when available, MCP when not)

Both transports can run simultaneously — SQLite WAL handles concurrent access.

If you prefer to manage the server yourself (launchd, systemd, manual),
omit `serverCommand` and start it however you like.

## Agent Tools

This gives the agent three tools:
- **memory_search** — Semantic search over stored memories
- **memory_store** — Save new memories with optional metadata
- **memory_get** — Get the count of stored memories

With `autoRecall` enabled, relevant memories are automatically injected
into context before each agent turn. With `autoCapture`, important user
messages are stored automatically after conversations.
