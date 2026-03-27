# OpenClaw Integration

Ragger integrates with [OpenClaw](https://github.com/openclaw/openclaw) as a
memory plugin via the HTTP server.

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
          "autoRecall": true,
          "autoCapture": true,
          "serverCommand": "/usr/local/bin/ragger",
          "serverArgs": ["serve"]
        }
      }
    }
  }
}
```

The `serverCommand` can point to either the C++ binary (`ragger serve`)
or a startup script that waits for external drives, sets environment,
etc. The plugin automatically starts the server if it's not already
running and waits up to 15 seconds for it to become ready.

If you prefer to manage the server yourself (launchd, systemd, manual),
just omit `serverCommand` and start it however you like.

## Agent Tools

This gives the agent three tools:
- **memory_search** — Semantic search over stored memories
- **memory_store** — Save new memories with optional metadata
- **memory_get** — Get the count of stored memories

With `autoRecall` enabled, relevant memories are automatically injected
into context before each agent turn. With `autoCapture`, important user
messages are stored automatically after conversations.
