# OpenClaw Integration

Ragger integrates with [OpenClaw](https://github.com/openclaw/openclaw) as a
memory plugin via HTTP (daemon) or MCP (per-user stdio).

## 1. Install the OpenClaw plugin

Copy the `ragger` plugin directory into `~/.openclaw/extensions/`:

```
~/.openclaw/extensions/ragger/
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
      "memory": "ragger"
    },
    "entries": {
      "ragger": {
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
      "memory": "ragger"
    },
    "entries": {
      "ragger": {
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

> **Caveat: untested.** This plugin was ported to the Ragger 0.17 agent API
> without a live OpenClaw to test against. It was only exercised against a stub
> HTTP server (every tool reaches its route, the recall and capture hooks fire).
> Expect rough edges; reports welcome.

The tool set is the 0.17 agent API, generated from `tools/*.json`
(`scripts/gen-openclaw-tools.py`; `--check` fails when the plugin is stale):

- **memory_search**: layered search (turn summaries, episodes, knowledge); handles + snippets
- **memory_get**: full rows by typed handle, or conversation pages by session
- **memory_timeline**: newest-first lists by turn / episode / session / knowledge
- **memory_note**: note or insight (insight when `evidence` is given)
- **memory_decision**: decision with a required `trigger`
- **memory_update**: metadata-only curation (status, tags)
- **document_search**: imported documents
- **memory_store**: flat store of facts, preferences, lessons (`category: "decision"` is rejected; use `memory_decision`)
- **memory_status**: daemon status

Upgrading from the 0.16 plugin: `memory_get` used to return the memory count
(now `memory_status`), and `memory_search` was a flat semantic search.
