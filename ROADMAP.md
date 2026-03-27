# Ragger — Roadmap

## Current Status

Both Python and C++ versions at feature parity. C++ is production server.

## Planned

### MCP Spec Compliance

Both Python and C++ MCP servers currently use custom JSON-RPC methods
(`memory_store`, `memory_search`) directly. This works but is **not
MCP-spec compliant** — no MCP client (Claude Desktop, OpenClaw, etc.)
can discover or use the server.

**Phase 1: Tools** (priority)
- `initialize` / `initialized` handshake with capability negotiation
- `tools/list` — expose `store` and `search` as discoverable tools
- `tools/call` — dispatch tool invocations
- Keep plain-text search shortcut for interactive use

**Phase 2: Resources** (later)
- Expose collections as MCP resources
- `resources/list` — enumerate available collections (maps to DB tables)
- `resources/read` — read collection contents
- Resource subscriptions for live updates (if useful)

### Potential Upgrades

- **Re-ranking:** Cross-encoder reranking after initial retrieval
  (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Web UI:** Browser interface for chat (`web→chat` edge, v0.9.0)
- **llama.cpp integration:** Embedded inference in C++ only
  (first intentional divergence from Python)
