# Ragger — Roadmap

## Current Status

Both Python and C++ versions at feature parity. C++ is production server.

### Transport Model

- **HTTP** — Primary transport for all modes (single-user and multi-user).
  Runs as a daemon via launchd/systemd. Handles auth, routing, and
  concurrent access to both user and common databases.
- **MCP** — Spec-compliant (protocol `2024-11-05`). Best suited for
  single-user/local use where no server is needed — agent fork+execs
  `ragger mcp` directly. In multi-user mode, HTTP is preferred since
  it already solves auth/routing/permissions.

## Planned

### MCP Resources (Phase 2)

- Expose collections as MCP resources
- `resources/list` — enumerate available collections
- `resources/read` — read collection contents

### Performance

- **Pre-load user configs:** At daemon startup/HUP, cache all user configurations
  from the users table. Eliminates per-request disk I/O for `~/.ragger/ragger.ini`.
  Tradeoff: config changes require HUP to reload.

### Potential Upgrades

- **Re-ranking:** Cross-encoder reranking after initial retrieval
  (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Web UI:** Browser interface for chat (`web→chat` edge, v0.9.0)
- **llama.cpp integration:** Embedded inference in C++ only
  (first intentional divergence from Python)
