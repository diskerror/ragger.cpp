# Ragger — Roadmap

## Current Status

Both Python and C++ versions at feature parity. C++ is production server.

### Transport Model

- **HTTP** — Primary transport. Runs as a per-user daemon via a user
  LaunchAgent (macOS) or systemd --user unit (Linux). Handles auth,
  routing, and shared access when the host user provisions additional
  sub-users via tokens.
- **MCP** — Spec-compliant (protocol `2024-11-05`). Best suited for
  purely local use where no daemon is needed — agent fork+execs
  `ragger mcp` directly against the same `~/.ragger/memories.db`.
  When sub-users access the daemon from other machines or accounts,
  HTTP is preferred since it already solves auth and routing.

## Planned

### MCP Resources (Phase 2)

- Expose collections as MCP resources
- `resources/list` — enumerate available collections
- `resources/read` — read collection contents

### Performance

- **Pre-load user configs:** At daemon startup/HUP, cache all user configurations
  from the users table. Eliminates per-request disk I/O for `~/.ragger/settings.ini`.
  Tradeoff: config changes require HUP to reload.

### Potential Upgrades

- **Re-ranking:** Cross-encoder reranking after initial retrieval
  (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Web UI:** Browser interface for chat (`web→chat` edge, v0.9.0)
- **llama.cpp integration:** Embedded inference in C++ only
  (first intentional divergence from Python)
