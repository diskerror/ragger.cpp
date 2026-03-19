# Ragger Memory — Roadmap

## Current: v0.6.0

Both Python and C++ versions at feature parity. C++ is production server.

## Potential Upgrades

These are natural next steps to improve retrieval quality:

- **Re-ranking:** After retrieving top-k candidates, re-score them with a
  cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) for
  more accurate relevance ranking. Slower per query but significantly
  better precision.

- **Query expansion / HyDE:** Generate a hypothetical answer to the query
  using the LLM, then embed *that* for retrieval. Often finds results that
  the original terse query would miss.

- **Memory lifecycle (decay/promotion):** Usage tracking is already in place.
  A natural extension would be to decay memories that are never accessed and
  promote frequently-used ones — mimicking human memory consolidation.

- **Native vector search:** Purpose-built vector databases like SQLite vector
  extensions would move similarity search into the database
  engine, eliminating the need to load all embeddings into Python. Worthwhile
  for a very large corpora (100K+).

None of these are necessary at a small scale (<50K chunks), but they
become worthwhile as your corpus grows or retrieval precision becomes
critical.

## Multi-User Architecture (v0.7.0+)

### System-Wide (`/var/ragger/` or platform equivalent)
- `ragger.conf` — system config (port, host, model path, search params, cache TTL)
- `memories.db` — shared knowledge base. Every record tagged with `user_id`.
  Documents imported here by default. Users can store/move project knowledge
  here for collaboration.
- `models/` — shared ONNX model files
- Logs

### Per-User (`~/.ragger/`)
- `ragger.conf` — personal additions (extends system config, does not override)
- `memories.db` — private conversations, personal memories
- `token` — bearer auth token, permissions 0600
- `USER.md` — about this user (preferences, working style)
- `AGENTS.md` — per-user agent customization
- Workspace files as needed

### Authentication
- **API clients**: Bearer token in `Authorization` header. Token generated on
  first run, stored in `~/.ragger/token` (0600). Daemon maps token → username
  (home directory name, e.g. "reid").
- **Browser (future `ragger chat` web UI)**: Hashed password + session cookies.
  Separate auth concern from API tokens.

### Runtime Model
- **Single process**, single port, single loaded embedding model
- Per-user backend cached with configurable TTL (default 12 hours)
- Search merges system DB + user's private DB, ranked together
- Multiple concurrent users, each with their own conversation context
- One persona, many simultaneous conversations

### Collaborative Features
- Users can store to system DB tagged with `user_id` and project (`collection`)
- Project-scoped search: filter by collection for team context
- Move/promote between private and system memory
- Only admin users can modify system-wide files through the agent
- System files also changeable manually with root/ragger user perms

### Permissions
- Daemon runs as `_ragger` (macOS) / `ragger` (Linux)
- System directory owned by ragger user
- Each user's `~/.ragger/` readable by ragger group
- Project-level permissions (future): who can read/write which collections

## Agent Modes

- **`ragger serve`** — headless memory + LLM backend. No workspace files,
  no agent loop — just HTTP endpoints. For OpenClaw or other frameworks.
- **`ragger chat`** — standalone agent with full tool loop, memory, and LLM.
  Loads workspace MD files from `~/.ragger/`. Web UI for browser access.

## Thin CLI Client

HTTP request to running server instead of loading the embedding model.
Sub-100ms queries from the terminal.
