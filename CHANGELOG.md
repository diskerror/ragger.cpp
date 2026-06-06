# Changelog

## v1.0.0rc1 — 59 commits since v0.9.4

The 0.9.x line was a working Python-port-in-C++ with a chat REPL and a
proxy mode bolted on. v1.0.0rc1 throws both of those away and rebuilds
around a single idea: **the agent owns the conversation; Ragger owns
the memory.** Turns flow in through one MCP/HTTP entry point, a
daemon-resident worker summarizes them in the background, and the
agent pulls a recipe-shaped context payload back out whenever it
needs one.

### Highlights

- **Fading-memory schema (v2).** Separate tables for `turns`,
  `summaries`, `documents`, and `decisions`, each with its own FTS5
  index. Summary rows are tagged by level — L1 raw turn, L2 turn
  summary, L3 session summary, L4 project summary, L6 decision —
  with timestamp-keystone linkage between turns and their L2
  summaries (no FK column needed). FTS5 replaces the old BM25 path.
- **Daemon-resident summarizer.** A worker thread inside the HTTP
  daemon owns L2 + L3 generation. Inherits the source turn's
  timestamp on L2 rows so the (session_id, timestamp) pair joins
  turn ↔ summary cleanly. Falls back to a heuristic `draft`-tagged
  row when inference is unreachable; housekeeping rewrites the
  draft once the endpoint comes back.
- **Recipe-based context assembly.** `build_context` (MCP tool /
  `GET /session/<id>`) walks the latest prompt backward and layers
  raw turns → turn summaries → session/project summaries →
  decisions according to a JSON recipe. Five built-in recipes ship
  (`natural_fading`, `reconnect`, `deep_recall`, `tldr`,
  `raw_only`); drop your own in `~/.ragger/recipes/`.
  `ragger recipe` is an interactive picker, persisting the choice
  to the DB.
- **`ragger onboard` verb.** Guided first-run setup: storage,
  capture/build flags, default recipe, inference endpoint + model
  (with a live `/v1/models` probe), optional separate memory model,
  daemon start. Idempotent — re-run any time to change one
  section. Surfaced at the top of `ragger help`.

### Agent integration

- **`capture_turn`** (MCP) / `POST /turn` — uniform write entry
  point for agents to push completed turns. Both replace the old
  LM-proxy path.
- **`build_context`** — read entry point for recipe-shaped
  payloads. Both surfaces are gated by their respective config
  flags; the MCP `tools/list` advertisement is filtered to match,
  so disabled tools don't appear to the agent.
- **Plugin install scripts.** `scripts/install-claude-desktop.sh`
  for the GUI app; new `scripts/install-claude-code.sh` for the
  CLI — registers the MCP server in `~/.claude.json` and installs
  a Stop hook (`~/.ragger/hooks/claude-code-capture-turn.sh`)
  that POSTs each completed turn over the unix socket. Both
  scripts are idempotent and dated-backup the user's config
  before any edit.

### Cross-platform build

- **CMakeLists.txt** is now Apple/Linux conditional. Linux build
  verified on Debian 13 x86_64.
- **ONNX Runtime auto-fetches** the right tarball for the host
  platform (macOS arm64/x86_64, Linux x86_64/aarch64) at configure
  time when no vendored copy is present.
- **`build.sh`** sources `~/.cargo/env` if present and checks
  `cargo`/`rustc` up front — no more silent failures mid-build at
  the Rust step.
- **`install.sh`** downloads the all-MiniLM-L6-v2 embedding model
  (~90 MB) into `~/.ragger/models/` on first run; idempotent.
- README + getting-started rewritten with real package-manager
  commands for MacPorts, Homebrew, apt, and dnf, plus the rustup
  one-liner.

### Summarization

- **Length contract:** summaries are never longer than the source
  and target roughly ¼ of it. Trivial exchanges (slash commands
  like `/exit`, single-line greetings) skip inference entirely and
  store verbatim. Post-check rejects bloat from non-compliant
  models.
- **Reasoning-content fallback** in `api_formats` — thinking models
  (Gemma 4, DeepSeek R1 shims) that return the visible answer in
  `choices[0].message.reasoning_content` now work alongside
  standard models without reconfiguration.
- **Session L3 pause:** a configurable idle-gap closes a running
  session summary.

### Import

- **Conversation importer ported to C++.** `ragger import
  conversations --format=code|web PATH [--since/--until/--session]`
  replaces the Python script. Imports land at `level="turn"`.
- **L4 summaries import.** `ragger import summaries <file>...`
  treats each file as one L4 row (filename stem → tag); or
  `--jsonl=FILE` for `{text, tags?, timestamp?}` per line.

### Storage & embeddings

- **f16 embedding storage** halves blob size; configurable via
  `[embedding] vector_type = f16 | f32` with drift guards that
  refuse to mix.
- **Drift guards** also cover model name and dimensions, with an
  empty-DB re-adopt path so a fresh install can switch defaults
  without manual intervention.
- **`rebuild-embeddings`** now covers all four embedded tables,
  not just summaries. `rebuild-bm25` removed.

### Refactors

- RAII `Stmt` wrapper for SQLite prepared statements.
- HTTP routes wrapped in a single `guarded()` adapter for auth +
  error handling.
- Daemon/service control extracted into `daemon_control.cpp`.
- Shared `summarize_transcript()` helper — one prompt, one place.
- `RaggerClient` backed by libcurl instead of raw sockets.
- Timestamp / filesystem / workspace helpers consolidated into
  `util/`.

### Removed

- **Chat REPL** and **web interface.** Ragger is now a memory
  server only; conversation lives in the agent.
- **LM proxy mode.** Replaced by `capture_turn`.
- **BM25 path.** Replaced by FTS5.
- **`scripts/import-claude-conversations.py`.** Replaced by the
  native `ragger import conversations` subcommand.

### Defaults

- `capture_turns` and `build_context` ship **enabled** so a
  freshly-installed Ragger is useful to an agent out of the box;
  `ragger onboard` asks before flipping either off.
- `[inference] api_url` defaults to `http://localhost:1234/v1`
  (LM Studio convention) so a new install with LM Studio running
  finds a model on its own.

### Known gaps

- The Fedora/RHEL and Linux-aarch64 prereq lines in the README are
  unverified — only macOS (arm64) and Debian 13 (x86_64) were
  smoke-tested end-to-end before tagging.
- No `ragger uninstall` verb yet; uninstall the plugin scripts
  with `--uninstall` and remove `~/.ragger/` by hand.
- The Stop hook script reads Claude Code's transcript and POSTs
  per turn — fine for the typical case, but big transcripts mean
  the hook re-reads the whole file each turn. A streaming variant
  is a future improvement.
