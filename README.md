# ragger.cpp

**C++ port of [Ragger Memory](https://github.com/diskerror/ragger) — local-first semantic memory for AI agents and humans.**

Same HTTP API, same database format, same config files. The C++ and Python versions are interchangeable — swap the binary, restart.

Faster startup, lower memory footprint, single static binary. Everything else — features, endpoints, config, behavior — is identical.

## Features

- **Local embeddings** — all-MiniLM-L6-v2 via ONNX Runtime (384-dim)
- **Hybrid search** — BM25 keyword + vector cosine similarity (Eigen3, configurable blend)
- **HTTP & MCP servers** — Crow REST API and JSON-RPC over stdin/stdout
- **Collection filtering** — Organize memories into searchable collections
- **Chat with memory** — REPL with turn persistence, summarization, dynamic context sizing
- **Token auth** — Bearer token authentication with SHA-256 hashing
- **Structured logging** — Four log files (query/http/mcp/error), config-toggleable, thread-safe
- **File import/export** — Heading-aware paragraph chunking, doc reassembly
- **i18n ready** — All user-facing strings in compiled-in language file

## Quick Start

```bash
# Build
cmake -B build -DBOOST_ROOT=/opt/local/libexec/boost/1.88
cmake --build build

# Store a memory
./build/ragger store "The deploy script requires Node 18+"

# Search
./build/ragger search "deployment requirements"

# Import a document
./build/ragger import notes.md --collection docs

# Start HTTP server
./build/ragger serve
```

## Build Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| **SQLite3** | Storage backend | System (MacPorts/apt) |
| **Eigen3** | Vector math (cosine similarity) | System (MacPorts/apt) |
| **Boost** | ProgramOptions, Asio (via Crow) | System (MacPorts/apt) |
| **Rust** | Required by tokenizers-cpp | System (MacPorts/apt) |
| **Crow** | HTTP server + routing | Vendored (`crow_all.h`) |
| **ONNX Runtime** | Embedding inference | Vendored (pre-built) |
| **tokenizers-cpp** | HuggingFace tokenizer | Vendored |
| **nlohmann/json** | JSON serialization | Vendored (header-only) |

```bash
# macOS (MacPorts)
sudo port install boost eigen3 sqlite3 rust

# Linux (apt)
sudo apt install libboost-all-dev libeigen3-dev libsqlite3-dev rustc cargo
```

## Documentation

All documentation is shared between the Python and C++ versions.

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Setup, first run, install locations |
| [Configuration](docs/configuration.md) | Config files, settings reference |
| [Collections](docs/collections.md) | Organizing memories into collections |
| [Search & RAG](docs/search-and-rag.md) | How hybrid search works |
| [HTTP API](docs/http-api.md) | REST endpoints, MCP server, auth |
| [Chat Persistence](docs/chat-persistence.md) | Turn storage, summaries, cleanup |
| [Deployment](docs/deployment.md) | Production setup, LaunchDaemon, multi-user |
| [Project Structure](docs/project-structure.md) | Code layout, database schema |
| [OpenClaw Integration](OPENCLAW.md) | Plugin setup for OpenClaw |
| [Agent Guide](docs/agent-integration.md) | Best practices for AI agents |
| [Testing Your Install](docs/testing-your-install.md) | Verify with your own data |
| [Design Decisions](docs/design-decisions.md) | Why things are the way they are |
| [Roadmap](ROADMAP.md) | Future plans |

## Test Coverage

6 test suites, all passing:

| Suite | Coverage | What's tested |
|-------|----------|---------------|
| `test_config` | INI parsing, defaults, layering |
| `test_bm25` | Indexing, scoring, tokenization |
| `test_sqlite_backend` | CRUD, search, metadata, delete, keep tag, user mgmt |
| `test_import` | Chunking, heading detection, edge cases |
| `test_auth` | SHA-256 hashing, token generation, file I/O |
| `test_server` | Server instantiation, pImpl validation |

```bash
cd build && ctest --output-on-failure
```

## Status

**v0.7.1** — Single-user, production-ready. Full feature parity with the Python version.

Multi-user framework in place (layered config, SERVER_LOCKED, system ceilings, token auth). Multi-user data support planned for a future release.

## License

GPL v3 — same as the Python version.
