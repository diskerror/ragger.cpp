# ragger.cpp

**C++ port of [Ragger Memory](https://github.com/diskerror/ragger) — local-first semantic memory for AI agents and humans.**

Same HTTP API, same database format, same config files. The C++ and Python versions are interchangeable — swap the binary, restart.

Faster startup, lower memory footprint, single static binary. Everything else — features, endpoints, config, behavior — is identical.

## Features

- **Local embeddings** — all-MiniLM-L6-v2 via ONNX Runtime (384-dim)
- **Hybrid search** — BM25 keyword + vector cosine similarity (Eigen3, configurable blend)
- **HTTP & MCP servers** — cpp-httplib REST API and JSON-RPC over stdin/stdout
- **Collection filtering** — Organize memories into searchable collections
- **Chat with memory** — REPL with turn persistence, summarization, dynamic context sizing
- **Token auth** — Bearer token authentication with SHA-256 hashing
- **Structured logging** — Four log files (query/http/mcp/error), config-toggleable, thread-safe
- **File import/export** — Heading-aware paragraph chunking, doc reassembly
- **i18n ready** — All user-facing strings in compiled-in language file

## Quick Start

**Production install** (creates system user/group, installs as service):
```bash
cd /path/to/ragger.cpp
./build.sh                # Check dependencies, build binary
sudo ./install.sh         # Install to /usr/local/bin, restart daemon
```

The installer is interactive: it asks for single-user or multi-user mode,
creates all system resources, installs the binary, then walks through
every user on the system — offering to add, remove, or configure client
integrations (OpenClaw, Claude Desktop) for each. Safe to re-run on
upgrades; only the binary is overwritten.

**Development build** (manual cmake):
```bash
cmake -B build -DBOOST_ROOT=/opt/local/libexec/boost/1.88
cmake --build build -j8

# Test the build
./build/ragger version
```

**Usage:**
```bash
# Store a memory
ragger store "The deploy script requires Node 18+"

# Search
ragger search "deployment requirements"

# Import a document
ragger import notes.md --collection docs

# Start HTTP server
ragger serve
```

## Build Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| **SQLite3** | Storage backend | System (MacPorts/apt) |
| **Eigen3** | Vector math (cosine similarity) | System (MacPorts/apt) |
| **Boost** | ProgramOptions | System (MacPorts/apt) |
| **OpenSSL** | SHA-256 hashing (token auth) | System (MacPorts/apt) |
| **libcurl** | HTTP client (inference proxy) | System (MacPorts/apt) |
| **Rust** | Required by tokenizers-cpp | System (MacPorts/apt) |
| **cpp-httplib** | HTTP server + routing | Vendored (`httplib.h`) |
| **ONNX Runtime** | Embedding inference | Vendored (pre-built) |
| **tokenizers-cpp** | HuggingFace tokenizer | Vendored |
| **nlohmann/json** | JSON serialization | Vendored (header-only) |

```bash
# macOS (MacPorts)
sudo port install boost eigen3 sqlite3 rust openssl

# Linux (apt)
sudo apt install libboost-all-dev libeigen3-dev libsqlite3-dev rustc cargo libssl-dev libcurl4-openssl-dev
```

**Platforms:** macOS and Linux are tested and supported. Windows should work (all libraries cross-compile) but needs porting:
- Replace `fork()` in background summarization with threads or `CreateProcess()`
- Build with MSVC or MinGW (CMake generates Visual Studio projects)
- Create PowerShell install script (current script uses bash, dscl, launchctl)

Contributions welcome.

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
| [OpenClaw Integration](docs/openclaw.md) | Plugin setup for OpenClaw |
| [Agent Guide](docs/agent-integration.md) | Best practices for AI agents |
| [Testing Your Install](docs/testing-your-install.md) | Verify with your own data |
| [Design Decisions](docs/design-decisions.md) | Why things are the way they are |

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

**v0.8.1** — Production-ready with unified install, OpenSSL auth, rebuild-embeddings backup/warning, and comprehensive testing.

Features: multi-user with per-user databases and search merging, common shared memory DB, bearer token authentication (OpenSSL SHA-256), automatic token rotation, per-user inference model selection, rebuild-embeddings verb with backup and confirmation, schema-driven API formats, chat persistence with background summarization, user provisioning CLI, and idempotent install script.

## License

GPL v3 — same as the Python version.

**Commercial licensing:** If you'd like to use Ragger in a proprietary
product without GPL obligations, commercial licenses are available.
Contact reid@diskerror.com.
