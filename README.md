# raggerc

C++ port of [Ragger Memory](https://github.com/diskerror/ragger) — a local-first semantic memory store with hybrid vector + BM25 search.

## Status

**Production-ready** — running as the live memory server, all 23 retrieval quality tests passing. Feature parity with the Python version's core functionality.

Same HTTP API, same database format, same embedding model. The C++ and Python versions share a database and config file.

## Features

- **Local embeddings** — all-MiniLM-L6-v2 via ONNX Runtime (384-dim)
- **Hybrid search** — BM25 keyword + vector cosine similarity (configurable blend)
- **Fast vector search** — Eigen3 cosine similarity
- **HTTP server** — Crow REST API on localhost
- **MCP server** — JSON-RPC over stdin/stdout with plain text fallback
- **File import** — Heading-aware paragraph chunking
- **Export** — Reassemble docs by collection, memories grouped by date/category
- **Collection filtering** — Organize into searchable collections
- **Usage tracking** — Per-memory access stats
- **Path normalization** — `$HOME` → `~/`
- **i18n ready** — All user-facing strings in compiled-in language file

## Configuration

Ragger uses a shared INI config file. Search order (first found wins):

1. `/etc/ragger.conf`
2. `~/.ragger/ragger.conf`
3. `--config-file=<path>`

A config file is required — there are no silent defaults. See `example-config.conf` for all options. The same config file works for both the C++ and Python versions.

## Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| **SQLite3** | Storage backend | System (MacPorts/apt) |
| **Eigen3** | Vector math (cosine similarity) | System (MacPorts/apt) |
| **Boost** | ProgramOptions, Asio (via Crow) | System (MacPorts/apt) |
| **Crow** | HTTP server + routing | Vendored header (`crow_all.h`) |
| **ONNX Runtime** | Embedding inference (all-MiniLM-L6-v2) | Vendored pre-built |
| **tokenizers-cpp** | HuggingFace tokenizer | Vendored (requires Rust) |
| **nlohmann/json** | JSON serialization | Vendored header-only |

## Building

```bash
# System deps (macOS/MacPorts)
sudo port install boost eigen3 sqlite3 rust

# Build
cmake -B build -DBOOST_ROOT=/opt/local/libexec/boost/1.88
cmake --build build

# Test
./build/tests/test_config
```

## Usage

```bash
# Start HTTP server (port 8432)
raggerc serve

# Search
raggerc search "deployment requirements"
raggerc search "transposition" --collection sibelius

# Store a memory
raggerc store "The deploy script requires Node 18+"

# Memory count
raggerc count

# Import files
raggerc import notes.md --collection docs
raggerc import doc1.md doc2.md --collection reference

# Export
raggerc export docs ./exported/ --collection orchestration
raggerc export memories ./exported/
raggerc export all ./exported/

# MCP server (JSON-RPC over stdin/stdout)
raggerc mcp

# Rebuild BM25 index
raggerc rebuild-bm25

# Help
raggerc help
raggerc          # no verb = help
```

## HTTP Endpoints

```
GET  /health  — {"status": "ok", "memories": 10622}
GET  /count   — {"count": 10622}
POST /store   — {"text": "...", "metadata": {...}}
POST /search  — {"query": "...", "limit": 5, "min_score": 0.0, "collections": [...]}
```

## Architecture

Mirrors the Python version's module structure:

| C++ | Python equivalent | Purpose |
|-----|-------------------|---------|
| `config.h/cpp` | `config.py` | Load INI config file |
| `embedder.h/cpp` | `embedding.py` | ONNX Runtime inference |
| `bm25.h/cpp` | `bm25.py` | BM25 keyword index |
| `sqlite_backend.h/cpp` | `sqlite_backend.py` | SQLite storage + hybrid search |
| `memory.h/cpp` | `memory.py` | High-level facade |
| `server.h/cpp` | `server.py` | HTTP server (Crow) |
| `main.cpp` | `cli.py` | CLI entry point |
| `lang/en.h` | `lang/en.py` | User-facing strings (i18n) |

## File Layout

```
~/.ragger/
├── ragger.conf         # Config file
├── memories.db         # SQLite database
└── models/             # ONNX model files (model.onnx, tokenizer.json, etc.)
```

## License

GPL v3 — same as the Python version.
