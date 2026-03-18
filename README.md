# raggerc

C++ port of [Ragger Memory](https://github.com/diskerror/ragger) — a local-first semantic memory store with hybrid vector + BM25 search.

## Status

**Early development** — scaffolding and build system in place, implementations in progress.

See the Python version's [README](https://github.com/diskerror/ragger)
for full feature documentation. This port aims for feature parity with
the same HTTP API, database format, and embedding model.

## Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| **SQLite3** | Storage backend | System (MacPorts/apt) |
| **Eigen3** | Vector math (cosine similarity, normalization) | System (MacPorts/apt) |
| **Boost** | Program options, Asio (networking) | System (MacPorts/apt) |
| **Crow** | HTTP server + routing | CMake FetchContent |
| **ONNX Runtime** | Embedding inference (all-MiniLM-L6-v2) | Vendored pre-built |
| **tokenizers-cpp** | HuggingFace tokenizer | Vendored (requires Rust) |
| **nlohmann/json** | JSON serialization for storage/config | Vendored header-only |

## Building

```bash
# System deps (macOS/MacPorts)
sudo port install boost188 sqlite3 eigen3

# Rust toolchain (needed for tokenizers-cpp)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Vendored deps (run once)
# Download ONNX Runtime pre-built binary into vendor/
# Clone tokenizers-cpp with submodules into vendor/

# Build
cmake -B build -DBOOST_ROOT=/opt/local/libexec/boost/1.88
cmake --build build

# Test
cd build && ctest
```

## Usage (planned)

```bash
# Start HTTP server (same API as Python version, port 8432)
raggerc serve

# One-shot search from CLI
raggerc search "deployment requirements"

# Store a memory
raggerc store "The deploy script requires Node 18+"

# Memory count
raggerc count
```

## Architecture

The C++ port mirrors the Python version's module structure:

| C++ | Python equivalent | Purpose |
|-----|-------------------|---------|
| `config.h/cpp` | `config.py` | Configuration and path helpers |
| `embedder.h/cpp` | `embedding.py` | ONNX Runtime inference |
| `bm25.h/cpp` | `bm25.py` | BM25 keyword index |
| `sqlite_backend.h/cpp` | `sqlite_backend.py` | SQLite storage + search |
| `memory.h/cpp` | `memory.py` | High-level facade |
| `server.h/cpp` | `server.py` | HTTP server (Crow) |
| `main.cpp` | `cli.py` | CLI entry point |

Same database format — the C++ and Python versions can share a database.

## Best Practices for AI Agents

If you're building with or contributing to raggerc using an AI coding
assistant, these patterns prevent lost context across sessions.

### Read the Docs First

Before making changes, read this README and [the Python version's
ROADMAP.md](https://github.com/diskerror/ragger/blob/master/ROADMAP.md).
Library choices, architecture decisions, and design rationale are
documented there. Don't rely solely on memory/conversation history.

### Store Decisions, Not Just Summaries

When you make a technical decision (library choice, API design, algorithm
selection), store it as a standalone, searchable entry — not buried in a
session summary. Include the rationale and a reference to where details
live in the project files.

### The Database Is the API Contract

The SQLite schema is shared between the Python and C++ versions. Any
changes to the database format must be coordinated across both. The
Python version is the reference implementation.

## License

GPL v3 — same as the Python version.
