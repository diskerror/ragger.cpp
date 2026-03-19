# ragger

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
ragger serve

# Search
ragger search "deployment requirements"
ragger search "transposition" --collection sibelius

# Store a memory
ragger store "The deploy script requires Node 18+"

# Memory count
ragger count

# Import files
ragger import notes.md --collection docs
ragger import doc1.md doc2.md --collection reference

# Export
ragger export docs ./exported/ --collection orchestration
ragger export memories ./exported/
ragger export all ./exported/

# MCP server (JSON-RPC over stdin/stdout)
ragger mcp

# Rebuild BM25 index
ragger rebuild-bm25

# Help
ragger help
ragger          # no verb = help
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

## Installation

### Per-user install (single user, no sudo)

| Platform | Binary location | Config location |
|----------|----------------|-----------------|
| macOS    | `~/.local/bin/ragger` | `~/.ragger/ragger.conf` |
| Linux    | `~/.local/bin/ragger` | `~/.ragger/ragger.conf` |
| Windows  | `%LOCALAPPDATA%\ragger\ragger.exe` | `%LOCALAPPDATA%\ragger\ragger.conf` |

On macOS/Linux, ensure `~/.local/bin` is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc or ~/.bashrc
```

After building:
```bash
mkdir -p ~/.local/bin
cp build/ragger ~/.local/bin/ragger
```

### Switching between C++ and Python versions

The Python version can be installed alongside as `ragger-py`:
```bash
cat > ~/.local/bin/ragger-py << 'EOF'
#!/bin/bash
RAGGER_PY_DIR="${RAGGER_PY_DIR:-$HOME/PyCharmProjects/Ragger}"
exec python3 "$RAGGER_PY_DIR/ragger_memory/cli.py" "$@"
EOF
chmod +x ~/.local/bin/ragger-py
```

To switch which version `ragger` points to:
```bash
# Use Python version
mv ~/.local/bin/ragger ~/.local/bin/ragger-cpp
ln -s ~/.local/bin/ragger-py ~/.local/bin/ragger

# Switch back to C++
rm ~/.local/bin/ragger
mv ~/.local/bin/ragger-cpp ~/.local/bin/ragger
```

### System-wide install (future, multi-user)

Reserved for future multi-user support. Will use `/usr/local/bin/ragger`,
`/etc/ragger.conf`, and `/var/ragger/` for data.

## macOS Deployment Note

When running Ragger as a LaunchDaemon (i.e., starting at boot before any user logs in),
be aware that if the user's home directory is on an external or non-default volume, that
volume may not be mounted until a user session starts. In this case, you may need to
enable **System Settings → Users & Groups → Automatically log in as…** for the
relevant user account to ensure the volume is available at boot.

A start script that waits for the volume to mount (with a timeout) can help, but is not
a substitute for the volume actually being mounted by the system.

## License

GPL v3 — same as the Python version.
