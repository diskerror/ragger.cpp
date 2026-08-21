# Ragger

Ragger is local-first semantic memory for AI agents — and for the humans
working with them. Think of it as **running compaction with tiered
recall**: every turn your agent has gets captured and summarized in the
background while it stays warm, then served back as a layered "what just
happened?" payload that fades the way a person's memory fades — while
keeping the verbatim transcript a single query away. All embeddings are
local. The whole database is one SQLite file you own.

C++ port of the original Python [Ragger Memory](https://github.com/diskerror/ragger),
diverged at v0.9.4 — now the sole focus.

## What you get

- **Captured turns, summarized live.** Turns land in the DB as they
  happen; an in-daemon worker writes per-turn summaries immediately and
  closes session summaries on idle, all without blocking the agent. If
  the summarizer model is unreachable, a draft is stored and rewritten
  later — the agent never waits on inference it didn't ask for.
- **Hybrid search.** BM25/FTS5 keyword search blended with dense vector
  cosine via Eigen3, plus an optional phonetic ("sounds-like") signal.
  Configurable weights; all signals normalized before blending.
- **Local embeddings.** `all-MiniLM-L6-v2` via ONNX Runtime (384-dim).
  Stored as IEEE half (f16) by default to halve disk; in-memory math
  stays f32. Re-encode any time with `ragger rebuild-embeddings`.
- **HTTP and MCP, same data.** Daemon serves a REST API; `ragger mcp`
  speaks JSON-RPC over stdio. Bearer-token auth on remote requests;
  unix-socket and loopback are pre-authenticated.
- **Markdown-aware import.** Heading-aware chunking; Claude Code and
  claude.ai conversation archives import with original timestamps.

Single binary. Single-user out of the box. No external services, no
network calls once the embedding model is on disk.

## Quick start

Per-user, no sudo (binary → `~/.local/bin`, data → `~/.ragger/`):

```bash
./scripts/build.sh        # check deps, build
./scripts/install.sh      # install binary + user service unit + recipes
ragger start              # start the daemon
```

`install.sh` is idempotent — re-run after a rebuild to refresh the
binary, the service unit, and the shipped recipes. Your config, database,
and any custom recipe files are preserved.

```bash
ragger store "The deploy script needs Node 18+"
ragger search "deployment requirements"
ragger import-docs notes.md

ragger recipe                # interactive picker over available recipes
ragger recipe reconnect      # set the active recipe (persisted in the DB)

ragger start | stop | restart | status
```

Dev build (manual cmake):

```bash
cmake --preset dev && cmake --build build -j8
./build/ragger version
```

## Layout

Every on-disk path is hardcoded relative to a single base directory
(`~/.ragger` by default). There's no per-path config — the only way to
relocate the whole tree is the hidden `--ragger-base <path>` flag
(testing only; not documented in `--help`). Symlink `~/.ragger`
elsewhere if you need the data on a different disk.

| What | Where |
|---|---|
| Binary | `~/.local/bin/ragger` |
| Config | `~/.ragger/settings.ini` |
| Database | `~/.ragger/memories.db` |
| Embedding model | `~/.ragger/models/` |
| Recipes | `~/.ragger/recipes/` |
| Inference formats | `~/.ragger/formats/` |
| Log | `~/.ragger/logs/activity.log` |
| Bearer token | `~/.ragger/token` |
| Unix socket | `~/.ragger/ragger.sock` |
| Retrieval stats (opt-in build) | `~/.ragger/stats.db` |

## Build dependencies

- **System:** SQLite3, Eigen3, Boost (program_options), OpenSSL, libcurl,
  Rust (for tokenizers-cpp; rustup stable is fine).
- **Auto-fetched at configure time:** ONNX Runtime (downloaded into the
  build dir for your platform), [c_lib](https://github.com/diskerror/c_lib)
  (shared C++ utilities). Drop a matching `vendor/onnxruntime/`
  in place if you want an offline build.
- **Vendored** (already in the repo): cpp-httplib, nlohmann/json,
  tokenizers-cpp source.

```bash
# macOS — MacPorts
sudo port install boost eigen3 sqlite3 rust openssl curl cmake

# macOS — Homebrew
brew install boost eigen sqlite openssl@3 curl cmake rustup-init && rustup-init -y

# Debian / Ubuntu
sudo apt install build-essential cmake pkg-config \
                 libboost-program-options-dev libeigen3-dev libsqlite3-dev \
                 libssl-dev libcurl4-openssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Fedora / RHEL
sudo dnf install gcc-c++ cmake pkgconf-pkg-config \
                 boost-devel eigen3-devel sqlite-devel \
                 openssl-devel libcurl-devel
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

Then:

```bash
./scripts/build.sh          # check deps, configure, build
./scripts/install.sh        # places binary + downloads ~90 MB embedding model
```

### Dev build (local c_lib)

If you have a local checkout of [c_lib](https://github.com/diskerror/c_lib),
create a `CMakeUserPresets.json` (gitignored) to use it instead of fetching
from GitHub:

```json
{
  "version": 6,
  "configurePresets": [{
    "name": "dev",
    "inherits": "default",
    "cacheVariables": {
      "FETCHCONTENT_SOURCE_DIR_C_LIB": "/path/to/your/c_lib"
    }
  }]
}
```

The build script auto-detects this file and passes the variable to cmake.
CLion also picks up the `dev` preset from its CMake profile dropdown.

Manual cmake (without the build script):

```bash
cmake --preset dev            # or: cmake -B build
cmake --build build -j8
./build/ragger version
```

macOS (arm64 and x86_64) and Linux (x86_64 and aarch64) are supported.
Windows needs porting (`fork()`, the bash/launchctl/systemd install scripts).

## Documentation

| Guide | |
|-------|--|
| [Getting started](docs/getting-started.md) | Setup and first run |
| [Configuration](docs/configuration.md) | `settings.ini` reference |
| [Search & RAG](docs/search-and-rag.md) | How hybrid search works |
| [HTTP API](docs/http-api.md) | REST endpoints, MCP, auth, `/turn` and `/session/<id>?recipe=` |
| [Importing conversations](docs/importing-conversations.md) | Claude Code / claude.ai history |
| [Deployment](docs/deployment.md) | Daemon lifecycle, sub-users, reverse proxy |
| [TLS setup](docs/tls-setup.md) | HTTPS via reverse proxy or native cert/key |
| [Agent integration](docs/agent-integration.md) | MCP and Claude Desktop |
| [Agent memory instructions](docs/agent-memory-instructions.md) | Guidance served to agents over MCP |
| [OpenClaw](docs/openclaw.md) | OpenClaw plugin setup |
| [Testing your install](docs/testing-your-install.md) | End-to-end smoke check |

## Tests

```bash
cd build && ctest --output-on-failure   # 14 suites
```

## License

GPL v3. Commercial (non-GPL) licenses available — contact
[reid@diskerror.com](mailto:reid@diskerror.com).
