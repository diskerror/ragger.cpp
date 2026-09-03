# Getting Started

## Requirements

- **macOS or Linux** (Windows not yet supported)
- **C++23 toolchain** (recent clang or gcc)
- **~1 GB disk** for the embedding model and build dependencies

First run downloads the embedding model (~90 MB) into
`~/.ragger/models/`. Everything else is offline after that.

## Build and install

```bash
cd /path/to/ragger.cpp
./scripts/build.sh        # check deps, build
./scripts/install.sh      # copy binary, write user service unit, install recipes
```

No `sudo`. The binary lands in `~/.local/bin` (already on `PATH` in
most modern shells); everything else under `~/.ragger/`:

| What                 | Where                       |
|----------------------|-----------------------------|
| Executable           | `~/.local/bin/ragger`       |
| Config               | `settings` table in `~/.ragger/memories.db` |
| Database             | `~/.ragger/memories.db`     |
| Embedding model      | `~/.ragger/models/`         |
| Recipes              | `~/.ragger/recipes/`        |
| Inference formats    | `~/.ragger/formats/`        |
| Log                  | `~/.ragger/logs/activity.log`    |
| Bearer token         | `~/.ragger/token`           |

`install.sh` is idempotent — re-run it after any rebuild to refresh the
binary, service unit, and shipped recipes. Your config, database, and
custom recipe files are preserved.

If `~/.local/bin` isn't on your `PATH`, the installer adds it to your
shell rc. Open a new shell (or `source ~/.zshrc` / `~/.bashrc`) to
pick the change up.

## Build dependencies

Installed once via your package manager:

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

Vendored (already in the repo): cpp-httplib, tokenizers-cpp sources,
nlohmann/json. ONNX Runtime is auto-downloaded at configure time for
your platform; drop a `vendor/onnxruntime/` override in place for
offline builds.

`scripts/install.sh` downloads the all-MiniLM-L6-v2 embedding model on
first run (~90 MB). Re-run the script to retry if the network was down.

## First run

```bash
# Store and recall by hand to confirm the round trip works.
ragger store "Test memory: deploy needs Node 18+"
ragger search "deployment requirements"
ragger count
```

Bring up the daemon if you want HTTP / MCP integrations or background
summarization:

```bash
ragger start
ragger status
```

## Recipes

`build_context` returns a recipe-shaped payload; five recipes ship out
of the box (`natural_fading`, `reconnect`, `deep_recall`, `tldr`,
`raw_only`). Browse them with:

```bash
ragger recipe              # interactive picker (↑/↓, Enter to set, q to quit)
ragger recipe reconnect    # set non-interactively
ragger recipe default      # revert to the configured default_recipe
```

The picker stores your choice in the DB so it survives daemon
restarts. Add your own JSON files under `~/.ragger/recipes/` to
extend the menu — see [Configuration → Recipes](configuration.md#recipes).

## Turn capture

`capture_turns` ships enabled; `build_context` ships **disabled**
(agents get `search`/`store` either way; the recipe-shaped context
payload is opt-in). To change either, use the dashboard or the CLI:

```bash
ragger config set capture_turns true    # write: ingest agent-pushed turns
ragger config set build_context false   # read: serve recipe-shaped context payloads
```

Note: `build_context` and `default_recipe` are still fully functional
but no longer documented in the shipped `default-settings.txt` template
— add them by hand if you want the read side turned on.

`ragger reload` (or `ragger restart`) picks up the change. With
capture on, every turn an agent hands Ragger gets summarized in the
background by the daemon-resident worker — no manual work for the
agent.

## Embedding model

The model is downloaded once by `install.sh` (~90 MB); re-run the
script if the network was down the first time. There's no separate
CLI command to fetch it — `install.sh` is the mechanism.

Files land in `~/.ragger/models/all-MiniLM-L6-v2/`.

## Next steps

- [Configuration](configuration.md) — Full config key reference
- [Search & RAG](search-and-rag.md) — How hybrid search works
- [HTTP API](http-api.md) — Run the daemon for tool integration
- [Agent integration](agent-integration.md) — `capture_turn`, recipes, MCP
- [Deployment](deployment.md) — Service units and the daemon lifecycle
