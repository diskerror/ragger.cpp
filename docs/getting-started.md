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
| Config               | `~/.ragger/settings.ini`    |
| Database             | `~/.ragger/memories.db`     |
| Embedding model      | `~/.ragger/models/`         |
| Recipes              | `~/.ragger/recipes/`        |
| Inference formats    | `~/.ragger/formats/`        |
| Logs                 | `~/.ragger/logs/`           |
| Persona              | `~/.ragger/SOUL.md`         |
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
# macOS (MacPorts)
sudo port install boost eigen3 sqlite3 rust openssl

# Linux (apt)
sudo apt install libboost-all-dev libeigen3-dev libsqlite3-dev \
                 rustc cargo libssl-dev libcurl4-openssl-dev
```

Vendored (already in the repo): cpp-httplib, ONNX Runtime,
tokenizers-cpp, nlohmann/json.

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
ragger recipe default      # revert to whatever settings.ini says
```

The picker stores your choice in the DB so it survives daemon
restarts. Add your own JSON files under `~/.ragger/recipes/` to
extend the menu — see [Configuration → Recipes](configuration.md#recipes).

## Turn capture (opt-in)

Both sides of the turn pipeline are off by default. Enable them in
`~/.ragger/settings.ini`:

```ini
[server]
capture_turns = true     # write: ingest agent-pushed turns
build_context = true     # read: assemble recipe payloads from those turns
```

`ragger reload` (or `ragger restart`) picks up the change. With
capture on, every turn an agent hands Ragger gets summarized in the
background by the daemon-resident worker — no manual work for the
agent.

## Embedding model

The model is downloaded on first use. To pull it explicitly:

```bash
ragger update-model
```

Files land in `~/.ragger/models/all-MiniLM-L6-v2/`.

## Next steps

- [Configuration](configuration.md) — Full `settings.ini` reference
- [Search & RAG](search-and-rag.md) — How hybrid search works
- [HTTP API](http-api.md) — Run the daemon for tool integration
- [Agent integration](agent-integration.md) — `capture_turn`, recipes, MCP
- [Deployment](deployment.md) — Service units and the daemon lifecycle
