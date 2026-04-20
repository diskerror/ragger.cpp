# Getting Started

## Requirements

- **macOS or Linux** (Windows support planned, not yet implemented)
- **C++23 toolchain** (recent clang or gcc)
- **~1 GB disk space** for the embedding model + dependencies

First run downloads the embedding model (~90 MB) into
`~/.ragger/models/`. After that, all memory operations are offline.

## Build & Install

```bash
cd /path/to/ragger.cpp
./build.sh           # check dependencies + cmake build
./install.sh         # copy binary to ~/.ragger/bin, write user service unit, update PATH
```

No `sudo`. Everything lives under `~/.ragger/`:

| What        | Where                     |
|-------------|---------------------------|
| Executable  | `~/.ragger/bin/ragger`    |
| Config      | `~/.ragger/settings.ini`  |
| Database    | `~/.ragger/memories.db`   |
| Logs        | `~/.ragger/logs/`         |
| Models      | `~/.ragger/models/`       |
| Persona     | `~/.ragger/SOUL.md`       |

Open a new terminal (or `source ~/.zshrc` / `~/.bashrc`) so
`~/.ragger/bin` is on `PATH`.

See [Deployment](deployment.md) for full details on the install
script, service unit, and daemon lifecycle.

## Build Dependencies

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

## First Run

```bash
# Store a test memory
ragger store "Test memory"

# Search for it
ragger search "test"

# Check count
ragger count
```

If these run without errors, you're ready.

Start the HTTP daemon whenever you want it available to tools:

```bash
ragger start
ragger status
```

## Downloading the Model

The embedding model downloads automatically on first use. To pull it
explicitly (useful for offline prep):

```bash
ragger update-model
```

The files land in `~/.ragger/models/all-MiniLM-L6-v2/`.

## Upgrading from Earlier Layouts

If you have an older install at `~/.local/share/ragger/memories.db`
or `/var/ragger/memories.db`:

```bash
mkdir -p ~/.ragger
mv ~/.local/share/ragger/memories.db ~/.ragger/memories.db     # or
sudo mv /var/ragger/memories.db ~/.ragger/memories.db && \
  sudo chown $USER ~/.ragger/memories.db
```

The DB format is backwards-compatible through v0.9.3; Ragger migrates
forward automatically on open.

If you used `ragger.ini`, rename it:

```bash
mv ~/.ragger/ragger.ini ~/.ragger/settings.ini
```

## Next Steps

- [Configuration](configuration.md) — Tune `settings.ini`
- [Collections](collections.md) — Organize your memories
- [Search & RAG](search-and-rag.md) — How hybrid search works
- [HTTP API](http-api.md) — Run the daemon for tool integration
- [Deployment](deployment.md) — Service units, sub-users, and deploy.sh
