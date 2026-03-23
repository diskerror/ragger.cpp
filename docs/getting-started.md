# Getting Started

## Requirements

- **Python 3.10+**
- **~1GB disk space** for model + dependencies
- **SQLite backend:** No extra dependencies (uses Python stdlib)

First run downloads the embedding model (~90MB) to your HuggingFace cache.
After that, all operations are offline.

## Installation

### Per-user Install (single user, no sudo)

| Platform | Executable location | Config location |
|----------|---------------------|-----------------|
| macOS    | `~/.local/bin/ragger` | `~/.ragger/ragger.ini` |
| Linux    | `~/.local/bin/ragger` | `~/.ragger/ragger.ini` |
| Windows  | `%LOCALAPPDATA%\ragger\ragger.exe` | `%LOCALAPPDATA%\ragger\ragger.ini` |

On macOS/Linux, ensure `~/.local/bin` is in your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc or ~/.bashrc
```

Install as the default `ragger` command:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/ragger << 'EOF'
#!/bin/bash
RAGGER_PY_DIR="${RAGGER_PY_DIR:-$HOME/PyCharmProjects/Ragger}"
exec python3 "$RAGGER_PY_DIR/ragger_memory/cli.py" "$@"
EOF
chmod +x ~/.local/bin/ragger
```

If the C++ version is also installed, this can coexist as `ragger-py`
while the C++ binary is the default `ragger`.

### Python Dependencies

```bash
cd /path/to/Ragger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads the embedding model (~90MB) to your HuggingFace cache.
After that, all operations are offline.

## Storage

SQLite is the default backend — zero setup, single-file database at
`~/.ragger/memories.db`. No configuration needed.

The abstract `MemoryBackend` base class makes it straightforward to add
new backends (Postgres, Qdrant, etc.) — see [Python API](python-api.md)
for details.

## Upgrading from v0.4.x

The database location moved from `~/.local/share/ragger/memories.db` to
`~/.ragger/memories.db`. Copy or move your database file to the new location:

```bash
mkdir -p ~/.ragger
mv ~/.local/share/ragger/memories.db ~/.ragger/memories.db
```

## First Run

Test your installation:

```bash
# Store a test memory
ragger store "Test memory"

# Search for it
ragger search "test"

# Check count
ragger count
```

If the commands run without errors, you're ready to go.

## Downloading the Model

The embedding model downloads automatically on first use. To download it
explicitly:

```bash
ragger update-model
```

This is useful for offline deployments or pre-warming a fresh install.

## Next Steps

- [Configuration](configuration.md) — Customize settings
- [Collections](collections.md) — Organize your memories
- [Search & RAG](search-and-rag.md) — Learn how search works
- [HTTP API](http-api.md) — Run the server for tool integration
