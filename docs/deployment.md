# Deployment

Ragger operates in two modes: **single-user** and **multi-user**.

## Single-user vs Multi-user

In **single-user mode**, everything lives in your home directory
(`~/.ragger/`). Your config, database, and token are all yours. No system
user, no shared resources. This is the default when you install per-user
or run `ragger mcp`.

In **multi-user mode**, a system user (`_ragger` on macOS, `ragger` on
Linux) owns shared resources under `/var/ragger/`. The common database
holds the **user table** — which manages authentication tokens and
per-user settings. Each user still has their own database
(`~/.ragger/memories.db`) for their actual memories. The common DB
doesn't need to be used for anything else beyond the user table.

`ragger serve` (the HTTP daemon) **always requires multi-user mode**.
It must run as the system user and needs the user table to authenticate
connections via bearer tokens. If you just want Ragger for yourself,
you don't need `ragger serve` — use `ragger mcp` or the CLI directly.

`ragger mcp` works in **either mode**. In a multi-user environment
(e.g., multiple users via SSH), each user can run their own `ragger mcp`
instance. This is functionally correct but less resource-efficient since
each instance loads its own embedding model.

## Single-user Setup (Per-user Install)

This is the default for personal use. No `sudo` required.

### Installation Locations

| Platform | Executable                         | Config                             | Database                            |
|----------|------------------------------------|------------------------------------|-------------------------------------|
| macOS    | `~/.local/bin/ragger`              | `~/.ragger/ragger.ini`             | `~/.ragger/memories.db`             |
| Linux    | `~/.local/bin/ragger`              | `~/.ragger/ragger.ini`             | `~/.ragger/memories.db`             |
| Windows  | `%LOCALAPPDATA%\ragger\ragger.exe` | `%LOCALAPPDATA%\ragger\ragger.ini` | `%LOCALAPPDATA%\ragger\memories.db` |

> **Note:** Windows support is planned but not yet implemented. macOS and Linux only for now.

### Install Script

On macOS/Linux, ensure `~/.local/bin` is in your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc or ~/.bashrc
```

Install the Python version as `ragger`:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/ragger << 'EOF'
#!/bin/bash
RAGGER_PY_DIR="${RAGGER_PY_DIR:-$HOME/PyCharmProjects/Ragger}"
exec python3 "$RAGGER_PY_DIR/ragger_memory/cli.py" "$@"
EOF
chmod +x ~/.local/bin/ragger
```

If you also have the C++ version installed, you can keep both:

- `ragger` → C++ binary (default)
- `ragger-py` → Python version

Or vice versa.

---

## Multi-user Deployment (System-wide Install)

### Step 1: Run the installer

```bash
cd /path/to/Ragger
sudo ./install.sh
```

The install script is idempotent (safe to run multiple times) and handles:

- Creates system user (`_ragger` on macOS, `ragger` on Linux) and `ragger` group
- Creates `/var/ragger/` (common database) and `/var/log/ragger/` with correct permissions
- Installs system config at `/etc/ragger.ini` if missing
- Installs the executable to `/usr/local/bin/ragger`
  - C++ version: copies the compiled binary directly
  - Python version: syncs code to `/usr/local/lib/ragger/`, creates venv, installs a wrapper script
- Creates the daemon configuration:
  - **macOS:** LaunchDaemon plist at `/Library/LaunchDaemons/com.diskerror.ragger.plist`
  - **Linux:** systemd unit at `/etc/systemd/system/ragger.service`
- Installs `SOUL.md` to `/var/ragger/` if not present (shared assistant persona)
- Installs the OpenClaw plugin if OpenClaw is detected

### Step 2: Add users

Register each user who will use Ragger:

```bash
sudo ragger add-user <username>
```

Or provision all users on the system at once:

```bash
sudo ragger add-all
```

This does three things for each user:

1. Creates `~/.ragger/` and generates an authentication token
2. Adds the user to the `ragger` OS group (for `/var/ragger/` access)
3. Registers the user in the common database (user table)

> **Note:** Users must log out and back in for group membership to take
> effect. If a user was already in the `ragger` group, this is a no-op.

To remove a user:

```bash
sudo ragger remove-user <username>
```

This removes the user from the `ragger` OS group, deletes their entry
from the common database, removes their token, and deletes `~/.ragger/`.
Use `--keep-data` to preserve `~/.ragger/` (only removes DB entry, token,
and group membership).

### Step 3: Start the daemon

**macOS:**

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.diskerror.ragger.plist
```

**Linux:**

```bash
sudo systemctl enable ragger
sudo systemctl start ragger
```

### Step 4: Verify

```bash
# Check the daemon is running
curl -s http://localhost:8432/health

# Test with your token
ragger search "test"
```

### Installation Locations

- `/usr/local/bin/ragger` — Executable (binary or wrapper script)
- `/usr/local/lib/ragger/` — Python code and venv (Python version only)
- `/etc/ragger.ini` — System configuration
- `/var/ragger/memories.db` — Common database (user table)
- `/var/ragger/SOUL.md` — Shared assistant persona
- `/var/ragger/models/` — Embedding model cache
- `/var/log/ragger/` — Daemon logs
- `~/.ragger/memories.db` — Per-user memory database
- `~/.ragger/token` — Per-user authentication token
- `~/.ragger/ragger.ini` — Per-user config overrides

### Managing the Daemon

**macOS:**

```bash
# Check status
sudo launchctl list | grep ragger

# View logs
tail -f /var/log/ragger/stdout.log
tail -f /var/log/ragger/stderr.log

# Restart
sudo launchctl bootout system/com.diskerror.ragger 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.diskerror.ragger.plist

# Stop
sudo launchctl bootout system/com.diskerror.ragger
```

**Linux:**

```bash
# Check status
sudo systemctl status ragger

# View logs
sudo journalctl -u ragger -f

# Restart
sudo systemctl restart ragger

# Stop
sudo systemctl stop ragger
```

---

## Switching from Single-user to Multi-user

No reinstall needed. Three steps:

1. Edit `/etc/ragger.ini` — set `single_user = false`
2. Add users: `sudo ragger add-user <username>` for each user
   (this handles group membership, token, and DB registration)
3. Restart the daemon

Client configs (OpenClaw, Claude Desktop) are set up manually — see
[OpenClaw Integration](openclaw.md) for transport options.

---

## Switching Between C++ and Python Versions

Both versions use the same database format, config files, HTTP API, and
default port (8432). You can swap between them without data migration.

For per-user installs, you can keep both as separate commands:

```bash
# C++ version as "ragger", Python as "ragger-py" (or vice versa)
cp ~/.local/bin/ragger-cpp ~/.local/bin/ragger
```

For system installs, just re-run the other version's `install.sh` — it
overwrites `/usr/local/bin/ragger` with the new version.

---

## Troubleshooting

**"Permission denied" accessing `/var/ragger/`:**
Make sure your user is in the `ragger` group (see Step 2 above). Log out
and back in after adding group membership.

**Daemon won't start:**
Check the logs. On macOS: `tail /var/log/ragger/stderr.log`. On Linux:
`journalctl -u ragger -e`. Common causes: port 8432 already in use,
missing SOUL.md in multi-user mode, permission issues on `/var/ragger/`.

**Embedding model download hangs:**
The model (~90MB) downloads on first run. If the daemon can't reach
HuggingFace, pre-download it: `ragger update-model`, then copy the
model cache to `/var/ragger/models/`.

**Token issues:**
Each user's token lives at `~/.ragger/token`. If it gets corrupted or
lost, re-register: `ragger user add <username>` (as admin or with sudo).

---

## Related

- [Configuration](configuration.md) — System vs user config
- [HTTP API](http-api.md) — Running the server
- [Getting Started](getting-started.md) — Installation basics
