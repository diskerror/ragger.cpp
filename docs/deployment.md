# Deployment

Ragger supports both single-user and multi-user deployments.

## Single-user Setup (Per-user Install)

This is the default for personal use. No `sudo` required.

### Installation Locations

| Platform | Executable                         | Config                             | Database                            |
|----------|------------------------------------|------------------------------------|-------------------------------------|
| macOS    | `~/.local/bin/ragger`              | `~/.ragger/ragger.ini`             | `~/.ragger/memories.db`             |
| Linux    | `~/.local/bin/ragger`              | `~/.ragger/ragger.ini`             | `~/.ragger/memories.db`             |
| Windows  | `%LOCALAPPDATA%\ragger\ragger.exe` | `%LOCALAPPDATA%\ragger\ragger.ini` | `%LOCALAPPDATA%\ragger\memories.db` |

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

## System-wide Install (Multi-user)

**Status:** Framework in place, per-user DB routing in progress.

Multi-user framework is complete (layered config, SERVER_LOCKED keys,
system ceilings, token auth). User-specific DB routing via authenticated
HTTP requests is planned.

**Locations:**

| Component       | Path                           |
|-----------------|--------------------------------|
| Executable      | `/usr/local/bin/ragger`        |
| System config   | `/etc/ragger.ini`              |
| Common database | `/var/ragger/memories.db`      |
| Embedding model | `/var/ragger/models/`          |
| System logs     | `/var/log/ragger/`             |
| User databases  | `~/.ragger/memories.db`        |
| User tokens     | `~/.ragger/token`              |

---

## macOS LaunchDaemon Setup

To run Ragger as a background service on macOS (starts at boot, runs as
a specific user):

### 1. Create a LaunchDaemon plist

`/Library/LaunchDaemons/com.diskerror.ragger.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.diskerror.ragger</string>

        <key>ProgramArguments</key>
        <array>
            <string>/usr/local/bin/ragger</string>
            <string>serve</string>
        </array>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <true/>

        <key>StandardOutPath</key>
        <string>/var/log/ragger/stdout.log</string>

        <key>StandardErrorPath</key>
        <string>/var/log/ragger/stderr.log</string>
    </dict>
</plist>
```

**Note:** The daemon uses `/etc/ragger.ini` for configuration. All paths
are system paths (`/var/ragger/*`, `/var/log/ragger/*`) — no `~` expansion.

### 2. Load the LaunchDaemon

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/com.diskerror.ragger.plist
```

**Check status:**

```bash
sudo launchctl list | grep ragger
curl -s http://localhost:8432/health
```

**View logs:**

```bash
tail -f /var/log/ragger/stdout.log
tail -f /var/log/ragger/stderr.log
```

**Unload:**

```bash
sudo launchctl bootout system/com.diskerror.ragger
```

---

## Switching Between C++ and Python Versions

Both versions use the same:

- Database format (`~/.ragger/memories.db`)
- Config file (`~/.ragger/ragger.ini`)
- HTTP API (identical endpoints)
- Port (8432 by default)

You can swap between them without data migration.

### Keep Both Installed

```bash
# C++ version as "ragger"
cp ~/.local/bin/ragger-cpp ~/.local/bin/ragger

# Python version as "ragger-py"
cat > ~/.local/bin/ragger-py << 'EOF'
#!/bin/bash
exec python3 ~/PyCharmProjects/Ragger/ragger_memory/cli.py "$@"
EOF
chmod +x ~/.local/bin/ragger-py
```

Or vice versa (Python as `ragger`, C++ as `ragger-cpp`).

### Swap the Default

```bash
# Stop the server
pkill ragger  # or: launchctl unload ...

# Replace the binary
cp ~/.local/bin/ragger-py ~/.local/bin/ragger

# Restart
ragger serve
```

No data loss — both versions read the same database.

---

## Install Script (`install.sh`)

Both Python and C++ versions include an idempotent `install.sh` script that:

- Creates system user (`_ragger` on macOS, `ragger` on Linux) and group if needed
- Creates `/var/ragger/` and `/var/log/ragger/` with correct permissions
- Installs `/etc/ragger.ini` system config if missing
- Installs LaunchDaemon (macOS) or systemd unit (Linux) if missing
- Builds and installs the binary to `/usr/local/bin/ragger`
- Restarts the daemon if running

**Usage:**

```bash
cd /path/to/Ragger
sudo ./install.sh
```

The script is idempotent — safe to run multiple times. It creates system resources on first run and updates the binary on subsequent runs.

---

## Process Managers (Linux)

For Linux deployments, use systemd to run Ragger as a service.

### systemd Unit File

`/etc/systemd/system/ragger.service`:

```ini
[Unit]
Description = Ragger Memory Server
After = network.target

[Service]
Type = simple
User = reid
ExecStart = /home/reid/.local/bin/ragger serve --host 127.0.0.1 --port 8432
Restart = on-failure
StandardOutput = journal
StandardError = journal

[Install]
WantedBy = multi-user.target
```

**Enable and start:**

```bash
sudo systemctl enable ragger
sudo systemctl start ragger
```

**Check status:**

```bash
sudo systemctl status ragger
```

**View logs:**

```bash
sudo journalctl -u ragger -f
```

---

## Related

- [Configuration](configuration.md) — System vs user config
- [HTTP API](http-api.md) — Running the server
- [Getting Started](getting-started.md) — Installation basics
