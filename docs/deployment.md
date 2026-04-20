# Deployment

Ragger installs per-user. Everything lives under `~/.ragger/`:

```
~/.ragger/bin/ragger        # executable
~/.ragger/settings.ini      # config
~/.ragger/memories.db       # SQLite database
~/.ragger/ragger.sock       # unix socket (daemon)
~/.ragger/logs/             # daemon logs
~/.ragger/models/           # embedding models
~/.ragger/formats/          # inference format definitions
~/.ragger/www/              # web UI assets
~/.ragger/SOUL.md           # assistant persona
```

No `sudo`, no system user, no `/etc/` or `/var/` paths. The daemon
runs as you.

## Multi-user: One Install, Many Clients

Ragger serves additional users over HTTP with bearer tokens. There's
still only one install and one database — the "daemon owner" runs
`ragger` under their own account, and other humans / agents connect
to that daemon with tokens. They don't need OS accounts on the host.

```
                    ┌──────────────────────────┐
  sub-user A ──HTTP▶│                          │
  sub-user B ──HTTP▶│  ragger daemon           │
  sub-user C ──HTTP▶│  (runs as daemon-owner)  │
  daemon-owner ─────│  ~/.ragger/memories.db   │
  (local CLI/MCP) ─▶│                          │
                    └──────────────────────────┘
```

Isolation between sub-users is enforced at the API layer, not at the
filesystem layer.

## Install

```bash
cd /path/to/ragger.cpp
./build.sh           # dependency check + cmake build
./install.sh         # copy binary, write LaunchAgent / user systemd unit, update PATH
ragger start         # bring the daemon up
```

`install.sh`:

- Creates `~/.ragger/{bin,logs,models,formats,www}` if missing
- Copies `example-settings.ini` → `~/.ragger/settings.ini` on first run
- Copies the built binary to `~/.ragger/bin/ragger` (codesigns on macOS)
- Adds `~/.ragger/bin` to `PATH` in your shell rc (`.zshrc` / `.bash_profile` / `.bashrc` / `.profile`)
- Writes a user service unit:
  - **macOS:** `~/Library/LaunchAgents/com.diskerror.ragger.plist`
  - **Linux:** `~/.config/systemd/user/ragger.service` (+ `systemctl --user enable ragger.service`)
- Installs the default `SOUL.md` to `~/.ragger/` if you don't already have one
- Copies bundled formats and web assets under `~/.ragger/`

It's idempotent — re-run after a rebuild to update the binary. Config,
database, SOUL.md, and custom formats are preserved.

### Installation locations

| Platform | Executable              | Config                    | Database                |
|----------|-------------------------|---------------------------|-------------------------|
| macOS    | `~/.ragger/bin/ragger`  | `~/.ragger/settings.ini`  | `~/.ragger/memories.db` |
| Linux    | `~/.ragger/bin/ragger`  | `~/.ragger/settings.ini`  | `~/.ragger/memories.db` |
| Windows  | not yet supported       | —                         | —                       |

## Daemon Lifecycle

```bash
ragger start        # bring the daemon up (via launchctl / systemctl --user)
ragger stop         # take it down
ragger restart      # bounce after editing settings.ini
ragger status       # is it running?
```

Under the hood these wrap `launchctl bootstrap/bootout gui/$UID` on
macOS and `systemctl --user start/stop/restart/status ragger.service`
on Linux. `serve` is the foreground entry the service unit itself
invokes — you rarely run it directly.

### Linger (Linux)

By default, a systemd user instance stops when you log out. To keep
the daemon running across logout / reboot:

```bash
sudo loginctl enable-linger $USER
```

This is the only time `install.sh` asks you to touch `sudo` on Linux,
and it's optional.

### Logs

- `~/.ragger/logs/stdout.log`
- `~/.ragger/logs/stderr.log`

`deploy.sh` truncates these on each deploy so a fresh run is easy to
read.

## Adding Sub-Users

```bash
ragger useradd <name>     # prints the generated token once
ragger userdel <name>
```

The token is shown exactly once at creation — hand it to the sub-user
via whatever channel you trust. They set it in their client:

```bash
# Example: curl with the token
curl -H "Authorization: Bearer <token>" \
     http://daemon-host:8432/health
```

For a sub-user on the same machine, the daemon can bypass token auth
on the unix socket — see `[server] auth_bypass_socket` in
`example-settings.ini`.

## Rebuilding and Redeploying

After code changes, rebuild then run `deploy.sh`:

```bash
cmake --build build --parallel
./deploy.sh
```

`deploy.sh` stops the daemon, copies the new binary in place,
codesigns on macOS, truncates logs, starts the daemon, and
health-checks it over the unix socket or `127.0.0.1:8432`. It does
**not** touch config, schema, or the install layout — it's a fast
binary swap only. Run `install.sh` again if the service unit or PATH
entry needs updating.

## Switching Between C++ and Python Versions

Both versions use the same database format, config file, HTTP API,
and default port (8432). You can swap between them without data
migration — re-run the other version's `install.sh` and it overwrites
`~/.ragger/bin/ragger` with the new binary.

If you want both side by side:

```bash
cp ~/.ragger/bin/ragger ~/.ragger/bin/ragger-cpp   # back up the current
# run the other install.sh, then:
mv ~/.ragger/bin/ragger ~/.ragger/bin/ragger-py
mv ~/.ragger/bin/ragger-cpp ~/.ragger/bin/ragger
```

## Troubleshooting

**Daemon won't start:**
Check `~/.ragger/logs/stderr.log`. Common causes: port 8432 already
in use, missing embedding model in `~/.ragger/models/`, invalid
`settings.ini` (run `ragger serve` in the foreground to see the
parse error).

**`ragger start` says "service loaded" but nothing's listening:**
On macOS, `launchctl print gui/$UID/com.diskerror.ragger` shows the
actual state. On Linux, `systemctl --user status ragger.service` and
`journalctl --user -u ragger.service` are your friends.

**Embedding model missing:**
The model (~90 MB) downloads on first run. If the daemon can't reach
HuggingFace, pre-download it somewhere with internet and copy to
`~/.ragger/models/`.

**Token issues:**
Sub-user tokens live in the database, not on disk. If a sub-user
loses their token, the daemon owner runs `ragger userdel <name>`
then `ragger useradd <name>` to issue a new one.

**Daemon stops after logout (Linux):**
Run `sudo loginctl enable-linger $USER`. See "Linger" above.

## Related

- [Configuration](configuration.md) — Single `settings.ini` reference
- [HTTP API](http-api.md) — Endpoints and auth
- [Getting Started](getting-started.md) — First run
