# Deployment

Ragger installs per-user. The executable lives on `PATH` under
`~/.local/bin`; data, config, and runtime files live under `~/.ragger/`:

```
~/.local/bin/ragger         # executable (on PATH)

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

- Creates `~/.local/bin` and `~/.ragger/{logs,models,formats,www}` if missing
- Copies `example-settings.ini` → `~/.ragger/settings.ini` on first run
- Copies the built binary to `~/.local/bin/ragger` (codesigns on macOS)
- Ensures `~/.local/bin` is on `PATH` — only edits your shell rc if it isn't
  already there (most modern shells include it by default)
- Writes a user service unit:
  - **macOS:** `~/Library/LaunchAgents/com.diskerror.ragger.plist`
  - **Linux:** `~/.config/systemd/user/ragger.service` (+ `systemctl --user enable ragger.service`)
- Installs the default `SOUL.md` to `~/.ragger/` if you don't already have one
- Copies bundled formats and web assets under `~/.ragger/`
- Removes a legacy `~/.ragger/bin/` directory if present (old install layout)

It's idempotent — re-run after a rebuild to update the binary. Config,
database, SOUL.md, and custom formats are preserved.

### Installation locations

| Platform | Executable              | Config                    | Database                |
|----------|-------------------------|---------------------------|-------------------------|
| macOS    | `~/.local/bin/ragger`   | `~/.ragger/settings.ini`  | `~/.ragger/memories.db` |
| Linux    | `~/.local/bin/ragger`   | `~/.ragger/settings.ini`  | `~/.ragger/memories.db` |
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

User management is split across three standard verbs:

```bash
ragger useradd <name>     # create user + mint bearer token (printed once)
ragger usermod <name>     # rotate an existing user's token (printed once)
ragger userdel <name>     # remove user and revoke their token
ragger passwd  <name>     # set (or clear) web-UI login password — only needed
                          # for remote browser sessions; loopback is auto-auth
```

`useradd` errors if the user already exists — use `usermod` to rotate. The
token is shown exactly once — hand it to the sub-user via whatever channel
you trust. They set it in their client:

```bash
# Example: curl with the token
curl -H "Authorization: Bearer <token>" \
     http://daemon-host:8432/health
```

**Local access needs no token.** Requests on the unix socket and from
`127.0.0.1` / `::1` are auto-authenticated as the daemon owner. Tokens
matter for remote clients (other hosts, OpenClaw on a different machine,
etc.). The daemon owner's own token is minted at install time and
written to `~/.ragger/token`.

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
`~/.local/bin/ragger` with the new binary.

If you want both side by side:

```bash
cp ~/.local/bin/ragger ~/.local/bin/ragger-cpp   # back up the current
# run the other install.sh, then:
mv ~/.local/bin/ragger ~/.local/bin/ragger-py
mv ~/.local/bin/ragger-cpp ~/.local/bin/ragger
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
