# OpenClaw Plugin

This directory contains the OpenClaw memory plugin for Ragger.

## Files

- `openclaw.plugin.json` — Plugin metadata and configuration schema
- `index.ts` — Plugin implementation (dual HTTP/MCP transport)

## Installation

The Ragger `install.sh` installs this plugin automatically when it detects
`~/.openclaw/`. It:

1. Copies `openclaw.plugin.json` and `index.ts` into
   `~/.openclaw/extensions/ragger/` — **always overwriting**, so the
   plugin stays in sync with the Ragger binary version.
2. Merges the minimal ragger hooks into `~/.openclaw/openclaw.json`:
   - `plugins.slots.memory = "ragger"` (only if the slot is unset —
     never clobbers a different choice)
   - `plugins.entries.ragger` with `transport: "mcp"` defaults (only
     if missing — existing config is preserved)

   A timestamped backup (`openclaw.json.bak-YYYY-MM-DD`) is created
   the first time install.sh changes the file on a given day.

### Manual Installation

If you can't use `install.sh`:

```bash
mkdir -p ~/.openclaw/extensions/ragger
cp openclaw-plugin/openclaw.plugin.json \
   openclaw-plugin/index.ts \
   ~/.openclaw/extensions/ragger/
```

Then add the plugin hooks to `~/.openclaw/openclaw.json` yourself (see
[docs/openclaw.md](../docs/openclaw.md)).

## Transport Modes

- **MCP** (local-only): Spawns `ragger mcp` on demand, no daemon needed
- **HTTP** (required for remote or shared daemon): Connects to running daemon with token auth
- **Auto** (development): Try HTTP, fall back to MCP

See [docs/openclaw.md](../docs/openclaw.md) for full configuration guide.
