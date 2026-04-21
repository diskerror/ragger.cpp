# OpenClaw Plugin

This directory contains the OpenClaw memory plugin for Ragger.

## Files

- `openclaw.plugin.json` — Plugin metadata and configuration schema
- `index.ts` — Plugin implementation (dual HTTP/MCP transport)

## Installation

The `install.sh` script automatically installs these files to `~/.openclaw/extensions/ragger/` if OpenClaw is detected.

### Manual Installation

```bash
mkdir -p ~/.openclaw/extensions/ragger
cp openclaw-plugin/* ~/.openclaw/extensions/ragger/
```

Then configure in `~/.openclaw/openclaw.json` (see [docs/openclaw.md](../docs/openclaw.md)).

## Transport Modes

- **MCP** (local-only): Spawns `ragger mcp` on demand, no daemon needed
- **HTTP** (required for remote or shared daemon): Connects to running daemon with token auth
- **Auto** (development): Try HTTP, fall back to MCP

See [docs/openclaw.md](../docs/openclaw.md) for full configuration guide.
