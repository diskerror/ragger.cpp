#!/bin/bash
# install-openclaw.sh — Wire the Ragger memory plugin into OpenClaw.
#
# Usage: ./install-openclaw.sh
#
# Run this AFTER install-bin.sh if you have OpenClaw installed.
# Requires: ragger already installed to ~/.local/bin/ragger
#
# What this does:
#   - Copies openclaw-plugin/ files into ~/.openclaw/extensions/ragger/
#   - Merges ragger entries into ~/.openclaw/openclaw.json (conservative
#     merge — never clobbers values you've already set)
#
# Idempotent — safe to re-run on update.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
    fail "Do NOT run install-openclaw.sh as root."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"

OPENCLAW_HOME="$HOME/.openclaw"
OPENCLAW_PLUGIN_SRC="$SRC/openclaw-plugin"
OPENCLAW_PLUGIN_DST="$OPENCLAW_HOME/extensions/ragger"
OPENCLAW_CFG="$OPENCLAW_HOME/openclaw.json"

if [ ! -d "$OPENCLAW_HOME" ]; then
    fail "OpenClaw not found at $OPENCLAW_HOME — install OpenClaw first."
fi

if [ ! -d "$OPENCLAW_PLUGIN_SRC" ]; then
    fail "Plugin source not found at $OPENCLAW_PLUGIN_SRC — run from the Ragger repo root."
fi

echo ""
info "Installing Ragger memory plugin into OpenClaw at $OPENCLAW_HOME"

mkdir -p "$OPENCLAW_PLUGIN_DST"
for f in openclaw.plugin.json index.ts; do
    if [ -f "$OPENCLAW_PLUGIN_SRC/$f" ]; then
        cp "$OPENCLAW_PLUGIN_SRC/$f" "$OPENCLAW_PLUGIN_DST/$f"
        info "  wrote $OPENCLAW_PLUGIN_DST/$f"
    fi
done

# Ensure openclaw.json wires the ragger memory plugin. We merge conservatively:
# only set keys that are missing, never clobber user-set values.
if [ ! -f "$OPENCLAW_CFG" ]; then
    warn "$OPENCLAW_CFG not found — run OpenClaw once to create it, then re-run this script."
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found — cannot merge openclaw.json hooks automatically."
    warn "Ensure plugins.slots.memory=\"ragger\" and plugins.entries.ragger"
    warn "are set in $OPENCLAW_CFG (see docs/openclaw.md)."
    exit 0
fi

python3 - "$OPENCLAW_CFG" << 'PYEOF'
import json, sys, pathlib, shutil, datetime

cfg_path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(cfg_path.read_text())
except Exception as e:
    print(f"[!] Could not parse {cfg_path} ({e}) — skipping hook merge", file=sys.stderr)
    sys.exit(0)

changed = False
plugins = data.setdefault("plugins", {})
slots   = plugins.setdefault("slots", {})
entries = plugins.setdefault("entries", {})
allow   = plugins.setdefault("allow", [])

# Trust allowlist: OpenClaw warns at startup about non-bundled plugins
# that aren't explicitly trusted. Add "ragger" if it isn't already there.
if not isinstance(allow, list):
    print(f"[!]   plugins.allow is not a list ({type(allow).__name__}) — leaving alone")
elif "ragger" not in allow:
    allow.append("ragger")
    changed = True
    print("[+]   plugins.allow += [\"ragger\"]")

if slots.get("memory") != "ragger":
    if "memory" not in slots:
        slots["memory"] = "ragger"
        changed = True
        print("[+]   plugins.slots.memory = \"ragger\"")
    else:
        print(f"[!]   plugins.slots.memory already set to \"{slots['memory']}\" — leaving alone")

if "ragger" not in entries:
    # autoRecall/autoCapture are OFF by default — Ragger proper will own
    # recall and turn capture directly (see FM roadmap / issues #40, #41).
    # The OC plugin's auto-inject was found to be too eager and polluted
    # payloads with recall that the user didn't ask for. Leaving these off
    # also makes the MCP transport strictly a memory-tools surface, which
    # matches the intended long-term architecture.
    entries["ragger"] = {
        "enabled": True,
        "config": {
            "transport": "mcp",
            "autoRecall": False,
            "autoCapture": False,
        },
    }
    changed = True
    print("[+]   plugins.entries.ragger added (transport=mcp, auto* off)")
else:
    rag = entries["ragger"]
    if not rag.get("enabled", False):
        rag["enabled"] = True
        changed = True
        print("[+]   plugins.entries.ragger.enabled = true")
    rag.setdefault("config", {})
    print("[+]   plugins.entries.ragger already present — kept existing config")

if changed:
    # Timestamped backup once per day
    bak = cfg_path.with_suffix(
        f".json.bak-{datetime.date.today().isoformat()}"
    )
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[+]   backed up → {bak.name}")
    cfg_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[+]   updated {cfg_path}")
else:
    print(f"[+]   {cfg_path} already wired — no changes")
PYEOF

echo ""
info "OpenClaw plugin installed."
echo ""
echo "  Start the Ragger daemon:        ragger start"
echo "  Restart the OpenClaw gateway to pick up the changes."
echo ""
