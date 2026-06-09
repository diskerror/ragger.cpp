#!/bin/bash
# install-hermes.sh — Wire Ragger as a memory provider plugin for Hermes Agent.
#
# Usage:
#   ./scripts/install-hermes.sh              # install / update
#   ./scripts/install-hermes.sh --uninstall  # remove plugin and revert config
#
# Run this AFTER install.sh if you have Hermes Agent installed.
# Requires: ragger already installed (~/.local/bin/ragger)
#
# What install does:
#   - Copies hermes-plugin/ files into ~/.hermes/plugins/ragger-memory/
#   - Sets memory.provider: ragger in ~/.hermes/config.yaml
#     (conservative merge — only sets the key if it is currently empty)
#
# What uninstall does:
#   - Removes ~/.hermes/plugins/ragger-memory/
#   - Clears memory.provider back to '' if it is currently 'ragger'
#
# Idempotent — safe to re-run.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

HERMES_HOME="$HOME/.hermes"
PLUGIN_DST="$HERMES_HOME/plugins/ragger"
HERMES_CFG="$HERMES_HOME/config.yaml"

# ============================================================
# Uninstall
# ============================================================

if [ "${1:-}" = "--uninstall" ]; then
    echo ""
    info "Uninstalling Ragger memory plugin from Hermes"

    # Remove plugin directory
    if [ -d "$PLUGIN_DST" ]; then
        rm -rf "$PLUGIN_DST"
        info "  removed $PLUGIN_DST"
    else
        info "  $PLUGIN_DST not found — already removed"
    fi

    # Revert memory.provider in config.yaml
    if [ -f "$HERMES_CFG" ] && command -v python3 >/dev/null 2>&1; then
        python3 - "$HERMES_CFG" << 'PYEOF'
import sys, pathlib, shutil, datetime, re

cfg_path = pathlib.Path(sys.argv[1])
lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

in_memory = False
changed = False

for i, line in enumerate(lines):
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if re.match(r'^memory:\s*(?:#.*)?$', line.rstrip()):
        in_memory = True
        continue

    if in_memory:
        if stripped and not stripped.startswith("#") and not line[0].isspace():
            break  # left the memory block

        m = re.match(r'^(\s+provider:\s*)(.*)$', line.rstrip())
        if m:
            current_val = m.group(2).strip().strip("'\"")
            if current_val == "ragger":
                lines[i] = f"{m.group(1)}''\n"
                changed = True
                print("[+]   memory.provider cleared (was 'ragger')")
            elif current_val:
                print(f"[+]   memory.provider is '{current_val}', not 'ragger' — leaving alone")
            else:
                print("[+]   memory.provider already empty — no change needed")
            break

if changed:
    bak = cfg_path.with_suffix(f".yaml.bak-{datetime.date.today().isoformat()}")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[+]   backed up → {bak.name}")
    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"[+]   updated {cfg_path}")
PYEOF
    elif [ -f "$HERMES_CFG" ]; then
        warn "python3 not found — clear memory.provider manually in $HERMES_CFG"
    fi

    echo ""
    info "Uninstall complete. Restart the Hermes gateway (hermes gateway restart) to pick up the change."
    exit 0
fi

# ============================================================
# Install
# ============================================================

if [ "$(id -u)" -eq 0 ]; then
    fail "Do NOT run install-hermes.sh as root."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"

PLUGIN_SRC="$SRC/hermes-plugin"

# --- Preflight ---

if [ ! -d "$HERMES_HOME" ]; then
    fail "Hermes not found at $HERMES_HOME — install Hermes Agent first."
fi

if [ ! -d "$PLUGIN_SRC" ]; then
    fail "Plugin source not found at $PLUGIN_SRC — run from the Ragger repo root."
fi

if [ ! -f "$HOME/.local/bin/ragger" ] && ! command -v ragger >/dev/null 2>&1; then
    warn "ragger binary not found on PATH — install Ragger first (./scripts/install.sh)."
    warn "Continuing anyway; plugin will be inactive until ragger is installed."
fi

echo ""
info "Installing Ragger memory plugin into Hermes at $HERMES_HOME"

# --- Copy plugin files ---

mkdir -p "$PLUGIN_DST"

for f in plugin.yaml __init__.py; do
    if [ -f "$PLUGIN_SRC/$f" ]; then
        cp "$PLUGIN_SRC/$f" "$PLUGIN_DST/$f"
        info "  wrote $PLUGIN_DST/$f"
    else
        fail "Missing $PLUGIN_SRC/$f — repo may be incomplete."
    fi
done

# --- Wire memory.provider in config.yaml ---

if [ ! -f "$HERMES_CFG" ]; then
    warn "$HERMES_CFG not found — run Hermes once to create it, then re-run this script."
    echo ""
    warn "When config.yaml exists, add manually:"
    warn "    memory:"
    warn "      provider: ragger"
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found — cannot update $HERMES_CFG automatically."
    warn "Add the following manually:"
    warn "    memory:"
    warn "      provider: ragger"
    exit 0
fi

python3 - "$HERMES_CFG" << 'PYEOF'
import sys, pathlib, shutil, datetime, re

cfg_path = pathlib.Path(sys.argv[1])
lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

in_memory = False
found = False
changed = False

for i, line in enumerate(lines):
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Detect the memory: section header (top-level key)
    if re.match(r'^memory:\s*(?:#.*)?$', line.rstrip()):
        in_memory = True
        continue

    if in_memory:
        # A new top-level key ends the memory block
        if stripped and not stripped.startswith("#") and not line[0].isspace():
            in_memory = False
            break

        # Look for "  provider: <value>"
        m = re.match(r'^(\s+provider:\s*)(.*)$', line.rstrip())
        if m:
            found = True
            current_val = m.group(2).strip().strip("'\"")
            if current_val in ("", "null"):
                lines[i] = f"{m.group(1)}ragger\n"
                changed = True
                print("[+]   memory.provider = ragger")
            elif current_val == "ragger":
                print("[+]   memory.provider already set to ragger — no change needed")
            else:
                print(f"[!]   memory.provider already set to '{current_val}' — leaving alone")
                print( "      To switch to Ragger, set it manually: memory.provider: ragger")
            break

if not found:
    print("[!]   memory.provider line not found in config.yaml")
    print("      Add manually under the memory: section:")
    print("          memory:")
    print("            provider: ragger")
    sys.exit(0)

if changed:
    bak = cfg_path.with_suffix(f".yaml.bak-{datetime.date.today().isoformat()}")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[+]   backed up → {bak.name}")
    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"[+]   updated {cfg_path}")
PYEOF

# --- Done ---

echo ""
info "Hermes plugin installed."
echo ""
echo "  Start the Ragger daemon:  ragger start"
echo "  Restart Hermes gateway:   hermes gateway restart  (or restart the Telegram bot)"
echo ""
echo "  To verify: send '/memory' or start a conversation — Hermes will show"
echo "  'Ragger Memory' in the system prompt context block."
echo ""
echo "  To remove:  ./scripts/install-hermes.sh --uninstall"
echo ""
if ! ragger status >/dev/null 2>&1; then
    warn "Ragger daemon is not currently running. Start it with: ragger start"
fi
