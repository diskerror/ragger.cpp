#!/bin/bash
# install-claude-desktop.sh — Register Ragger as an MCP memory server for the
# Claude desktop app.
#
# Usage:
#   ./scripts/install-claude-desktop.sh              # install / update
#   ./scripts/install-claude-desktop.sh --uninstall  # remove the ragger entry
#
# Run this AFTER install.sh. Requires: ragger already installed
# (~/.local/bin/ragger).
#
# What install does:
#   - Installs docs/agent-memory-instructions.md → ~/.ragger/ (only if absent,
#     so your edits are preserved) — `ragger mcp` serves it to clients via the
#     MCP `initialize` instructions field.
#   - Merges an "mcpServers.ragger" entry into the Claude desktop config
#     (conservative — never clobbers other servers or settings; daily backup).
#
# Restart the Claude desktop app afterward — it reads MCP config only at launch.
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
    fail "Do NOT run install-claude-desktop.sh as root."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"

UNINSTALL=0
[ "${1:-}" = "--uninstall" ] && UNINSTALL=1

command -v python3 >/dev/null 2>&1 || fail "python3 is required to edit the Claude config."

# --- Locate the Claude desktop config for this OS ---
case "$(uname -s)" in
    Darwin) CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
    Linux)  CFG="$HOME/.config/Claude/claude_desktop_config.json" ;;
    *)      fail "Unsupported OS '$(uname -s)' — unknown Claude desktop config location." ;;
esac

INSTR_SRC="$SRC/docs/agent-memory-instructions.md"
INSTR_DST="$HOME/.ragger/agent-memory-instructions.md"

# ============================================================
# Uninstall
# ============================================================
if [ "$UNINSTALL" -eq 1 ]; then
    [ -f "$CFG" ] || { warn "No Claude config at $CFG — nothing to remove."; exit 0; }
    python3 - "$CFG" <<'PYEOF'
import json, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text())
except Exception as e:
    print(f"[!] Could not parse {p} ({e})", file=sys.stderr); sys.exit(0)
servers = data.get("mcpServers", {})
if "ragger" in servers:
    bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
    if not bak.exists(): shutil.copy2(p, bak)
    del servers["ragger"]
    if not servers: data.pop("mcpServers", None)  # don't leave an empty section
    p.write_text(json.dumps(data, indent=2) + "\n")
    print("[+]   removed mcpServers.ragger")
else:
    print("[+]   no ragger entry present — nothing to do")
PYEOF
    info "Done. The instructions file at $INSTR_DST was left in place"
    info "(it is also used by other MCP clients). Restart the Claude app."
    exit 0
fi

# ============================================================
# Install
# ============================================================

# Resolve an absolute path to the ragger binary — the Claude app launches MCP
# servers with a minimal PATH, so a bare "ragger" or "~/..." won't resolve.
RAGGER_BIN="$HOME/.local/bin/ragger"
if [ ! -x "$RAGGER_BIN" ]; then
    RAGGER_BIN="$(command -v ragger || true)"
fi
[ -n "$RAGGER_BIN" ] && [ -x "$RAGGER_BIN" ] || \
    fail "ragger binary not found — run ./scripts/install.sh first."
RAGGER_BIN="$(cd "$(dirname "$RAGGER_BIN")" && pwd)/$(basename "$RAGGER_BIN")"

echo ""
info "Registering Ragger MCP memory for the Claude desktop app"
info "  binary: $RAGGER_BIN"

# 1. Agent usage instructions (served by `ragger mcp` via initialize)
mkdir -p "$HOME/.ragger"
if [ ! -f "$INSTR_SRC" ]; then
    warn "Instructions source missing at $INSTR_SRC — skipping (built-in fallback will be used)."
elif [ -f "$INSTR_DST" ]; then
    info "  kept existing $INSTR_DST (edit it to customize agent guidance)"
else
    cp "$INSTR_SRC" "$INSTR_DST"
    info "  installed $INSTR_DST"
fi

# 2. Merge the mcpServers.ragger entry (create the config if needed)
if [ ! -f "$CFG" ]; then
    mkdir -p "$(dirname "$CFG")"
    printf '{}\n' > "$CFG"
    info "  created $CFG"
fi

RAGGER_BIN="$RAGGER_BIN" python3 - "$CFG" <<'PYEOF'
import json, os, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text() or "{}")
except Exception as e:
    print(f"[!] Could not parse {p} ({e}) — leaving it untouched", file=sys.stderr); sys.exit(1)

servers = data.setdefault("mcpServers", {})
desired = {"command": os.environ["RAGGER_BIN"], "args": ["mcp"]}

if servers.get("ragger") == desired:
    print("[+]   mcpServers.ragger already up to date — no changes")
    sys.exit(0)

bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[+]   backed up → {bak.name}")

if "ragger" in servers:
    servers["ragger"].update(desired)   # refresh command/args, keep extras
    print("[+]   updated mcpServers.ragger")
else:
    servers["ragger"] = desired
    print("[+]   added mcpServers.ragger")

p.write_text(json.dumps(data, indent=2) + "\n")
PYEOF

echo ""
info "Done. Restart the Claude desktop app to load the Ragger memory tools."
