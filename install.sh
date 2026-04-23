#!/bin/bash
# install.sh — Install Ragger for the current user.
#
# Usage: ./install.sh
#
# No sudo needed. Layout follows the XDG convention:
#
#     ~/.local/bin/ragger         executable (on PATH)
#
#     ~/.ragger/settings.ini      config
#     ~/.ragger/logs/             logs
#     ~/.ragger/models/           embedding models
#     ~/.ragger/formats/          inference format definitions
#     ~/.ragger/www/              web UI assets
#     ~/.ragger/memories.db       SQLite database
#     ~/.ragger/ragger.sock       unix socket (runtime)
#
# Idempotent — re-run to update the binary, daemon file, or PATH entry.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
    fail "Do NOT run install.sh as root. It installs into your own ~/.ragger/."
fi

cd "$(dirname "$0")"
SRC="$(pwd)"
OS="$(uname -s)"

BINARY="build/ragger"
[ -x "$BINARY" ] || fail "No binary at $BINARY — build first: cmake --build build --parallel"

# --- Paths (single source of truth) ---
RAGGER_HOME="$HOME/.ragger"
BIN_DIR="$HOME/.local/bin"          # XDG user executables
LOG_DIR="$RAGGER_HOME/logs"
MODEL_DIR="$RAGGER_HOME/models"
FORMATS_DIR="$RAGGER_HOME/formats"
WEB_DIR="$RAGGER_HOME/www"
CONF_FILE="$RAGGER_HOME/settings.ini"
DEST="$BIN_DIR/ragger"

# ============================================================
# PHASE 1: Directory layout
# ============================================================

for d in "$RAGGER_HOME" "$BIN_DIR" "$LOG_DIR" "$MODEL_DIR" "$FORMATS_DIR" "$WEB_DIR"; do
    if [ ! -d "$d" ]; then
        info "Creating $d"
        mkdir -p "$d"
    fi
done

# Migration: older installs put the binary under ~/.ragger/bin. If that
# directory exists, clean it up so we have one authoritative location.
OLD_BIN_DIR="$RAGGER_HOME/bin"
if [ -d "$OLD_BIN_DIR" ]; then
    info "Removing legacy $OLD_BIN_DIR (binary now lives at $DEST)"
    rm -rf "$OLD_BIN_DIR"
fi

# ============================================================
# PHASE 2: Config
# ============================================================

if [ ! -f "$CONF_FILE" ]; then
    if [ -f "$SRC/example-settings.ini" ]; then
        info "Installing $CONF_FILE from example-settings.ini"
        cp "$SRC/example-settings.ini" "$CONF_FILE"
    else
        fail "Missing $SRC/example-settings.ini — cannot bootstrap config."
    fi
    chmod 0644 "$CONF_FILE"
else
    info "Keeping existing $CONF_FILE"
fi

# Default SOUL.md (user persona) if missing
if [ ! -f "$RAGGER_HOME/SOUL.md" ] && [ -f "$SRC/SOUL.md" ]; then
    info "Installing default SOUL.md"
    cp "$SRC/SOUL.md" "$RAGGER_HOME/SOUL.md"
fi

# Web UI
if [ -d "$SRC/web" ]; then
    info "Installing web UI to $WEB_DIR"
    cp -R "$SRC/web/." "$WEB_DIR/"
fi

# Inference format definitions (shipped JSON files)
if [ -d "$SRC/formats" ]; then
    info "Installing inference formats to $FORMATS_DIR"
    cp -R "$SRC/formats/." "$FORMATS_DIR/"
fi

# ============================================================
# PHASE 3: Install executable
# ============================================================

info "Installing $DEST"
cp "$BINARY" "$DEST"
chmod 0755 "$DEST"
if [ "$OS" = "Darwin" ]; then
    codesign --force --sign - "$DEST" 2>/dev/null || true
fi

"$DEST" version 2>/dev/null || true

# --- First-run: mint a bearer token for the daemon owner ---
# ~/.ragger/token holds the raw bearer token the local owner uses against
# their own daemon (needed for remote clients + any tool that speaks HTTP
# over a non-loopback address). Local browser and unix-socket access are
# already bypassed by the server, so this token is optional for most uses
# — but it's cheap to create and avoids a "wait, I need a token" moment
# the first time someone wires up an OpenClaw / Claude Desktop / curl client.
if [ ! -f "$RAGGER_HOME/token" ]; then
    info "Bootstrapping bearer token at $RAGGER_HOME/token"
    "$DEST" add-self >/dev/null 2>&1 || \
        warn "add-self failed — run 'ragger add-self' manually after install"
fi

# ============================================================
# PHASE 4: Ensure ~/.local/bin is on PATH
# ============================================================
#
# Most modern shells already put ~/.local/bin on PATH (macOS path_helper,
# Debian/Ubuntu /etc/profile, Fedora default .bash_profile, Nix, etc.),
# so we only touch the user's rc if the directory is NOT currently on PATH.
# That keeps the install a no-op on systems that already follow the convention.

PATH_LINE='[ -d "$HOME/.local/bin" ] && export PATH="$HOME/.local/bin:$PATH"'
PATH_MARKER='# Added by Ragger installer — ensure ~/.local/bin is on PATH'

add_to_rc() {
    local rc="$1"
    if [ -f "$rc" ] && grep -Fq "$PATH_MARKER" "$rc"; then
        return  # already present
    fi
    {
        echo ""
        echo "$PATH_MARKER"
        echo "$PATH_LINE"
    } >> "$rc"
    info "Added ~/.local/bin to PATH in $rc"
}

case ":$PATH:" in
    *":$BIN_DIR:"*)
        info "~/.local/bin already on PATH — no shell rc changes needed"
        ;;
    *)
        # Pick the appropriate rc file for the user's current shell.
        case "$(basename "${SHELL:-}")" in
            zsh)   add_to_rc "$HOME/.zshrc" ;;
            bash)  [ -f "$HOME/.bash_profile" ] && add_to_rc "$HOME/.bash_profile" \
                                                || add_to_rc "$HOME/.bashrc" ;;
            *)     add_to_rc "$HOME/.profile" ;;
        esac
        warn "Open a new terminal (or 'source' your rc file) to pick up the new PATH."
        ;;
esac

# ============================================================
# PHASE 5: User daemon (LaunchAgent / systemd --user)
# ============================================================

if [ "$OS" = "Darwin" ]; then
    AGENT_DIR="$HOME/Library/LaunchAgents"
    PLIST="$AGENT_DIR/com.diskerror.ragger.plist"
    mkdir -p "$AGENT_DIR"
    info "Writing $PLIST"
    cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.diskerror.ragger</string>
    <key>ProgramArguments</key>
    <array>
        <string>$DEST</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/stdout.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
EOF
    chmod 0644 "$PLIST"

    echo ""
    echo "To start the daemon now:  ragger start"

elif [ "$OS" = "Linux" ]; then
    UNIT_DIR="$HOME/.config/systemd/user"
    UNIT="$UNIT_DIR/ragger.service"
    mkdir -p "$UNIT_DIR"
    info "Writing $UNIT"
    cat > "$UNIT" << EOF
[Unit]
Description=Ragger Memory Server (user)
After=default.target

[Service]
Type=simple
ExecStart=$DEST serve
Restart=on-failure
StandardOutput=append:$LOG_DIR/stdout.log
StandardError=append:$LOG_DIR/stderr.log

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload 2>/dev/null || true

    # Ensure the unit will start automatically next login
    systemctl --user enable ragger.service 2>/dev/null || true

    echo ""
    echo "To start the daemon now:  ragger start"
    echo ""
    echo "To keep it running when you're not logged in:"
    echo "    sudo loginctl enable-linger $USER"
else
    warn "Unknown OS '$OS' — daemon file not installed. Run 'ragger serve' manually."
fi

# ============================================================
# PHASE 6: OpenClaw plugin (if OpenClaw is installed)
# ============================================================
#
# If the user has OpenClaw (~/.openclaw/ exists), install the Ragger memory
# plugin into ~/.openclaw/extensions/ragger/ and ensure its entries are
# wired in ~/.openclaw/openclaw.json.
#
# Plugin files are ALWAYS overwritten — they're versioned with the Ragger
# repo and must match the binary. openclaw.json is merged in place: we add
# missing ragger hooks without touching unrelated config.

OPENCLAW_HOME="$HOME/.openclaw"
OPENCLAW_PLUGIN_SRC="$SRC/openclaw-plugin"
OPENCLAW_PLUGIN_DST="$OPENCLAW_HOME/extensions/ragger"
OPENCLAW_CFG="$OPENCLAW_HOME/openclaw.json"

if [ -d "$OPENCLAW_HOME" ] && [ -d "$OPENCLAW_PLUGIN_SRC" ]; then
    echo ""
    info "OpenClaw detected at $OPENCLAW_HOME — installing memory plugin"

    mkdir -p "$OPENCLAW_PLUGIN_DST"
    for f in openclaw.plugin.json index.ts; do
        if [ -f "$OPENCLAW_PLUGIN_SRC/$f" ]; then
            cp "$OPENCLAW_PLUGIN_SRC/$f" "$OPENCLAW_PLUGIN_DST/$f"
            info "  wrote $OPENCLAW_PLUGIN_DST/$f"
        fi
    done

    # Ensure openclaw.json wires the ragger memory plugin. We merge conservatively:
    # only set keys that are missing, never clobber user-set values.
    if [ -f "$OPENCLAW_CFG" ]; then
        if command -v python3 >/dev/null 2>&1; then
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
    entries["ragger"] = {
        "enabled": True,
        "config": {
            "transport": "mcp",
            "autoRecall": True,
            "autoCapture": True,
        },
    }
    changed = True
    print("[+]   plugins.entries.ragger added (transport=mcp)")
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
        else
            warn "  python3 not found — cannot merge openclaw.json hooks automatically."
            warn "  Ensure plugins.slots.memory=\"ragger\" and plugins.entries.ragger"
            warn "  are set in $OPENCLAW_CFG (see docs/openclaw.md)."
        fi
    else
        warn "  $OPENCLAW_CFG not found — run OpenClaw once to create it, then re-run install.sh"
    fi
fi

echo ""
info "Installed to $RAGGER_HOME"
