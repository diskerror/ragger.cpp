#!/bin/bash
# install.sh — Install Ragger to the current user's ~/.ragger directory.
#
# Usage: ./install.sh
#
# No sudo needed. Everything lives under $HOME/.ragger:
#
#     ~/.ragger/bin/ragger        executable
#     ~/.ragger/settings.ini        config
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
BIN_DIR="$RAGGER_HOME/bin"
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

# ============================================================
# PHASE 4: Add ~/.ragger/bin to PATH
# ============================================================

PATH_LINE='export PATH="$HOME/.ragger/bin:$PATH"'
PATH_MARKER='# Added by Ragger installer'

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
    info "Added ~/.ragger/bin to PATH in $rc"
}

# Figure out which shell rc files to touch.  Prefer the user's current shell.
case "$(basename "${SHELL:-}")" in
    zsh)   add_to_rc "$HOME/.zshrc" ;;
    bash)  [ -f "$HOME/.bash_profile" ] && add_to_rc "$HOME/.bash_profile" \
                                        || add_to_rc "$HOME/.bashrc" ;;
    *)     add_to_rc "$HOME/.profile" ;;
esac

# Check whether the current shell already sees it
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "Open a new terminal (or 'source' your rc file) to pick up the new PATH." ;;
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

echo ""
info "Installed to $RAGGER_HOME"
