#!/bin/bash
# install.sh — Install Ragger C++ to this machine
#
# Usage: sudo ./install.sh
#
# Idempotent install: creates system user/group/directories if needed,
# builds the binary, copies to /usr/local/bin, restarts daemon.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[!] This script must be run as root: sudo ./install.sh" >&2
    exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }

cd "$(dirname "$0")"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin) RAGGER_USER="_ragger"; RAGGER_GROUP="ragger" ;;
    Linux)  RAGGER_USER="ragger";  RAGGER_GROUP="ragger" ;;
    *)      echo "[!] Unsupported OS: $OS" >&2; exit 1 ;;
esac

# --- Paths (single source of truth) ---
LOG_DIR="/var/log/ragger"
DATA_DIR="/var/ragger"
CONF_FILE="/etc/ragger.ini"

# --- Create group if missing ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Groups/$RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        LAST_GID=$(dscl . -list /Groups PrimaryGroupID | awk '{print $2}' | sort -n | tail -1)
        NEXT_GID=$((LAST_GID + 1))
        if [ "$NEXT_GID" -lt 400 ]; then NEXT_GID=400; fi
        dscl . -create /Groups/$RAGGER_GROUP
        dscl . -create /Groups/$RAGGER_GROUP PrimaryGroupID "$NEXT_GID"
    fi
else
    if ! getent group $RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        groupadd --system $RAGGER_GROUP
    fi
fi

# --- Create user if missing ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Users/$RAGGER_USER &>/dev/null; then
        info "Creating user: $RAGGER_USER"
        LAST_UID=$(dscl . -list /Users UniqueID | awk '{print $2}' | sort -n | tail -1)
        NEXT_UID=$((LAST_UID + 1))
        if [ "$NEXT_UID" -lt 400 ]; then NEXT_UID=400; fi
        GID=$(dscl . -read /Groups/$RAGGER_GROUP PrimaryGroupID | awk '{print $2}')
        dscl . -create /Users/$RAGGER_USER
        dscl . -create /Users/$RAGGER_USER UniqueID "$NEXT_UID"
        dscl . -create /Users/$RAGGER_USER PrimaryGroupID "$GID"
        dscl . -create /Users/$RAGGER_USER UserShell /usr/bin/false
        dscl . -create /Users/$RAGGER_USER NFSHomeDirectory /var/empty
        dscl . -create /Users/$RAGGER_USER IsHidden 1
    fi
else
    if ! id $RAGGER_USER &>/dev/null; then
        info "Creating user: $RAGGER_USER"
        useradd --system --no-create-home --shell /usr/bin/false \
                --gid $RAGGER_GROUP --home-dir /var/empty $RAGGER_USER
    fi
fi

# --- Create directories and fix ownership/permissions ---
for dir in "$DATA_DIR" "$LOG_DIR"; do
    if [ ! -d "$dir" ]; then
        info "Creating $dir"
        mkdir -p "$dir"
    fi
    chown -R "$RAGGER_USER:$RAGGER_GROUP" "$dir"
    chmod 0750 "$dir"
    # Fix subdirectory permissions
    find "$dir" -type d -exec chmod 0750 {} +
done

# Fix file permissions: DB files group-writable (for MCP users), logs 0660
info "Fixing file permissions"
find "$DATA_DIR" -name "*.db" -o -name "*.db-shm" -o -name "*.db-wal" 2>/dev/null | while read f; do
    chmod 0660 "$f"
done
find "$LOG_DIR" -type f 2>/dev/null | while read f; do
    chmod 0660 "$f"
done

# --- Install system config if missing ---
if [ ! -f "$CONF_FILE" ]; then
    info "Creating $CONF_FILE"
    if [ -f example-system.ini ]; then
        cp example-system.ini "$CONF_FILE"
    else
        cat > "$CONF_FILE" << EOF
[server]
host = 127.0.0.1
port = 8432
single_user = true

[logging]
log_dir = $LOG_DIR

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_weight = 0.3
vector_weight = 0.7

[auth]
token_rotation_minutes = 1440
EOF
    fi
    chmod 0644 "$CONF_FILE"
fi

# --- Install default SOUL.md if missing ---
SOUL_DEST="$DATA_DIR/SOUL.md"
if [ ! -f "$SOUL_DEST" ] && [ -f SOUL.md ]; then
    info "Installing default SOUL.md to $DATA_DIR"
    cp SOUL.md "$SOUL_DEST"
    chown "$RAGGER_USER:$RAGGER_GROUP" "$SOUL_DEST"
    chmod 0644 "$SOUL_DEST"
fi

# --- Install/update LaunchDaemon (macOS) ---
if [ "$OS" = "Darwin" ]; then
    PLIST="/Library/LaunchDaemons/com.diskerror.ragger.plist"
    info "Writing LaunchDaemon plist"
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
        <string>/usr/local/bin/ragger</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
EOF
    chmod 0644 "$PLIST"
fi

# --- Install binary ---
BINARY="build/ragger"
if [ ! -x "$BINARY" ]; then
    echo "[!] Build failed — no binary at $BINARY" >&2
    exit 1
fi

info "Installing /usr/local/bin/ragger"
cp "$BINARY" /usr/local/bin/ragger
chmod 0755 /usr/local/bin/ragger
codesign -f -s - /usr/local/bin/ragger 2>/dev/null || true

echo ""
info "Installed: /usr/local/bin/ragger"
/usr/local/bin/ragger version 2>/dev/null || true

# --- Install OpenClaw plugin (if OpenClaw is present) ---
# Detect the actual user who ran sudo (not root)
if [ -n "${SUDO_USER:-}" ]; then
    ACTUAL_USER="$SUDO_USER"
    ACTUAL_HOME=$(eval echo ~"$SUDO_USER")
else
    ACTUAL_USER="$USER"
    ACTUAL_HOME="$HOME"
fi

OPENCLAW_DIR="$ACTUAL_HOME/.openclaw"
PLUGIN_SRC="openclaw-plugin"

if [ -d "$OPENCLAW_DIR" ] && [ -d "$PLUGIN_SRC" ]; then
    PLUGIN_DEST="$OPENCLAW_DIR/extensions/memory-ragger"
    info "Installing OpenClaw plugin to $PLUGIN_DEST"
    mkdir -p "$PLUGIN_DEST"
    cp "$PLUGIN_SRC/openclaw.plugin.json" "$PLUGIN_DEST/"
    cp "$PLUGIN_SRC/index.ts" "$PLUGIN_DEST/"
    chown -R "$ACTUAL_USER" "$PLUGIN_DEST"
    echo "    Plugin installed. Restart OpenClaw to load it."
elif [ -d "$PLUGIN_SRC" ]; then
    echo ""
    echo -e "${GREEN}[+]${NC} OpenClaw plugin available at: $(pwd)/$PLUGIN_SRC"
    echo "    To install later, copy to: ~/.openclaw/extensions/memory-ragger/"
fi

# --- Print restart commands ---
echo ""
echo -e "${GREEN}[+]${NC} To (re)start the daemon, run:"
if [ "$OS" = "Darwin" ]; then
    echo "    sudo launchctl bootout system/com.diskerror.ragger 2>/dev/null || true"
    echo "    sudo launchctl bootstrap system $PLIST"
elif [ "$OS" = "Linux" ]; then
    echo "    sudo systemctl restart ragger"
fi
