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

# --- Create directories and fix ownership ---
for dir in "$DATA_DIR" "$LOG_DIR"; do
    if [ ! -d "$dir" ]; then
        info "Creating $dir"
        mkdir -p "$dir"
    fi
    chown -R "$RAGGER_USER:$RAGGER_GROUP" "$dir"
    chmod 0750 "$dir"
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
        <string>/bin/bash</string>
        <string>/usr/local/bin/ragger-start.sh</string>
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

info "Installing /usr/local/bin/ragger-cpp"
cp "$BINARY" /usr/local/bin/ragger-cpp
chmod 0755 /usr/local/bin/ragger-cpp

info "Installing /usr/local/bin/ragger"
cp "$BINARY" /usr/local/bin/ragger
chmod 0755 /usr/local/bin/ragger

# --- Restart daemon ---
if [ "$OS" = "Darwin" ]; then
    info "Loading daemon"
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
elif [ "$OS" = "Linux" ]; then
    if systemctl is-active --quiet ragger; then
        info "Restarting daemon"
        systemctl restart ragger
    fi
fi

echo ""
info "Installed: /usr/local/bin/ragger"
/usr/local/bin/ragger version 2>/dev/null || true
