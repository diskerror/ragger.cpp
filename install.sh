#!/bin/bash
# install.sh — Build and install Ragger C++ to this machine
#
# Usage: sudo ./install.sh
#
# Builds if needed, copies binary to /usr/local/bin/ragger,
# creates log directory, and restarts the daemon if it's running.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[!] This script must be run as root: sudo ./install.sh" >&2
    exit 1
fi

cd "$(dirname "$0")"
BINARY="build/ragger"
DEST="/usr/local/bin/ragger"

# Build first
./build.sh

if [ ! -x "$BINARY" ]; then
    echo "[!] Build failed — no binary at $BINARY" >&2
    exit 1
fi

echo ""
echo "[+] Installing to $DEST"
cp "$BINARY" "$DEST"
chmod 0755 "$DEST"

# Detect OS and user/group
OS="$(uname -s)"
case "$OS" in
    Darwin) RAGGER_USER="_ragger"; RAGGER_GROUP="ragger" ;;
    Linux)  RAGGER_USER="ragger";  RAGGER_GROUP="ragger" ;;
    *)      RAGGER_USER="ragger";  RAGGER_GROUP="ragger" ;;
esac

# Create log directory if needed
LOG_DIR="/var/log/ragger"
if [ ! -d "$LOG_DIR" ]; then
    echo "[+] Creating $LOG_DIR"
    mkdir -p "$LOG_DIR"
    chown "$RAGGER_USER:$RAGGER_GROUP" "$LOG_DIR"
    chmod 0750 "$LOG_DIR"
fi

# Restart daemon if running
if [ "$OS" = "Darwin" ]; then
    PLIST="com.diskerror.ragger"
    if launchctl list "$PLIST" &>/dev/null; then
        echo "[+] Restarting daemon..."
        launchctl kickstart -k "system/$PLIST"
    else
        echo "[*] Daemon not loaded — skipping restart"
    fi
elif [ "$OS" = "Linux" ]; then
    if systemctl is-active --quiet ragger; then
        echo "[+] Restarting daemon..."
        systemctl restart ragger
    else
        echo "[*] Service not active — skipping restart"
    fi
fi

echo ""
echo "✓ Installed: $DEST"
"$DEST" --version 2>/dev/null || true
