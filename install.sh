#!/bin/bash
# install.sh — Build and install Ragger C++ to this machine
#
# Usage: ./install.sh
#
# Builds if needed, copies binary to /usr/local/bin/ragger,
# and restarts the daemon if it's running.
# Requires sudo for the install step.

set -euo pipefail

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
echo "[+] Installing to $DEST (requires sudo)"
sudo cp "$BINARY" "$DEST"
sudo chmod 0755 "$DEST"

# Restart daemon if running (macOS)
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    PLIST="com.diskerror.ragger"
    if sudo launchctl list "$PLIST" &>/dev/null; then
        echo "[+] Restarting daemon..."
        sudo launchctl kickstart -k "system/$PLIST"
    else
        echo "[*] Daemon not loaded — skipping restart"
    fi
elif [ "$OS" = "Linux" ]; then
    if systemctl is-active --quiet ragger; then
        echo "[+] Restarting daemon..."
        sudo systemctl restart ragger
    else
        echo "[*] Service not active — skipping restart"
    fi
fi

echo ""
echo "✓ Installed: $DEST"
"$DEST" --version 2>/dev/null || true
