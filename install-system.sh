#!/bin/bash
# install-system.sh — Set up system-level resources for Ragger Memory
# Run with: sudo bash install-system.sh
#
# What this does:
#   1. Creates _ragger system user and ragger group (macOS / Linux)
#   2. Creates /var/ragger/ with correct ownership and permissions
#   3. Installs LaunchDaemon plist (macOS) or systemd unit (Linux)
#   4. Copies the ragger binary to /usr/local/bin/ (optional, for system-wide)
#
# What this does NOT do:
#   - Touch ~/.ragger/ (created automatically on first user run)
#   - Load models (user downloads them separately)
#   - Start the service (do that after config is set)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[!]${NC} $*" >&2; }

if [ "$(id -u)" -ne 0 ]; then
    error "This script must be run as root (sudo)."
    exit 1
fi

RAGGER_DIR="/var/ragger"
RAGGER_GROUP="ragger"

# --- Detect platform ---
OS="$(uname -s)"
case "$OS" in
    Darwin) RAGGER_USER="_ragger" ;;
    Linux)  RAGGER_USER="ragger" ;;
    *)      error "Unsupported OS: $OS"; exit 1 ;;
esac

# --- Create group ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Groups/$RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        # Find next available GID >= 400 (system range on macOS)
        LAST_GID=$(dscl . -list /Groups PrimaryGroupID | awk '{print $2}' | sort -n | tail -1)
        NEXT_GID=$((LAST_GID + 1))
        if [ "$NEXT_GID" -lt 400 ]; then NEXT_GID=400; fi
        dscl . -create /Groups/$RAGGER_GROUP
        dscl . -create /Groups/$RAGGER_GROUP PrimaryGroupID "$NEXT_GID"
        dscl . -create /Groups/$RAGGER_GROUP RealName "Ragger Memory Service"
    else
        info "Group $RAGGER_GROUP already exists"
    fi
else
    if ! getent group $RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        groupadd --system $RAGGER_GROUP
    else
        info "Group $RAGGER_GROUP already exists"
    fi
fi

# --- Create system user ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Users/$RAGGER_USER &>/dev/null; then
        info "Creating system user: $RAGGER_USER"
        # Find next available UID >= 400
        LAST_UID=$(dscl . -list /Users UniqueID | awk '{print $2}' | sort -n | tail -1)
        NEXT_UID=$((LAST_UID + 1))
        if [ "$NEXT_UID" -lt 400 ]; then NEXT_UID=400; fi
        GID=$(dscl . -read /Groups/$RAGGER_GROUP PrimaryGroupID | awk '{print $2}')
        dscl . -create /Users/$RAGGER_USER
        dscl . -create /Users/$RAGGER_USER UniqueID "$NEXT_UID"
        dscl . -create /Users/$RAGGER_USER PrimaryGroupID "$GID"
        dscl . -create /Users/$RAGGER_USER UserShell /usr/bin/false
        dscl . -create /Users/$RAGGER_USER RealName "Ragger Memory Service"
        dscl . -create /Users/$RAGGER_USER NFSHomeDirectory /var/empty
        # Hide from login window
        dscl . -create /Users/$RAGGER_USER IsHidden 1
    else
        info "User $RAGGER_USER already exists"
    fi
else
    if ! id $RAGGER_USER &>/dev/null; then
        info "Creating system user: $RAGGER_USER"
        useradd --system --no-create-home --shell /usr/bin/false \
                --gid $RAGGER_GROUP --home-dir /var/empty $RAGGER_USER
    else
        info "User $RAGGER_USER already exists"
    fi
fi

# --- Add invoking user to ragger group ---
REAL_USER="${SUDO_USER:-}"
if [ -n "$REAL_USER" ] && [ "$REAL_USER" != "root" ]; then
    if [ "$OS" = "Darwin" ]; then
        if ! dscl . -read /Groups/$RAGGER_GROUP GroupMembership 2>/dev/null | grep -qw "$REAL_USER"; then
            info "Adding $REAL_USER to $RAGGER_GROUP group"
            dscl . -append /Groups/$RAGGER_GROUP GroupMembership "$REAL_USER"
        else
            info "$REAL_USER already in $RAGGER_GROUP group"
        fi
    else
        if ! id -nG "$REAL_USER" | grep -qw "$RAGGER_GROUP"; then
            info "Adding $REAL_USER to $RAGGER_GROUP group"
            usermod -aG $RAGGER_GROUP "$REAL_USER"
        else
            info "$REAL_USER already in $RAGGER_GROUP group"
        fi
    fi
fi

# --- Create /var/ragger/ ---
info "Creating $RAGGER_DIR/"
mkdir -p "$RAGGER_DIR"
mkdir -p "$RAGGER_DIR/models"
mkdir -p "$RAGGER_DIR/logs"

chown -R $RAGGER_USER:$RAGGER_GROUP "$RAGGER_DIR"
chmod 0750 "$RAGGER_DIR"
chmod 0750 "$RAGGER_DIR/models"
chmod 0750 "$RAGGER_DIR/logs"

# --- Install system config ---
ETC_CONF="/etc/ragger.ini"
if [ ! -f "$ETC_CONF" ]; then
    EXAMPLE_CONF="$SCRIPT_DIR/example-system.ini"
    if [ -f "$EXAMPLE_CONF" ]; then
        info "Installing system config to $ETC_CONF"
        cp "$EXAMPLE_CONF" "$ETC_CONF"
    else
        info "Writing default system config to $ETC_CONF"
        cat > "$ETC_CONF" << 'CONF'
# /etc/ragger.ini — Ragger Memory system configuration
# Per-user overrides go in ~/.ragger/ragger.ini

[server]
host = 127.0.0.1
port = 8432
single_user = true

[storage]
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 0.3
vector_weight = 0.7

[logging]
log_dir = /var/log/ragger
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300
CONF
    fi
    chown root:$RAGGER_GROUP "$ETC_CONF"
    chmod 0644 "$ETC_CONF"
else
    info "System config already exists at $ETC_CONF"
fi

# --- Install LaunchDaemon (macOS) ---
if [ "$OS" = "Darwin" ]; then
    PLIST="/Library/LaunchDaemons/com.diskerror.ragger.plist"
    START_SCRIPT="/usr/local/bin/ragger-start.sh"

    if [ ! -f "$START_SCRIPT" ]; then
        warn "Start script not found at $START_SCRIPT"
        warn "Create it before loading the LaunchDaemon."
    fi

    info "Installing LaunchDaemon plist"
    cat > "$PLIST" << PLIST
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
        <string>$START_SCRIPT</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/ragger/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/ragger/stderr.log</string>
</dict>
</plist>
PLIST
    chown root:wheel "$PLIST"
    chmod 0644 "$PLIST"
    info "LaunchDaemon installed at $PLIST"
    warn "To load: sudo launchctl load $PLIST"
fi

# --- Install systemd unit (Linux) ---
if [ "$OS" = "Linux" ]; then
    UNIT="/etc/systemd/system/ragger.service"
    info "Installing systemd unit"
    cat > "$UNIT" << UNIT
[Unit]
Description=Ragger Memory Server
After=network.target

[Service]
Type=simple
User=$RAGGER_USER
Group=$RAGGER_GROUP
ExecStart=/usr/local/bin/ragger serve --config-file=/etc/ragger.ini
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
    chmod 0644 "$UNIT"
    systemctl daemon-reload
    info "Systemd unit installed. Enable with: systemctl enable --now ragger"
fi

# --- Install binary or wrapper ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALLED=""

# Check for compiled C++ binary
for candidate in \
    "$SCRIPT_DIR/build/ragger" \
    "$SCRIPT_DIR/cmake-build-release/ragger" \
    "$SCRIPT_DIR/cmake-build-debug/ragger"; do
    if [ -x "$candidate" ]; then
        info "Installing C++ binary from $candidate"
        cp "$candidate" /usr/local/bin/ragger
        chown root:$RAGGER_GROUP /usr/local/bin/ragger
        chmod 0755 /usr/local/bin/ragger
        INSTALLED="cpp"
        break
    fi
done

# If no C++ binary, check for Python package
if [ -z "$INSTALLED" ]; then
    if python3 -c "import ragger_memory" 2>/dev/null; then
        info "Installing Python wrapper to /usr/local/bin/ragger"
        cat > /usr/local/bin/ragger << 'WRAPPER'
#!/bin/bash
exec python3 -m ragger_memory "$@"
WRAPPER
        chown root:$RAGGER_GROUP /usr/local/bin/ragger
        chmod 0755 /usr/local/bin/ragger
        INSTALLED="python"
    fi
fi

if [ -z "$INSTALLED" ]; then
    warn "No compiled binary or Python package found."
    warn "Build the C++ version or pip install the Python version first."
fi

echo ""
info "System setup complete."
info ""
info "Next steps:"
if [ -z "$INSTALLED" ]; then
    info "  1. Build C++ or pip install Python, then re-run this script"
elif [ "$INSTALLED" = "cpp" ]; then
    info "  1. C++ binary installed ✓"
else
    info "  1. Python wrapper installed ✓"
fi
info "  2. Copy/symlink ONNX model to $RAGGER_DIR/models/"
info "  3. Edit /etc/ragger.ini as needed"
if [ "$OS" = "Darwin" ]; then
    info "  4. sudo launchctl load /Library/LaunchDaemons/com.diskerror.ragger.plist"
else
    info "  4. sudo systemctl enable --now ragger"
fi
if [ -n "$REAL_USER" ]; then
    warn ""
    warn "NOTE: $REAL_USER was added to the $RAGGER_GROUP group."
    warn "You may need to log out and back in for group membership to take effect."
fi
