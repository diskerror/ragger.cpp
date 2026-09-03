#!/bin/bash
# install-bin.sh — Install the Ragger binary + daemon for the current user.
#
# Usage: ./install-bin.sh [--stats|--no-stats]
#   --stats     install the instrumented binary from build-stats/
#   --no-stats  install the default binary from build/ (default)
#
# No sudo needed. Layout follows the XDG convention:
#
#     ~/.local/bin/ragger         executable (on PATH)
#
#     ~/.ragger/memories.db       memories + config (settings table)
#     ~/.ragger/logs/activity.log logs (auto-rotated: see log_max_size_mb /
#                                 log_max_age_days config settings)
#     ~/.ragger/models/           embedding models
#     ~/.ragger/formats/          inference format definitions
#     ~/.ragger/recipes/          build_context recipes (layered assembly)
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
    fail "Do NOT run install-bin.sh as root. It installs into your own ~/.ragger/."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"
OS="$(uname -s)"

# --stats selects the instrumented build tree (build-stats/) produced by
# `./scripts/build.sh --stats`. Default is the stats-off tree (build/).
BUILD_DIR="build"
for arg in "$@"; do
    case "$arg" in
        --stats)     BUILD_DIR="build-stats" ;;
        --no-stats)  BUILD_DIR="build" ;;
    esac
done

BINARY="$BUILD_DIR/ragger"
if [ ! -x "$BINARY" ]; then
    if [ "$BUILD_DIR" = "build-stats" ]; then
        fail "No binary at $BINARY — build first: ./scripts/build.sh --stats"
    else
        fail "No binary at $BINARY — build first: ./scripts/build.sh"
    fi
fi

# --- Paths (single source of truth) ---
RAGGER_BASE="$HOME/.ragger"
BIN_DIR="$HOME/.local/bin"          # XDG user executables
LOG_DIR="$RAGGER_BASE/logs"
MODEL_DIR="$RAGGER_BASE/models"
FORMATS_DIR="$RAGGER_BASE/formats"
RECIPES_DIR="$RAGGER_BASE/recipes"
WEB_DIR="$RAGGER_BASE/www"
DEST="$BIN_DIR/ragger"

# ============================================================
# PHASE 1: Directory layout
# ============================================================

for d in "$RAGGER_BASE" "$BIN_DIR" "$LOG_DIR" "$MODEL_DIR" "$FORMATS_DIR" "$RECIPES_DIR" "$WEB_DIR"; do
    if [ ! -d "$d" ]; then
        info "Creating $d"
        mkdir -p "$d"
    fi
done

# Migration: older installs put the binary under ~/.ragger/bin. If that
# directory exists, clean it up so we have one authoritative location.
OLD_BIN_DIR="$RAGGER_BASE/bin"
if [ -d "$OLD_BIN_DIR" ]; then
    info "Removing legacy $OLD_BIN_DIR (binary now lives at $DEST)"
    rm -rf "$OLD_BIN_DIR"
fi

# ============================================================
# PHASE 2: Config
# ============================================================
# No config file is written. Defaults are compiled into the binary from
# default-settings.txt (see cmake/embed_ini.cmake) and the settings table in
# memories.db is the only store, so there is nothing here to install, template,
# or leave behind stale. Configuration is done from the dashboard.

# Agent memory-usage instructions (served by `ragger mcp` via the MCP
# initialize `instructions` field). Install if missing — preserves user edits.
if [ ! -f "$RAGGER_BASE/agent-memory-instructions.md" ] && \
   [ -f "$SRC/docs/agent-memory-instructions.md" ]; then
    info "Installing agent memory instructions"
    cp "$SRC/docs/agent-memory-instructions.md" "$RAGGER_BASE/agent-memory-instructions.md"
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

# build_context recipes (shipped JSON files). Same pattern as formats/:
# refresh on every install so the shipped catalog tracks the binary.
# User-added recipes in this directory are preserved; user edits to a
# shipped recipe are overwritten (copy a renamed variant to keep changes).
if [ -d "$SRC/recipes" ]; then
    info "Installing recipes to $RECIPES_DIR"
    cp -R "$SRC/recipes/." "$RECIPES_DIR/"
fi

# Embedding models — fetch from HuggingFace if not already present.
# Provider/model layout mirrors LM Studio and HuggingFace conventions:
#   ~/.ragger/models/<provider>/<model>/
#
# Legacy flat dirs (e.g. all-MiniLM-L6-v2/ at the top level) still work
# for backward compat but aren't offered as new choices in the dashboard.
# New installs use the provider layout exclusively.

# Helper: download one ONNX embedding model from HuggingFace.
# Usage: download_model <hf_repo> <model_dir> [onnx_subdir]
#   hf_repo    — HuggingFace repo (e.g. sentence-transformers/all-MiniLM-L6-v2)
#   model_dir  — local target directory under $MODEL_DIR
#   onnx_subdir— subdirectory containing model.onnx in the HF repo (default: "")
#                If set, the ONNX file is stored at $model_dir/onnx/model.onnx
download_model() {
    local hf_repo="$1" model_dir="$2" onnx_sub="${3:-}"
    local dest="$MODEL_DIR/$model_dir"
    local hf_base="https://huggingface.co/$hf_repo/resolve/main"

    # Determine where model.onnx goes locally
    local onnx_local="$dest/model.onnx"
    local onnx_remote="$hf_base/model.onnx"
    if [ -n "$onnx_sub" ]; then
        onnx_local="$dest/$onnx_sub/model.onnx"
        onnx_remote="$hf_base/$onnx_sub/model.onnx"
    fi

    if [ -f "$onnx_local" ]; then
        info "✓ $model_dir already present"
        return
    fi

    info "Downloading $model_dir from $hf_repo"
    mkdir -p "$(dirname "$onnx_local")"
    mkdir -p "$dest"

    for f in config.json special_tokens_map.json tokenizer_config.json tokenizer.json vocab.txt; do
        curl -sSLfo "$dest/$f" "$hf_base/$f" || \
            warn "  failed: $f (rerun install-bin.sh to retry)"
    done
    curl -#Lfo "$onnx_local" "$onnx_remote" || \
        warn "  failed: model.onnx (rerun install-bin.sh once you have a network)"
}

download_model "sentence-transformers/all-MiniLM-L6-v2" \
               "sentence-transformers/all-MiniLM-L6-v2" "onnx"

download_model "sentence-transformers/all-MiniLM-L12-v2" \
               "sentence-transformers/all-MiniLM-L12-v2" "onnx"

download_model "onnx-community/all-MiniLM-L12-v2-qa-all-ONNX" \
               "onnx-community/all-MiniLM-L12-v2-qa-all-ONNX" "onnx"

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
if [ ! -f "$RAGGER_BASE/token" ]; then
    info "Bootstrapping bearer token at $RAGGER_BASE/token"
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
    # StandardOutPath/StandardErrorPath point at the same fixed path Ragger
    # itself rotates (see the log_max_size_mb/log_max_age_days config settings).
    # Ragger writes its own log lines by opening/appending/closing the file
    # on every call (not via this stdout/stderr redirect) so its content is
    # unaffected by rotation; only the launchd-level duplicate of ERROR/
    # CRITICAL lines Ragger also echoes to stderr keeps writing to whatever
    # inode this fd was opened against, so after a rotation those stderr
    # echoes trail into the rotated-out backup until the next daemon
    # restart. Harmless — the authoritative copy is always the live file.
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
    <key>StandardOutPath</key><string>$LOG_DIR/activity.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/activity.log</string>
</dict>
</plist>
EOF
    chmod 0644 "$PLIST"


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
StandardOutput=append:$LOG_DIR/activity.log
StandardError=append:$LOG_DIR/activity.log

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload 2>/dev/null || true

    # Ensure the unit will start automatically next login
    systemctl --user enable ragger.service 2>/dev/null || true

    echo ""
    echo "To keep it running when you're not logged in:"
    echo "    sudo loginctl enable-linger $USER"
else
    warn "Unknown OS '$OS' — daemon file not installed. Run 'ragger serve' manually."
fi

echo ""
info "Installed to $RAGGER_BASE"
echo ""
echo "  To start the daemon:              ragger start"
echo "  To open the configuration dashboard:  ragger dashboard --browser"
echo ""
echo "  (\`ragger dashboard\` prints the URL instead of opening a browser.)"

if [ -d "$HOME/.openclaw" ]; then
    echo ""
    info "OpenClaw detected — run ./install-openclaw.sh to wire the memory plugin."
fi
