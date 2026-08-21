#!/bin/bash
# build.sh — Build Ragger C++ from source
#
# Usage: ./build.sh [clean] [--stats|--no-stats]
#   clean      — remove build directory first
#   --stats    — enable RAGGER_STATS instrumentation (default: ON)
#   --no-stats — disable RAGGER_STATS instrumentation

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m'

# rustup installs cargo under ~/.cargo/bin and writes an env file to source.
# Pull it in early so the cargo/rustc check below sees it without forcing
# the user to open a new shell after `rustup-init`.
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

missing=()
check() { command -v "$1" &>/dev/null || missing+=("$1"); }

check cmake
check make
check c++
check pkg-config
# Rust toolchain — required by vendor/tokenizers-cpp. Cargo invokes
# rustc, so checking one would do, but naming both makes the fix
# obvious in the error message.
check cargo
check rustc

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required tools:${NC} ${missing[*]}"
    echo ""
    case " ${missing[*]} " in
        *" cargo "*|*" rustc "*)
            echo "    Install Rust with rustup (recommended):"
            echo "      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
            echo "    Then open a new shell, or run:  . \$HOME/.cargo/env"
            echo "" ;;
    esac
    echo "    System packages on macOS: sudo port install cmake pkg-config rust"
    echo "    System packages on Debian/Ubuntu: sudo apt install build-essential cmake pkg-config"
    exit 1
fi

cd "$(dirname "$0")/.."

RAGGER_STATS_FLAG="OFF"   # default: stats off
DO_CLEAN=0

for arg in "$@"; do
    case "$arg" in
        clean)       DO_CLEAN=1 ;;
        --stats)     RAGGER_STATS_FLAG="ON" ;;
        --no-stats)  RAGGER_STATS_FLAG="OFF" ;;
    esac
done

# Dual build dirs: each flag gets its own tree so their CMake caches never go
# stale against each other. build/ = stats off, build-stats/ = stats on.
if [ "$RAGGER_STATS_FLAG" = "ON" ]; then
    BUILD_DIR="build-stats"
else
    BUILD_DIR="build"
fi

if [ "$DO_CLEAN" = "1" ]; then
    echo "[+] Clean build"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check for required libraries
lib_missing=()
pkg-config --exists sqlite3 2>/dev/null || lib_missing+=("sqlite3")
pkg-config --exists eigen3 2>/dev/null || lib_missing+=("eigen3")
pkg-config --exists libcurl 2>/dev/null || lib_missing+=("libcurl")
pkg-config --exists openssl 2>/dev/null || lib_missing+=("openssl")

if [ ${#lib_missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required libraries:${NC} ${lib_missing[*]}"
    echo "    macOS (MacPorts): sudo port install ${lib_missing[*]}"
    echo "    Linux (apt):      sudo apt install $(printf 'lib%s-dev ' "${lib_missing[@]}")"
    exit 1
fi

# Detect platform-specific cmake flags
CMAKE_FLAGS=""
OS="$(uname -s)"
case "$OS" in
    Darwin)
        # MacPorts Boost — find latest installed version
        BOOST_BASE="/opt/local/libexec/boost"
        if [ -d "$BOOST_BASE" ]; then
            BOOST_DIR=$(ls -d "$BOOST_BASE"/[0-9]* 2>/dev/null | sort -V | tail -1)
            if [ -n "$BOOST_DIR" ]; then
                CMAKE_FLAGS="-DBOOST_ROOT=$BOOST_DIR"
            fi
        fi
        if [ -z "$CMAKE_FLAGS" ]; then
            echo -e "${RED}[!] Boost not found.${NC} Install with: sudo port install boost"
            exit 1
        fi
        JOBS=$(/usr/sbin/sysctl -n hw.ncpu)
        ;;
    Linux)
        JOBS=$(nproc)
        ;;
    *)
        JOBS=4
        ;;
esac

# Each flag has its own build dir, so the cache never goes stale against a
# different flag value. Re-run cmake only when unconfigured or CMakeLists changed.
if [ ! -f Makefile ] || [ ../CMakeLists.txt -nt Makefile ]; then
    echo "[+] Configuring (RAGGER_STATS=$RAGGER_STATS_FLAG) in $BUILD_DIR/..."

    # Use dev preset (local c_lib) if CMakeUserPresets.json exists,
    # otherwise fall back to default (fetches c_lib from GitHub).
    LOCAL_C_LIB=""
    if [ -f ../CMakeUserPresets.json ]; then
        # Extract FETCHCONTENT_SOURCE_DIR_C_LIB from the user presets
        C_LIB_PATH=$(python3 -c "import json,sys; d=json.load(open('../CMakeUserPresets.json')); print(d['configurePresets'][0].get('cacheVariables',{}).get('FETCHCONTENT_SOURCE_DIR_C_LIB',''))" 2>/dev/null || true)
        if [ -n "$C_LIB_PATH" ] && [ -d "$C_LIB_PATH" ]; then
            LOCAL_C_LIB="-DFETCHCONTENT_SOURCE_DIR_C_LIB=$C_LIB_PATH"
        fi
    fi

    cmake .. $CMAKE_FLAGS $LOCAL_C_LIB -DRAGGER_STATS="$RAGGER_STATS_FLAG"
fi

echo "[+] Building with $JOBS threads..."
make -j"$JOBS"

echo ""
echo "✓ Built: $(pwd)/ragger"
echo "  Install with: ./scripts/install.sh"
