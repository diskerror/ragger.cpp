#!/bin/bash
# build.sh — Build Ragger C++ from source
#
# Usage: ./build.sh [clean]
#   clean  — remove build directory first

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m'

missing=()
check() { command -v "$1" &>/dev/null || missing+=("$1"); }

check cmake
check make
check c++
check pkg-config

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required tools:${NC} ${missing[*]}"
    echo "    Install them before building."
    exit 1
fi

cd "$(dirname "$0")"
BUILD_DIR="build"

if [ "${1:-}" = "clean" ]; then
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
        JOBS=$(sysctl -n hw.ncpu)
        ;;
    Linux)
        JOBS=$(nproc)
        ;;
    *)
        JOBS=4
        ;;
esac

# Only re-run cmake if needed
if [ ! -f Makefile ] || [ ../CMakeLists.txt -nt Makefile ]; then
    echo "[+] Configuring..."
    cmake .. $CMAKE_FLAGS
fi

echo "[+] Building with $JOBS threads..."
make -j"$JOBS"

echo ""
echo "✓ Built: $(pwd)/ragger"
echo "  Install with: sudo ./install.sh"
