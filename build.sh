#!/bin/bash
# build.sh — Build Ragger C++ from source
#
# Usage: ./build.sh [clean]
#   clean  — remove build directory first

set -euo pipefail

cd "$(dirname "$0")"
BUILD_DIR="build"

if [ "${1:-}" = "clean" ]; then
    echo "[+] Clean build"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect platform-specific cmake flags
CMAKE_FLAGS=""
OS="$(uname -s)"
case "$OS" in
    Darwin)
        # MacPorts Boost
        if [ -d /opt/local/libexec/boost/1.88 ]; then
            CMAKE_FLAGS="-DBOOST_ROOT=/opt/local/libexec/boost/1.88"
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
echo "  Install with: ./install.sh"
