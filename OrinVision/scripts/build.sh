#!/usr/bin/env bash
# =====================================================================
# build.sh - Configure, build and test OrinVision
# Usage: ./scripts/build.sh [Release|Debug] [--no-test]
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_TYPE="${1:-Release}"
RUN_TESTS=1
[[ "${2:-}" == "--no-test" || "${1:-}" == "--no-test" ]] && RUN_TESTS=0
[[ "$BUILD_TYPE" == "--no-test" ]] && BUILD_TYPE=Release

BUILD_DIR="build"
[[ "$BUILD_TYPE" == "Debug" ]] && BUILD_DIR="build-debug"

# nvcc is often missing from PATH in non-interactive SSH sessions
export PATH="/usr/local/cuda/bin:$PATH"

command -v cmake >/dev/null || { echo "cmake not found: sudo apt install cmake"; exit 1; }
command -v nvcc  >/dev/null || { echo "nvcc not found: Jetson -> sudo apt install nvidia-jetpack, PC -> install the CUDA Toolkit"; exit 1; }

echo "==> Configuring ($BUILD_TYPE) in $BUILD_DIR/"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "==> Building with $(nproc) jobs"
cmake --build "$BUILD_DIR" -j"$(nproc)"

if [[ $RUN_TESTS -eq 1 ]]; then
    echo "==> Running tests"
    (cd "$BUILD_DIR" && ctest --output-on-failure)
fi

echo
echo "Build OK. Try:"
echo "  ./$BUILD_DIR/orin_vision --help"
echo "  ./$BUILD_DIR/orin_vision"
