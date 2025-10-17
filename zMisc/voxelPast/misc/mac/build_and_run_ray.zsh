#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJECT_ROOT"

if command -v brew >/dev/null 2>&1; then
  brew list nlohmann-json >/dev/null 2>&1 || brew install nlohmann-json
  brew list stb >/dev/null 2>&1 || brew install stb
else
  echo "[warn] Homebrew not found. Ensure headers nlohmann/json.hpp and stb_image.h are available in include paths."
fi

# Determine Homebrew include prefix for Apple Silicon
BREW_PREFIX=${HOMEBREW_PREFIX:-/opt/homebrew}
INCLUDES=("-I${BREW_PREFIX}/include" "-I${BREW_PREFIX}/include/stb")

echo "[info] Compiling ray_voxel.cpp"
clang++ -std=c++17 -O2 ray_voxel.cpp ${INCLUDES[@]} -o ray_voxel
echo "[info] Built ./ray_voxel"

if [ $# -eq 3 ]; then
  ./ray_voxel "$1" "$2" "$3"
  exit $?
fi

echo "Usage: mac/build_and_run_ray.zsh <metadata.json> <image_folder> <output_voxel_bin>"
echo "Example: mac/build_and_run_ray.zsh motionimages/metadata.json motionimages voxel_grid.bin"

