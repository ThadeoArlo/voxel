#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

echo "[info] Running spacevoxelviewer.py (requires FITS + compiled extension)"
python spacevoxelviewer.py || true

echo "[info] Running voxelmotionviewer.py to visualize voxel_grid.bin with PyVista"
python voxelmotionviewer.py || true

