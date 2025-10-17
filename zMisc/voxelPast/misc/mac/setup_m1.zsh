#!/bin/zsh
set -euo pipefail

# Simple macOS (Apple Silicon) setup for this project
# - Creates a venv
# - Installs Homebrew deps (llvm libomp) if available
# - Installs Python deps
# - Builds the C++ extension

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate

# Prefer Homebrew python if present
if command -v brew >/dev/null 2>&1; then
  echo "[info] Homebrew found"
  # OpenMP and newer clang (optional)
  brew list libomp >/dev/null 2>&1 || brew install libomp
  # For visualization libs that need vtk runtime
  # brew install vtk # optional; pyvista wheels often bundle
else
  echo "[warn] Homebrew not found; proceeding without it"
fi

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r mac/requirements-macos.txt

# Build the extension with Apple Silicon friendly flags
export CFLAGS="${CFLAGS:-} -O3"
export CXXFLAGS="${CXXFLAGS:-} -O3"
export LDFLAGS="${LDFLAGS:-}"

python setup.py build_ext --inplace

echo "[done] Environment ready. Activate with: source .venv/bin/activate"

