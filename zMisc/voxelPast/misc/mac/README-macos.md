### macOS (Apple Silicon) quickstart

Requirements: macOS 13+, Xcode CLT, Homebrew recommended.

1. Clone and cd into the repo, then run:

```bash
chmod +x mac/setup_m1.zsh mac/run_viewers.zsh
./mac/setup_m1.zsh
```

This creates `.venv`, installs deps, and builds the C++ extension
(`process_image_cpp`).

2. Activate environment for future shells:

```bash
source .venv/bin/activate
```

3. To visualize results:

```bash
./mac/run_viewers.zsh
```

Notes:

- OpenMP on mac uses Homebrew `libomp`. If you see link errors, run
  `brew install libomp` and re-run setup.
- For 3D interactive plotting (`pyvista`), a modern VTK wheel is installed. If
  rendering fails headless, PyVista will still save screenshots.
- To rebuild the extension after code changes: Vendored headers (if Homebrew
  headers are missing):

```bash
mkdir -p third_party/stb third_party/nlohmann
curl -L -o third_party/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -L -o third_party/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
```

```bash
python setup.py build_ext --inplace
```
