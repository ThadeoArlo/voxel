### Voxel Projector – minimal pipeline

Three steps: 1) make motion masks, 2) triangulate a 3D track, 3) build a voxel
grid and (optionally) extract a voxel-based track.

Input videos live in `src/<scene>/` (e.g., `src/A1/` contains three camera
videos like `A11.mp4`, `A12.mp4`, `A13.mp4`).

---

### mask/

Generates per-frame RGB frames and binary motion masks for each camera.

Outputs to `mask/output/<scene>/<cam>/{frames,masks}/` plus `motion_mask.mp4`
and `motion_overlay.mp4`.

Basic usage:

```bash
python3 mask/mask.py --scene A1
# Common flags: --stride N  --thresh T  --blur K  --downscale S  --no-overlay
```

You can also process a single video:

```bash
python3 mask/mask.py --input /abs/path/to/video.mp4 --out /abs/path/to/out
```

---

### render/

Triangulates a 3D point per frame from the three mask streams and writes a
`track.json` for visualization.

Reads masks from `mask/output/<scene>/` and cameras from
`render/cam_config.json` (auto-selects `setup1_*`, `setup2_*`, `setup3_*` from
the scene name A1/A2/A3, etc.).

Basic usage:

```bash
python3 render/render.py --scene A1
# Outputs: render/output/A1/track.json
```

Advanced (explicit mask folders or config):

```bash
python3 render/render.py --masks /abs/m1 /abs/m2 /abs/m3 --config render/cam_config.json --out render/output
```

Saving camera configs directly from the browser UI (no file download):

1. Start the local save server in a terminal:

```bash
python3 render/save_server.py --port 8787
```

2. Open `render/index.html` in your browser. Click "Save Camera Configs".
   - If the save server is running, it updates `render/cam_config.json` in-place
     and flashes green.
   - If the server isn’t running, it falls back to downloading `cam_config.json`
     (amber flash).

Serverless alternative (no Python server):

- Click "Choose cam_config.json…" in the HUD and select your existing
  `render/cam_config.json`.
- Then "Save Camera Configs" will write directly to that file using the browser
  File System Access API.
  - Supported in Chromium-based browsers in secure contexts (localhost or
    `file://` with user selection).
  - If not supported or permission denied, it will fall back to the server or
    download flow above.

---

### voxel/

Builds a dense voxel grid by ray-accumulating through masks, then optionally
extracts a trajectory using local projection voting.

Outputs under `voxel/output/<scene>/`: `voxel_grid.bin`, `voxel_grid.npy`,
`voxel_grid.json`, and `track.json` (from the tracker).

1. Build voxel grid:

```bash
python3 voxel/voxel.py --scene A1
# Useful flags: --grid-size N --voxel-size M --grid-center x,y,z
#               --pixel-stride S --use-motion-gating --mask-blur K --mask-open K
```

2. Extract a voxel-based track:

```bash
python3 voxel/track.py --scene A1 --use-global-prior --seed-triangulation
# Useful flags: --local-radius R --tri-thresh PX --kf-smooth --kf-q Q --kf-r Rm
```

Notes:

- Camera config lives at `render/cam_config.json`. Scene names like A1/A2/A3 map
  to setups 1/2/3.
- If your subject is off-center, adjust `--grid-center` in voxel commands
  (defaults to `0,0,300`).
- All commands prefer absolute paths for robustness.
