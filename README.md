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

Saving camera configs directly from the browser UI (serverless):

1. Open `render/index.html` in a Chromium-based browser (Chrome, Edge, Arc).
2. Click "Save Camera Configs". On first save you’ll be prompted to select
   `cam_config.json`. Subsequent saves write silently to that same file.
   - Uses the File System Access API. If permission is denied or unsupported,
     saving will not proceed.

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

CONFIG NOTES

- cam_config.json
  - Setup1: 200,30,800
    - when zoomed better result and got complete path, from far away resulting
      path truncated / shorter than actual simulation path length.
  - Setup2: 120,100,0,800
    - POVs narrow, json empty, even after zoom i get result but very bad.
- cam_config_1.json:
  - Setup1: 0,30,800
    - reconstructed path still shorter than actual sim
  - Setup2: 120,100,-10,800
    - testing -/+ flare
    - rendering spawned very far from sim location
- cam_config_2.json
  - Setup1: 0,30,720
    - reconstructed path still shorter than actual sim
  - Setup2: 100, 1, -15, 480
    - rendering spawned very far from sim location
- cam_config_3.json
  - Setup1: 0,30,720
    - reconstructed path still shorter than actual sim though a bit
      fuller/longer
  - Setup2: 100, 1, -15, 480
    - rendering spawned very far from sim location
- cam_config_4.json
  - Setup1: 700,30,560
    - Results very bad, short path, skewed, offset, etc.
  - Setup2: 100, 400, 0, 460
    - track now resides in the grid instead of very far away in the void, but
      still the reconstruction is offset and contain more irrelevant tracks.
