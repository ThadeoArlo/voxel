Quick command cheatsheet

Mask (generate frames + binary motion masks)

```bash
python3 mask/mask.py --scene A1
# Options: --stride N --thresh T --blur K --downscale S --no-overlay
```

Triangulated track (JSON)

```bash
# Auto-runs after mask when --scene is used, or run directly:
python3 render/render.py --scene A1
# Uses render/cam_config.json (setup1/2/3) and mask/output/<scene>
```

Voxel reconstruction (ray accumulation) and viewing.

```bash
# Build voxel grid from masks (writes to zMisc/Pixeltovoxelprojector/output/<scene>/)
python3 zMisc/Pixeltovoxelprojector/test.py --scene A3

# Extract trajectory from voxel grid (writes track.json for web viewer)
python3 zMisc/Pixeltovoxelprojector/track.py --scene A3 \
  --temporal-lock --local-radius 3 --seed-triangulation --tri-thresh 4.0

# View voxel point cloud (auto-reads voxel_grid.json if present)
python3 zMisc/Pixeltovoxelprojector/voxelmotionviewer.py \
  --bin zMisc/Pixeltovoxelprojector/output/A3/voxel_grid.bin \
  --percentile 99.95 --center 0,0,300 --rotate 90,270,0
```

Web viewer (compare simulation vs reconstruction)

```bash
# Open render/index.html in browser
# Load track.json from zMisc/Pixeltovoxelprojector/output/<scene>/
# Press Start to animate; check "Include simulation" to compare
```

Notes

- Camera setups are in `render/cam_config.json`:
  - setup1* (close baseline), setup2* (wide baseline), setup3\* (360° ring)
- Mask improvements: running background model, morphological closing,
  largest-component filtering; tune with `--bg-alpha`, `--close`,
  `--largest-only`.
- Voxel reconstruction: more robust than triangulation, handles multiple objects
- Track extraction: uses local projection voting for temporal consistency
- Web viewer: supports comparing simulation vs reconstruction in real-time
