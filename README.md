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
python3 zMisc/voxelMisc/voxelViewer.py \
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

## Updated voxel and track usage (concise)

voxel/voxel.py builds a 3D voxel grid from per-camera masks.

New useful flags:

- --use-motion-gating: gate accumulation by frame-to-frame mask change
- --mask-blur N: Gaussian blur kernel size (odd); 0 off
- --mask-open N: morphological open kernel size; 0 off
- --attenuation-alpha A: distance attenuation w = w/(1+A\*t)
- --soft-splat: add a fraction to 6-neighbour voxels
- --soft-splat-weight W: neighbour fraction (default 0.25)
- --precompute-rays: vectorize rays for strided pixels
- --save-percentile P: also save top-P% voxel mask to voxel_grid_top.npy

Example:

```bash
python3 voxel/voxel.py --scene A1 --pixel-stride 2 --use-motion-gating \
  --mask-blur 3 --mask-open 3 --attenuation-alpha 0.002 --precompute-rays \
  --save-percentile 99.5
```

voxel/track.py extracts a per-frame 3D trajectory; can leverage the voxel grid
as a prior.

New useful flags:

- --prior-percentile P: trim prior to top-P% before normalization
- --mask-blur N, --mask-open N: preprocess masks before voting
- --kf-smooth: apply constant-velocity Kalman smoothing
- --kf-q Q, --kf-r R: KF process/measurement noise scales

Example:

```bash
python3 voxel/track.py --scene A1 --use-global-prior --prior-percentile 99.0 \
  --seed-triangulation --tri-thresh 4.0 --local-radius 3 \
  --mask-blur 3 --mask-open 3 --kf-smooth --kf-q 0.05 --kf-r 0.5
```
