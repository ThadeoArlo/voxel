#!/usr/bin/env python3
import os, sys, json, glob, math
import argparse
from pathlib import Path
import numpy as np
import cv2


def load_cam_config(path: str):
    cfg = json.load(open(path))
    cams = []
    for c in cfg["cameras"]:
        pos = np.asarray(c["position"], dtype=np.float64)
        fwd = np.asarray(c["forward"], dtype=np.float64)
        up  = np.asarray(c["up"], dtype=np.float64)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        up = up / (np.linalg.norm(up) + 1e-12)
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, fwd)
        fov_deg = float(c.get("fov_deg", 70.0))
        cams.append({
            "name": c.get("name"),
            "pos": pos,
            "fwd": fwd,
            "right": right,
            "up": up,
            "fov": math.radians(fov_deg),
        })
    return cams


def select_cams_by_scene(all_cams, scene: str):
    setup_num = None
    if scene and len(scene) >= 2 and scene[1:].isdigit():
        setup_num = int(scene[1:])
    if setup_num in (1, 2):
        wanted = [f"setup{setup_num}_{i}" for i in (1, 2, 3)]
        name_to_cam = {c.get("name"): c for c in all_cams}
        cams = []
        for n in wanted:
            if n not in name_to_cam:
                raise SystemExit(f"Camera '{n}' not found in config")
            cams.append(name_to_cam[n])
        return cams
    # fallback to first 3
    if len(all_cams) < 3:
        raise SystemExit("Need at least 3 cameras in config")
    return all_cams[:3]


def find_mask_folders_for_scene(mask_root: Path, scene: str):
    base = mask_root / scene
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Scene masks not found at {base}")
    # Prefer folders that end in ...1, ...2, ...3 (e.g., a11,a12,a13 or 1,2,3)
    subs = [p for p in base.iterdir() if p.is_dir()]
    def last_digit_key(p: Path):
        s = p.name.strip().lower()
        return s[-1] if s and s[-1] in '123' else None
    groups = {d: None for d in '123'}
    for p in subs:
        d = last_digit_key(p)
        if d in groups and groups[d] is None:
            # prefer subfolder that contains a 'masks' dir
            m = p / 'masks'
            groups[d] = m if m.exists() else p
    ordered = []
    for d in '123':
        if groups[d] is None:
            raise SystemExit(f"Could not find POV folder ending with '{d}' under {base}")
        # resolve masks folder inside
        m = groups[d]
        if m.is_dir() and m.name != 'masks':
            mm = m / 'masks'
            if mm.exists():
                m = mm
        ordered.append(m)
    return ordered  # [left, mid, right]


def list_mask_images(folder: Path):
    files = sorted(glob.glob(str(folder / "*.png")))
    if not files:
        raise SystemExit(f"No PNG masks found in {folder}")
    return files


def pixel_ray(cam, width: int, height: int, px: float, py: float):
    cx = width * 0.5
    cy = height * 0.5
    focal = (width * 0.5) / math.tan(cam["fov"] * 0.5)
    x = (px - cx)
    y = (py - cy)
    z = focal
    dir_cam = np.array([x, -y, z], dtype=np.float64)
    dir_cam /= (np.linalg.norm(dir_cam) + 1e-12)
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)
    dir_world = R @ dir_cam
    dir_world /= (np.linalg.norm(dir_world) + 1e-12)
    return dir_world


def ray_aabb(camera_pos, ray_dir, grid_min, grid_max):
    tmin = -1e30
    tmax = 1e30
    for i in range(3):
        origin = camera_pos[i]
        d = ray_dir[i]
        mn = grid_min[i]
        mx = grid_max[i]
        if abs(d) < 1e-12:
            if origin < mn or origin > mx:
                return None
        else:
            t1 = (mn - origin) / d
            t2 = (mx - origin) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmin < 0.0:
        tmin = 0.0
    return tmin, tmax


def dda_voxel_indices(camera_pos, ray_dir, N: int, voxel_size: float, grid_center):
    half = 0.5 * (N * voxel_size)
    grid_min = np.array(grid_center, dtype=np.float64) - half
    grid_max = np.array(grid_center, dtype=np.float64) + half
    hit = ray_aabb(camera_pos, ray_dir, grid_min, grid_max)
    if hit is None:
        return []
    tmin, tmax = hit
    start = camera_pos + tmin * ray_dir
    fx = (start[0] - grid_min[0]) / voxel_size
    fy = (start[1] - grid_min[1]) / voxel_size
    fz = (start[2] - grid_min[2]) / voxel_size
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
        return []

    step_x = 1 if ray_dir[0] >= 0 else -1
    step_y = 1 if ray_dir[1] >= 0 else -1
    step_z = 1 if ray_dir[2] >= 0 else -1

    def boundary_x(i): return grid_min[0] + i * voxel_size
    def boundary_y(i): return grid_min[1] + i * voxel_size
    def boundary_z(i): return grid_min[2] + i * voxel_size

    nx_x = ix + (1 if step_x > 0 else 0)
    nx_y = iy + (1 if step_y > 0 else 0)
    nx_z = iz + (1 if step_z > 0 else 0)

    next_bx = boundary_x(nx_x)
    next_by = boundary_y(nx_y)
    next_bz = boundary_z(nx_z)

    t_max_x = (next_bx - camera_pos[0]) / (ray_dir[0] + 1e-30)
    t_max_y = (next_by - camera_pos[1]) / (ray_dir[1] + 1e-30)
    t_max_z = (next_bz - camera_pos[2]) / (ray_dir[2] + 1e-30)

    t_delta_x = voxel_size / (abs(ray_dir[0]) + 1e-30)
    t_delta_y = voxel_size / (abs(ray_dir[1]) + 1e-30)
    t_delta_z = voxel_size / (abs(ray_dir[2]) + 1e-30)

    t_current = tmin
    voxels = []
    # conservative max steps
    max_steps = int((tmax - tmin) / (min(t_delta_x, t_delta_y, t_delta_z) + 1e-30)) + 2
    for _ in range(max_steps):
        if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
            break
        if t_current > tmax:
            break
        voxels.append((ix, iy, iz))
        if t_max_x < t_max_y and t_max_x < t_max_z:
            ix += step_x
            t_current = t_max_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_z:
            iy += step_y
            t_current = t_max_y
            t_max_y += t_delta_y
        else:
            iz += step_z
            t_current = t_max_z
            t_max_z += t_delta_z
    return voxels


def dda_voxel_steps(camera_pos, ray_dir, N: int, voxel_size: float, grid_center):
    """
    Similar to dda_voxel_indices but also returns the parametric distance t at each step.
    Returns a list of (ix, iy, iz, t_current). Assumes ray_dir is unit length so t is meters.
    """
    half = 0.5 * (N * voxel_size)
    grid_min = np.array(grid_center, dtype=np.float64) - half
    grid_max = np.array(grid_center, dtype=np.float64) + half
    hit = ray_aabb(camera_pos, ray_dir, grid_min, grid_max)
    if hit is None:
        return []
    tmin, tmax = hit
    start = camera_pos + tmin * ray_dir
    fx = (start[0] - grid_min[0]) / voxel_size
    fy = (start[1] - grid_min[1]) / voxel_size
    fz = (start[2] - grid_min[2]) / voxel_size
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
        return []

    step_x = 1 if ray_dir[0] >= 0 else -1
    step_y = 1 if ray_dir[1] >= 0 else -1
    step_z = 1 if ray_dir[2] >= 0 else -1

    def boundary_x(i): return grid_min[0] + i * voxel_size
    def boundary_y(i): return grid_min[1] + i * voxel_size
    def boundary_z(i): return grid_min[2] + i * voxel_size

    nx_x = ix + (1 if step_x > 0 else 0)
    nx_y = iy + (1 if step_y > 0 else 0)
    nx_z = iz + (1 if step_z > 0 else 0)

    next_bx = boundary_x(nx_x)
    next_by = boundary_y(nx_y)
    next_bz = boundary_z(nx_z)

    t_max_x = (next_bx - camera_pos[0]) / (ray_dir[0] + 1e-30)
    t_max_y = (next_by - camera_pos[1]) / (ray_dir[1] + 1e-30)
    t_max_z = (next_bz - camera_pos[2]) / (ray_dir[2] + 1e-30)

    t_delta_x = voxel_size / (abs(ray_dir[0]) + 1e-30)
    t_delta_y = voxel_size / (abs(ray_dir[1]) + 1e-30)
    t_delta_z = voxel_size / (abs(ray_dir[2]) + 1e-30)

    t_current = tmin
    out = []
    max_steps = int((tmax - tmin) / (min(t_delta_x, t_delta_y, t_delta_z) + 1e-30)) + 2
    for _ in range(max_steps):
        if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
            break
        if t_current > tmax:
            break
        out.append((ix, iy, iz, t_current))
        if t_max_x < t_max_y and t_max_x < t_max_z:
            ix += step_x
            t_current = t_max_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_z:
            iy += step_y
            t_current = t_max_y
            t_max_y += t_delta_y
        else:
            iz += step_z
            t_current = t_max_z
            t_max_z += t_delta_z
    return out


def precompute_dirs_for_grid(cam, width: int, height: int, xs: np.ndarray, ys: np.ndarray):
    """Vectorize pixel_ray over a strided grid of xs, ys for one camera."""
    cx = width * 0.5
    cy = height * 0.5
    focal = (width * 0.5) / math.tan(cam["fov"] * 0.5)
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)
    X, Y = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64), indexing="xy")
    x = (X - cx)
    y = (Y - cy)
    z = np.full_like(x, focal)
    dir_cam = np.stack([x, -y, z], axis=-1)
    norm = np.linalg.norm(dir_cam, axis=-1, keepdims=True) + 1e-12
    dir_cam /= norm
    dir_world = dir_cam @ R.T
    dir_world /= (np.linalg.norm(dir_world, axis=-1, keepdims=True) + 1e-12)
    return dir_world

def accumulate_voxels(mask_folders, cams, out_bin: Path, grid_size: int, voxel_size: float, grid_center, pixel_stride: int,
                      use_motion_gating: bool = False,
                      mask_blur: int = 0,
                      mask_open: int = 0,
                      attenuation_alpha: float = 0.0,
                      soft_splat: bool = False,
                      soft_splat_weight: float = 0.25,
                      precompute_rays: bool = False):
    # Load first frame to get dimensions
    first = None
    for f in mask_folders:
        imgs = list_mask_images(f)
        if imgs:
            first = imgs[0]
            break
    if first is None:
        raise SystemExit("No mask images found")
    sample = cv2.imread(first, cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise SystemExit("Could not read sample mask image")
    height, width = sample.shape[:2]

    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    half = 0.5 * (grid_size * voxel_size)
    grid_min = np.array(grid_center, dtype=np.float64) - half
    grid_max = np.array(grid_center, dtype=np.float64) + half

    # iterate frames
    mask_paths = [list_mask_images(f) for f in mask_folders]
    min_len = min(len(x) for x in mask_paths)
    print(f"[info] Accumulating {min_len} frames; grid N={grid_size}, voxel={voxel_size}m, center={grid_center}")

    prev_masks = [None, None, None]

    for i in range(min_len):
        if min_len > 0 and i % max(1, (min_len // 20)) == 0:
            pct = int((i * 100) / min_len)
            print(f"[progress] {pct}% ({i}/{min_len})")
        for k in range(3):
            m = cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            # Optional preprocessing: blur/open
            if mask_blur and mask_blur > 0:
                bsz = int(max(1, mask_blur)) | 1
                m = cv2.GaussianBlur(m, (bsz, bsz), 0)
            if mask_open and mask_open > 0:
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(mask_open), int(mask_open)))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)

            # Optional motion gating from mask diffs
            motion_mask = None
            if use_motion_gating and prev_masks[k] is not None:
                diff = cv2.absdiff(m, prev_masks[k])
                _, motion_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
            prev_masks[k] = m.copy()

            # Subsample pixels for speed
            ys = np.arange(0, m.shape[0], pixel_stride)
            xs = np.arange(0, m.shape[1], pixel_stride)
            dirs_grid = None
            if precompute_rays:
                dirs_grid = precompute_dirs_for_grid(cams[k], width, height, xs, ys)
            for yy in ys:
                row = m[yy]
                for xx in xs:
                    if row[xx] == 0:
                        continue
                    if motion_mask is not None and motion_mask[yy, xx] == 0:
                        continue
                    if dirs_grid is not None:
                        j = int(yy // pixel_stride)
                        i2 = int(xx // pixel_stride)
                        d = dirs_grid[j, i2]
                    else:
                        d = pixel_ray(cams[k], width, height, float(xx), float(yy))
                    steps = dda_voxel_steps(cams[k]["pos"], d, grid_size, voxel_size, grid_center)
                    if not steps:
                        continue
                    val_w = float(row[xx]) / 255.0
                    for (ix, iy, iz, tcur) in steps:
                        w = val_w
                        if attenuation_alpha and attenuation_alpha > 0:
                            w = w / (1.0 + float(attenuation_alpha) * float(tcur))
                        voxel_grid[iz, iy, ix] += float(w)
                        if soft_splat:
                            sw = float(soft_splat_weight) * float(w)
                            if sw > 0:
                                if ix > 0:
                                    voxel_grid[iz, iy, ix-1] += sw
                                if ix < grid_size-1:
                                    voxel_grid[iz, iy, ix+1] += sw
                                if iy > 0:
                                    voxel_grid[iz, iy-1, ix] += sw
                                if iy < grid_size-1:
                                    voxel_grid[iz, iy+1, ix] += sw
                                if iz > 0:
                                    voxel_grid[iz-1, iy, ix] += sw
                                if iz < grid_size-1:
                                    voxel_grid[iz+1, iy, ix] += sw

    # Write binary in (N, voxel_size, data) row-major order
    with open(out_bin, "wb") as f:
        np.array([grid_size], dtype=np.int32).tofile(f)
        np.array([voxel_size], dtype=np.float32).tofile(f)
        voxel_grid.astype(np.float32).ravel(order='C').tofile(f)
    print(f"[ok] Wrote {out_bin}")
    return voxel_grid


def main():
    ap = argparse.ArgumentParser(description="Voxel accumulation test (mac-compatible) from masks")
    ap.add_argument("--scene", required=True, help="Scene name under mask/output (e.g., A1, B1)")
    ap.add_argument("--mask-root", default=str(Path(__file__).resolve().parents[2] / "mask" / "output"), help="Root of mask/output")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "render" / "cam_config.json"), help="Camera config JSON")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "output"), help="Output root directory (per-scene subfolder will be created)")
    ap.add_argument("--grid-size", type=int, default=300, help="Voxel grid size N (N^3 floats)")
    ap.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size (meters per voxel)")
    ap.add_argument("--grid-center", default="0,0,300", help="Grid center x,y,z in meters")
    ap.add_argument("--pixel-stride", type=int, default=2, help="Subsample stride for mask pixels (>=1)")
    ap.add_argument("--use-motion-gating", action="store_true", help="Gate accumulation by mask frame-to-frame motion")
    ap.add_argument("--mask-blur", type=int, default=0, help="Gaussian blur kernel size (odd) for masks (0=off)")
    ap.add_argument("--mask-open", type=int, default=0, help="Morphological open kernel size for masks (0=off)")
    ap.add_argument("--attenuation-alpha", type=float, default=0.0, help="Distance attenuation alpha (0=off)")
    ap.add_argument("--soft-splat", action="store_true", help="Soft-splat energy to 6-neighbour voxels")
    ap.add_argument("--soft-splat-weight", type=float, default=0.25, help="Fraction of weight for neighbours if soft-splat")
    ap.add_argument("--precompute-rays", action="store_true", help="Precompute ray directions for strided grid")
    ap.add_argument("--save-percentile", type=float, default=None, help="If set, also save a top-percentile voxel mask (e.g., 99.5)")
    args = ap.parse_args()

    mask_root = Path(args.mask_root)
    all_cams = load_cam_config(args.config)
    cams = select_cams_by_scene(all_cams, args.scene)

    center = tuple(float(x) for x in args.grid_center.split(','))
    if len(center) != 3:
        raise SystemExit("--grid-center must be 'x,y,z'")

    mask_folders = find_mask_folders_for_scene(mask_root, args.scene)
    print("[info] Mask folders (L,M,R):")
    for p in mask_folders:
        print("  ", p)

    out_dir = Path(args.out_root) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = out_dir / "voxel_grid.bin"
    voxel_grid = accumulate_voxels(
        mask_folders,
        cams,
        out_bin,
        args.grid_size,
        args.voxel_size,
        center,
        max(1, args.pixel_stride),
        args.use_motion_gating,
        int(args.mask_blur),
        int(args.mask_open),
        float(args.attenuation_alpha),
        bool(args.soft_splat),
        float(args.soft_splat_weight),
        bool(args.precompute_rays),
    )
    # Also save as .npy for quick analysis
    np.save(out_dir / "voxel_grid.npy", voxel_grid)
    # Write a simple JSON with viewer params
    meta = {
        "grid_size": args.grid_size,
        "voxel_size": args.voxel_size,
        "grid_center": center,
        "scene": args.scene,
        "mask_folders": [str(p) for p in mask_folders],
        "config": str(Path(args.config).resolve()),
    }
    with open(out_dir / 'voxel_grid.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] Wrote {out_dir / 'voxel_grid.json'} (viewer params). Update grid_center in voxelViewer.py accordingly.")

    # Optional: save top-percentile mask
    if args.save_percentile is not None:
        p = float(args.save_percentile)
        flat = voxel_grid.ravel()
        if np.any(flat > 0):
            thresh = np.percentile(flat[flat > 0], p)
            vox_mask = (voxel_grid >= thresh).astype(np.uint8)
            np.save(out_dir / "voxel_grid_top.npy", vox_mask)
            print(f"[ok] Saved top-percentile ({p}%) voxel mask to {out_dir / 'voxel_grid_top.npy'}")


if __name__ == "__main__":
    main()


