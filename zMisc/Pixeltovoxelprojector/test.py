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


def accumulate_voxels(mask_folders, cams, out_bin: Path, grid_size: int, voxel_size: float, grid_center, pixel_stride: int):
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

    for i in range(min_len):
        if min_len > 0 and i % max(1, (min_len // 20)) == 0:
            pct = int((i * 100) / min_len)
            print(f"[progress] {pct}% ({i}/{min_len})")
        for k in range(3):
            m = cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            # Subsample pixels for speed
            ys = np.arange(0, m.shape[0], pixel_stride)
            xs = np.arange(0, m.shape[1], pixel_stride)
            for yy in ys:
                row = m[yy]
                for xx in xs:
                    if row[xx] == 0:
                        continue
                    d = pixel_ray(cams[k], width, height, float(xx), float(yy))
                    vox_idx = dda_voxel_indices(cams[k]["pos"], d, grid_size, voxel_size, grid_center)
                    for (ix, iy, iz) in vox_idx:
                        voxel_grid[iz, iy, ix] += 1.0  # note: store as (Z,Y,X) for viewer convenience

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
    voxel_grid = accumulate_voxels(mask_folders, cams, out_bin, args.grid_size, args.voxel_size, center, max(1, args.pixel_stride))
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
    print(f"[ok] Wrote {out_dir / 'voxel_grid.json'} (viewer params). Update grid_center in voxelmotionviewer.py accordingly.")


if __name__ == "__main__":
    main()


