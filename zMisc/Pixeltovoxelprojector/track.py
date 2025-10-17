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
    if setup_num in (1, 2, 3):
        wanted = [f"setup{setup_num}_{i}" for i in (1, 2, 3)]
        name_to_cam = {c.get("name"): c for c in all_cams}
        cams = []
        for n in wanted:
            if n not in name_to_cam:
                raise SystemExit(f"Camera '{n}' not found in config")
            cams.append(name_to_cam[n])
        return cams
    if len(all_cams) < 3:
        raise SystemExit("Need at least 3 cameras in config")
    return all_cams[:3]


def find_mask_folders_for_scene(mask_root: Path, scene: str):
    base = mask_root / scene
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Scene masks not found at {base}")
    subs = [p for p in base.iterdir() if p.is_dir()]
    def last_digit_key(p: Path):
        s = p.name.strip().lower()
        return s[-1] if s and s[-1] in '123' else None
    groups = {d: None for d in '123'}
    for p in subs:
        d = last_digit_key(p)
        if d in groups and groups[d] is None:
            m = p / 'masks'
            groups[d] = m if m.exists() else p
    ordered = []
    for d in '123':
        if groups[d] is None:
            raise SystemExit(f"Could not find POV folder ending with '{d}' under {base}")
        m = groups[d]
        if m.is_dir() and m.name != 'masks':
            mm = m / 'masks'
            if mm.exists():
                m = mm
        ordered.append(m)
    return ordered


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


def detect_largest_centroid(mask: np.ndarray, min_area: int = 20):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)


def least_squares_point_from_rays(origins, dirs):
    # Solve min || (I - d d^T)(X - o) ||^2 across rays (o,d)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for o, d in zip(origins, dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        I = np.eye(3)
        P = I - np.outer(d, d)
        A += P
        b += P @ o
    try:
        X = np.linalg.solve(A, b)
        return X
    except np.linalg.LinAlgError:
        return None


def project_point_to_pixel(P, cam, width, height):
    # Camera basis
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)
    rel = P - cam["pos"]
    pc = R.T @ rel  # camera space
    if pc[2] <= 1e-6:
        return None
    focal = (width * 0.5) / math.tan(cam["fov"] * 0.5)
    x = (pc[0] * focal) / pc[2]
    y = (pc[1] * focal) / pc[2]
    u = x + width * 0.5
    v = -y + height * 0.5
    return np.array([u, v], dtype=np.float64)


# --- New: local voxel scoring by projecting into masks ---
def score_voxel_local(ix: int, iy: int, iz: int, grid_min: np.ndarray, voxel_size: float,
                      cams, masks_this_frame, width: int, height: int) -> tuple:
    # Convert voxel index to world coords (center)
    x = grid_min[0] + (ix + 0.5) * voxel_size
    y = grid_min[1] + (iy + 0.5) * voxel_size
    z = grid_min[2] + (iz + 0.5) * voxel_size
    P = np.array([x, y, z], dtype=np.float64)
    support = 0
    intensity_sum = 0
    for k in range(3):
        mk = masks_this_frame[k]
        if mk is None:
            continue
        uv = project_point_to_pixel(P, cams[k], width, height)
        if uv is None:
            continue
        u = int(round(uv[0])); v = int(round(uv[1]))
        if u < 0 or v < 0 or u >= mk.shape[1] or v >= mk.shape[0]:
            continue
        val = int(mk[v, u])
        if val > 0:
            support += 1
            intensity_sum += val
    return support, intensity_sum, (x, y, z)


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

    t_delta_x = 1.0 * (voxel_size / (abs(ray_dir[0]) + 1e-30))
    t_delta_y = 1.0 * (voxel_size / (abs(ray_dir[1]) + 1e-30))
    t_delta_z = 1.0 * (voxel_size / (abs(ray_dir[2]) + 1e-30))

    t_current = tmin
    voxels = []
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


def load_voxel_grid_bin(bin_path: Path):
    with open(bin_path, "rb") as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        vox_size = np.frombuffer(f.read(4), dtype=np.float32)[0]
        data = np.frombuffer(f.read(N * N * N * 4), dtype=np.float32)
        grid = data.reshape((N, N, N))
    return grid, float(vox_size)


def main():
    ap = argparse.ArgumentParser(description="Extract per-frame trajectory from masks using volumetric intersection; writes track.json")
    ap.add_argument("--scene", required=True, help="Scene name under mask/output (e.g., A1, B1)")
    ap.add_argument("--mask-root", default=str(Path(__file__).resolve().parents[2] / "mask" / "output"), help="Root of mask/output")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "render" / "cam_config.json"), help="Camera config JSON")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "output"), help="Output root directory (per-scene subfolder)")
    ap.add_argument("--grid-json", default=None, help="Optional path to voxel_grid.json (otherwise inferred from out-root/scene)")
    ap.add_argument("--grid-size", type=int, default=None, help="Override grid size N (default: from voxel_grid.json)")
    ap.add_argument("--voxel-size", type=float, default=None, help="Override voxel size (default: from voxel_grid.json)")
    ap.add_argument("--grid-center", default=None, help="Override grid center 'x,y,z' (default: from voxel_grid.json)")
    ap.add_argument("--pixel-stride", type=int, default=2, help="Subsample stride for mask pixels (>=1)")
    ap.add_argument("--use-global-prior", action="store_true", help="Use global voxel_grid as a prior to stabilize per-frame argmax")
    ap.add_argument("--prior-weight", type=float, default=0.4, help="Weight [0..1] blending frame accumulator with normalized global prior")
    ap.add_argument("--temporal-lock", action="store_true", help="Prefer peaks near last frame's position (enforces continuity)")
    ap.add_argument("--local-radius", type=int, default=3, help="Neighborhood radius (voxels) for local search and centroid")
    ap.add_argument("--seed-triangulation", action="store_true", help="Seed first frame from 3-view triangulation of mask centroids")
    ap.add_argument("--tri-thresh", type=float, default=4.0, help="Max reprojection error (pixels) to accept triangulation seed")
    args = ap.parse_args()

    scene = args.scene
    mask_root = Path(args.mask_root)
    out_dir = Path(args.out_root) / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve grid params
    meta_path = Path(args.grid_json) if args.grid_json else (out_dir / "voxel_grid.json")
    if not meta_path.exists():
        raise SystemExit(f"voxel_grid.json not found: {meta_path}")
    meta = json.load(open(meta_path))
    N = int(args.grid_size if args.grid_size is not None else meta.get("grid_size", 300))
    voxel_size = float(args.voxel_size if args.voxel_size is not None else meta.get("voxel_size", 1.0))
    if args.grid_center is not None:
        grid_center = tuple(float(x) for x in args.grid_center.split(','))
    else:
        gc = meta.get("grid_center", [0, 0, 300])
        grid_center = (float(gc[0]), float(gc[1]), float(gc[2]))

    # Optional global prior
    global_prior = None
    if args.use_global_prior:
        bin_path = out_dir / "voxel_grid.bin"
        if bin_path.exists():
            global_prior, _ = load_voxel_grid_bin(bin_path)
        elif (out_dir / "voxel_grid.npy").exists():
            global_prior = np.load(out_dir / "voxel_grid.npy")
        if global_prior is not None:
            if global_prior.shape[0] != N:
                # resize by simple interpolation along axes if needed
                # fallback: normalize and ignore size mismatch
                mx = float(global_prior.max() + 1e-12)
                global_prior = (global_prior / mx).astype(np.float32)
            gp_max = float(global_prior.max() + 1e-12)
            global_prior = (global_prior / gp_max).astype(np.float32)

    print(f"[info] Grid: N={N}, voxel={voxel_size}m, center={grid_center}; prior={'on' if global_prior is not None else 'off'}")

    # Cameras and masks
    all_cams = load_cam_config(args.config)
    cams = select_cams_by_scene(all_cams, scene)
    mask_folders = find_mask_folders_for_scene(mask_root, scene)
    print("[info] Mask folders (L,M,R):")
    for p in mask_folders:
        print("  ", p)

    mask_paths = [list_mask_images(f) for f in mask_folders]
    min_len = min(len(x) for x in mask_paths)
    # Dimensions from first image
    sample = cv2.imread(mask_paths[0][0], cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise SystemExit("Could not read sample mask image")
    height, width = sample.shape[:2]

    print(f"[info] Extracting trajectory over {min_len} frames (stride={args.pixel_stride})")

    half = 0.5 * (N * voxel_size)
    grid_min = np.array(grid_center, dtype=np.float64) - half

    def voxel_to_world(ix, iy, iz):
        x = grid_min[0] + (ix + 0.5) * voxel_size
        y = grid_min[1] + (iy + 0.5) * voxel_size
        z = grid_min[2] + (iz + 0.5) * voxel_size
        return float(x), float(y), float(z)

    track = []
    last_vox = None  # (ix, iy, iz)
    last_pct = -1
    for i in range(min_len):
        if min_len > 0:
            pct = int((i * 100) / min_len)
            if pct % 5 == 0 and pct != last_pct:
                print(f"[progress] {pct}% ({i}/{min_len})")
                last_pct = pct
        # Seed first frame via triangulation of centroids (optional)
        if i == 0 and args.seed_triangulation:
            cents = []
            for k in range(3):
                mk = cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE)
                c = detect_largest_centroid(mk) if mk is not None else None
                cents.append(c)
            if all(c is not None for c in cents):
                rays = [pixel_ray(cams[k], width, height, cents[k][0], cents[k][1]) for k in range(3)]
                origins = [cams[k]["pos"] for k in range(3)]
                X = least_squares_point_from_rays(origins, rays)
                if X is not None:
                    # gate by reprojection error
                    errs = []
                    for k in range(3):
                        uv = project_point_to_pixel(X, cams[k], width, height)
                        if uv is None:
                            errs.append(1e9)
                        else:
                            errs.append(np.linalg.norm(uv - np.array(cents[k])))
                    if max(errs) < float(args.tri_thresh):
                        rel = (X - grid_min) / voxel_size
                        ix0 = int(np.clip(rel[0], 0, N - 1))
                        iy0 = int(np.clip(rel[1], 0, N - 1))
                        iz0 = int(np.clip(rel[2], 0, N - 1))
                        last_vox = (ix0, iy0, iz0)

        # Load current masks once
        masks_this = [cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE) for k in range(3)]
        # Determine base voxel to search around
        base_vox = last_vox
        if base_vox is None:
            # fallback: use simple argmax of a single-frame quick accumulator to get a coarse seed
            acc = np.zeros((N, N, N), dtype=np.float32)
            for k in range(3):
                m = masks_this[k]
                if m is None:
                    continue
                ys = np.arange(0, m.shape[0], max(1, args.pixel_stride))
                xs = np.arange(0, m.shape[1], max(1, args.pixel_stride))
                for yy in ys:
                    row = m[yy]
                    if row.max() == 0:
                        continue
                    for xx in xs:
                        if row[xx] == 0:
                            continue
                        d = pixel_ray(cams[k], width, height, float(xx), float(yy))
                        vox_idx = dda_voxel_indices(cams[k]["pos"], d, N, voxel_size, grid_center)
                        for (ix, iy, iz) in vox_idx:
                            acc[iz, iy, ix] += 1.0
            flat_idx = int(np.argmax(acc))
            if acc.ravel()[flat_idx] > 0:
                iz = flat_idx // (N * N); rem = flat_idx % (N * N)
                iy = rem // N; ix = rem % N
                base_vox = (ix, iy, iz)

        # Local search around base voxel using projection voting
        if base_vox is None:
            # no evidence
            continue
        r = max(1, int(args.local_radius))
        bix, biy, biz = base_vox
        best = (-1, -1, (None, None, None), (None, None, None))  # (support, intensity, (x,y,z), (ix,iy,iz))
        for iz in range(max(0, biz - r), min(N - 1, biz + r) + 1):
            for iy in range(max(0, biy - r), min(N - 1, biy + r) + 1):
                for ix in range(max(0, bix - r), min(N - 1, bix + r) + 1):
                    support, inten, world = score_voxel_local(ix, iy, iz, grid_min, voxel_size, cams, masks_this, width, height)
                    if support > best[0] or (support == best[0] and inten > best[1]):
                        best = (support, inten, world, (ix, iy, iz))
        if best[0] <= 0:
            # fallback: keep base_vox
            ix, iy, iz = base_vox
            x = grid_min[0] + (ix + 0.5) * voxel_size
            y = grid_min[1] + (iy + 0.5) * voxel_size
            z = grid_min[2] + (iz + 0.5) * voxel_size
        else:
            (x, y, z) = best[2]
            (ix, iy, iz) = best[3]
        last_vox = (ix, iy, iz)
        track.append({"t": i, "x": float(x), "y": float(y), "z": float(z)})

    out_json = out_dir / "track.json"
    with open(out_json, 'w') as f:
        json.dump(track, f, indent=2)
    print(f"[ok] Wrote {out_json} with {len(track)} points")


if __name__ == "__main__":
    main()


