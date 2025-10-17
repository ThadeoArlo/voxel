#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import numpy as np
import math
import cv2
import argparse


# Local tools
from tools.extract_frames import extract_frames
from tools.make_metadata import build_entries


PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_dirs():
    (PROJECT_ROOT / "motionimages").mkdir(exist_ok=True)
    (PROJECT_ROOT / "screenshots").mkdir(exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)


def compile_ray_voxel() -> Path:
    """Compile ray_voxel.cpp using clang++ with Homebrew includes if available."""
    src = PROJECT_ROOT / "ray_voxel.cpp"
    out = PROJECT_ROOT / "ray_voxel"
    if not src.exists():
        raise SystemExit("ray_voxel.cpp not found")

    # Pick compiler
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        raise SystemExit("No C++ compiler found (need clang++ or g++)")

    # Homebrew prefix for Apple Silicon
    brew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    include_dirs = [f"-I{brew_prefix}/include", f"-I{brew_prefix}/include/stb"]
    # Also search local vendored headers: third_party/{stb,nlohmann}
    include_dirs += [f"-I{PROJECT_ROOT/'third_party'}",
                     f"-I{PROJECT_ROOT/'third_party/stb'}",
                     f"-I{PROJECT_ROOT/'third_party/nlohmann'}"]

    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        str(src),
        *include_dirs,
        "-o",
        str(out),
    ]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


def run_ray_voxel(exe: Path, metadata: Path, images_dir: Path, out_bin: Path):
    cmd = [str(exe), str(metadata), str(images_dir), str(out_bin)]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

 
def ray_aabb_intersection(ray_origin, ray_dir, box_min, box_max):
    tmin = -float("inf")
    tmax = float("inf")
    for i in range(3):
        if abs(ray_dir[i]) > 1e-8:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
        else:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None
    return max(0.0, tmin), tmax


def python_voxelize_single_camera(meta_path: Path, images_dir: Path, out_bin: Path):
    with open(meta_path) as f:
        entries = json.load(f)
    # sort by frame_index
    entries.sort(key=lambda e: (e.get("camera_index", 0), e.get("frame_index", 0)))
    if len(entries) < 2:
        raise SystemExit("Need at least 2 frames for motion detection")

    # Load first image to get size
    first_path = str(images_dir / entries[0]["image_file"])
    im0 = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if im0 is None:
        raise SystemExit(f"Failed to read {first_path}")
    h, w = im0.shape[:2]

    # Grid params (smaller N for speed in Python)
    N = 256
    voxel_size = 6.0
    grid_center = np.array([0.0, 0.0, 500.0], dtype=np.float64)
    half = 0.5 * (N * voxel_size)
    box_min = grid_center - half
    box_max = grid_center + half

    voxel = np.zeros((N, N, N), dtype=np.float32)

    # Camera intrinsics from FOV
    fov_deg = float(entries[0].get("fov_degrees", 60.0))
    fov_rad = math.radians(fov_deg)
    focal_len = (w * 0.5) / math.tan(fov_rad * 0.5)
    cx = w * 0.5
    cy = h * 0.5

    def dir_for_uv(u, v):
        x = (u - cx)
        y = -(v - cy)
        z = -focal_len
        vec = np.array([x, y, z], dtype=np.float64)
        n = np.linalg.norm(vec)
        if n == 0:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return vec / n

    # Process consecutive frames
    prev = im0.astype(np.float32)
    for i in range(1, len(entries)):
        path = str(images_dir / entries[i]["image_file"])
        cur = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if cur is None:
            continue
        cur = cur.astype(np.float32)

        diff = cv2.absdiff(prev, cur)
        # Threshold for motion; tune if needed
        _, mask = cv2.threshold(diff, 5.0, 255.0, cv2.THRESH_BINARY)
        ys, xs = np.where(mask > 0)

        for (v, u) in zip(ys, xs):
            ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            ray_dir = dir_for_uv(float(u), float(v))
            t_pair = ray_aabb_intersection(ray_origin, ray_dir, box_min, box_max)
            if t_pair is None:
                continue
            t0, t1 = t_pair
            if t1 <= t0:
                continue
            step = voxel_size  # coarse step
            t = t0
            while t <= t1:
                p = ray_origin + t * ray_dir
                # Convert to voxel indices
                xn = (p[0] - box_min[0]) / voxel_size
                yn = (p[1] - box_min[1]) / voxel_size
                zn = (p[2] - box_min[2]) / voxel_size
                xi = int(xn)
                yi = int(yn)
                zi = int(zn)
                if 0 <= xi < N and 0 <= yi < N and 0 <= zi < N:
                    voxel[zi, yi, xi] += float(diff[v, u])
                t += step

        prev = cur

    # Write voxel_grid.bin
    with open(out_bin, "wb") as f:
        np.array([N], dtype=np.int32).tofile(f)
        np.array([voxel_size], dtype=np.float32).tofile(f)
        voxel.astype(np.float32).ravel().tofile(f)


def save_voxel_top_points(voxel_bin: Path, out_json: Path, top_n: int = 500):
    with open(voxel_bin, "rb") as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        voxel_size = np.frombuffer(f.read(4), dtype=np.float32)[0]
        data = np.frombuffer(f.read(N*N*N*4), dtype=np.float32)
        grid = data.reshape((N, N, N))

    flat = grid.ravel()
    idx = np.argpartition(flat, -top_n)[-top_n:]
    mask = np.zeros_like(grid, dtype=bool)
    mask.ravel()[idx] = True
    coords = np.argwhere(mask)
    vals = grid[mask]

    # world coords assuming grid centered at origin
    half = (N * voxel_size) * 0.5
    z = coords[:, 0] + 0.5
    y = coords[:, 1] + 0.5
    x = coords[:, 2] + 0.5
    xw = -half + x * voxel_size
    yw = -half + y * voxel_size
    zw = -half + z * voxel_size
    pts = np.column_stack([xw, yw, zw])

    order = np.argsort(-vals)
    out = [
        {"x": float(pts[i, 0]), "y": float(pts[i, 1]), "z": float(pts[i, 2]), "v": float(vals[i])}
        for i in order
    ]
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] Wrote {out_json}")


def save_voxel_screenshot(voxel_bin: Path, out_png: Path):
    import pyvista as pv

    with open(voxel_bin, "rb") as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        voxel_size = np.frombuffer(f.read(4), dtype=np.float32)[0]
        data = np.frombuffer(f.read(N*N*N*4), dtype=np.float32)
        grid = data.reshape((N, N, N))

    # pick top percentile for visualization
    thresh = float(np.percentile(grid.ravel(), 99.7))
    coords = np.argwhere(grid > thresh)
    if coords.size == 0:
        print("[warn] No voxels above threshold to display; skipping screenshot")
        return
    intensities = grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    # world coords (z,y,x -> x,y,z)
    half = (N * voxel_size) * 0.5
    z = coords[:, 0] + 0.5
    y = coords[:, 1] + 0.5
    x = coords[:, 2] + 0.5
    xw = -half + x * voxel_size
    yw = -half + y * voxel_size
    zw = -half + z * voxel_size
    points = np.column_stack([xw, yw, zw])

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    cloud = pv.PolyData(points)
    cloud["intensity"] = intensities
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.15,
    )
    plotter.add_scalar_bar(title="Brightness", n_labels=5)
    plotter.show(window_size=[1920, 1080], auto_close=True, screenshot=str(out_png))
    print(f"[ok] Saved screenshot to {out_png}")


def main():
    ap = argparse.ArgumentParser(description="One-shot voxelizer: extract frames, build metadata, run voxel grid, output JSON+PNG")
    ap.add_argument("--videos", nargs="*", default=[], help="Input videos (e.g., 1.mp4 2.mp4)")
    ap.add_argument("--headings", nargs="*", type=float, default=None, help="Compass headings (degrees from North) for each video, same order as --videos")
    ap.add_argument("--baseline", type=float, default=3.0, help="Meters between consecutive cameras along +x")
    ap.add_argument("--heights", nargs="*", type=float, default=None, help="Per-camera heights (meters). Default 1.5 for all")
    ap.add_argument("--pitch", type=float, default=0.0, help="Pitch (deg). Use negative for tilting up if cloud appears upside down")
    ap.add_argument("--fov", type=float, default=60.0, help="Horizontal field of view in degrees")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame during extraction")
    args = ap.parse_args()

    ensure_dirs()
    images_dir = PROJECT_ROOT / "motionimages"

    videos = args.videos
    if not videos:
        # default to single 1.mov if no args provided
        v = PROJECT_ROOT / "1.mov"
        if not v.exists():
            raise SystemExit("Provide --videos or place 1.mov in project root")
        videos = [str(v)]

    # 1) Extract frames for all videos
    print("[step] Extracting frames → motionimages/")
    prefixes = []
    for i, vp in enumerate(videos):
        prefix = f"cam{i}"
        prefixes.append(prefix)
        n = extract_frames(vp, str(images_dir), prefix=prefix, stride=max(1, args.stride), start=0, end=-1)
        print(f"[ok] {Path(vp).name}: {n} frames → {prefix}_*.png")

    # 2) Generate metadata.json from simple poses
    print("[step] Generating metadata.json")
    H = args.heights if args.heights else [1.5] * len(videos)
    if len(H) < len(videos):
        H = (H + [H[-1]] * len(videos))[:len(videos)]

    # Yaws: use headings if available, setting cam0 yaw=0 and camk = heading[k]-heading[0]
    if args.headings and len(args.headings) >= len(videos):
        base = args.headings[0]
        def wrap(a):
            x = (a + 180.0) % 360.0 - 180.0
            return x
        yaws = [wrap(h - base) for h in args.headings[:len(videos)]]
    else:
        yaws = [0.0] * len(videos)

    entries = []
    for i, prefix in enumerate(prefixes):
        pos = [args.baseline * i, H[i], 0.0]
        entries += build_entries(
            images_dir=str(images_dir),
            pattern=f"{prefix}_*.png",
            camera_index=i,
            camera_position=pos,
            yaw=float(yaws[i]),
            pitch=float(args.pitch),
            roll=0.0,
            fov_degrees=float(args.fov),
            prefix="",
        )

    meta_path = images_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[ok] Wrote {meta_path}")

    out_bin = PROJECT_ROOT / "voxel_grid.bin"
    # 3) Try C++ path, else Python fallback (single camera only)
    try:
        print("[step] Compiling ray_voxel.cpp")
        exe = compile_ray_voxel()
        print(f"[ok] Built {exe}")
        print("[step] Running ray-voxel to produce voxel_grid.bin")
        run_ray_voxel(exe, meta_path, images_dir, out_bin)
        if not out_bin.exists():
            raise RuntimeError("voxel_grid.bin not produced by C++ path")
        print("[ok] voxel_grid.bin ready")
    except subprocess.CalledProcessError:
        if len(videos) > 1:
            raise SystemExit("C++ build failed and Python fallback supports only single-camera. Vendor headers per README and retry.")
        print("[warn] C++ build failed (likely missing stb). Falling back to slower Python voxelizer…")
        python_voxelize_single_camera(meta_path, images_dir, out_bin)
        print("[ok] voxel_grid.bin ready (python)")

    # 5) Save results
    print("[step] Writing top points JSON")
    save_voxel_top_points(out_bin, PROJECT_ROOT / "outputs" / "voxel_top_points.json", top_n=500)

    print("[step] Rendering screenshot")
    save_voxel_screenshot(out_bin, PROJECT_ROOT / "screenshots" / "voxel_app.png")

    print("[done] All outputs ready:")
    print(" - voxel_grid.bin")
    print(" - outputs/voxel_top_points.json")
    print(" - screenshots/voxel_app.png (if enough voxels)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)



