#!/usr/bin/env python3
import os, json, glob
import numpy as np
import cv2
from pathlib import Path
import argparse


def load_cam_config(path: str):
    cfg = json.load(open(path))
    cams = []
    for c in cfg["cameras"]:
        pos = np.array(c["position"], dtype=np.float64)
        fwd = np.array(c["forward"], dtype=np.float64)
        up  = np.array(c["up"], dtype=np.float64)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        up = up / (np.linalg.norm(up) + 1e-9)
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, fwd)
        fov_deg = float(c.get("fov_deg", 70.0))
        cams.append({
            "name": c.get("name"),
            "pos": pos,
            "fwd": fwd,
            "right": right,
            "up": up,
            "fov": np.deg2rad(fov_deg),
        })
    return cams


def select_cams_by_names(all_cams, names):
    name_to_cam = {cam.get("name"): cam for cam in all_cams}
    selected = []
    for n in names:
        cam = name_to_cam.get(n)
        if cam is None:
            raise SystemExit(f"Camera '{n}' not found in config")
        selected.append(cam)
    return selected


def largest_component_centroid(mask: np.ndarray, min_area: int = 50, close_kernel: int = 5):
    # Optional morphological closing to merge motion lobes
    if close_kernel and close_kernel > 1:
        k = max(1, int(close_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest area contour above threshold
    best = None
    best_area = 0.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area >= min_area and area > best_area:
            best = c
            best_area = area
    if best is None:
        return None
    M = cv2.moments(best)
    if M.get("m00", 0.0) == 0.0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy


def pixel_ray(cam, width, height, px, py):
    cx = width * 0.5
    cy = height * 0.5
    focal = (width * 0.5) / np.tan(cam["fov"] * 0.5)
    x = (px - cx)
    y = (py - cy)
    z = focal
    dir_cam = np.array([x, -y, z], dtype=np.float64)
    dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-9)
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)
    dir_world = R @ dir_cam
    dir_world = dir_world / (np.linalg.norm(dir_world) + 1e-9)
    return dir_world


def project_point(cam, width: int, height: int, P_world: np.ndarray):
    # Build rotation with columns [right, up, fwd]; world->cam is R^T
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)
    v = P_world - cam["pos"]
    v_cam = R.T @ v
    z = float(v_cam[2])
    if z <= 1e-6:
        return None  # behind camera or at plane
    focal = (width * 0.5) / np.tan(cam["fov"] * 0.5)
    cx = width * 0.5
    cy = height * 0.5
    px = cx + (v_cam[0] / z) * focal
    py = cy - (v_cam[1] / z) * focal
    return float(px), float(py)


def reprojection_error(cams, width: int, height: int, centroids, P_world: np.ndarray):
    # Returns RMS pixel error across available centroids
    errs = []
    for cam, c in zip(cams, centroids):
        if c is None:
            continue
        pp = project_point(cam, width, height, P_world)
        if pp is None:
            return float("inf")
        ex = pp[0] - c[0]
        ey = pp[1] - c[1]
        errs.append(ex * ex + ey * ey)
    if not errs:
        return float("inf")
    return float(np.sqrt(np.mean(errs)))


def tri_intersection(cams, rays):
    A = np.zeros((3,3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for cam, d in zip(cams, rays):
        d = d / (np.linalg.norm(d) + 1e-9)
        I = np.eye(3)
        A += I - np.outer(d, d)
        b += (I - np.outer(d, d)) @ cam["pos"]
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return P


def load_masks(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return files


def reconstruct_from_masks(mask_folders, config_path, out_dir: Path, select_names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Using camera config: {config_path}")
    all_cams = load_cam_config(str(config_path))
    if select_names is not None:
        cams = select_cams_by_names(all_cams, select_names)
    else:
        if len(all_cams) != 3:
            raise SystemExit("Need exactly 3 cameras in config or provide select_names")
        cams = all_cams

    print("[info] Mask folders:")
    for idx, mf in enumerate(mask_folders):
        print(f"  - cam{idx}: {mf}")

    mask_paths = [load_masks(str(mf)) for mf in mask_folders]
    min_len = min(len(x) for x in mask_paths)
    if min_len == 0:
        raise SystemExit("No masks found.")

    for idx, paths in enumerate(mask_paths):
        print(f"[info] cam{idx} frames: {len(paths)}")
    print(f"[info] Frames to process (synced): {min_len}")

    track = []
    last_pct = -1
    for i in range(min_len):
        centroids = []
        dims = None
        for k in range(3):
            m = cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE)
            if m is None:
                centroids.append(None)
                continue
            if dims is None:
                dims = (m.shape[1], m.shape[0])
            # robust centroid from largest connected component
            c = largest_component_centroid(m, min_area=50, close_kernel=5)
            centroids.append(c)
        if any(c is None for c in centroids):
            continue
        width, height = dims
        rays = []
        for k in range(3):
            px, py = centroids[k]
            d = pixel_ray(cams[k], width, height, px, py)
            rays.append(d)
        P = tri_intersection(cams, rays)
        # Reprojection gating
        accept = False
        if P is not None:
            err = reprojection_error(cams, width, height, centroids, P)
            if err <= 4.0:  # pixels
                accept = True
        if not accept:
            # best-pair fallback among camera pairs
            best_P = None
            best_err = float("inf")
            pairs = [(0, 1), (0, 2), (1, 2)]
            for a, b in pairs:
                P2 = tri_intersection([cams[a], cams[b]], [rays[a], rays[b]])
                if P2 is None:
                    continue
                e2 = reprojection_error(cams, width, height, centroids, P2)
                if e2 < best_err:
                    best_err = e2
                    best_P = P2
            if best_P is not None and best_err <= 4.0:
                P = best_P
                accept = True
        if not accept:
            continue
        track.append({"t": i, "x": float(P[0]), "y": float(P[1]), "z": float(P[2])})

        # progress every ~5%
        pct = int(((i + 1) * 100) / max(1, min_len))
        if pct % 5 == 0 and pct != last_pct:
            print(f"[progress] {pct}% ({i+1}/{min_len})")
            last_pct = pct

    out_json = out_dir / "track.json"
    json.dump(track, open(out_json, "w"), indent=2)
    print(f"[ok] Wrote {out_json} with {len(track)} points")
    return out_json


def find_mask_folders_for_scene(scene: str, mask_root: Path):
    base = mask_root / scene
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Scene masks not found at {base}")

    preferred = [base / "1" / "masks", base / "2" / "masks", base / "3" / "masks"]
    if all(p.exists() and p.is_dir() for p in preferred):
        print(f"[info] Using preferred mask layout under {base}")
        return preferred

    # Otherwise, pick first three subdirs that contain a 'masks' folder; sort by name for stable order
    candidates = []
    for child in sorted([p for p in base.iterdir() if p.is_dir()]):
        m = child / "masks"
        if m.exists() and m.is_dir():
            candidates.append(m)
    if len(candidates) < 3:
        raise SystemExit(f"Expected at least 3 mask folders under {base}")
    print(f"[info] Using fallback mask layout; selected first 3 mask folders under {base}")
    return candidates[:3]


def reconstruct_scene(scene: str, mask_root: Path, config_path: Path, out_dir: Path):
    print(f"[info] Reconstructing scene '{scene}'")
    mask_folders = find_mask_folders_for_scene(scene, mask_root)
    # Derive setup number from scene like 'A1','B2' etc.
    setup_num = None
    if scene and len(scene) >= 2 and scene[1:].isdigit():
        setup_num = int(scene[1:])
    select_names = None
    if setup_num in (1, 2, 3):
        select_names = [f"setup{setup_num}_{i}" for i in (1, 2, 3)]
        print(f"[info] Selecting cameras: {select_names}")
    return reconstruct_from_masks(mask_folders, config_path, out_dir, select_names=select_names)


def main():
    ap = argparse.ArgumentParser(description="Reconstruct a 3D track from three mask folders and camera config")
    ap.add_argument("--scene", help="Scene name under mask/output (e.g., A1, B1)")
    ap.add_argument("--mask-root", default="mask/output", help="Root folder containing mask outputs")
    ap.add_argument("--masks", nargs=3, help="Explicit three mask folders (overrides --scene)")
    ap.add_argument("--config", help="cam_config.json with exactly 3 cameras")
    ap.add_argument("--out", default="render/output", help="Output folder (or folder per scene)")
    args = ap.parse_args()

    render_dir = Path(__file__).resolve().parent
    project_root = render_dir.parent
    default_config = project_root / "render/cam_config.json"
    config_path = Path(args.config) if args.config else default_config

    if args.masks:
        mask_folders = [Path(m) for m in args.masks]
        out_dir = Path(args.out)
        print("[info] Reconstructing from explicit mask folders")
        # If config has more than 3 cameras, no selection is applied here; pass select_names=None
        reconstruct_from_masks(mask_folders, config_path, out_dir, select_names=None)
        return

    if not args.scene:
        raise SystemExit("Provide --scene or --masks")

    mask_root = Path(args.mask_root)
    out_dir = Path(args.out) / args.scene
    reconstruct_scene(args.scene, mask_root, config_path, out_dir)


if __name__ == "__main__":
    main()