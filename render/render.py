#!/usr/bin/env python3
import os, json, glob
import re
# --- Default settings (edit here for project-wide defaults) ---
DEFAULT_MIN_AREA = 50
DEFAULT_CLOSE_KERNEL = 5
DEFAULT_REPROJ_THRESH = 4.0
DEFAULT_PAIRWISE_REPROJ_THRESH = None  # None means use DEFAULT_REPROJ_THRESH
DEFAULT_LOG_STATS = True
DEFAULT_VERBOSE = True

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


def reconstruct_from_masks(mask_folders, config_path, out_path: Path, select_names=None,
                          min_area: int = DEFAULT_MIN_AREA,
                          close_kernel: int = DEFAULT_CLOSE_KERNEL,
                          reproj_thresh_px: float = DEFAULT_REPROJ_THRESH,
                          pairwise_reproj_thresh_px: float = DEFAULT_PAIRWISE_REPROJ_THRESH,
                          log_stats: bool = DEFAULT_LOG_STATS,
                          verbose: bool = DEFAULT_VERBOSE):
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    angle_stats = []  # collect per-frame pairwise angles if logging
    rms_stats = []
    n_frames = min_len
    n_missing = 0
    n_rejected = 0
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
            # robust centroid from largest connected component (tunable)
            c = largest_component_centroid(m, min_area=int(min_area), close_kernel=int(close_kernel))
            centroids.append(c)
        if any(c is None for c in centroids):
            n_missing += 1
            if verbose and (i % max(1, n_frames // 10) == 0):
                print(f"[debug] frame {i}: missing centroid(s) {centroids}")
            continue
        width, height = dims
        rays = []
        for k in range(3):
            px, py = centroids[k]
            d = pixel_ray(cams[k], width, height, px, py)
            rays.append(d)
        # log baseline angles if requested
        if log_stats:
            def ang(u, v):
                dn = max(1e-9, np.linalg.norm(u) * np.linalg.norm(v))
                cc = np.clip(float(np.dot(u, v)) / dn, -1.0, 1.0)
                return float(np.degrees(np.arccos(cc)))
            a01 = ang(rays[0], rays[1]); a02 = ang(rays[0], rays[2]); a12 = ang(rays[1], rays[2])
            angle_stats.append((a01, a02, a12))

        P = tri_intersection(cams, rays)
        # Reprojection gating
        accept = False
        if P is not None:
            err = reprojection_error(cams, width, height, centroids, P)
            if log_stats:
                rms_stats.append(err)
            if err <= float(reproj_thresh_px):  # pixels
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
                # compute error using only the two cameras contributing
                e2 = reprojection_error([cams[a], cams[b]], width, height, [centroids[a], centroids[b]], P2)
                if e2 < best_err:
                    best_err = e2
                    best_P = P2
            # choose threshold for pairwise if provided
            thr2 = float(pairwise_reproj_thresh_px) if pairwise_reproj_thresh_px is not None else float(reproj_thresh_px)
            if best_P is not None and best_err <= thr2:
                P = best_P
                accept = True
        if not accept:
            n_rejected += 1
            if verbose and (i % max(1, n_frames // 10) == 0):
                print(f"[debug] frame {i}: reprojection err too high ({err if 'err' in locals() else 'n/a'}), best_pair={best_err if 'best_err' in locals() else 'n/a'}")
            continue
        if log_stats:
            # pairwise angles between rays (degrees)
            def ang(u, v):
                dn = max(1e-9, np.linalg.norm(u) * np.linalg.norm(v))
                cc = np.clip(float(np.dot(u, v)) / dn, -1.0, 1.0)
                return float(np.degrees(np.arccos(cc)))
            a01 = ang(rays[0], rays[1]); a02 = ang(rays[0], rays[2]); a12 = ang(rays[1], rays[2])
            angle_stats.append((a01, a02, a12))
            rms_stats.append(err)
        track.append({"t": i, "x": float(P[0]), "y": float(P[1]), "z": float(P[2])})

        # progress every ~5%
        pct = int(((i + 1) * 100) / max(1, min_len))
        if pct % 5 == 0 and pct != last_pct:
            print(f"[progress] {pct}% ({i+1}/{min_len})")
            last_pct = pct

    out_json = out_path
    json.dump(track, open(out_json, "w"), indent=2)
    print(f"[ok] Wrote {out_json} with {len(track)} points")
    if verbose:
        print(f"[debug] frames: {n_frames}, missing: {n_missing}, rejected: {n_rejected}, accepted: {len(track)}")
    if log_stats and angle_stats:
        A = np.array(angle_stats, dtype=np.float64)
        rms = np.array(rms_stats, dtype=np.float64)
        a_min = A.min(axis=0); a_mean = A.mean(axis=0); a_max = A.max(axis=0)
        print(f"[stats] Pairwise ray angles (deg) min/mean/max: ")
        print(f"        0-1: {a_min[0]:.1f}/{a_mean[0]:.1f}/{a_max[0]:.1f}")
        print(f"        0-2: {a_min[1]:.1f}/{a_mean[1]:.1f}/{a_max[1]:.1f}")
        print(f"        1-2: {a_min[2]:.1f}/{a_mean[2]:.1f}/{a_max[2]:.1f}")
        print(f"[stats] RMS reprojection error (px) mean: {float(rms.mean()):.2f};  p50: {float(np.percentile(rms,50)):.2f}; p90: {float(np.percentile(rms,90)):.2f}")
    return out_json


def find_mask_folders_for_scene(scene: str, mask_root: Path):
    base = mask_root / scene
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Scene masks not found at {base}")

    # Prefer explicit POV digits 1/2/3 anywhere in the subfolder name (A21,A22,A23 etc.)
    subs = [p for p in base.iterdir() if p.is_dir()]
    def last_digit_key(p: Path):
        s = p.name.strip().lower()
        for ch in reversed(s):
            if ch in '123':
                return ch
        return None
    groups = {d: None for d in '123'}
    for p in subs:
        d = last_digit_key(p)
        if d in groups and groups[d] is None:
            m = p / 'masks'
            groups[d] = m if m.exists() else p
    ordered = []
    for d in '123':
        if groups[d] is None:
            continue
        m = groups[d]
        if m.is_dir() and m.name != 'masks':
            mm = m / 'masks'
            if mm.exists():
                m = mm
        ordered.append(m)
    if len(ordered) == 3:
        print(f"[info] Using POV-inferred mask layout under {base} (1,2,3)")
        return ordered

    # Fallback: first 3 subdirs that contain 'masks' folder, sorted by name
    candidates = []
    for child in sorted([p for p in subs if p.is_dir()]):
        m = child / 'masks'
        if m.exists() and m.is_dir():
            candidates.append(m)
    if len(candidates) < 3:
        raise SystemExit(f"Expected at least 3 mask folders under {base}")
    print(f"[info] Using fallback mask layout; selected first 3 mask folders under {base}")
    return candidates[:3]


def reconstruct_scene(scene: str, mask_root: Path, config_path: Path, out_root: Path,
                      min_area: int = DEFAULT_MIN_AREA,
                      close_kernel: int = DEFAULT_CLOSE_KERNEL,
                      reproj_thresh_px: float = DEFAULT_REPROJ_THRESH,
                      pairwise_reproj_thresh_px: float = DEFAULT_PAIRWISE_REPROJ_THRESH,
                      log_stats: bool = DEFAULT_LOG_STATS,
                      verbose: bool = DEFAULT_VERBOSE):
    print(f"[info] Reconstructing scene '{scene}'")
    mask_folders = find_mask_folders_for_scene(scene, mask_root)
    # Derive setup number from scene like 'A1', 'B2', or 'A1_1' (use first digit)
    setup_num = None
    if scene:
        m = re.search(r"(\d)", scene)
        if m:
            setup_num = int(m.group(1))
    select_names = None
    if setup_num in (1, 2, 3, 4):
        select_names = [f"setup{setup_num}_{i}" for i in (1, 2, 3)]
        print(f"[info] Selecting cameras: {select_names}")
    # Adaptive defaults for setup 2 if user didn't override
    if setup_num == 2 and (min_area == DEFAULT_MIN_AREA and close_kernel == DEFAULT_CLOSE_KERNEL and abs(reproj_thresh_px - DEFAULT_REPROJ_THRESH) < 1e-6):
        print("[info] Setup 2 detected; using adaptive defaults: --min-area 120 --close-kernel 7 --reproj-thresh 10.0 --pairwise-reproj-thresh 8.0")
        min_area = 120
        close_kernel = 7
        reproj_thresh_px = 10.0
        pairwise_reproj_thresh_px = 8.0
    out_path = out_root / f"{scene}.json"
    return reconstruct_from_masks(mask_folders, config_path, out_path, select_names=select_names,
                                  min_area=min_area, close_kernel=close_kernel,
                                  reproj_thresh_px=reproj_thresh_px, pairwise_reproj_thresh_px=pairwise_reproj_thresh_px,
                                  log_stats=log_stats, verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description="Reconstruct a 3D track from three mask folders and camera config")
    ap.add_argument("--scene", help="Scene name under mask/output (e.g., A1, B1)")
    ap.add_argument("--mask-root", default="mask/output", help="Root folder containing mask outputs")
    ap.add_argument("--masks", nargs=3, help="Explicit three mask folders (overrides --scene)")
    ap.add_argument("--config", help="cam_config.json with exactly 3 cameras")
    ap.add_argument("--out", default="render/output", help="Output folder (or folder per scene)")
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="Min area for largest-component centroid")
    ap.add_argument("--close-kernel", type=int, default=DEFAULT_CLOSE_KERNEL, help="Morphological close kernel (odd pixels)")
    ap.add_argument("--reproj-thresh", type=float, default=DEFAULT_REPROJ_THRESH, help="Reprojection error threshold in pixels (triple)")
    ap.add_argument("--pairwise-reproj-thresh", type=float, default=DEFAULT_PAIRWISE_REPROJ_THRESH, help="Reprojection threshold for pairwise fallback (defaults to triple)")
    ap.add_argument("--log-stats", action="store_true", default=DEFAULT_LOG_STATS, help="Print baseline angle and RMS reprojection stats")
    ap.add_argument("--verbose", action="store_true", default=DEFAULT_VERBOSE, help="Log per-frame rejection/missing diagnostics")
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
    out_root = Path(args.out)
    reconstruct_scene(args.scene, mask_root, config_path, out_root,
                      min_area=int(args.min_area), close_kernel=int(args.close_kernel),
                      reproj_thresh_px=float(args.reproj_thresh),
                      pairwise_reproj_thresh_px=(float(args.pairwise_reproj_thresh) if args.pairwise_reproj_thresh is not None else None),
                      log_stats=bool(args.log_stats), verbose=bool(args.verbose))


if __name__ == "__main__":
    main()