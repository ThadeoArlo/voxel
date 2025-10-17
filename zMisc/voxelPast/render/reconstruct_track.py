#!/usr/bin/env python3
import os, json, glob
import numpy as np
import cv2
from pathlib import Path


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
        cams.append({"pos":pos, "fwd":fwd, "right":right, "up":up, "fov":np.deg2rad(fov_deg)})
    return cams


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None
    return float(xs.mean()), float(ys.mean())


def pixel_ray(cam, width, height, px, py):
    cx = width * 0.5
    cy = height * 0.5
    focal = (width * 0.5) / np.tan(cam["fov"] * 0.5)
    x = (px - cx)
    y = (py - cy)
    z = focal
    dir_cam = np.array([x, -y, z], dtype=np.float64)
    dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-9)
    # camera basis: right, up, fwd
    R = np.stack([cam["right"], cam["up"], cam["fwd"]], axis=1)  # columns
    dir_world = R @ dir_cam
    dir_world = dir_world / (np.linalg.norm(dir_world) + 1e-9)
    return dir_world


def tri_intersection(cams, rays):
    # Least-squares closest point to 3 skew lines using linear algebra
    # minimize sum || P - (Ci + ti*Di) ||^2.
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


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Reconstruct a 3D flight path from three mask folders and camera config")
    ap.add_argument("--config", required=True, help="cam_config.json exported from simulation")
    ap.add_argument("--masks", nargs=3, required=True, help="Three mask folders (sim0, sim1, sim2)")
    ap.add_argument("--out", default="render/output", help="Output folder")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    cams = load_cam_config(args.config)
    if len(cams) != 3:
        raise SystemExit("Need exactly 3 cameras in config")

    mask_paths = [load_masks(mf) for mf in args.masks]
    min_len = min(len(x) for x in mask_paths)
    if min_len == 0:
        raise SystemExit("No masks found.")

    track = []
    for i in range(min_len):
        centroids = []
        dims = None
        for k in range(3):
            m = cv2.imread(mask_paths[k][i], cv2.IMREAD_GRAYSCALE)
            if m is None: centroids.append(None); continue
            if dims is None: dims = (m.shape[1], m.shape[0])
            c = mask_centroid(m)
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
        if P is None: continue
        track.append({"t": i, "x": float(P[0]), "y": float(P[1]), "z": float(P[2])})

    out_json = os.path.join(args.out, "track.json")
    json.dump(track, open(out_json, "w"), indent=2)
    print(f"[ok] Wrote {out_json} with {len(track)} points")


if __name__ == "__main__":
    main()


