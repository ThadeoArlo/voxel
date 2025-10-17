#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np


def load_voxel_grid(filename):
    with open(filename, "rb") as f:
        N = np.frombuffer(f.read(4), dtype=np.int32)[0]
        voxel_size = np.frombuffer(f.read(4), dtype=np.float32)[0]
        data = np.frombuffer(f.read(N*N*N*4), dtype=np.float32)
        grid = data.reshape((N, N, N))
    return grid, float(voxel_size)


def top_points(grid: np.ndarray, voxel_size: float, num: int = 100, percentile: float = None):
    flat = grid.ravel()
    if percentile is not None:
        thresh = np.percentile(flat, percentile)
        mask = grid >= thresh
    else:
        mask = np.zeros_like(grid, dtype=bool)
        idx = np.argpartition(flat, -num)[-num:]
        mask.ravel()[idx] = True
    coords = np.argwhere(mask)
    vals = grid[mask]
    # world coords assuming grid is centered at origin
    N = grid.shape[0]
    half = (N * voxel_size) * 0.5
    # z,y,x -> world x,y,z
    z = coords[:, 0] + 0.5
    y = coords[:, 1] + 0.5
    x = coords[:, 2] + 0.5
    xw = -half + x * voxel_size
    yw = -half + y * voxel_size
    zw = -half + z * voxel_size
    points = np.column_stack([xw, yw, zw])
    order = np.argsort(-vals)
    return points[order], vals[order]


def main():
    ap = argparse.ArgumentParser(description="Analyze voxel_grid.bin and output top points as JSON")
    ap.add_argument("voxel_bin", help="Path to voxel_grid.bin")
    ap.add_argument("--top", type=int, default=200, help="Top N points by intensity")
    ap.add_argument("--percentile", type=float, default=None, help="Percentile cutoff (overrides --top)")
    ap.add_argument("--out", default="voxel_top_points.json", help="Output JSON path")
    args = ap.parse_args()

    grid, vox = load_voxel_grid(args.voxel_bin)
    pts, vals = top_points(grid, vox, num=args.top, percentile=args.percentile)

    out = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "v": float(v)} for p, v in zip(pts, vals)]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} ({len(out)} points)")


if __name__ == "__main__":
    main()


