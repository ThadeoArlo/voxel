#!/usr/bin/env python3
import os
import json
import glob
import argparse
from typing import List, Dict, Any


def build_entries(images_dir: str, pattern: str, camera_index: int,
                  camera_position: List[float], yaw: float, pitch: float, roll: float,
                  fov_degrees: float, prefix: str = "") -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(images_dir, pattern)))
    entries = []
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        # allow optional prefix for different cameras
        entries.append({
            "camera_index": int(camera_index),
            "frame_index": int(i),
            "camera_position": [float(camera_position[0]), float(camera_position[1]), float(camera_position[2])],
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "fov_degrees": float(fov_degrees),
            "image_file": f"{prefix}{fname}" if prefix else fname
        })
    return entries


def main():
    ap = argparse.ArgumentParser(description="Create metadata.json for ray_voxel.cpp from image frames and simple pose inputs")
    ap.add_argument("images_dir", help="Directory containing image frames")
    ap.add_argument("--pattern", default="*.png", help="Glob pattern for frames (default: *.png)")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--pos", nargs=3, type=float, default=[0, 0, 0], metavar=("X", "Y", "Z"), help="Camera position meters")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=60.0, help="Horizontal FOV in degrees")
    ap.add_argument("--prefix", default="", help="Optional image name prefix written into image_file")
    ap.add_argument("--out", default="metadata.json")
    args = ap.parse_args()

    entries = build_entries(
        images_dir=args.images_dir,
        pattern=args.pattern,
        camera_index=args.camera_index,
        camera_position=args.pos,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        fov_degrees=args.fov,
        prefix=args.prefix,
    )

    if not entries:
        raise SystemExit("No images matched. Check images_dir and pattern.")

    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {args.out} with {len(entries)} entries")


if __name__ == "__main__":
    main()


