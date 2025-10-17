#!/usr/bin/env python3
import os
import cv2
import argparse
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, prefix: str, stride: int, start: int, end: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    end_frame = total - 1 if end < 0 else min(end, total - 1)
    start_frame = max(0, start)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    idx = -1
    written = 0
    while True:
        ok, frame = cap.read()
        idx += 1
        if not ok:
            break
        if idx < start_frame:
            continue
        if idx > end_frame:
            break
        if stride > 1 and ((idx - start_frame) % stride != 0):
            continue
        out_name = f"{prefix}_{idx:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)
        written += 1

    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser(description="Extract frames from a video to a folder")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("outdir", help="Output directory for frames")
    ap.add_argument("--prefix", default="frame", help="Output filename prefix")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    ap.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="End frame index (inclusive, -1 = till end)")
    args = ap.parse_args()

    n = extract_frames(args.video, args.outdir, args.prefix, max(1, args.stride), args.start, args.end)
    print(f"Wrote {n} frames to {args.outdir}")


if __name__ == "__main__":
    main()


