#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from pathlib import Path

# ============================
# Defaults (can be overridden via CLI)
# ============================
# If neither --input nor --scene are provided, these are used as a fallback.
# Prefer using --scene <name> to process all videos in src/<name>.
INPUT_VIDEOS = []
OUTPUT_SUBDIR_DEFAULT = "output"  # used only when no --scene provided
STRIDE = 1
THRESHOLD = 20  # lower default to be sensitive to small objects
BLUR = 3
ERODE_ITERS = 0
DILATE_ITERS = 0
DOWNSCALE = 1.0
WRITE_OVERLAY = True
BG_ALPHA = 0.02  # running average background update rate (0..1)
CLOSE_KERNEL = 5  # morphological closing kernel (odd); 0 to disable
KEEP_LARGEST = True  # keep only the largest connected component per frame

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_video_init(path: Path, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (width, height), True)


def process_video(
    input_path: str,
    out_root: Path,
    stride: int = 1,
    thresh: int = 25,
    blur: int = 3,
    erode_iter: int = 1,
    dilate_iter: int = 2,
    downscale: float = 1.0,
    write_overlay: bool = True,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    base = Path(input_path).stem
    out_dir = out_root / base
    frames_dir = out_dir / "frames"
    masks_dir = out_dir / "masks"
    ensure_dir(frames_dir)
    ensure_dir(masks_dir)

    prev_gray = None
    bg_f32 = None  # running background model (float32)
    idx_in = -1
    idx_kept = 0

    writer = None
    mask_writer = None
    overlay_path = out_dir / "motion_overlay.mp4"
    mask_video_path = out_dir / "motion_mask.mp4"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx_in += 1
        if stride > 1 and (idx_in % stride) != 0:
            continue

        # Progress logging (~every 5%) based on input frame index
        if total > 0:
            pct = int(((idx_in + 1) * 100) / total)
            # print only at 0,5,10,...,100
            if pct % 5 == 0:
                # avoid spamming the same percentage
                # store last printed pct on the out_dir marker file
                marker = out_dir / ".last_pct"
                last = -1
                if marker.exists():
                    try:
                        last = int(marker.read_text().strip() or "-1")
                    except Exception:
                        last = -1
                if pct != last:
                    print(f"[progress] {pct}% ({idx_in+1}/{total})")
                    try:
                        marker.write_text(str(pct))
                    except Exception:
                        pass

        if downscale != 1.0:
            frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur and blur > 0:
            k = max(1, int(blur))
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Save frame for reference
        cv2.imwrite(str(frames_dir / f"frame_{idx_kept:06d}.png"), frame)

        # Initialize background on first kept frame
        if bg_f32 is None:
            bg_f32 = gray.astype(np.float32)
            prev_gray = gray
            idx_kept += 1
            continue

        # Background subtraction (running average) to avoid double-lobes from prev-frame differencing
        gray_f32 = gray.astype(np.float32)
        diff = cv2.absdiff(gray_f32, bg_f32)
        # Threshold (fixed or Otsu if thresh<=0)
        if thresh is not None and thresh > 0:
            _, mask = cv2.threshold(diff.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to merge leading/trailing lobes into one solid blob
        if CLOSE_KERNEL and CLOSE_KERNEL > 1:
            ck = int(CLOSE_KERNEL)
            if ck % 2 == 0:
                ck += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Morphology cleanup
        if erode_iter > 0:
            mask = cv2.erode(mask, None, iterations=erode_iter)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, None, iterations=dilate_iter)

        # Keep only largest connected component (optional) to avoid interpreting split lobes as separate objects
        if KEEP_LARGEST:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best = max(contours, key=cv2.contourArea)
                mask_largest = np.zeros_like(mask)
                cv2.drawContours(mask_largest, [best], -1, 255, thickness=cv2.FILLED)
                mask = mask_largest

        # Save binary mask (white = motion)
        cv2.imwrite(str(masks_dir / f"mask_{idx_kept:06d}.png"), mask)

        # Optional overlay with contours
        if write_overlay:
            overlay = frame.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 25:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if writer is None:
                writer = write_video_init(overlay_path, overlay.shape[1], overlay.shape[0], fps / max(1, stride))
            writer.write(overlay)

        # Always write a black-and-white mask video for localization
        if mask_writer is None:
            # Convert to 3-channel for broad codec compatibility
            mask_writer = write_video_init(mask_video_path, mask.shape[1], mask.shape[0], fps / max(1, stride))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_writer.write(mask_bgr)

        # Update the running background
        cv2.accumulateWeighted(gray_f32, bg_f32, BG_ALPHA)
        prev_gray = gray
        idx_kept += 1

    cap.release()
    if writer is not None:
        writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"[done] Frames: {idx_kept}  out: {out_dir}")
    if write_overlay:
        print(f"[info] Overlay video: {overlay_path}")
    print(f"[info] Mask video: {mask_video_path}")


def main():
    # If CLI args provided, they override defaults; otherwise use hardcoded config
    ap = argparse.ArgumentParser(description="Extract frames and generate binary motion masks (white = moving pixels)")
    ap.add_argument("--input", help="Input video path (e.g., myvideo.mp4). Overrides --scene if provided.")
    ap.add_argument("--scene", help="Scene folder name under src (e.g., 'wide') to process all videos inside")
    ap.add_argument("--src", help="Root folder containing scene folders (defaults to <project_root>/src)", default=None)
    ap.add_argument("--out", help="Output root folder (defaults to mask/<scene> or mask/output)", default=None)
    ap.add_argument("--stride", type=int, help="Keep every Nth frame", default=None)
    ap.add_argument("--thresh", type=int, help="Binary threshold", default=None)
    ap.add_argument("--blur", type=int, help="Gaussian blur kernel size (odd); 0 to disable", default=None)
    ap.add_argument("--erode", type=int, help="Erode iterations", default=None)
    ap.add_argument("--dilate", type=int, help="Dilate iterations", default=None)
    ap.add_argument("--downscale", type=float, help="Resize factor (e.g., 0.5)", default=None)
    ap.add_argument("--no-overlay", action="store_true", help="Disable overlay video with contours")
    ap.add_argument("--bg-alpha", type=float, help="Exponential background update rate (0..1)", default=None)
    ap.add_argument("--close", type=int, help="Morphological closing kernel (odd); 0 to disable", default=None)
    ap.add_argument("--largest-only", action="store_true", help="Keep only largest connected component")
    ap.add_argument("--no-largest-only", action="store_true", help="Disable largest component filtering")
    args = ap.parse_args()

    mask_dir = Path(__file__).resolve().parent
    project_root = mask_dir.parent

    # Resolve output root. Scene outputs always live under mask/output/<scene>
    if args.out:
        out_root = Path(args.out)
    elif args.scene:
        out_root = mask_dir / OUTPUT_SUBDIR_DEFAULT / args.scene
    else:
        out_root = mask_dir / OUTPUT_SUBDIR_DEFAULT
    ensure_dir(out_root)

    # Resolve input videos
    videos = []
    if args.input:
        videos = [args.input]
    elif args.scene:
        src_root = Path(args.src) if args.src else (project_root / "src")
        scene_dir = src_root / args.scene
        if not scene_dir.exists() or not scene_dir.is_dir():
            raise SystemExit(f"Scene folder not found: {scene_dir}")
        allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
        videos = sorted([str(p) for p in scene_dir.iterdir() if p.suffix in allowed_exts])
        if not videos:
            raise SystemExit(f"No videos found in scene folder: {scene_dir}")
        print(f"[info] Processing scene '{args.scene}' with {len(videos)} video(s) from {scene_dir}")
    else:
        videos = INPUT_VIDEOS
        if not videos:
            raise SystemExit("No inputs provided. Use --scene <name> or --input <video>.")

    for vid in videos:
        if not vid:
            continue
        process_video(
            input_path=vid,
            out_root=out_root,
            stride=max(1, args.stride if args.stride is not None else STRIDE),
            thresh=int(args.thresh if args.thresh is not None else THRESHOLD),
            blur=int(args.blur if args.blur is not None else BLUR),
            erode_iter=max(0, int(args.erode if args.erode is not None else ERODE_ITERS)),
            dilate_iter=max(0, int(args.dilate if args.dilate is not None else DILATE_ITERS)),
            downscale=float(args.downscale if args.downscale is not None else DOWNSCALE),
            write_overlay=(False if args.no_overlay else WRITE_OVERLAY),
        )


if __name__ == "__main__":
    main()