#!/usr/bin/env python3
import os
import cv2
import argparse
from pathlib import Path


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

        if prev_gray is None:
            prev_gray = gray
            idx_kept += 1
            continue

        # Frame differencing
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        # Morphology cleanup
        if erode_iter > 0:
            mask = cv2.erode(mask, None, iterations=erode_iter)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, None, iterations=dilate_iter)

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
    ap = argparse.ArgumentParser(description="Extract frames and generate binary motion masks (white = moving pixels)")
    ap.add_argument("--input", required=True, help="Input video path (e.g., 1.mp4)")
    ap.add_argument("--out", default="videoProcessor/output", help="Output root folder (results within per-video subfolder)")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    ap.add_argument("--thresh", type=int, default=25, help="Binary threshold for motion mask")
    ap.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size (odd); 0 to disable")
    ap.add_argument("--erode", type=int, default=1, help="Erode iterations")
    ap.add_argument("--dilate", type=int, default=2, help="Dilate iterations")
    ap.add_argument("--downscale", type=float, default=1.0, help="Resize factor (e.g., 0.5 for half)")
    ap.add_argument("--no-overlay", action="store_true", help="Disable overlay video with contours")
    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)
    process_video(
        input_path=args.input,
        out_root=out_root,
        stride=max(1, args.stride),
        thresh=int(args.thresh),
        blur=int(args.blur),
        erode_iter=max(0, int(args.erode)),
        dilate_iter=max(0, int(args.dilate)),
        downscale=float(args.downscale),
        write_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()


