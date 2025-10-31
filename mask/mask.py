#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import csv

# ============================
# Defaults (overridable via CLI)
# ============================
STRIDE = 1
DOWNSCALE = 1.0
WRITE_OVERLAY = True
BLUR = 5

# Background subtractor (MOG2) params tuned for small movers in sky scenes
MOG2_HISTORY = 300          # frames kept in model
MOG2_VAR_THRESH = 16.0      # sensitivity; lower -> more sensitive
MOG2_DETECT_SHADOWS = False # we want pure binary

# Morphology / filtering
OPEN_KERNEL = 3             # noise cleanup; 0 disables
CLOSE_KERNEL = 5            # merge small gaps
ERODE_ITERS = 0
DILATE_ITERS = 2
MIN_COMPONENT_AREA = 25     # keep tiny dots (tune as needed)
KEEP_LARGEST = True         # for now, track one object

# Learning rate: 0 < lr <= 1, or -1 for auto (OpenCV chooses 1/history)
LEARNING_RATE = -1

# Output root
OUTPUT_SUBDIR_DEFAULT = "output"  # under mask/
THRESH_POST = 1                   # hard floor after MOG2 to ensure [0/255]

# Try CUDA path on Jetson if available
def make_bg_subtractor_cuda():
    try:
        if not hasattr(cv2, "cuda"):
            return None
        # Some OpenCV builds on Jetson include CUDA MOG2
        return cv2.cuda.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESH,
            detectShadows=MOG2_DETECT_SHADOWS
        )
    except Exception:
        return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_video_init(path: Path, width: int, height: int, fps: float, is_color=True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (width, height), is_color)

def morphology(mask: np.ndarray,
               open_k: int,
               close_k: int,
               erode_iters: int,
               dilate_iters: int) -> np.ndarray:
    out = mask
    if open_k and open_k > 1:
        k = open_k + (open_k % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_k and close_k > 1:
        k = close_k + (close_k % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if erode_iters > 0:
        out = cv2.erode(out, None, iterations=erode_iters)
    if dilate_iters > 0:
        out = cv2.dilate(out, None, iterations=dilate_iters)
    return out

def to_gray_blur(frame: np.ndarray, blur: int) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur and blur > 0:
        k = blur + (blur % 2 == 0)
        g = cv2.GaussianBlur(g, (k, k), 0)
    return g

def process_video(
    input_path: str,
    out_root: Path,
    stride: int = 1,
    downscale: float = 1.0,
    write_overlay: bool = True,
    blur: int = 5,
    open_kernel: int = 3,
    close_kernel: int = 5,
    erode_iter: int = 0,
    dilate_iter: int = 2,
    min_component_area: int = 25,
    keep_largest: bool = True,
    mog2_history: int = 300,
    mog2_var: float = 16.0,
    mog2_shadows: bool = False,
    learning_rate: float = -1.0,
    quality_log_path: Path = None
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(w0 * downscale) if downscale != 1.0 else w0
    height = int(h0 * downscale) if downscale != 1.0 else h0

    base = Path(input_path).stem
    out_dir = out_root / base
    frames_dir = out_dir / "frames"
    masks_dir = out_dir / "masks"
    ensure_dir(frames_dir)
    ensure_dir(masks_dir)

    overlay_path = out_dir / "motion_overlay.mp4"
    mask_video_path = out_dir / "motion_mask.mp4"
    writer = write_overlay and write_video_init(overlay_path, width, height, fps)
    mask_writer = write_video_init(mask_video_path, width, height, fps, is_color=True)

    # Quality CSV (optional)
    qfp = None
    qwriter = None
    if quality_log_path:
        ensure_dir(quality_log_path.parent)
        qfp = open(quality_log_path, "w", newline="")
        qwriter = csv.writer(qfp)
        qwriter.writerow(["frame","mask_area","num_components","largest_area","bbox_x","bbox_y","bbox_w","bbox_h"])

    # Choose CUDA subtractor if possible, else CPU
    cuda_sub = make_bg_subtractor_cuda()
    if cuda_sub is None:
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var,
            detectShadows=mog2_shadows
        )
    else:
        mog2 = None  # using CUDA path

    idx_in = -1
    idx_kept = 0
    last_pct = -1

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        idx_in += 1
        if stride > 1 and (idx_in % stride) != 0:
            continue

        if downscale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)

        gray = to_gray_blur(frame_bgr, blur)

        # Background subtraction → raw foreground mask
        if cuda_sub is not None:
            # Upload to GPU
            g_gpu = cv2.cuda_GpuMat()
            g_gpu.upload(gray)
            fg_gpu = cuda_sub.apply(g_gpu, learningRate=learning_rate)
            mask_raw = fg_gpu.download()
        else:
            mask_raw = mog2.apply(gray, learningRate=learning_rate)

        # Enforce binary (remove any shadow codes)
        _, mask_bin = cv2.threshold(mask_raw, THRESH_POST, 255, cv2.THRESH_BINARY)

        # Morphology to stabilize dot
        mask_bin = morphology(mask_bin, open_kernel, close_kernel, erode_iter, dilate_iter)

        # Keep only the largest blob (single target for now) or filter by min area
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_final = np.zeros_like(mask_bin)
        largest_area = 0
        bbox = (0,0,0,0)
        if contours:
            if keep_largest:
                best = max(contours, key=cv2.contourArea)
                largest_area = float(cv2.contourArea(best))
                if largest_area >= max(1, min_component_area):
                    cv2.drawContours(mask_final, [best], -1, 255, thickness=cv2.FILLED)
                    x,y,w,h = cv2.boundingRect(best)
                    bbox = (x,y,w,h)
                    if write_overlay:
                        cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
            else:
                # keep all blobs above min area
                kept = []
                for c in contours:
                    a = cv2.contourArea(c)
                    if a >= max(1, min_component_area):
                        kept.append(c)
                        largest_area = max(largest_area, float(a))
                if kept:
                    cv2.drawContours(mask_final, kept, -1, 255, thickness=cv2.FILLED)
                    if write_overlay:
                        for c in kept:
                            x,y,w,h = cv2.boundingRect(c)
                            cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), (0,255,0), 1)

        # Write per-frame artifacts
        cv2.imwrite(str(frames_dir / f"frame_{idx_kept:06d}.png"), frame_bgr)
        cv2.imwrite(str(masks_dir / f"mask_{idx_kept:06d}.png"), mask_final)

        # Write videos
        mask_writer.write(cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR))
        if write_overlay:
            writer.write(frame_bgr)

        # Quality CSV row
        if qwriter is not None:
            mask_area = int(cv2.countNonZero(mask_final))
            qwriter.writerow([idx_kept, mask_area, len(contours), int(largest_area), *bbox])

        # Progress (~every 5%)
        if total > 0:
            pct = int(((idx_in + 1) * 100) / total)
            if pct % 5 == 0 and pct != last_pct:
                print(f"[progress] {pct}% ({idx_in+1}/{total})")
                last_pct = pct

        idx_kept += 1

    cap.release()
    if write_overlay:
        writer.release()
    mask_writer.release()
    if qfp is not None:
        qfp.close()

    print(f"[done] Frames: {idx_kept}  out: {out_dir}")
    if write_overlay:
        print(f"[info] Overlay video: {overlay_path}")
    print(f"[info] Mask video: {mask_video_path}")

def main():
    ap = argparse.ArgumentParser(description="Real-time-grade motion masking (MOG2/CUDA) with same I/O as original.")
    ap.add_argument("output_folder", nargs="?", help="Output folder name (e.g., output1, output2). If omitted, uses default behavior.")
    ap.add_argument("--input", help="Input video path; if omitted, scans record/{output_folder}/", default=None)
    ap.add_argument("--src", help="Root folder containing videos (defaults to <project_root>/record/{output_folder})", default=None)
    ap.add_argument("--out", help="Output root (defaults to mask/output)", default=None)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--downscale", type=float, default=DOWNSCALE)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--blur", type=int, default=BLUR)
    ap.add_argument("--open", type=int, default=OPEN_KERNEL)
    ap.add_argument("--close", type=int, default=CLOSE_KERNEL)
    ap.add_argument("--erode", type=int, default=ERODE_ITERS)
    ap.add_argument("--dilate", type=int, default=DILATE_ITERS)
    ap.add_argument("--min-component-area", type=int, default=MIN_COMPONENT_AREA)
    ap.add_argument("--largest-only", action="store_true", help="Force keep largest blob")
    ap.add_argument("--no-largest-only", action="store_true", help="Keep all blobs above min area")
    ap.add_argument("--history", type=int, default=MOG2_HISTORY)
    ap.add_argument("--var", type=float, default=MOG2_VAR_THRESH)
    ap.add_argument("--shadows", action="store_true", help="Enable MOG2 shadow detection (off by default)")
    ap.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate; -1 for auto")
    ap.add_argument("--quality-log", type=str, help="CSV path for metrics")
    args = ap.parse_args()

    mask_dir = Path(__file__).resolve().parent
    project_root = mask_dir.parent

    # Output root under mask/output
    out_root = Path(args.out) if args.out else (mask_dir / OUTPUT_SUBDIR_DEFAULT)
    ensure_dir(out_root)

    # Input videos: default to record/output_folder/ if specified, else record/output/
    if args.input:
        videos = [args.input]
    else:
        # Use output_folder if provided, otherwise default to "output"
        record_folder = args.output_folder if args.output_folder else "output"
        if args.src:
            src_root = Path(args.src)
        else:
            src_root = project_root / "record" / record_folder
        if not src_root.exists() or not src_root.is_dir():
            raise SystemExit(f"Video folder not found: {src_root}")
        allowed = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
        videos = sorted([str(p) for p in src_root.iterdir() if p.suffix in allowed and not p.name.startswith('.')])
        if not videos:
            raise SystemExit(f"No videos found in folder: {src_root}")
        print(f"[info] Processing {len(videos)} video(s) from {src_root}")

    # Resolve keep-largest flag precedence (default True like now)
    keep_largest = True
    if args.no_largest_only:
        keep_largest = False
    elif args.largest_only:
        keep_largest = True

    for vid in videos:
        if not vid:
            continue
        process_video(
            input_path=vid,
            out_root=out_root,
            stride=max(1, int(args.stride)),
            downscale=float(args.downscale),
            write_overlay=(False if args.no_overlay else WRITE_OVERLAY),
            blur=int(args.blur),
            open_kernel=int(args.open),
            close_kernel=int(args.close),
            erode_iter=max(0, int(args.erode)),
            dilate_iter=max(0, int(args.dilate)),
            min_component_area=int(args.min_component_area),
            keep_largest=keep_largest,
            mog2_history=int(args.history),
            mog2_var=float(args.var),
            mog2_shadows=bool(args.shadows),
            learning_rate=float(args.lr),
            quality_log_path=(Path(args.quality_log) if args.quality_log else None),
        )

if __name__ == "__main__":
    # Jetson: fewer CPU threads can sometimes reduce scheduling overhead
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass
    main()
