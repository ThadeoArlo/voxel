import os
import time
import threading

import cv2


# ====== Editable settings ======
# Auto-discover working cameras under /dev/video* (recommended)
AUTO_DISCOVER = True

# If not auto-discovering, specify camera indices here
CAMERA_INDICES = [0, 1, 2]

# Output directory and filenames (saved as MP4)
OUTPUT_DIR = "rec"
OUTPUT_FILENAMES = ["1.mp4", "2.mp4", "3.mp4"]

# Duration of recording in seconds
DURATION_SECONDS = 5

# Basic capture settings (very light for stress-free triple capture)
TARGET_FPS = 10.0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Video codec for MP4 files. 'mp4v' is broadly compatible.
FOURCC = "mp4v"
INPUT_FOURCC = "MJPG"  # ask webcams for MJPG to reduce USB bandwidth


def try_open_capture(source):
    """Try to open a camera source with robust fallbacks.

    Source can be an int index (e.g., 0) or a string path (e.g., "/dev/video0").
    """
    # Try V4L2 first
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()

    # Fallback to ANY backend
    cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if cap.isOpened():
        return cap
    cap.release()

    # If given an int index, also try the device path explicitly
    if isinstance(source, int):
        dev = f"/dev/video{source}"
        cap = cv2.VideoCapture(dev, cv2.CAP_ANY)
        if cap.isOpened():
            return cap
        cap.release()

    # If given a device path, try a simple GStreamer pipeline as last resort
    if isinstance(source, str) and os.path.exists(source):
        gst = f"v4l2src device={source} ! videoconvert ! appsink"
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture()  # unopened


def discover_cameras(required: int = 3, max_scan: int = 10):
    found = []
    for i in range(max_scan):
        dev = f"/dev/video{i}"
        if not os.path.exists(dev):
            continue
        cap = try_open_capture(dev)
        if cap.isOpened():
            cap.release()
            found.append(dev)
            if len(found) >= required:
                break
    return found


def record_from_camera(source, output_path: str, duration_seconds: float) -> None:
    start_time = time.time()

    # Aggressive retries to open the camera to avoid race/USB contention
    cap = None
    for _ in range(50):  # ~5s total with 0.1s sleep
        cap = try_open_capture(source)
        if cap.isOpened():
            break
        time.sleep(0.1)
    if cap is None or not cap.isOpened():
        print(f"[{source}] ERROR: Could not open camera after retries.")
        return

    # Try to set basic properties; some cameras might ignore these.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*INPUT_FOURCC))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
    fps_for_writer = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    if fps_for_writer <= 0:
        fps_for_writer = TARGET_FPS

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(output_path, fourcc, fps_for_writer, (actual_width, actual_height))
    if not writer.isOpened():
        print(f"[{source}] ERROR: Could not open writer for {output_path}.")
        cap.release()
        return

    print(f"[{source}] Recording {os.path.basename(output_path)} at {actual_width}x{actual_height}@{fps_for_writer:.1f}fps")

    try:
        consecutive_failures = 0
        while (time.time() - start_time) < duration_seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    # Attempt to reinitialize the capture mid-run
                    cap.release()
                    time.sleep(0.1)
                    cap = try_open_capture(source)
                    consecutive_failures = 0
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
        print(f"[{source}] Done.")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build list of sources
    if AUTO_DISCOVER:
        sources = discover_cameras(required=3, max_scan=10)
        if len(sources) < 3:
            print("WARNING: Fewer than 3 working cameras discovered; proceeding with available.")
    else:
        sources = CAMERA_INDICES[:3]

    # Align sources with desired output filenames (first 3 entries)
    pairs = list(zip(sources[:3], OUTPUT_FILENAMES[:3]))
    if len(pairs) < 3:
        print("WARNING: Less than 3 sources or filenames configured; proceeding with available pairs.")

    threads = []
    for cam_idx, name in pairs:
        output_path = os.path.join(OUTPUT_DIR, name)
        t = threading.Thread(target=record_from_camera, args=(cam_idx, output_path, DURATION_SECONDS), daemon=True)
        threads.append(t)

    print(f"Starting {len(threads)} cameras for {DURATION_SECONDS}s...")
    for t in threads:
        t.start()
        time.sleep(0.3)  # stagger starts to reduce USB bandwidth spikes

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")

    print("All recordings complete.")


if __name__ == "__main__":
    main()


