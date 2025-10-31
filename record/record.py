import os
import time
import threading
import subprocess
from pathlib import Path

import cv2


# ====== Editable settings ======
# Auto-discover icspring cameras (recommended for Jetson)
AUTO_DISCOVER = True

# If not auto-discovering, specify camera indices here
CAMERA_INDICES = [0, 1, 2]

# Output directory and filenames (saved as MP4)
OUTPUT_BASE_DIR = Path(__file__).parent
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


def find_icspring_cameras():
    """Find cameras with 'icspring' in their name using lsusb."""
    icspring_devices = []
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        # Look for lines containing "icspring" (case insensitive)
        for line in lines:
            if 'icspring' in line.lower():
                # Extract bus and device numbers: Bus 002 Device 003
                parts = line.split()
                if len(parts) >= 6:
                    bus = parts[1]
                    device = parts[3].rstrip(':')
                    icspring_devices.append(f"Bus {bus} Device {device}")
        
        print(f"Found {len(icspring_devices)} icspring USB devices via lsusb")
        for dev in icspring_devices:
            print(f"  {dev}")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"lsusb not available ({e}), will fall back to /dev/video* discovery")
    
    return icspring_devices


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


def discover_icspring_cameras(required: int = 3, max_scan: int = 10):
    """Discover icspring cameras on Jetson, looking for /dev/video* devices."""
    found = []
    
    # First, look for icspring cameras via USB
    icspring_usb_devices = find_icspring_cameras()
    print(f"Scanning /dev/video* devices (0-{max_scan-1}) for working cameras...")
    
    # Scan /dev/video* devices
    for i in range(max_scan):
        dev = f"/dev/video{i}"
        if not os.path.exists(dev):
            continue
        
        cap = try_open_capture(dev)
        if cap.isOpened():
            cap.release()
            found.append(dev)
            print(f"  Found working camera: {dev}")
            if len(found) >= required:
                break
    
    return found


def record_from_camera(source, output_path: str, duration_seconds: float, start_barrier: threading.Barrier) -> None:

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

    # Synchronize start across all cameras
    start_barrier.wait()
    start_time = time.time()

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


def get_next_output_dir():
    """Find the next available output directory (output1, output2, etc.)."""
    base = OUTPUT_BASE_DIR
    i = 1
    while True:
        output_dir = base / f"output{i}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Creating new output directory: {output_dir}")
            return output_dir
        i += 1


def main() -> None:
    print("Running on: Jetson (Linux)")
    
    OUTPUT_DIR = get_next_output_dir()

    # Build list of sources
    if AUTO_DISCOVER:
        sources = discover_icspring_cameras(required=3, max_scan=10)
        if len(sources) < 3:
            print("WARNING: Fewer than 3 icspring cameras discovered; proceeding with available.")
    else:
        sources = [f"/dev/video{i}" for i in CAMERA_INDICES if os.path.exists(f"/dev/video{i}")]
        print(f"Using manual camera indices: {sources}")

    # Align sources with desired output filenames (first 3 entries)
    pairs = list(zip(sources[:3], OUTPUT_FILENAMES[:3]))
    if len(pairs) < 3:
        print("WARNING: Less than 3 sources or filenames configured; proceeding with available pairs.")

    if len(pairs) == 0:
        print("No camera/output pairs available. Exiting.")
        return

    # Create a barrier so all threads start recording at the same time
    start_barrier = threading.Barrier(len(pairs) + 1)

    threads = []
    for cam_idx, name in pairs:
        output_path = str(OUTPUT_DIR / name)
        t = threading.Thread(
            target=record_from_camera,
            args=(cam_idx, output_path, DURATION_SECONDS, start_barrier),
            daemon=True,
        )
        threads.append(t)

    print(f"Preparing {len(threads)} cameras; will start simultaneously for {DURATION_SECONDS}s...")
    for t in threads:
        t.start()

    # Release all threads to start recording at the same instant
    start_barrier.wait()

    # Show a simple overall time-based progress bar while recording
    start_time = time.time()
    total = float(DURATION_SECONDS)
    last_pct = -1
    bar_width = 30
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed < 0:
                elapsed = 0.0
            if elapsed > total:
                elapsed = total
            pct = int((elapsed / max(1e-6, total)) * 100)
            # throttle to ~10 updates per second and avoid redundant prints
            if pct != last_pct:
                filled = int((elapsed / max(1e-6, total)) * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(f"\r[record] [{bar}] {elapsed:4.1f}/{total:.0f}s  {pct:3d}%", end="", flush=True)
                last_pct = pct
            if elapsed >= total:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
    finally:
        # Ensure the progress line ends cleanly
        print()

    # Wait for all camera threads to finish
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted by user while joining threads.")

    print("All recordings complete.")


if __name__ == "__main__":
    main()
