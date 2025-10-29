import os
import time
import threading
import subprocess
from pathlib import Path

import cv2


# ====== Editable settings ======
# Auto-discover icspring cameras (recommended for Mac)
AUTO_DISCOVER = True

# If not auto-discovering, specify camera indices here
CAMERA_INDICES = [0, 1, 2]

# Output directory and filenames (saved as MP4)
OUTPUT_DIR = Path(__file__).parent / "rec"
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


def find_icspring_cameras_mac():
    """Find icspring cameras on Mac using system_profiler."""
    icspring_devices = []
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Look for icspring in camera info
        if 'icspring' in output.lower():
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'icspring' in line.lower():
                    # Extract camera name
                    if ':' in line:
                        camera_name = line.split(':')[0].strip()
                        icspring_devices.append(camera_name)
                    else:
                        icspring_devices.append(line.strip())
        
        print(f"Found {len(icspring_devices)} icspring cameras via system_profiler")
        for dev in icspring_devices:
            print(f"  {dev}")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"system_profiler not available ({e}), will try camera indices directly")
    
    return icspring_devices


def try_open_capture(source):
    """Try to open a camera source on Mac.

    Source should be an int index (e.g., 0, 1, 2).
    """
    # Try AVFoundation first (Mac's camera backend)
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    cap.release()
    
    # Fallback to ANY backend
    cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if cap.isOpened():
        return cap
    cap.release()

    return cv2.VideoCapture()  # unopened


def get_camera_name(cap):
    """Try to get the camera name/description."""
    try:
        # Try to get backend name
        backend = cap.getBackendName()
        return backend
    except:
        return "Unknown"


def discover_icspring_cameras_mac(required: int = 3, max_scan: int = 10):
    """Discover icspring cameras on Mac by trying camera indices."""
    found = []
    
    # First, check system_profiler for icspring cameras
    icspring_info = find_icspring_cameras_mac()
    print(f"Scanning camera indices (0-{max_scan-1}) for working cameras...")
    
    # Try opening cameras by index
    for i in range(max_scan):
        cap = try_open_capture(i)
        if cap.isOpened():
            # Try to read a test frame to ensure it's really working
            ret, frame = cap.read()
            if ret and frame is not None:
                camera_name = get_camera_name(cap)
                print(f"  Found working camera at index {i}: {camera_name}")
                cap.release()
                found.append(i)
                if len(found) >= required:
                    break
            else:
                cap.release()
        else:
            cap.release()
    
    if len(icspring_info) > 0 and len(found) >= len(icspring_info):
        print(f"Note: Assuming first {len(icspring_info)} working cameras are icspring cameras")
    
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
        print(f"[Camera {source}] ERROR: Could not open camera after retries.")
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
        print(f"[Camera {source}] ERROR: Could not open writer for {output_path}.")
        cap.release()
        return

    print(f"[Camera {source}] Recording {os.path.basename(output_path)} at {actual_width}x{actual_height}@{fps_for_writer:.1f}fps")

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
        print(f"[Camera {source}] Done.")


def main() -> None:
    print("Running on: macOS")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of sources
    if AUTO_DISCOVER:
        sources = discover_icspring_cameras_mac(required=3, max_scan=10)
        if len(sources) < 3:
            print("WARNING: Fewer than 3 cameras discovered; proceeding with available.")
    else:
        sources = CAMERA_INDICES[:3]
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

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")

    print("All recordings complete.")


if __name__ == "__main__":
    main()

