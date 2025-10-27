import cv2
import multiprocessing
import time
import os

# ===== CONFIGURATION =====
NUM_CAMERAS = 3
RECORDING_DURATION = 5  # seconds
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rec')
STAGGER_INIT_DELAY = 2  # Seconds between camera initializations
# =========================

def record_camera(camera_index, duration, fps, width, height, output_folder, ready_queue, start_barrier):
    """Record video from a single camera"""
    
    print(f"Camera {camera_index + 1}: Initializing...")
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Camera {camera_index + 1}: ERROR - Could not open")
        ready_queue.put((camera_index, False))
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {camera_index + 1}: Opened at {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    
    # Warm up camera - read and discard frames
    print(f"Camera {camera_index + 1}: Warming up...")
    warmup_count = 0
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            warmup_count += 1
    
    if warmup_count < 20:
        print(f"Camera {camera_index + 1}: WARNING - Only {warmup_count}/30 warmup frames successful")
    
    # Create output file
    filename = os.path.join(output_folder, f"{camera_index + 1}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (actual_width, actual_height))
    
    if not out.isOpened():
        print(f"Camera {camera_index + 1}: ERROR - Could not create video file")
        cap.release()
        ready_queue.put((camera_index, False))
        return
    
    print(f"Camera {camera_index + 1}: ✓ READY and waiting for sync...")
    
    # Signal this camera is ready
    ready_queue.put((camera_index, True))
    
    # Wait for all cameras to be ready
    start_barrier.wait()
    
    print(f"Camera {camera_index + 1}: 🔴 RECORDING NOW!")
    
    start_time = time.time()
    frame_count = 0
    failed_reads = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        
        if ret:
            out.write(frame)
            frame_count += 1
            failed_reads = 0
        else:
            failed_reads += 1
            if failed_reads > 30:
                print(f"Camera {camera_index + 1}: ERROR - Too many failed reads")
                break
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    actual_fps_recorded = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"Camera {camera_index + 1}: ✓ COMPLETED - {frame_count} frames in {elapsed:.2f}s ({actual_fps_recorded:.1f} fps)")

def main():
    print(f"\n{'='*60}")
    print(f"  Recording from {NUM_CAMERAS} cameras for {RECORDING_DURATION} seconds")
    print(f"{'='*60}\n")
    
    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}\n")
    
    # Create synchronization primitives
    ready_queue = multiprocessing.Queue()
    start_barrier = multiprocessing.Barrier(NUM_CAMERAS)
    
    # Create processes for each camera
    processes = []
    for i in range(NUM_CAMERAS):
        p = multiprocessing.Process(
            target=record_camera,
            args=(i, RECORDING_DURATION, FPS, FRAME_WIDTH, FRAME_HEIGHT, OUTPUT_FOLDER, ready_queue, start_barrier)
        )
        processes.append(p)
    
    # Start all camera processes
    print("Starting camera initialization...\n")
    for p in processes:
        p.start()
    
    # Wait for all cameras to report ready
    ready_cameras = []
    failed_cameras = []
    
    for _ in range(NUM_CAMERAS):
        cam_index, success = ready_queue.get()
        if success:
            ready_cameras.append(cam_index + 1)
        else:
            failed_cameras.append(cam_index + 1)
    
    print(f"\n{'='*60}")
    if failed_cameras:
        print(f"⚠️  WARNING: Camera(s) {failed_cameras} failed to initialize")
    if ready_cameras:
        print(f"✓ All {len(ready_cameras)} camera(s) ready: {ready_cameras}")
        print(f"{'='*60}")
        print(f"\n🎬 Starting synchronized recording...\n")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\n{'='*60}")
    print(f"  ✓ All recordings completed!")
    print(f"  Videos saved in: {OUTPUT_FOLDER}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()