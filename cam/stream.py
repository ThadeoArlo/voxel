import os
import time
import threading
from typing import List, Dict, Any

import cv2
from flask import Flask, Response, jsonify


# ====== Editable settings ======
# Auto-discover working cameras under /dev/video* (recommended)
AUTO_DISCOVER = True

# If not auto-discovering, specify camera indices or device paths here
CAMERA_SOURCES: List[Any] = [0, 1, 2]

# Basic capture settings
TARGET_FPS = 10.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Input request to cameras (helps USB bandwidth)
INPUT_FOURCC = "MJPG"

# JPEG encode quality for MJPEG streaming (lower => smaller size)
JPEG_QUALITY = 80

# HTTP server binding
HOST = "0.0.0.0"
PORT = 8000


def try_open_capture(source):
    """Try to open a camera source with robust fallbacks.

    Source can be an int index (e.g., 0) or a string path (e.g., "/dev/video0").
    """
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if cap.isOpened():
        return cap
    cap.release()

    if isinstance(source, int):
        dev = f"/dev/video{source}"
        cap = cv2.VideoCapture(dev, cv2.CAP_ANY)
        if cap.isOpened():
            return cap
        cap.release()

    if isinstance(source, str) and os.path.exists(source):
        gst = f"v4l2src device={source} ! videoconvert ! appsink"
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture()  # unopened


def discover_cameras(required: int = 3, max_scan: int = 10) -> List[str]:
    found: List[str] = []
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


class CameraStream:
    """Background reader that keeps the most recent JPEG-encoded frame in memory."""

    def __init__(self, source: Any, name: str):
        self.source = source
        self.name = name
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_jpeg = None  # bytes
        self.frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.05

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    def _open(self) -> bool:
        # multiple retries to avoid transient failures
        for _ in range(50):
            cap = try_open_capture(self.source)
            if cap.isOpened():
                # set desired properties (best-effort)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*INPUT_FOURCC))
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                self.cap = cap
                return True
            time.sleep(0.1)
        return False

    def _run(self) -> None:
        if not self._open():
            print(f"[{self.name}] ERROR: Could not open camera.")
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
        next_time = time.time()
        consecutive_failures = 0

        while self.running:
            # simple pacing to reduce CPU
            now = time.time()
            if now < next_time:
                time.sleep(max(0.0, next_time - now))
            next_time = now + self.frame_interval

            ok, frame = self.cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    # attempt reopen
                    if self.cap is not None:
                        self.cap.release()
                    time.sleep(0.1)
                    if not self._open():
                        time.sleep(0.4)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0

            ok, buf = cv2.imencode('.jpg', frame, encode_params)
            if not ok:
                continue
            jpeg_bytes = buf.tobytes()
            with self.lock:
                self.latest_jpeg = jpeg_bytes

    def get_latest_frame(self) -> bytes:
        with self.lock:
            return self.latest_jpeg


def create_app() -> Flask:
    app = Flask(__name__)

    # Build list of sources
    if AUTO_DISCOVER:
        # Try to find at least 3; but we will use all discovered
        sources = discover_cameras(required=3, max_scan=16)
    else:
        sources = CAMERA_SOURCES

    if not sources:
        print("WARNING: No cameras discovered/configured. Server will start but streams will be empty.")

    # Create camera workers
    cameras: List[CameraStream] = []
    for idx, src in enumerate(sources):
        name = str(src)
        cameras.append(CameraStream(src, name))

    # Start all cameras
    for cam in cameras:
        cam.start()

    def mjpeg_generator(camera: CameraStream):
        boundary = b"--frame"
        headers = b"Content-Type: image/jpeg\r\n\r\n"
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            yield boundary + b"\r\n" + headers + frame + b"\r\n"

    @app.get("/")
    def index():
        # Minimal inline HTML to display all streams
        blocks = []
        for i, cam in enumerate(cameras):
            label = cam.name
            blocks.append(
                f'<div style="margin:8px;text-align:center">'
                f'<div style="font-family:sans-serif;margin-bottom:4px">Camera {i}: {label}</div>'
                f'<img src="/stream/{i}" style="max-width:48%;height:auto;border:1px solid #ccc"/>'
                f"</div>"
            )
        html = (
            "<!doctype html>\n"
            "<html><head><title>Multi-Camera Live</title>"
            "<meta name=viewport content='width=device-width, initial-scale=1'/>"
            "</head>"
            "<body style='margin:0;padding:12px;background:#111;color:#eee'>"
            "<h2 style='font-family:sans-serif;margin:8px 0'>Live Cameras</h2>"
            "<div style='display:flex;flex-wrap:wrap'>"
            + "".join(blocks) +
            "</div>"
            "</body></html>"
        )
        return html

    @app.get("/stream/<int:idx>")
    def stream(idx: int):
        if idx < 0 or idx >= len(cameras):
            return jsonify({"error": "camera index out of range"}), 404
        return Response(
            mjpeg_generator(cameras[idx]),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/health")
    def health():
        ok = any(cam.get_latest_frame() is not None for cam in cameras)
        return jsonify({"status": "ok" if ok else "starting", "cameras": len(cameras)})

    return app


def main() -> None:
    app = create_app()
    print(f"Starting server on http://{HOST}:{PORT}  (open / in your browser)")
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()


