#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import List, Any
import threading
import json
import time
import os

import cv2

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
# JSON tracks are still produced under render/output
OUTPUT_DIR = ROOT.parent / "render" / "output"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Camera streaming (integrated from cam/stream.py) ======

# Editable settings
AUTO_DISCOVER = True
CAMERA_SOURCES: List[Any] = [0, 1, 2]
TARGET_FPS = 10.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INPUT_FOURCC = "MJPG"
JPEG_QUALITY = 80


def try_open_capture(source):
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


def discover_cameras(required: int = 3, max_scan: int = 16) -> List[str]:
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
        for _ in range(50):
            cap = try_open_capture(self.source)
            if cap.isOpened():
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
            now = time.time()
            if now < next_time:
                time.sleep(max(0.0, next_time - now))
            next_time = now + self.frame_interval

            ok, frame = self.cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
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


# Global camera registry
CAMERAS: List[CameraStream] = []


@app.on_event("startup")
def start_cameras():
    sources = discover_cameras(required=3, max_scan=16) if AUTO_DISCOVER else CAMERA_SOURCES
    if not sources:
        print("WARNING: No cameras discovered/configured. Streams will be empty.")
    for src in sources:
        name = str(src)
        cam = CameraStream(src, name)
        CAMERAS.append(cam)
    for cam in CAMERAS:
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


@app.get("/api/cameras")
def list_cameras():
    return [
        {"idx": i, "name": cam.name, "ready": cam.get_latest_frame() is not None}
        for i, cam in enumerate(CAMERAS)
    ]


@app.get("/stream/{idx}")
def stream(idx: int):
    if idx < 0 or idx >= len(CAMERAS):
        raise HTTPException(404, detail="camera index out of range")
    return StreamingResponse(
        mjpeg_generator(CAMERAS[idx]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cam_health")
def cam_health():
    ok = any(cam.get_latest_frame() is not None for cam in CAMERAS)
    return {"status": "ok" if ok else "starting", "cameras": len(CAMERAS)}

# Serve the web app (index.html and assets)
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")


@app.get("/api/scenes")
def list_scenes():
    if not OUTPUT_DIR.exists():
        return []
    return sorted([p.stem for p in OUTPUT_DIR.glob("*.json")])


@app.get("/api/scene/{name}")
def get_scene(name: str):
    fp = OUTPUT_DIR / f"{name}.json"
    if not fp.exists():
        raise HTTPException(404, detail="Not found")
    try:
        return json.loads(fp.read_text())
    except Exception:
        raise HTTPException(500, detail="Failed to read scene JSON")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


