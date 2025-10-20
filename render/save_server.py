#!/usr/bin/env python3
"""
Tiny local HTTP server to update render/cam_config.json from the browser UI.

Usage:
  python3 render/save_server.py --port 8787

Then open render/index.html in your browser. The "Save Camera Configs" button
will POST to this server and update cam_config.json in-place.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CAM_CONFIG_PATH = PROJECT_DIR / "cam_config.json"


def _send_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class SaveHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Quieter logs
        sys.stderr.write("[save_server] " + (format % args) + "\n")

    def do_OPTIONS(self):  # noqa: N802 (BaseHTTPRequestHandler naming)
        self.send_response(200)
        _send_cors_headers(self)
        self.end_headers()

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler naming)
        if self.path.rstrip("/") != "/save_cams":
            self.send_response(404)
            _send_cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{\"ok\":false,\"error\":\"Not found\"}")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            _send_cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{\"ok\":false,\"error\":\"Invalid JSON\"}")
            return

        cameras = payload.get("cameras")
        if not isinstance(cameras, list):
            self.send_response(400)
            _send_cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{\"ok\":false,\"error\":\"Payload must include 'cameras' list\"}")
            return

        # Preserve coordinate_system if present, otherwise default
        coord = "Z_up_XY_ground"
        if CAM_CONFIG_PATH.exists():
            try:
                existing = json.loads(CAM_CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("coordinate_system"), str):
                    coord = existing["coordinate_system"]
            except Exception:
                pass

        new_cfg = {"coordinate_system": coord, "cameras": cameras}
        CAM_CONFIG_PATH.write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")

        self.send_response(200)
        _send_cors_headers(self)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {
            "ok": True,
            "path": str(CAM_CONFIG_PATH),
            "count": len(cameras),
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local save server for cam_config.json")
    ap.add_argument("--port", type=int, default=8787, help="Port to listen on (default: 8787)")
    args = ap.parse_args()

    addr = ("127.0.0.1", int(args.port))
    httpd = HTTPServer(addr, SaveHandler)
    print(f"[save_server] Listening on http://{addr[0]}:{addr[1]}  → writing {CAM_CONFIG_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[save_server] Shutting down…")


if __name__ == "__main__":
    main()


