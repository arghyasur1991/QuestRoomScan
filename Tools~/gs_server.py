#!/usr/bin/env python3
"""
HTTP training server for QuestRoomScan Gaussian Splatting.

Wraps gs_pipeline.py: receives keyframes+points from Quest via WiFi,
runs COLMAP conversion + GS training, serves back the trained PLY.

Usage:
  python gs_server.py --port 8420
  python gs_server.py --port 8420 --iterations 10000

Endpoints:
  POST /upload   -- receives a ZIP (images/, frames.jsonl, points3d.ply)
  GET  /status   -- {"state": "idle|training|done|error", "progress": 0-1, "message": "..."}
  GET  /download -- returns trained splat.ply (only when state=done)
  POST /cancel   -- cancels in-progress training
"""

import argparse
import io
import json
import shutil
import threading
import time
import traceback
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import gs_pipeline


class TrainingState:
    IDLE = "idle"
    TRAINING = "training"
    DONE = "done"
    ERROR = "error"


class TrainingManager:
    def __init__(self, work_dir: Path, iterations: int):
        self.work_dir = work_dir
        self.iterations = iterations
        self.state = TrainingState.IDLE
        self.progress = 0.0
        self.message = "Ready"
        self.capture_dir: Path | None = None
        self.output_ply: Path | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def start_training(self, zip_data: bytes) -> bool:
        with self._lock:
            if self.state == TrainingState.TRAINING:
                return False
            self._cancel.clear()
            self.state = TrainingState.TRAINING
            self.progress = 0.0
            self.message = "Extracting upload..."
            self.output_ply = None

        self._thread = threading.Thread(target=self._run, args=(zip_data,), daemon=True)
        self._thread.start()
        return True

    def cancel(self):
        self._cancel.set()
        with self._lock:
            if self.state == TrainingState.TRAINING:
                self.state = TrainingState.IDLE
                self.message = "Cancelled"
                self.progress = 0.0

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "progress": round(self.progress, 3),
                "message": self.message,
            }

    def _run(self, zip_data: bytes):
        try:
            capture_dir = self.work_dir / "current_run"
            if capture_dir.exists():
                shutil.rmtree(capture_dir)
            capture_dir.mkdir(parents=True)

            with self._lock:
                self.message = "Extracting ZIP..."
                self.progress = 0.05

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                zf.extractall(capture_dir)

            if self._cancel.is_set():
                return

            if not (capture_dir / "frames.jsonl").exists():
                raise FileNotFoundError("frames.jsonl not found in upload")

            with self._lock:
                self.message = "Converting to COLMAP format..."
                self.progress = 0.1
                self.capture_dir = capture_dir

            frames = gs_pipeline.parse_frames(capture_dir)
            gs_pipeline.convert_to_colmap(capture_dir, frames)

            if self._cancel.is_set():
                return

            with self._lock:
                self.message = f"Training ({self.iterations} iterations)..."
                self.progress = 0.2

            args = argparse.Namespace(
                iterations=self.iterations,
                gs_repo=None,
            )
            output_dir = gs_pipeline.train(capture_dir, args)

            if self._cancel.is_set():
                return

            trained_ply = output_dir / "splat.ply"
            if not trained_ply.exists():
                candidates = list(output_dir.rglob("*.ply"))
                trained_ply = candidates[0] if candidates else None

            if trained_ply and trained_ply.exists():
                with self._lock:
                    self.state = TrainingState.DONE
                    self.progress = 1.0
                    self.message = f"Training complete: {trained_ply.name}"
                    self.output_ply = trained_ply
            else:
                with self._lock:
                    self.state = TrainingState.ERROR
                    self.message = "Training finished but no PLY output found"

        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self.state = TrainingState.ERROR
                self.message = str(e)


manager: TrainingManager | None = None


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/upload":
            self._handle_upload()
        elif self.path == "/cancel":
            self._handle_cancel()
        else:
            self._send_json(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/download":
            self._handle_download()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_upload(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty body"})
            return
        if content_length > 2 * 1024 * 1024 * 1024:  # 2GB limit
            self._send_json(413, {"error": "Upload too large"})
            return

        zip_data = self.rfile.read(content_length)
        if not manager.start_training(zip_data):
            self._send_json(409, {"error": "Training already in progress"})
            return

        self._send_json(200, {"status": "started"})

    def _handle_cancel(self):
        manager.cancel()
        self._send_json(200, {"status": "cancelled"})

    def _handle_status(self):
        self._send_json(200, manager.get_status())

    def _handle_download(self):
        status = manager.get_status()
        if status["state"] != TrainingState.DONE or manager.output_ply is None:
            self._send_json(404, {"error": "No trained model available"})
            return

        ply_path = manager.output_ply
        if not ply_path.exists():
            self._send_json(404, {"error": "PLY file not found on disk"})
            return

        data = ply_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{ply_path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[gs_server] {self.client_address[0]} - {format % args}")


def main():
    global manager

    parser = argparse.ArgumentParser(description="GS Training HTTP Server")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--work-dir", default="gs_server_work",
                        help="Working directory for uploads and training")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    manager = TrainingManager(work_dir, args.iterations)

    server = HTTPServer((args.host, args.port), RequestHandler)
    print(f"[gs_server] Listening on {args.host}:{args.port}")
    print(f"[gs_server] Work directory: {work_dir}")
    print(f"[gs_server] Training iterations: {args.iterations}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gs_server] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
