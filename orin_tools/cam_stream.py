#!/usr/bin/env python3
"""Orin USB webcam 檢查：以 MJPEG over HTTP 串流，在電腦瀏覽器觀看（不需要螢幕或 X11）。

用法：
  python3 cam_stream.py                   # /dev/video0，瀏覽器開 http://<orin-ip>:8080
  python3 cam_stream.py -d 1 --port 8081
  python3 cam_stream.py --snapshot snap.jpg   # 只拍一張存檔後結束
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

latest = None
lock = threading.Lock()


def open_cam(dev, w, h):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 /dev/video{dev}，先用 `v4l2-ctl --list-devices` 確認裝置編號")
    return cap


def grab_loop(cap):
    global latest
    frames, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest = jpg.tobytes()
        frames += 1
        if time.time() - t0 >= 5:
            print(f"擷取 {frames / (time.time() - t0):.1f} fps, {frame.shape[1]}x{frame.shape[0]}")
            frames, t0 = 0, time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    jpg = latest
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--snapshot", help="拍一張存成此檔名後結束")
    args = ap.parse_args()

    cap = open_cam(args.device, args.width, args.height)

    if args.snapshot:
        for _ in range(10):  # 丟掉前幾張讓曝光穩定
            ok, frame = cap.read()
        if not ok:
            raise SystemExit("讀取影像失敗")
        cv2.imwrite(args.snapshot, frame)
        print(f"已存 {args.snapshot} ({frame.shape[1]}x{frame.shape[0]})")
        return

    threading.Thread(target=grab_loop, args=(cap,), daemon=True).start()
    print(f"串流中：在電腦瀏覽器開 http://<orin-ip>:{args.port}/  (Ctrl+C 結束)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
