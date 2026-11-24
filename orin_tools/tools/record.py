#!/usr/bin/env python3
"""推車錄影：錄下鏡頭畫面給離線開發（main.py --dry-run --source、lane_view_check.py --source）用。

錄影前先把鏡頭固定在最終位置、關掉自動曝光，錄影時鏡頭設定不要再改。

  python3 record.py ../runs/track.mp4               # Ctrl+C 結束
  python3 record.py ../runs/curve.mp4 -t 20         # 錄 20 秒
  python3 record.py ../runs/a.mp4 --ctrl auto_exposure=1 --ctrl exposure_time_absolute=150
"""
import argparse
import os
import subprocess
import sys
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("-t", "--time", type=float, default=0, help="錄幾秒（0 = 直到 Ctrl+C）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--ctrl", action="append", default=[], help="v4l2-ctl 設定，例如 auto_exposure=1")
    args = ap.parse_args()

    for c in args.ctrl:
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{args.device}", "-c", c])
    api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.device, api)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 /dev/video{args.device}")
    for _ in range(10):
        cap.read()
    ok, f = cap.read()
    if not ok:
        raise SystemExit("讀取影像失敗")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (f.shape[1], f.shape[0]))
    print(f"錄影中 {f.shape[1]}x{f.shape[0]} → {args.out}（Ctrl+C 結束）")
    n, t0 = 0, time.time()
    try:
        while not args.time or time.time() - t0 < args.time:
            ok, f = cap.read()
            if not ok:
                continue
            w.write(f)
            n += 1
            if n % int(args.fps * 5) == 0:
                print(f"  {n} 張，{n / (time.time() - t0):.1f} fps")
    except KeyboardInterrupt:
        pass
    w.release()
    cap.release()
    dur = time.time() - t0
    print(f"完成：{n} 張，{dur:.1f} 秒，實際 {n / max(dur, 1e-3):.1f} fps")
    if n / max(dur, 1e-3) < args.fps * 0.8:
        print("注意：實際 fps 比設定低，影片播放速度會偏快；離線測試時速度相關的數字要打折看。")


if __name__ == "__main__":
    main()
