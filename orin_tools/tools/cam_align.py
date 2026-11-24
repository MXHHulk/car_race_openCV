#!/usr/bin/env python3
"""把鏡頭調回原本的俯仰角：畫面下方的車身木板是固定的，它上緣在第幾列，就知道鏡頭有沒有歪。

穩定版（run_20261101_141924）木板上緣在第 367 列。摔車後變 359，整個畫面往上移 8 列，
偏移換算、虛線位置、右線位置都會跟著錯。一邊跑這支程式一邊慢慢扳鏡頭，調到 TARGET ±1。

  python3 tools/cam_align.py
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lane"))
from camera import Camera  # noqa: E402

# 2026-11-03 狀況最好的三趟（193617、200840、210400）都在 374–377 列；2026-11-04 最後兩趟掉到 365
TARGET = 375


def plate_row(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col = g[300:470, 250:390].mean(1)
    return int(np.argmax(np.diff(col))) + 300


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(os.path.dirname(HERE), "lane", "config.yaml"))
    ap.add_argument("-t", "--target", type=int, default=TARGET)
    args = ap.parse_args()
    cc = yaml.safe_load(open(args.config, encoding="utf-8"))["camera"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    print(f"木板上緣目標第 {args.target} 列（±1 就好），Ctrl+C 結束")
    last, rows, t0 = 0, [], time.time()
    try:
        while True:
            got = cam.read(last)
            if got is None:
                continue
            last, _, frame = got
            rows.append(plate_row(frame))
            if time.time() - t0 >= 0.5:
                r = int(np.median(rows))
                d = r - args.target
                hint = "OK" if abs(d) <= 1 else ("鏡頭要往上抬" if d < 0 else "鏡頭要往下壓")
                print(f"第 {r} 列（差 {d:+d}）{hint}")
                rows, t0 = [], time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
