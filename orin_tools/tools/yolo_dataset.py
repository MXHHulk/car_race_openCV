#!/usr/bin/env python3
"""從實跑紀錄的影片自動產生 YOLO 分割資料集（車道線），給 tools/yolo_train.py 訓練。

沒有人工標註：用現在的辨識程式（lane_view_check.analyse）當老師，只取「兩條線都找到」的畫面，
標註是被找線程式認定為車道線的白色像素（Otsu 二值化 ∩ 找線視窗），雜訊、膠帶、木條邊緣不算。
模型學到的是「現在程式認得的白線」，期望在只看到一條線、光線差的畫面也能把線抓出來。

  python3 tools/yolo_dataset.py                          # 預設：logs/2026-10-3*、2026-11-* 的影片
  python3 tools/yolo_dataset.py --every 10 --out ../yolo_lane
"""
import argparse
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane"))
import lane_view_check as lvc  # noqa: E402
from vision import LaneDetector  # noqa: E402


def polygons(mask, min_area=15):
    """遮罩 → YOLO 分割標註的多邊形（座標除以寬高）。"""
    h, w = mask.shape
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        out.append(" ".join(f"{x / w:.4f} {y / h:.4f}" for x, y in c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=os.path.join(ROOT, "..", "orin_snapshot", "config_5laps.yaml"))
    ap.add_argument("--baseline", default=os.path.join(ROOT, "..", "orin_snapshot", "baseline.json"))
    ap.add_argument("--logs", nargs="*", default=None, help="影片（預設 logs/2026-10-3*、2026-11-* 的 mp4）")
    ap.add_argument("--every", type=int, default=10, help="每幾張取一張")
    ap.add_argument("--max-per-video", type=int, default=60)
    ap.add_argument("--val", type=float, default=0.15, help="多少比例的影片當驗證集")
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "yolo_lane"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["vision"]["baseline"] = os.path.abspath(args.baseline)
    det = LaneDetector(cfg)
    vids = args.logs or sorted(glob.glob(os.path.join(ROOT, "logs", "2026-10-3*", "*.mp4"))
                               + glob.glob(os.path.join(ROOT, "logs", "2026-11-*", "*.mp4")))
    random.seed(0)
    val = set(random.sample(vids, max(1, int(len(vids) * args.val))))
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    n_img = {"train": 0, "val": 0}
    for v in vids:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            continue
        split = "val" if v in val else "train"
        name = os.path.splitext(os.path.basename(v))[0]
        # 只取正在跑（RUN）的畫面：待命時車子常被拿在手上，會把手、背景的白色也標進去
        states = []
        if os.path.exists(v[:-4] + ".csv"):
            with open(v[:-4] + ".csv", encoding="utf-8") as f:
                states = [row["state"] for row in csv.DictReader(f)]
        i = kept = 0
        while kept < args.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % args.every or (states and (i - 1 >= len(states) or states[i - 1] != "RUN")):
                continue
            r = lvc.analyse(frame, det.acfg, det.base)
            if r["n_lines"] != 2:
                continue
            mask = ((r["_bin"] > 0) & r["_claimed"]).astype(np.uint8) * 255
            polys = polygons(mask)
            if not polys:
                continue
            stem = f"{name}_{i:05d}"
            cv2.imwrite(os.path.join(args.out, "images", split, stem + ".jpg"), r["_img"])
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                f.write("".join(f"0 {p}\n" for p in polys))
            kept += 1
        n_img[split] += kept
        print(f"{name}: {kept} 張（{split}）")
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: lane\n")
    print(f"完成：train {n_img['train']} 張、val {n_img['val']} 張 → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
