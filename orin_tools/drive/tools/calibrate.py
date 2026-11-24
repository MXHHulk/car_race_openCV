#!/usr/bin/env python3
"""鏡頭校正：車子擺在平坦長直線正中間、車頭對齊車道，跑這支（主程式要先關，鏡頭才不會被佔用）。

兩條直線在畫面上的交點 = 地平線列 + 正前方那一欄；再用量到的路寬反推鏡頭高度。
鏡頭角度每動一次就重跑。結果寫到 drive/calib.json。

  python3 tools/calibrate.py               # 用鏡頭，拍 60 張
  python3 tools/calibrate.py --source x.mp4 --start 0 --frames 60   # 用錄影（離線測試）
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def lines_in(frame, W, H, roi_bottom, top_frac=0.15):
    g = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    t0, t1 = int(top_frac * H), int(roi_bottom * H)
    roi = cv2.medianBlur(g[t0:t1], 3)
    thr, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(110, min(200, thr))
    n, lab, stt, _ = cv2.connectedComponentsWithStats((roi > thr).astype(np.uint8), 8)
    fits = []
    for i in range(1, n):
        if stt[i, cv2.CC_STAT_HEIGHT] < 20 or stt[i, cv2.CC_STAT_AREA] < 40:
            continue
        ys, xs = np.nonzero(lab == i)
        rows, inv = np.unique(ys, return_inverse=True)
        cx = np.bincount(inv, xs) / np.bincount(inv)       # 每列的中心
        vv = rows + t0
        keep = np.ones(len(vv), bool)
        for _ in range(3):                                 # 去掉黏在線旁邊的雜物（白色物品、地板）
            a, b = np.polyfit(vv[keep], cx[keep], 1)       # u = a·v + b
            err = np.abs(cx - (a * vv + b))
            keep = err <= max(1.5, 2 * np.median(err[keep]))
        if keep.sum() < max(15, 0.6 * len(vv)) or err[keep].mean() > 2.0:   # 不夠直
            continue
        fits.append((stt[i, cv2.CC_STAT_AREA], a, b))
    left = max((f for f in fits if f[1] < -0.2), default=None)
    right = max((f for f in fits if f[1] > 0.2), default=None)
    return left, right


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--source")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=float, default=0.15, help="從畫面高度的多少比例開始找線（背景太亮時調大）")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    v, g = cfg["vision"], cfg["geometry"]
    W, H = v["work_w"], v["work_h"]
    f = (W / 2) / math.tan(math.radians(g["hfov"] / 2))
    if args.source:
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        read = lambda: cap.read()[1]
        close = cap.release
    else:
        from camera import Camera
        cc = cfg["camera"]
        cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
        state = {"id": 0}

        def read():
            got = cam.read(state["id"])
            if got is None:
                return None
            state["id"] = got[0]
            return got[2]
        close = cam.close
        time.sleep(1.0)
    vhs, cxs, used = [], [], 0
    lines = []
    for _ in range(args.frames):
        fr = read()
        if fr is None:
            break
        l, r = lines_in(fr, W, H, v["roi_bottom"], args.top)
        if not l or not r:
            continue
        (_, a1, b1), (_, a2, b2) = l, r
        vy = (b2 - b1) / (a1 - a2)
        vx = a1 * vy + b1
        vhs.append(vy)
        cxs.append(vx)
        lines.append((a1, b1, a2, b2))
        used += 1
    close()
    if used < max(5, args.frames // 3):
        print(f"只有 {used}/{args.frames} 張同時看到兩條直線。車子要擺在平坦直線正中間、兩條線都要在畫面裡。")
        sys.exit(1)
    vh, cx = st.median(vhs), st.median(cxs)
    # 路寬：在 ROI 下緣往上 20 列量兩條線的距離，反推鏡頭高度
    vrow = v["roi_bottom"] * H - 20
    ws = []
    for a1, b1, a2, b2 in lines:
        du = (a2 * vrow + b2) - (a1 * vrow + b1)
        x = g["cam_height"] * f / (vrow - vh)
        ws.append(du * x / f)
    w = st.median(ws)
    h = g["cam_height"] * g["lane_width"] / w
    yaw = math.degrees(math.atan((cx - W / 2) / f))
    out = {"horizon": round(vh, 2), "center": round(cx, 2), "w": W, "h": H, "cam_height": round(h, 4),
           "lane_width_at_h0": round(w, 3), "frames": used, "spread_horizon": round(st.pstdev(vhs), 2),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = args.out or os.path.join(HERE, g.get("calib", "calib.json"))
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"地平線第 {vh:.1f} 列（{H} 列高），正前方第 {cx:.1f} 欄（鏡頭往{'右' if yaw > 0 else '左'}偏 {abs(yaw):.1f}°），"
          f"鏡頭等效高度 {h:.3f} m（設定 {g['cam_height']}）。{used} 張，地平線抖動 ±{st.pstdev(vhs):.1f} 列")
    print(f"已存 {path}")
    if vh < 0:
        print("⚠ 地平線在畫面外（鏡頭太往下）：遠處看不到，大直線判斷會變弱")


if __name__ == "__main__":
    main()
