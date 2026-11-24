#!/usr/bin/env python3
"""鏡頭對準：拿車上固定不動的前板 / 底板當基準，看鏡頭和「校正當時」差多少度。
車子停著就能用，不用擺在賽道上（主程式要先關，鏡頭才不會被佔用）。

  python3 tools/cam_align.py              # 每 0.5 秒印一行，一邊轉鏡頭一邊看，三項都 OK 就對準了；Ctrl+C 結束
  python3 tools/cam_align.py --once       # 只量一次
  python3 tools/cam_align.py --save-ref   # 用現在的鏡頭重做基準（重跑 tools/calibrate.py 之後才做）

基準檔 tools/plate_ref.npz：2026-11-18 15:19 校正（calib.json：地平線 8.3、中心 175.9）之後
10 次出發前靜止畫面的前板邊緣中位數。鏡頭被動到 → 前板在畫面上的位置跟著移 → 換算成度數。
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
REF = os.path.join(HERE, "tools", "plate_ref.npz")

Y_CUR = 260                  # 現在的畫面從這一列往下比（640×480）
Y_REF, Y_REF_END = 330, 410  # 基準：前板上半部這幾列（下半部留給鏡頭往上抬時的搜尋空間，約 ±8°）
X_M = 60                     # 左右各留 60 px 搜尋（約 ±7.5°）
TOL = 0.5                    # 三項都在 ±0.5° 內算對準
MIN_SCORE = 0.4              # 相似度低於這個 = 比對不到（車子在動、前板被擋住、鏡頭動太多）
GOOD_SCORE = 0.6             # 低於這個還是印數字，但標「參考用」（前板反光會讓相似度掉到 0.5 左右，數字仍準：和車道線量的一致）


def edges(gray):
    g = gray.astype(np.float32)
    m = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 2)
    return m / (m.max() + 1e-6)


def template_from(grays):
    """多張靜止畫面的前板邊緣取中位數（地面每次不同會被濾掉），只留最強的 15%。"""
    tpl = np.median(np.stack([edges(g[Y_REF:Y_REF_END]) for g in grays]), 0)
    tpl = np.where(tpl > np.percentile(tpl, 85), tpl, 0)
    return tpl[:, X_M:-X_M].astype(np.float32)


def measure(gray, tpl, f):
    """回傳 (左右°, 上下°, 歪斜°, 相似度)。正 = 鏡頭往右轉 / 往下轉 / 往右歪 了。"""
    cur = edges(gray[Y_CUR:])
    H, W = cur.shape
    center = (W / 2, gray.shape[0] / 2 - Y_CUR)     # 鏡頭歪斜是繞畫面中心轉
    best = None
    for roll in np.arange(-3.0, 3.01, 0.25):
        R = cv2.getRotationMatrix2D(center, roll, 1.0)
        res = cv2.matchTemplate(cv2.warpAffine(cur, R, (W, H)), tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if best is None or mv > best[0]:
            best = (mv, ml[0] - X_M, ml[1] - (Y_REF - Y_CUR), roll)
    mv, dx, dy, roll = best
    # 前板在畫面上往右移 dx = 鏡頭往左轉；往下移 dy = 鏡頭往上抬；
    # 要把現在的畫面逆時針轉 roll 才對得上 = 畫面順時針歪了 = 鏡頭往左歪（從車後往前看）
    yaw = -math.degrees(math.atan(dx / f))
    pitch = -math.degrees(math.atan(dy / f))
    return yaw, pitch, -roll, mv


def hint(v, pos, neg):
    if abs(v) <= TOL:
        return "OK"
    return f"{pos if v > 0 else neg} {abs(v):.1f}°"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--save-ref", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    cc = cfg["camera"]
    from camera import Camera
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    f = (cc["width"] / 2) / math.tan(math.radians(cfg["geometry"]["hfov"] / 2))
    time.sleep(1.0)
    last = 0

    def grab(n):
        nonlocal last
        out = []
        while len(out) < n:
            got = cam.read(last)
            if got is None:
                continue
            last, _, fr = got
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        return out

    try:
        if args.save_ref:
            print("重做基準：車子停著不要動，拍 30 張……")
            tpl = template_from(grab(30)[::3])
            np.savez_compressed(REF, tpl=tpl.astype(np.float16), created=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"已存 {REF}")
            return
        z = np.load(REF)
        tpl = z["tpl"].astype(np.float32)
        print(f"基準：{str(z['created'])}。車子停著；目標：左右、上下、歪斜都 OK（±{TOL}°）")
        while True:
            vals = [measure(g, tpl, f) for g in grab(3)]
            yaw, pitch, roll, sc = [float(np.median([v[i] for v in vals])) for i in range(4)]
            if sc < MIN_SCORE:
                print(f"比對不到（相似度 {sc:.2f}）：車子要停著、前板不要被擋住；鏡頭差太多就先大概轉回來")
            else:
                ok = max(abs(yaw), abs(pitch), abs(roll)) <= TOL
                print(f"左右 {yaw:+5.1f}°（{hint(yaw, '鏡頭偏右，往左轉', '鏡頭偏左，往右轉')}）  "
                      f"上下 {pitch:+5.1f}°（{hint(pitch, '鏡頭太低，往上抬', '鏡頭太高，往下壓')}）  "
                      f"歪斜 {roll:+5.1f}°（{hint(roll, '往右歪，轉回左', '往左歪，轉回右')}）  "
                      f"相似度 {sc:.2f}" + ("（參考用）" if sc < GOOD_SCORE else "") + ("  ← 對準了" if ok else ""), flush=True)
            if args.once:
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
