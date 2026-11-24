#!/usr/bin/env python3
"""小角度轉彎測試：不用繞圈，在一段直線車道上量出車子實際的轉彎半徑與「等效軸距」。

做法：
  1. 車子停在直線車道上、鏡頭看得到兩條線，先量「車子相對車道的角度與左右位置」
  2. 前輪打到指定角度往前開一小段（預設 1.0 秒，車頭轉 20–35° 最好），停車
  3. 再量一次。車頭轉了 θ、後輪軸往旁邊移了 Δ，圓弧半徑 R = Δ / (1 − cos θ)
  等效軸距 = R × tan(前輪指令角度)，直接填進 config.yaml 的 geometry.wheelbase（建議再 × 1.07）。
  打滑、連桿轉不足都會算進去，量的是在賽道墊上真正轉得多急。

用法（在 ~/orin_tools，車子放在直線起點正中央，前方至少 1 m 直線）：
  python3 tools/turn_test.py --fwa 31 --speed 26                 # 兩輪同速
  python3 tools/turn_test.py --fwa 31 --bwl 45 --bwr 5           # 加後輪差速（右轉時左輪快、右輪慢）
  python3 tools/turn_test.py --fwa -31 --bwl 5 --bwr 45          # 左轉
  python3 tools/turn_test.py --selftest                          # 用合成畫面驗證計算（不需硬體）

車頭只轉了 < 10°：加長 --time；量測後看不到線：縮短 --time 或把車往轉彎的反方向放一點。
鏡頭同時只能給一支程式用，先關掉 main.py / cam_tuner.py。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane"))
from vision import LaneDetector  # noqa: E402


def lane_pose(res):
    """車子（後輪軸中心）相對車道中線：(車道方向角 度，正 = 車道往左偏；中線在車子左邊多遠 m)。"""
    if res.center is None or res.pts is None or len(res.pts) < 6:
        return None
    # 測試用的是直線車道：中線用直線 y = b x + c 擬合（車身座標，x 前 y 左）。
    # 二次曲線往後外插到後輪軸（x = 0）時，一點點彎曲就會被放大成好幾公分的誤差。
    b, c = np.polyfit(res.pts[:, 0], res.pts[:, 1], 1)
    h = math.atan(b)
    return math.degrees(h), c * math.cos(h)   # 後輪軸到中線的垂直距離


def measure(grab, det, n=20):
    """取 n 張畫面量車道相對位置，回傳中位數；看不到線的張數太多就回傳 None。"""
    hs, ps = [], []
    for _ in range(n):
        det._coef = None                  # 每張獨立量，不要被前一張平滑影響
        pose = lane_pose(det.process(grab()))
        if pose:
            hs.append(pose[0])
            ps.append(pose[1])
    if len(hs) < n // 2:
        return None
    return float(np.median(hs)), float(np.median(ps)), len(hs)


def radius(p0, p1):
    """由前後兩次量測算圓弧半徑。回傳 (轉過角度 度，正 = 右轉；橫向移動 m；半徑 m)。"""
    theta = p1[0] - p0[0]                 # 車子右轉 θ → 原本正前方的車道看起來往左偏 θ
    shift = p1[1] - p0[1]                 # 車子往右移 → 中線相對變得更偏左
    one_minus_cos = 1 - math.cos(math.radians(theta))
    R = abs(shift) / one_minus_cos if one_minus_cos > 1e-4 else float("inf")
    return theta, shift, R


def report(theta, shift, R, fwa):
    print(f"\n車頭轉了 {theta:+.1f}°（正 = 右轉），後輪軸橫向移動 {shift * 100:+.1f} cm")
    if abs(theta) < 8:
        print("轉的角度太小，算出來誤差會很大：請加長 --time 再測一次。")
        return
    if theta * shift < 0:
        print("⚠ 轉向和橫移方向不一致（可能量測時線看不清楚），這次結果不可信，請再測一次。")
    L = R * math.tan(math.radians(abs(fwa)))
    print(f"轉彎半徑（後輪軸中心）約 {R:.2f} m")
    print(f"等效軸距 = {R:.2f} × tan({abs(fwa)}°) = {L:.3f} m → config 建議填 wheelbase: {L * 1.07:.2f}")
    print("直角彎（路寬 54 cm、車寬 30 cm）大約需要半徑 ≤ 0.6 m：" + ("✓ 做得到" if R <= 0.6 else "✗ 目前轉不夠急"))


def selftest(cfg):
    """用合成畫面驗證：給定真實的轉彎半徑，算出來的應該要一樣。"""
    import json
    import tempfile
    import lane_synth
    import lane_view_check as lvc
    g = cfg["geometry"]
    rend = dict(cam_h=g["cam_height"], pitch=18.7, hfov=g["hfov"], lane_w=g["lane_width"])
    det = LaneDetector(cfg, os.path.join(ROOT, "lane"))
    r0 = lvc.analyse(lane_synth.render(**rend), det.acfg)
    bf = os.path.join(tempfile.gettempdir(), "turn_test_base.json")
    json.dump({"h_near": r0["h_near"], "slope_deg": 0, "bow_l": 0, "bow_r": 0}, open(bf, "w"))
    cfg["vision"]["baseline"] = bf
    d = g["cam_to_rear_axle"]

    def shot(x_rear, yaw_deg):
        # x_rear：後輪軸在車道中線右邊多遠；yaw 正 = 車頭偏左。鏡頭在後輪軸前方 d
        cam_x = x_rear - d * math.sin(math.radians(yaw_deg))
        img = lane_synth.render(**rend, cam_x=cam_x, yaw=yaw_deg, noise=4.0)
        return lambda: img

    print("真實半徑  轉角   →  量到的半徑")
    ok = True
    for R_true, th in ((0.6, 25), (0.6, 35), (1.0, 30), (1.5, 25), (1.5, 20)):
        det = LaneDetector(cfg, os.path.join(ROOT, "lane"))
        p0 = measure(shot(0.0, 0.0), det, 3)
        for k in range(1, 11):                                # 行駛中一路餵畫面（實車也是），才分得出剩下的是哪條線
            a = th * k / 10
            det.process(shot(R_true * (1 - math.cos(math.radians(a))), -a)())
        x1 = R_true * (1 - math.cos(math.radians(th)))       # 右轉：後輪軸往右移
        p1 = measure(shot(x1, -th), det, 3)
        if not p0 or not p1:
            print(f"  {R_true:.1f} m  {th}°  →  看不到線")
            ok = False
            continue
        theta, shift, R = radius(p0, p1)
        err = (R - R_true) / R_true
        ok &= abs(err) < 0.15
        print(f"  {R_true:.1f} m  {th:2d}°  →  {R:.2f} m（轉角量到 {theta:+.1f}°，誤差 {err:+.0%}）")
    print("自我測試", "通過" if ok else "未通過（誤差 > 15%）")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(ROOT, "lane", "config.yaml"))
    ap.add_argument("--fwa", type=float, default=31, help="前輪角度（正 = 右轉）")
    ap.add_argument("--speed", type=float, default=26, help="兩輪同速時的速度 %%")
    ap.add_argument("--bwl", type=float, help="左後輪 %%（和 --bwr 一起給就用差速）")
    ap.add_argument("--bwr", type=float, help="右後輪 %%")
    ap.add_argument("--time", type=float, default=1.0, help="往前開幾秒")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.selftest:
        sys.exit(0 if selftest(cfg) else 1)

    from camera import Camera
    from motor_cmd import BMduinoLink
    cc, lk = cfg["camera"], cfg["link"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    det = LaneDetector(cfg, os.path.join(ROOT, "lane"))
    last = [0]

    def grab():
        got = cam.read(last[0])
        last[0] = got[0]
        return got[2]

    bwl = args.bwl if args.bwl is not None else args.speed
    bwr = args.bwr if args.bwr is not None else args.speed
    link = BMduinoLink(lk["port"], lk["baud"])
    try:
        link.stop()
        time.sleep(0.5)
        p0 = measure(grab, det)
        if not p0:
            raise SystemExit("起點看不到車道線：把車放在直線上、確認除錯畫面兩條線都有抓到。")
        print(f"起點：車道方向 {p0[0]:+.1f}°、中線在左邊 {p0[1] * 100:+.1f} cm（{p0[2]} 張有效）")
        if abs(p0[1]) > 0.06 or abs(p0[0]) > 6:
            raise SystemExit("起點車子沒有擺正（偏離中線超過 6 cm 或車頭歪超過 6°）：結束後線容易認錯、結果不準。\n"
                             "請把車放在兩條線正中間、車頭對準前方再測。")
        print(f"出發：前輪 {args.fwa:+.0f}°、左輪 {bwl:.0f}%、右輪 {bwr:.0f}%、{args.time:.1f} 秒")
        for _ in range(5):                        # 先打方向再走
            link.send(args.fwa, 0, 0)
            time.sleep(0.05)
        t_end = time.time() + args.time
        # 行駛中也一直辨識：只剩一條線時要靠前一張的中線判斷它是左線還是右線，中間不能斷掉
        t_send = 0.0
        while time.time() < t_end:
            if time.time() - t_send >= 0.05:
                link.send(args.fwa, bwl, bwr)
                link.poll()
                t_send = time.time()
            det.process(grab())
        t_stop = time.time() + 0.5
        while time.time() < t_stop:               # 停車、前輪維持不動，等車停穩
            if time.time() - t_send >= 0.05:
                link.send(args.fwa, 0, 0)
                t_send = time.time()
            det.process(grab())
        p1 = measure(grab, det)
        link.stop()
        if not p1:
            raise SystemExit("停下來後看不到車道線：縮短 --time，或出發前把車往轉彎的反方向放一點。")
        print(f"終點：車道方向 {p1[0]:+.1f}°、中線在左邊 {p1[1] * 100:+.1f} cm（{p1[2]} 張有效）")
        report(*radius(p0, p1), args.fwa)
    finally:
        link.close()
        cam.close()


if __name__ == "__main__":
    main()
