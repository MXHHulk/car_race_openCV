#!/usr/bin/env python3
"""後輪滿載轉速實測：車子放地上，用固定出力直走 t 秒後煞停，用捲尺量跑了多遠。
跑兩次不同秒數（例如 0.7 s、1.4 s），兩次距離相減 ÷ 秒數差 = 穩定車速（起步、煞車的距離兩次一樣，相減抵銷）。
主程式要先關（序列埠才不會被佔用）。車前要有足夠直線空地（100% 約 3–4 m）。

  python3 tools/speed_test.py --lane --t 1.5        # 循線（鏡頭修方向、不踩線）100% 跑 1.5 秒（大直線用這個）
  python3 tools/speed_test.py --lane --t 4.5        # 循線 100% 跑 4.5 秒（10.8 m 大直線約跑 8 m）
  python3 tools/speed_test.py --t 0.7               # 不循線（只用陀螺儀維持方向）100% 跑 0.7 秒
  python3 tools/speed_test.py --calc 0.7 1.20 1.4 2.35   # 兩次的秒數和距離（m）→ 車速和後輪轉速

跑的時候用陀螺儀把車頭維持在出發時的方向（後輪差速 + 前輪），不會跑歪。
"""
import argparse
import math
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

WHEEL_D = 0.115   # 後輪直徑（m）
EDGE = 0.17       # 循線模式：離中線超過這個（m）就中止（快踩線了）
LOST_S = 0.3      # 循線模式：看不到線超過這麼久就中止


def calc(t1, d1, t2, d2, diam):
    v = (d2 - d1) / (t2 - t1)
    rpm = v / (math.pi * diam) * 60
    print(f"穩定車速 {v:.2f} m/s（{v * 3.6:.1f} km/h）→ 後輪 {rpm:.0f} rpm（直徑 {diam * 100:.1f} cm）")
    return v, rpm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pct", type=float, default=100.0, help="出力 %（預設 100）")
    ap.add_argument("--t", type=float, default=0.7, help="全速跑幾秒（不含煞車）")
    ap.add_argument("--diam", type=float, default=WHEEL_D, help="後輪直徑（m）")
    ap.add_argument("--calc", type=float, nargs=4, metavar=("T1", "D1", "T2", "D2"), help="只計算，不跑")
    ap.add_argument("--lane", action="store_true", help="循線：用鏡頭跟車道線修方向（大直線用，確保不踩線）")
    args = ap.parse_args()
    if args.calc:
        calc(*args.calc, args.diam)
        return
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    if args.lane:
        run_lane(args, cfg)
        return
    from motor_cmd import BMduinoLink
    from imu import IMU
    lk, sc, ic = cfg["link"], cfg["steer"], cfg["imu"]
    imu = IMU(ic.get("bus", 1), ic.get("addr", 0x68), forward=ic.get("forward", "+x"),
              calib_path=os.path.join(HERE, "imu_calib.json"))
    link = BMduinoLink(lk["port"], lk["baud"])
    t0 = time.time()
    while imu.read() is None and time.time() - t0 < 20:
        link.send(sc["fwa_trim"], 0, 0)
        link.poll()
        time.sleep(0.05)
    r = imu.read()
    if r is None:
        print("IMU 沒校正好（車子要靜止），不跑")
        return
    yaw0 = r[1]
    trim, bias = sc["fwa_trim"], sc.get("diff_trim0", 0.0)
    v = max(0.0, min(100.0, args.pct))
    print(f"3 秒後用 {v:.0f}% 直走 {args.t:.2f} 秒，然後煞停。先在車子後輪軸的位置做記號！", flush=True)
    for k in (3, 2, 1):
        print(f"  {k}", flush=True)
        end = time.time() + 1.0
        while time.time() < end:
            link.send(trim, 0, 0)
            link.poll()
            time.sleep(0.05)
    try:
        start = time.time()
        while time.time() - start < args.t:
            e = imu.read()[1] - yaw0                       # 正 = 車頭往左偏了
            u = max(-15.0, min(15.0, bias * v - 1.5 * e))   # 右輪多 = 往左帶
            fwa = max(trim - 10, min(trim + 10, trim + 0.8 * e))
            l, rr = v - u, v + u
            hi = max(l, rr)
            if hi > 100:
                l, rr = l - (hi - 100), rr - (hi - 100)
            link.send(fwa, l, rr)
            link.poll()
            time.sleep(0.02)
        end = time.time() + 0.3
        while time.time() < end:                           # 反打煞車
            link.send(trim, -30, -30)
            link.poll()
            time.sleep(0.02)
    finally:
        for _ in range(10):
            link.send(trim, 0, 0)
            link.poll()
            time.sleep(0.05)
        link.stop()
        drift = imu.read()[1] - yaw0
        imu.close()
    print(f"完成：{v:.0f}% 跑 {args.t:.2f} 秒（車頭偏了 {drift:+.0f}°）。請量後輪軸從記號到現在的距離（m）。", flush=True)


def run_lane(args, cfg):
    """循線滿速：方向交給主程式的循線（含左右舵機補償），速度固定 pct。偏離中線 / 看不到線就中止煞停。"""
    from camera import Camera
    from driver import Driver
    from imu import IMU
    from link import Sender
    cc, lk, sc, ic = cfg["camera"], cfg["link"], cfg["steer"], cfg["imu"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    drv = Driver(cfg, HERE)
    imu = IMU(ic.get("bus", 1), ic.get("addr", 0x68), forward=ic.get("forward", "+x"),
              calib_path=os.path.join(HERE, "imu_calib.json"))
    snd = Sender(lk["port"], lk["baud"], lk["hz"])
    trim = sc["fwa_trim"]
    v = max(0.0, min(100.0, args.pct))
    bias = sc.get("diff_trim0", 0.0)
    snd.set(trim, 0, 0)
    t0 = time.time()
    while imu.read() is None and time.time() - t0 < 20:
        time.sleep(0.05)
    if imu.read() is None:
        print("IMU 沒校正好（車子要靜止），不跑")
        snd.close(); cam.close(); imu.close()
        return
    last = 0
    for _ in range(15):                                   # 先讓鏡頭和循線熱身
        got = cam.read(last)
        if got:
            last, tf, fr = got
            r = imu.read()
            drv.step(fr, tf, r[3], r[1], snd.sta_age())
    print(f"3 秒後循線用 {v:.0f}% 跑 {args.t:.2f} 秒，然後煞停。先在後輪軸位置做記號！"
          f"（離中線 > {EDGE * 100:.0f} cm 或看不到線 > {LOST_S} s 會自動中止）", flush=True)
    for k in (3, 2, 1):
        print(f"  {k}", flush=True)
        time.sleep(1.0)
    drv.start()
    start, lost_since, offs, why = time.time(), None, [], ""
    try:
        while time.time() - start < args.t:
            got = cam.read(last)
            if got is None:
                continue
            last, tf, fr = got
            r = imu.read()
            drv.v = v                                     # 前視距離照滿速算
            fwa, _, _ = drv.step(fr, tf, r[3], r[1], snd.sta_age())
            s, now = drv.s, time.time()
            if s.conf == 0:
                lost_since = lost_since or now
                if now - lost_since > LOST_S:
                    why = f"看不到線超過 {LOST_S} s"
                    break
            else:
                lost_since = None
                offs.append(s.offset)
                if abs(s.offset) > EDGE:
                    why = f"離中線 {s.offset * 100:+.0f} cm（快踩線）"
                    break
            u = bias * v
            l, rr = v - u, v + u
            hi = max(l, rr)
            if hi > 100:
                l, rr = l - (hi - 100), rr - (hi - 100)
            snd.set(fwa, l, rr)
        ran = time.time() - start
        end = time.time() + 0.3
        while time.time() < end:                          # 反打煞車
            snd.set(trim, -30, -30)
            time.sleep(0.02)
    finally:
        snd.set(trim, 0, 0)
        time.sleep(0.3)
        snd.close()
        cam.close()
        imu.close()
    if offs:
        import statistics as st
        print(f"偏離中線：平均 {st.mean(abs(o) for o in offs) * 100:.1f} cm、最大 {max(abs(o) for o in offs) * 100:.1f} cm")
    if why:
        print(f"⚠ 跑了 {ran:.2f} 秒就中止：{why}。這次距離不能用，修好再重跑", flush=True)
    else:
        print(f"完成：循線 {v:.0f}% 跑 {args.t:.2f} 秒。請量後輪軸從記號到現在的距離（m）", flush=True)


if __name__ == "__main__":
    main()
