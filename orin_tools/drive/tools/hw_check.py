#!/usr/bin/env python3
"""硬體檢查（測試前跑）：序列埠 / IMU / 鏡頭 / 舵機 / 馬達。主程式要先關（鏡頭、序列埠才不會被佔用）。

  python3 tools/hw_check.py                 # 1–3 自動檢查：序列埠、IMU、鏡頭（車子擺在直線正中間）
  python3 tools/hw_check.py --servo         # 4. 舵機：左 → 中 → 右 → 中，各停 1.5 s（看前輪）
  python3 tools/hw_check.py --motor         # 5. 馬達：後輪架高！左右輪各正轉 / 反轉 1 s（看輪子）
  python3 tools/hw_check.py --ground        # 6. 地上實測：車前要有 1.5 m 空地，慢速直走 + 左 / 右小彎，用陀螺儀量轉向

鏡頭檢查會和 calib.json 比：地平線、正前方差太多 = 鏡頭被撞歪了，要調回去或重跑 tools/calibrate.py。
"""
import argparse
import math
import os
import statistics as st
import sys
import time

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))

OK, BAD = "OK  ", "異常"
results = []


def report(name, ok, msg):
    results.append((name, ok))
    print(f"[{OK if ok else BAD}] {name}：{msg}", flush=True)


def check_link(cfg):
    from motor_cmd import BMduinoLink
    lk = cfg["link"]
    link = BMduinoLink(lk["port"], lk["baud"])
    t0 = time.time()
    while time.time() - t0 < 1.5:
        link.send(cfg["steer"]["fwa_trim"], 0, 0)
        link.poll()
        time.sleep(0.05)
    age = time.time() - link.sta_time if link.sta_time else 99
    report("序列埠 / BMduino", age < 0.5, f"STA 有效 {link.good}、無效 {link.bad}，最後 {age:.2f} s 前 {link.sta}")
    return link


def check_imu(cfg):
    ic = cfg["imu"]
    try:
        from imu import IMU
        imu = IMU(ic.get("bus", 1), ic.get("addr", 0x68), forward=ic.get("forward", "+x"))
    except OSError as e:
        report("IMU", False, f"開不起來：{e}")
        return None
    time.sleep(1.5)
    r0 = imu.read()
    time.sleep(2.0)
    r1 = imu.read()
    if r0 is None or r1 is None:
        report("IMU", False, "讀不到資料")
        return imu
    drift = (r1[1] - r0[1]) / 2.0
    report("IMU 俯仰", abs(r1[3]) < 5, f"{r1[3]:+.1f}°（平地應在 ±5° 內）")
    report("IMU 陀螺儀", abs(drift) < 1.5, f"靜止時車頭角漂移 {drift:+.2f}°/s（應 < 1.5）")
    return imu


def check_camera(cfg, n=60):
    from camera import Camera
    from calibrate import lines_in
    import json
    cc, v = cfg["camera"], cfg["vision"]
    W, H = v["work_w"], v["work_h"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    time.sleep(1.0)
    last, ts, vhs, cxs, bright = 0, [], [], [], []
    t_end = time.time() + 6
    import cv2
    while len(ts) < n and time.time() < t_end:
        got = cam.read(last)
        if got is None:
            continue
        last, t, f = got
        ts.append(t)
        g = cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2GRAY)
        bright.append(float(g[H // 4:int(v["roi_bottom"] * H)].mean()))
        l, r = lines_in(f, W, H, v["roi_bottom"])
        if l and r:
            (_, a1, b1), (_, a2, b2) = l, r
            vy = (b2 - b1) / (a1 - a2)
            vhs.append(vy)
            cxs.append(a1 * vy + b1)
    cam.close()
    if len(ts) < 10:
        report("鏡頭", False, f"{len(ts)} 張畫面（鏡頭沒畫面？）")
        return
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    report("鏡頭畫面", fps >= 25, f"{fps:.1f} fps（應 ≥ 25），亮度 {st.median(bright):.0f}")
    path = os.path.join(HERE, cfg["geometry"].get("calib", "calib.json"))
    if len(vhs) < n // 3:
        report("鏡頭角度", False, f"只有 {len(vhs)}/{len(ts)} 張看到兩條直線：車子要擺在直線正中間；"
                              "擺好了還是看不到 = 鏡頭歪了")
        return
    vh, cx = st.median(vhs), st.median(cxs)
    if not os.path.exists(path):
        report("鏡頭角度", False, f"沒有 {path}，先跑 tools/calibrate.py")
        return
    c = json.load(open(path, encoding="utf-8"))
    dv, du = vh - c["horizon"], cx - c["center"]
    f = (W / 2) / math.tan(math.radians(cfg["geometry"]["hfov"] / 2))
    dpitch, dyaw = math.degrees(math.atan(dv / f)), math.degrees(math.atan(du / f))
    ok = abs(dpitch) < 1.5 and abs(dyaw) < 2.0
    report("鏡頭角度", ok, f"和校正時比：上下差 {dpitch:+.1f}°、左右差 {dyaw:+.1f}°（應在 ±1.5° / ±2° 內）"
                          + ("" if ok else " → 鏡頭被動到了：調回去，或車子擺直線中間重跑 tools/calibrate.py"))


def hold(link, fwa, bwl, bwr, sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        link.send(int(round(fwa)), int(bwl), int(bwr))
        link.poll()
        time.sleep(0.05)


def check_servo(cfg, link):
    trim = cfg["steer"]["fwa_trim"]
    for name, a in (("左", trim - 30), ("中", trim), ("右", trim + 30), ("中", trim)):
        print(f"  前輪打{name}（指令 {a:+.0f}°）", flush=True)
        hold(link, a, 0, 0, 2.5)
    print("  看前輪：左右打的角度要差不多、回中要直。連桿有晃動 / 一邊打不到 = 鎖緊連桿", flush=True)


def check_motor(cfg, link):
    trim = cfg["steer"]["fwa_trim"]
    for name, l, r in (("左輪正轉", 30, 0), ("左輪反轉", -30, 0), ("右輪正轉", 0, 30), ("右輪反轉", 0, -30),
                       ("兩輪正轉", 30, 30)):
        print(f"  {name}（{l}/{r}%）", flush=True)
        hold(link, trim, l, r, 2.0)
        hold(link, trim, 0, 0, 1.0)
    print("  看後輪：每一步只有該轉的輪子轉、方向對、聲音正常", flush=True)


def check_ground(cfg, link, imu, raw=False):
    if imu is None:
        report("地上實測", False, "沒有 IMU，量不到")
        return
    sc = cfg["steer"]
    trim, v = sc["fwa_trim"], 35
    # 照行駛時的左右補償打（steer_k_left / steer_k_right）；--raw 不補償
    kl, kr = (1.0, 1.0) if raw else (sc.get("steer_k_left", 1.0), sc.get("steer_k_right", 1.0))
    lim = sc["fwa_limit"]
    al, ar = max(-lim, trim - 25 * kl), min(lim, trim + 25 * kr)
    print(f"  前輪：左彎 {al:+.0f}°、右彎 {ar:+.0f}°（置中 {trim:+.1f}°，補償 左×{kl} 右×{kr}）", flush=True)
    out = {}
    for name, a in (("直走", trim), ("左彎", al), ("右彎", ar)):
        hold(link, a, 0, 0, 0.6)
        y0 = imu.read()[1]
        hold(link, a, v, v, 0.5)                 # 起步
        y1 = imu.read()[1]
        hold(link, a, v, v, 1.0)
        y2 = imu.read()[1]
        hold(link, trim, -10, -10, 0.3)
        hold(link, trim, 0, 0, 1.0)
        out[name] = y2 - y1
        print(f"  {name}：車頭 1 s 轉了 {y2 - y1:+.0f}°（起步 {y1 - y0:+.0f}°）", flush=True)
    report("地上直走", abs(out["直走"]) < 8, f"直走時車頭 {out['直走']:+.0f}°/s（應 < 8，太大 = fwa_trim 不對或前輪歪）")
    l, r = out["左彎"], -out["右彎"]
    ok = l > 10 and r > 10 and 0.75 < l / max(r, 1e-3) < 1.33
    report("地上轉向", ok, f"左 {l:.0f}°/s、右 {r:.0f}°/s（左右應差不多；一邊很小 = 舵機 / 連桿有問題）")
    if l > 3 and r > 3:
        tgt = (l + r) / 2
        print(f"  建議補償（讓左右都約 {tgt:.0f}°/s）：config.yaml steer: steer_k_left: {kl * tgt / l:.2f}、"
              f"steer_k_right: {kr * tgt / r:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--servo", action="store_true")
    ap.add_argument("--motor", action="store_true")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--raw", action="store_true", help="--ground 不套用左右補償（量舵機原本的樣子）")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sys.path.insert(0, os.path.join(HERE, "tools"))
    link = check_link(cfg)
    imu = check_imu(cfg)
    try:
        if not (args.servo or args.motor or args.ground):
            check_camera(cfg)
        if args.servo:
            check_servo(cfg, link)
        if args.motor:
            check_motor(cfg, link)
        if args.ground:
            check_ground(cfg, link, imu, args.raw)
    finally:
        link.stop()
        link.close() if hasattr(link, "close") else None
        if imu:
            imu.close()
    bad = [n for n, ok in results if not ok]
    print("\n全部正常" if not bad else f"\n有問題：{'、'.join(bad)}", flush=True)


if __name__ == "__main__":
    main()
