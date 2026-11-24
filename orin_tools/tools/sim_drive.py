#!/usr/bin/env python3
"""閉迴路模擬：用 lane_synth 依車子目前位置畫出鏡頭畫面 → 交給 main.Pilot → 用自行車模型移動車子。

不需要硬體，用來確認轉向方向正確、Pure Pursuit 參數穩不穩、狀態機與速度規劃有沒有照預期動作。
模擬的鏡頭參數要和 config.yaml 的 geometry 一致（這裡會自動套用）。

  python3 sim_drive.py                      # 預設 S 彎賽道
  python3 sim_drive.py --track slope        # 直線 → 上坡 → 坡頂平台 → 下坡
  python3 sim_drive.py --video sim.mp4      # 輸出疊圖影片
  python3 sim_drive.py --e0 0.08 --psi0 8   # 從偏 8 cm、歪 8 度出發
"""
import argparse
import copy
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane"))
import lane_synth  # noqa: E402
import lane_view_check as lvc  # noqa: E402
from main import Pilot  # noqa: E402

# 賽道：[(起點 s, 曲率 1/m, 坡度 度), ...]，每段一直維持到下一段起點
TRACKS = {
    "s_curve": [(0, 0, 0), (1.0, 1 / 1.0, 0), (2.6, 0, 0), (3.2, -1 / 0.9, 0), (4.6, 0, 0)],
    "tight":   [(0, 0, 0), (0.8, 1 / 0.6, 0), (2.4, 0, 0)],
    "slope":   [(0, 0, 0), (1.0, 0, 12), (2.0, 0, 0), (2.6, 0, -12), (3.6, 0, 0)],
    "field":   [(0, 0, 0), (1.6, -1 / 0.45, 0), (2.3, 0, 0)],   # 測試場地：1.6 m 直線接半徑約 45 cm 右彎
    "field_tight": [(0, 0, 0), (1.6, -1 / 0.33, 0), (2.1, 0, 0)],  # 實車 log 量到的彎：半徑約 33 cm
    "corner_r":    [(0, 0, 0), (1.6, -1 / 0.28, 0), (1.6 + 0.44, 0, 0)],   # 直角右彎：內側線是尖角（中線半徑 0.28 ≈ 半個路寬）
    "corner_l":    [(0, 0, 0), (1.6, 1 / 0.28, 0), (1.6 + 0.44, 0, 0)],    # 直角左彎
    "s_rl":        [(0, 0, 0), (1.0, -1 / 0.9, 0), (2.6, 1 / 0.9, 0), (4.2, 0, 0)],   # 右彎馬上接左彎：右彎限制左修時，左彎要轉得過來
    "long_r":      [(0, 0, 0), (1.0, -1 / 1.2, 0), (5.0, 0, 0)],   # 長右彎（正式賽道有好幾段）：看彎道裡會不會一直偏外側
    "straight":    [(0, 0, 0), (7.0, 0, 0)],   # 長直線：看高速直線會不會蛇行（從偏移出發）
    "tunnel":      [(0, 0, 0), (3.0, 1 / 1.5, 0), (4.6, 0, 0)],   # 隧道前：長直線接緩左彎（12/30 實車撞柱子的地方）
    "mixed":       [(0, 0, 0), (2.0, -1 / 1.5, 0), (3.6, 0, 0), (4.6, 1 / 0.8, 0), (5.6, 0, 0), (6.6, -1 / 0.5, 0), (7.3, 0, 0)],   # 緩右彎 → 中彎左 → 急右彎：看不同緩急的彎速度和誤差
}


def profile(track, col):
    starts = np.array([t[0] for t in track])
    vals = np.array([t[col] for t in track], float)
    return lambda s: vals[np.clip(np.searchsorted(starts, s, side="right") - 1, 0, None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=os.path.join(ROOT, "lane", "config.yaml"))
    ap.add_argument("--track", choices=TRACKS, default="s_curve")
    ap.add_argument("--length", type=float, default=0, help="跑多遠（m），0 = 賽道最後一段起點 + 1 m")
    ap.add_argument("--speed-scale", type=float, default=1.5, help="速度 100%% 對應多少 m/s")
    ap.add_argument("--pitch", type=float, default=28.0, help="模擬鏡頭俯角（度）")
    ap.add_argument("--servo-tau", type=float, default=0.08, help="舵機反應時間常數（s）")
    ap.add_argument("--delay", type=int, default=2, help="指令延遲幾張畫面（模擬取像 + 傳輸延遲）")
    ap.add_argument("--fps", type=float, default=30.0, help="鏡頭幀率（實車 log 的 fps 欄位）")
    ap.add_argument("--e0", type=float, default=0.0, help="起始橫向偏移 m（正 = 偏左）")
    ap.add_argument("--psi0", type=float, default=0.0, help="起始車頭偏角 度（正 = 偏左）")
    ap.add_argument("--video", help="輸出疊圖影片")
    ap.add_argument("--roi-top", type=float, help="ROI 上緣（預設 0.1）")
    ap.add_argument("--roi-bottom", type=float, help="ROI 下緣（預設 1.0；畫面下方是車身時照實車設定）")
    ap.add_argument("--real-wheelbase", type=float, help="實車軸距 m（和 config 不同時，模擬 config 填錯）")
    ap.add_argument("--steer-gain", type=float, default=1.0, help="實際輪子轉角 / 指令轉角（< 1 = 連桿轉不足）")
    ap.add_argument("--steer-bias", type=float, default=0.0,
                    help="舵機中心偏差（度，加在實際輪子角度上；負 = 指令 0 時往左偏，實車約 −2.7）")
    ap.add_argument("--steer-max", type=float, default=33.0, help="輪子機構最大轉角（度）")
    ap.add_argument("--car-width", type=float, default=0.0, help="車寬 m：給了就檢查車身有沒有壓過白線（中心）")
    ap.add_argument("--min-move", type=float, default=0.0, help="速度低於幾 %% 車子推不動（模擬馬達扭力不足）")
    ap.add_argument("--imu", action="store_true", help="把模擬的車頭角當成 IMU 陀螺儀給 Pilot")
    ap.add_argument("--set", action="append", default=[], metavar="段.參數=值",
                    help="覆寫 config，例如 --set control.lookahead=0.6 --set geometry.cam_height=0.15")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        key, _, val = kv.partition("=")
        *path, name = key.split(".")          # 可以多層，例如 control.corner.enable=true
        d = cfg
        for k in path:
            d = d.setdefault(k, {})
        d[name] = yaml.safe_load(val)
    g = cfg["geometry"]
    track = TRACKS[args.track]
    kap, grd = profile(track, 1), profile(track, 2)
    length = args.length or track[-1][0] + 1.0
    rend = dict(cam_h=g["cam_height"], pitch=args.pitch, hfov=g["hfov"], lane_w=g["lane_width"], noise=4.0)

    # 和實車一樣：先在平坦直線上建立基準
    cfg = copy.deepcopy(cfg)
    roi_t = args.roi_top if args.roi_top is not None else 0.1
    roi_b = args.roi_bottom if args.roi_bottom is not None else 1.0
    cfg["vision"].update(roi_top=roi_t, roi_bottom=roi_b)
    acfg = Pilot(cfg).det.acfg
    r0 = lvc.analyse(lane_synth.render(**rend), acfg)
    base = {"h_near": r0["h_near"], "slope_deg": 0.0, "bow_l": 0.0, "bow_r": 0.0, "roi_top": roi_t, "roi_bottom": roi_b}
    r1 = lvc.analyse(lane_synth.render(**rend), acfg, base)
    base.update(slope_deg=r1["slope_deg"] or 0.0, bow_l=r1["bow_l"] or 0.0, bow_r=r1["bow_r"] or 0.0)
    bfile = os.path.join(tempfile.gettempdir(), "sim_baseline.json")
    with open(bfile, "w", encoding="utf-8") as f:
        json.dump(base, f)
    cfg["vision"]["baseline"] = bfile
    pilot = Pilot(cfg)
    pilot.start()

    L, d_cam = args.real_wheelbase or g["wheelbase"], g["cam_to_rear_axle"]
    dt = 1 / args.fps
    s, e, psi = 0.0, args.e0, math.radians(args.psi0)   # 後輪軸在賽道上的里程、偏移（左正）、偏角（左正）
    delta = 0.0
    theta = 0.0                                          # 賽道本身轉了多少（rad），加上 psi 就是車頭的絕對方向
    queue = [(0.0, 0.0, 0.0)] * args.delay
    log, writer, t = [], None, 0.0
    while s < length and t < 60:
        # 鏡頭位置（相對中線）
        e_cam = e + d_cam * math.sin(psi)
        s_cam = s + d_cam * math.cos(psi)
        g_here = float(grd(np.array([s]))[0])
        img = lane_synth.render(**rend, cam_x=-e_cam, yaw=math.degrees(psi),
                                kappa_fn=lambda q: kap(q + s_cam),
                                grade_fn=lambda q: grd(q + s_cam) - g_here,   # 車身跟著坡道傾斜
                                seed=len(log))
        if args.imu:
            pilot.imu_yaw = math.degrees(theta + psi)
            pilot.pitch = g_here                          # 車身俯仰 = 賽道坡度（平地 0，進坡加速不會在平地急彎誤觸發）
        cmd = pilot.step(img, t)
        queue.append(cmd)
        fwa, bwl, bwr = queue.pop(0)
        if args.video:
            vis = pilot.overlay()
            if writer is None:
                writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        # 自行車模型（後輪軸）
        v_pct = (bwl + bwr) / 2
        v = 0.0 if v_pct < args.min_move else v_pct / 100 * args.speed_scale
        want = max(-args.steer_max, min(args.steer_max, -(fwa + args.steer_bias) * args.steer_gain))
        delta += (math.radians(want) - delta) * min(1.0, dt / args.servo_tau)
        k = float(kap(np.array([s]))[0])
        ds = v * math.cos(psi) / (1 - k * e) * dt
        e += v * math.sin(psi) * dt
        psi += v * math.tan(delta) / L * dt - k * ds
        s += ds
        theta += k * ds
        res = pilot.res
        log.append((t, s, e, math.degrees(psi), fwa, (bwl + bwr) / 2, pilot.state, res.confidence,
                    res.offset, res.shape, res.slope, k, g_here))
        t += dt
        if pilot.state == "STOP":
            break
    if writer:
        writer.release()

    arr = np.array([(r[1], r[2], r[4], r[5]) for r in log])
    half = g["lane_width"] / 2
    print(f"賽道 {args.track}：跑了 {s:.2f} / {length:.2f} m，{t:.1f} 秒，{len(log)} 張")
    print(f"橫向誤差：最大 {np.abs(arr[:, 1]).max() * 100:.1f} cm，RMS {np.sqrt((arr[:, 1] ** 2).mean()) * 100:.1f} cm"
          f"（路寬一半 {half * 100:.0f} cm）")
    est = np.array([r[8] for r in log if r[7] > 0])
    true = np.array([-r[2] for r in log if r[7] > 0])  # 偏左 e>0 → 中線在右 → offset 應為負
    if len(est):
        print(f"偏移量估計誤差（辨識 vs 真值，後輪軸處）：RMS {np.sqrt(((est - true) ** 2).mean()) * 100:.1f} cm")
    sw = arr[len(arr) // 3:, 1]                         # 前 1/3 是從偏移修回來，之後看還有沒有在擺
    print(f"後 2/3 橫向擺動：{(sw.max() - sw.min()) * 100:.1f} cm（峰對峰）")
    print(f"轉角範圍 {arr[:, 2].min():+.0f} ~ {arr[:, 2].max():+.0f} 度；速度 {arr[:, 3].min():.0f} ~ {arr[:, 3].max():.0f} %")
    print(f"狀態：{ {st: sum(r[6] == st for r in log) for st in ('RUN', 'LOST', 'STOP')} }  結束原因：{pilot.reason or '跑完'}")
    print("\n   s(m)  偏移cm  偏角°   fwa  速度  狀態 線  路段")
    for r in log[:: max(1, len(log) // 25)]:
        print(f"  {r[1]:5.2f}  {r[2] * 100:+6.1f}  {r[3]:+5.1f}  {r[4]:+4.0f}  {r[5]:4.0f}  {r[6]:<4} {r[7]:.1f}  "
              f"{r[9]}/{r[10]}  (真實曲率 {r[11]:+.2f}, 坡 {r[12]:+.0f}°)")
    if s < length * 0.98:
        print(f"沒有跑完：停在 {s:.2f} m（速度低於 --min-move 時車子推不動，或時間到了）")
    limit = half - args.car_width / 2 if args.car_width else half * 0.6
    if args.car_width:
        print(f"車寬 {args.car_width * 100:.0f} cm：車子中心最多只能偏 {limit * 100:.1f} cm，"
              + ("車身沒有壓線" if np.abs(arr[:, 1]).max() < limit else "車身壓到白線了"))
    ok = pilot.state != "STOP" and np.abs(arr[:, 1]).max() < limit and s >= length * 0.98
    print("\n結果：", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
