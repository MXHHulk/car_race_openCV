#!/usr/bin/env python3
"""量鏡頭與整套程式的實際幀率，找出 fps 低是卡在鏡頭、鏡頭設定，還是程式。

每一項量 5 秒：
  [0] 純辨識速度：同一張畫面重複辨識（不碰鏡頭），看 CPU 算一張要多久
  [1] 直接讀鏡頭（不做處理）
  [1b] 同上，但鏡頭緩衝設成 1 張（舊版 camera.py 的設定，會不會掉一半幀）
  [2] 套用 config.yaml 的鏡頭設定後讀鏡頭
  [3] 取像執行緒 + 辨識（OpenCV 預設執行緒數）
  [4] 同上，但 OpenCV 只用 1 個執行緒
  [5] 不用取像執行緒：讀一張、辨識一張
  [6] 和 main.py 完全一樣：取像執行緒 + 辨識 + 控制 + 除錯疊圖（用 [3]/[4] 比較快的設定）

用法（先關掉 main.py、cam_tuner.py 等用到鏡頭的程式）：
  cd ~/orin_tools && python3 tools/cam_fps.py
"""
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane"))
SEC = 5.0


def open_cap(dev, w, h, fps, bufsize=None):
    api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(dev, api)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if bufsize:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 /dev/video{dev}（其他程式還在用鏡頭？）")
    for _ in range(10):
        cap.read()
    return cap


def ctrl_state(dev):
    if not shutil.which("v4l2-ctl"):
        return "（沒有 v4l2-ctl）"
    r = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{dev}", "--get-ctrl", "auto_exposure,exposure_time_absolute"],
                       capture_output=True, text=True)
    return " ".join((r.stdout + r.stderr).split())


def report(name, times, proc):
    """times：每張畫面的取得時間；proc：每張的處理時間（s）。"""
    if len(times) < 2:
        print(f"{name}：沒有畫面")
        return 0.0
    gaps = np.diff(times)
    fps = (len(times) - 1) / (times[-1] - times[0])
    extra = f"，處理 {np.mean(proc) * 1000:.1f} ms/張" if proc else ""
    print(f"{name}：{fps:5.1f} fps（畫面間隔 平均 {gaps.mean() * 1000:.0f} ms、最大 {gaps.max() * 1000:.0f} ms{extra}）")
    return fps


def raw(cap):
    times = []
    t0 = time.time()
    while time.time() - t0 < SEC:
        ok, _ = cap.read()
        if ok:
            times.append(time.time())
    return times


def threaded(cfg, step, overlay=None):
    from camera import Camera
    cc = cfg["camera"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"])
    last, times, proc, t_ov = 0, [], [], 0.0
    t0 = time.time()
    while time.time() - t0 < SEC:
        got = cam.read(last)
        if got is None:
            continue
        last, t, frame = got
        t1 = time.time()
        step(frame, t)
        if overlay and t1 - t_ov >= 0.1:          # main.py 的除錯串流：每秒 10 次疊圖 + JPEG 編碼
            cv2.imencode(".jpg", overlay(), [cv2.IMWRITE_JPEG_QUALITY, 80])
            t_ov = t1
        proc.append(time.time() - t1)
        times.append(t)
    cam.close()
    return times, proc


def main():
    with open(os.path.join(ROOT, "lane", "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cc = cfg["camera"]
    dev = cc["device"]
    from vision import LaneDetector
    det = LaneDetector(cfg, os.path.join(ROOT, "lane"))
    print(f"OpenCV {cv2.__version__}，預設執行緒數 {cv2.getNumThreads()}，CPU 核心 {os.cpu_count()}")
    print(f"目前鏡頭設定：{ctrl_state(dev)}\n")

    cap = open_cap(dev, cc["width"], cc["height"], cc["fps"])
    ok, frame = cap.read()
    proc = []
    for _ in range(50):
        t1 = time.time()
        det.process(frame)
        proc.append(time.time() - t1)
    print(f"[0] 純辨識速度：{np.mean(proc) * 1000:.1f} ms/張（理論上限 {1 / np.mean(proc):.0f} fps）")
    report("[1] 直接讀鏡頭", raw(cap), None)
    cap.release()
    cap = open_cap(dev, cc["width"], cc["height"], cc["fps"], bufsize=1)
    times = []
    t0 = time.time()
    while time.time() - t0 < SEC:            # 模擬「讀一張、處理一下」：每張停 7 ms（約等於辨識時間）
        ok, _ = cap.read()
        if ok:
            times.append(time.time())
        time.sleep(0.007)
    report("[1b] 鏡頭緩衝 1 張、每張處理 7 ms", times, None)
    cap.release()

    ctrls = cc.get("v4l2_ctrls") or {}
    if ctrls and shutil.which("v4l2-ctl"):
        for k, v in ctrls.items():
            subprocess.run(["v4l2-ctl", "-d", f"/dev/video{dev}", "-c", f"{k}={v}"], capture_output=True)
        cap = open_cap(dev, cc["width"], cc["height"], cc["fps"])
        report("[2] 套用 config 設定後讀鏡頭", raw(cap), None)
        print(f"    套用後鏡頭設定：{ctrl_state(dev)}")
        cap.release()
    else:
        print("[2] config.yaml 沒有 v4l2_ctrls，略過")

    n_default = cv2.getNumThreads()
    det = LaneDetector(cfg, os.path.join(ROOT, "lane"))
    f3 = report("[3] 取像執行緒 + 辨識（OpenCV 預設執行緒）",
                *threaded(cfg, lambda f, t: det.process(f)))
    cv2.setNumThreads(1)
    f4 = report("[4] 取像執行緒 + 辨識（OpenCV 1 個執行緒）",
                *threaded(cfg, lambda f, t: det.process(f)))

    cap = open_cap(dev, cc["width"], cc["height"], cc["fps"])
    times, proc = [], []
    t0 = time.time()
    while time.time() - t0 < SEC:
        ok, f = cap.read()
        if not ok:
            continue
        t = time.time()
        det.process(f)
        proc.append(time.time() - t)
        times.append(t)
    cap.release()
    report("[5] 不用取像執行緒（OpenCV 1 個執行緒）", times, proc)

    best = 1 if f4 >= f3 else n_default
    cv2.setNumThreads(best)
    from main import Pilot
    pilot = Pilot(cfg, os.path.join(ROOT, "lane"))
    report(f"[6] 和 main.py 一樣（含除錯疊圖，OpenCV {best} 個執行緒）",
           *threaded(cfg, lambda f, t: pilot.step(f, t), overlay=pilot.overlay))

    print("\n判讀：")
    print("  [1] 低            → 鏡頭本身（光線太暗自動降幀、USB 頻寬）")
    print("  [1b] 只有 [1] 一半 → 證實緩衝 1 張會掉幀（新版 camera.py 已改掉）")
    print("  [1] 高、[2] 低    → config.yaml 的鏡頭設定拖慢（曝光時間太長）")
    print("  [4] 明顯比 [3] 快 → OpenCV 多執行緒互搶，要在 main.py 設成 1 個執行緒")
    print("  [6] 比 [3]/[4] 慢 → 除錯疊圖太花時間，比賽時要關掉串流")
    print("請把整段輸出貼回來。")


if __name__ == "__main__":
    main()
