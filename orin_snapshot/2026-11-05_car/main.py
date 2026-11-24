#!/usr/bin/env python3
"""循跡主程式：取像 → 辨識 → Pure Pursuit / 速度規劃 → 狀態機 → UART 20 Hz。

⚠ 第一次請把車架高（輪子離地）測試轉向方向與安全保護。

用法（在 ~/orin_tools/lane 底下）：
  python3 main.py                        # 待命；終端機按 Enter 出發、再按 Enter 停車，輸入 q 離開
  python3 main.py --go                   # 一啟動就出發
  python3 main.py --dry-run --source ../runs/track.mp4 --save-video out.mp4
  python3 main.py --record              # 另外把鏡頭原始畫面錄成影片（和紀錄檔同名 .mp4），可拿回電腦重播除錯
                                         # 離線：讀影片跑完整流程，不開序列埠，輸出疊圖影片
狀態機：待命 IDLE →(Enter)→ 行駛 RUN ⇄ 找不到線 LOST →(超過 lost_timeout)→ 停止 STOP →(Enter)→ 待命
"""
import argparse
import csv
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camera import Camera, VideoFile  # noqa: E402
from controller import Controller  # noqa: E402
from hilltop import HilltopMode, Junction  # noqa: E402
from tunnel import TunnelGuard  # noqa: E402
from stall import StallGuard  # noqa: E402
from vision import LaneDetector  # noqa: E402
from yaw import ImuYaw, YawEstimator  # noqa: E402
import lane_view_check as lvc  # noqa: E402  (vision 已把上一層加進 sys.path)

IDLE, RUN, LOST, STOP = "IDLE", "RUN", "LOST", "STOP"


class Pilot:
    """每張畫面呼叫一次 step()，回傳要送出的 (fwa, bwl, bwr)。不碰硬體，離線模擬也用它。"""

    def __init__(self, cfg, base_dir=HERE):
        self.cfg = cfg
        self.det = LaneDetector(cfg, base_dir)
        self.ctl = Controller(cfg)
        self.stall = StallGuard(cfg)
        self.hill = HilltopMode(cfg)
        self.junc = Junction(cfg)
        self.tunnel = TunnelGuard(cfg)   # 岔路接隧道：限速、目標往左偏（見 tunnel.py）
        self.junc.gmap = self.det.gmap   # 直走時把虛線換算成公尺（見 Junction.dash_y）
        self.yaw = YawEstimator()
        self.imu_yaw = None       # IMU 陀螺儀的車頭角（度），沒有 IMU 時 None（main() 每張畫面設定）
        self.gyro = ImuYaw()      # 有 IMU 時路口、坡道用這個（比畫面估的準）
        self.state, self.reason = IDLE, ""
        self.t_prev = self.t_lost = None
        self.v_lost = 0.0
        self.res = None
        self.cmd = (0, 0, 0)
        self.pitch = None         # IMU 量到的俯仰角（度，上坡正），沒有 IMU 時 None（main() 每張畫面設定）

    def start(self):
        if self.state in (IDLE, STOP):
            self.state, self.reason = RUN, ""
            self.tunnel.start(time.time() if self.t_prev is None else self.t_prev, self.imu_yaw)
            print("[pilot] 出發")

    def stop(self, reason="手動停止"):
        if self.state in (RUN, LOST):
            print(f"[pilot] 停止：{reason}")
        self.state, self.reason = STOP, reason

    def toggle(self):
        if self.state in (RUN, LOST):
            self.stop()
        elif self.state == STOP:
            self.state, self.reason = IDLE, ""
            print("[pilot] 待命（再按 Enter 出發）")
        else:
            self.start()

    def step(self, frame, t, sta_age=0.0):
        s, c = self.cfg["safety"], self.cfg["control"]
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        res = self.res = self.det.process(frame)
        # IMU 車頭角：轉向前先更新（阻尼要用這張的轉速）
        if self.imu_yaw is not None:
            if self.gyro.hist and t > self.gyro.hist[-1][0]:
                r = (self.imu_yaw - self.gyro.value) / (t - self.gyro.hist[-1][0])
                self.ctl.yaw_rate = r if self.ctl.yaw_rate is None else 0.5 * self.ctl.yaw_rate + 0.5 * r
            self.gyro.update(self.imu_yaw, t)

        # --- 狀態轉移
        if self.state in (RUN, LOST) and sta_age > s["sta_timeout"]:
            self.stop(f"超過 {s['sta_timeout']} 秒沒收到 STA")
        if self.state == RUN and res.confidence == 0 and not self.ctl.corner_active and not self.junc.active:
            # 減速但不低於 v_min：低於馬達推得動的速度，車子會停在原地，永遠轉不回來看到線
            self.state, self.t_lost = LOST, t
            self.v_lost = max(c["v_min"], self.ctl.v * s["lost_speed_factor"])
        elif self.state == LOST:
            if res.confidence > 0:
                self.state = RUN
            # 坡道和下坡不能一看不到線就停：車子停在坡上會掉下去（run_20261101_143948）；
            # run_20261101_165649 第二圈 111.6 s 也是爬坡時看不到線 0.3 s 就停
            elif t - self.t_lost > (s.get("down_lost_s", 1.0) if self.hill.active or self.tunnel.active or self.tunnel.in_dark or self.junc.after_turn(t, c["hilltop"]["junction"].get("post_s", 2.5)) else s["lost_timeout"]):
                self.stop(f"超過 {s['lost_timeout']} 秒找不到線")

        # --- 指令
        fwa = self.ctl.fwa
        # 卡住（畫面幾乎不動）時前輪維持不動：畫面沒變，重新算也只會被積分、雜訊慢慢帶偏，
        # 等加速一推動，車子就照著那個偏掉的角度衝出去（run_20261029_180918 坡頂往左衝出賽道）
        stuck = self.state == RUN and self.stall.on and self.stall.med < self.stall.thresh and self.ctl.v > 0
        # 只有真的不動時才暫停積分。用「還有卡住加速」判斷會讓整段爬坡都不累積（加速退得慢），
        # run_20261029_184312 坡道上偏右 13–25 cm 好幾秒，前輪只修 −2 ~ −6°，最後開出右邊
        self.ctl.hold_i = stuck
        self.ctl.no_bias = self.hill.active
        # 下坡到底的路口要往左打 −25~−30°（run_20261029_201418 40–43 s），路口動作做完後一段時間不限制左修
        free_s = (c.get("inner_bias") or {}).get("left_off_s", 15.0)
        self.ctl.left_free = self.junc.t_done is not None and t - self.junc.t_done < free_s
        # 爬坡中到路口判斷結束前也不限制：坡頂要靠前輪打左觸發路口動作，
        # run_20261030_201818 被限制在 −15°，第二次打左觸發不了，車子開過路口
        near_top = self.junc.t_climb is not None and t - self.junc.t_climb <= self.junc.window_s
        self.ctl.left_free = self.ctl.left_free or self.hill.active or near_top
        # 下坡前畫面上的線不可靠，偏移算錯時積分會一路累積（run_20261101_143948 前輪 +15° 打到下坡右側）
        if self.junc.after_turn(t, self.junc.post_s):
            self.ctl.hold_i = True
        # 下坡前畫面上線很雜，近處一點誤差前輪就左右大擺（run_20261031_154631 33–40 s，−22° ~ +44°）：看遠一點
        sp = c.get("speed_profile") or {}
        self.ctl.ld_extra = 0.0
        if sp.get("enable", False) and self.junc.t_done is not None and t - self.junc.t_done < sp.get("down_s", 14.0):
            self.ctl.ld_extra = sp.get("down_ld", 0.0)
        # 隧道段前視縮短（上一張判斷的）：拱門前車道往左拐，看遠了修正太軟，貼上右牆（194348）
        self.ctl.ld_extra += self.tunnel.ld_extra
        if self.state == RUN and not stuck:
            fwa = self.ctl.steer(res, self.ctl.v, t)
            fwa = self.ctl.fwa if fwa is None else fwa
            v = self.ctl.speed(res, dt, t)
        elif self.state == RUN:
            v = self.ctl.speed(res, dt, t)
        elif self.state == LOST:                       # 維持最後轉角、減速
            v = self.ctl.ramp(self.v_lost, dt)
        else:                                          # 待命 / 停止：速度 0
            self.ctl.v = v = 0.0
            fwa = 0.0
            if s["idle_steer"] and res.confidence > 0:
                f = self.ctl.steer(res, 0)
                fwa = 0.0 if f is None else f
        # 卡住偵測：行駛中畫面不動 → 加速、關差速，兩輪一起往前推（上坡推不動時）
        driving = self.state in (RUN, LOST) and v > 0
        extra = self.stall.update(frame, t, driving, fwa, self.pitch)
        self.yaw.update(frame, t)
        hd = self.gyro if self.imu_yaw is not None else self.yaw
        # 爬坡 / 坡頂平台：只沿左線走；正常在動時限制總速度，真的卡住時不限（見 hilltop.py）
        floor, cap = self.hill.update(self.stall, t, driving, self.pitch)
        self.det.only_left = self.hill.active
        # 速度規劃不看坡度：坡頂到路口、下坡這段要自己限速（直走路段前輪是正的，不限會衝到 v_straight）
        down = self.junc.t_done is not None and t - self.junc.t_done < sp.get("down_s", 14.0)
        # 坡底右轉做完（bend 結束）後的大彎道不必再 30%：193243 落地後整個大彎 43–50 s 都被限在 30%。
        # 改成 after_bend_v；彎裡實際速度再由 speed_profile 依彎度決定
        dv = sp.get("down_v", 30.0)
        if down and self.junc.bend_done and self.junc.mode is None:
            dv = max(dv, sp.get("after_bend_v", dv))
        if sp.get("enable", False) and not self.hill.active and (near_top or down) and v > dv:
            v = self.ctl.v = dv
        if cap is not None and v > cap:
            v = self.ctl.v = cap
        if extra:
            # 卡住加速加上直線速度不能太高：起步 55 + 16 = 71%，車子一衝就偏掉（run_20261101_192936）
            v = min(100.0, v + extra, c["stall"].get("total_max", 100.0) if cap is None else cap)
        if floor is not None and v < floor:
            v = floor              # 最低速度和卡住加速取大的，不要疊加（疊加會到 85%，一推動就暴衝）
        ramp = self.junc.ramp_steer(fwa, self.hill.active, t, hd, frame)
        if self.state == RUN and self.hill.active:
            fwa = self.ctl.fwa = ramp
        elif self.state == RUN:
            fwa = self.ctl.fwa = self.junc.post_limit(fwa, t)
        # 岔路接隧道：限速（目標往左偏在下一張的 steer 生效，見 tunnel.py）
        running = self.state in (RUN, LOST) and not self.hill.active and not self.junc.active
        v2 = self.tunnel.update(t, self.imu_yaw, v, running, frame)
        if v2 < v:
            v = self.ctl.v = v2
        self.ctl.y_extra = self.tunnel.y_extra
        # 坡頂路口：固定左轉 → 直走（見 hilltop.Junction）
        jf, jv = self.junc.update(fwa, self.hill.active, self.stall.active, t, driving and self.state == RUN,
                                  running=self.state in (RUN, LOST), yaw=hd, frame=frame, motion=self.stall.motion,
                                  pitch=self.pitch, v=v)
        if jf is not None:
            fwa, v, extra = jf, jv, 0.0
            self.ctl.fwa, self.ctl.v = fwa, v
        elif self.junc.v_cap is not None and v > self.junc.v_cap:
            v = self.ctl.v = self.junc.v_cap        # 看到坡頂虛線：先降到左轉的速度
            extra = 0.0
        if extra or (self.hill.active and jf is None):     # 爬坡不用差速：45% 打到底會變 81/9（run_20261030_202526）
            bwl = bwr = v
        else:
            bwl, bwr = self.ctl.wheels(v, fwa)
        self.cmd = (fwa, bwl, bwr)
        return self.cmd

    def overlay(self):
        """在 lane_view_check 的疊圖上加中線、目標點、狀態。"""
        r, res = self.res.raw, self.res
        vis = lvc.draw(r)
        gm = self.det.gmap
        if gm is not None and res.center is not None:
            xs = np.linspace(*res.x_range, 30)
            u, v = gm.to_image(xs, np.polyval(res.center, xs))
            pts = np.stack([u, v], 1)
            pts = pts[np.isfinite(pts).all(1)].astype(np.int32)
            cv2.polylines(vis, [pts], False, (0, 165, 255), 2)
            if self.ctl.target is not None:
                tu, tv = gm.to_image(*self.ctl.target)
                if np.isfinite(tu) and np.isfinite(tv):
                    cv2.circle(vis, (int(tu), int(tv)), 5, (0, 0, 255), -1)
        fwa, bwl, bwr = self.cmd
        col = {RUN: (80, 220, 80), LOST: (0, 200, 255), STOP: (60, 60, 255), IDLE: (200, 200, 200)}[self.state]
        cv2.putText(vis, f"{self.state} fwa={fwa:+.0f} L={bwl:.0f} R={bwr:.0f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(vis, f"off={res.offset * 100:+.1f}cm hd={res.heading:+.0f} k={res.curvature:+.2f}",
                    (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        # 把車身座標的中線另外畫成俯視小圖（右上角），方便確認換算是否合理
        bev = np.zeros((120, 90, 3), np.uint8)
        sc = 120 / 1.5                                                  # 1.5 m 高
        to_px = lambda x, y: (int(45 - y * sc), int(119 - x * sc))
        if res.pts is not None:
            for x, y in res.pts[::3]:
                cv2.circle(bev, to_px(x, y), 1, (0, 165, 255), -1)
        if self.ctl.target is not None:
            cv2.circle(bev, to_px(*self.ctl.target), 3, (0, 0, 255), -1)
        cv2.rectangle(bev, (40, 112), (50, 119), (200, 200, 200), -1)  # 後輪軸
        vis[4:124, 2 * lvc.W - 94:2 * lvc.W - 4] = bev
        return vis


LOG_KEYS = ["t", "state", "confidence", "offset", "heading", "curvature", "shape", "slope", "slope_deg",
            "lane_width", "fwa", "bwl", "bwr", "proc_ms", "latency_ms", "fps", "x_near",
            "corner_mode", "corner", "corner_dbg", "motion", "boost", "hill", "junction", "bias", "yaw", "imu_tilt", "imu_pitch", "imu_yaw", "imu_acc", "dash_y", "tunnel"]


def keyboard(pilot, quit_evt, race=None):
    """終端機：Enter 出發 / 停車，r 開關起跑模式（等中控台綠燈），q 離開。"""
    while not quit_evt.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        cmd = line.strip().lower()
        if cmd == "q":
            quit_evt.set()
        elif cmd == "r" and race is not None:
            if race.armed:
                race.disarm("關閉（手動）")
            elif pilot.state in (RUN, LOST):
                print("[race] 行駛中不能開起跑模式，先按 Enter 停車")
            else:
                if pilot.state == STOP:
                    pilot.toggle()           # STOP → 待命，綠燈時才能出發
                race.arm()
        else:
            if race is not None and race.armed:
                race.disarm("關閉（手動按 Enter）")   # 手動出發 / 停車就不再等綠燈
            pilot.toggle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--source", help="影片檔（不給就用鏡頭）")
    ap.add_argument("--dry-run", action="store_true", help="不開序列埠")
    ap.add_argument("--go", action="store_true", help="啟動後直接出發")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--save-video", help="把疊圖存成影片")
    ap.add_argument("--record", action="store_true", help="把鏡頭原始畫面錄成影片（logs/ 底下，和紀錄檔同名）")
    ap.add_argument("--race", action="store_true",
                    help="圈數賽起跑模式：載入完成後等中控台綠燈（/go 由 0 變 1）才出發，一次性；終端機輸入 r 可再開")
    ap.add_argument("--race-url", default=None, help="起跑旗標網址，預設 http://192.168.0.149:8080/go（測試時指到 fake_go.py）")
    ap.add_argument("--practice", action="store_true", help="燈號練習的綠燈也算（只限測試，正式比賽不要加）")
    ap.add_argument("--race-stop", action="store_true", help="起跑後旗標回 0（裁判按停止）就停車")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    cc, lk, dbg = cfg["camera"], cfg["link"], cfg["debug"]
    if cc.get("cv_threads"):
        cv2.setNumThreads(int(cc["cv_threads"]))  # 見 config.yaml：OpenCV 執行緒太多會和取像執行緒互搶

    src = VideoFile(args.source) if args.source else Camera(cc["device"], cc["width"], cc["height"],
                                                          cc["fps"], cc.get("v4l2_ctrls"))
    pilot = Pilot(cfg, base_dir)
    from link import Sender
    sender = Sender(lk["port"], lk["baud"], lk["hz"], dry_run=args.dry_run)
    imu = None
    ic = cfg.get("imu") or {}
    if ic.get("enable", False) and src.live:
        try:
            from imu import IMU   # 延後 import：電腦上沒有 fcntl / I2C 也能重播
            imu = IMU(ic.get("bus", 1), ic.get("addr", 0x68), forward=ic.get("forward", "+x"))
        except OSError as e:
            print(f"⚠ IMU 開不起來（{e}），這趟不用 IMU")
    stream = lvc.Stream(dbg["port"]) if dbg["stream"] and src.live and not args.no_stream else None
    if stream:
        print(f"除錯畫面：http://<orin-ip>:{dbg['port']}/")

    log_dir = os.path.join(base_dir, dbg["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, time.strftime("run_%Y%m%d_%H%M%S.csv"))
    flog = open(log_path, "w", newline="", encoding="utf-8")
    wlog = csv.DictWriter(flog, LOG_KEYS)
    wlog.writeheader()
    writer = raw_writer = None

    quit_evt = threading.Event()
    # 起跑旗標：鏡頭、IMU、序列埠都準備好之後才開始等（CAR-INTEGRATION.md 規則 1）
    from race_go import RaceStart, PI_GO_URL
    url = args.race_url or PI_GO_URL
    if args.practice:
        url += ("&" if "?" in url else "?") + "practice=1"
        print("⚠ 起跑旗標帶 practice=1：燈號練習也會出發，正式比賽不要加 --practice")
    race = RaceStart(pilot.start, pilot.stop, url=url, stop_on_zero=args.race_stop) if src.live else None
    if args.go:
        pilot.start()
    elif args.race and race is not None:
        race.arm()
    if src.live and sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(pilot, quit_evt, race), daemon=True).start()
        print("按 Enter 出發 / 停車，輸入 r 開關起跑模式（等綠燈），輸入 q 再按 Enter 離開")

    last_id, n, t_report, t_stream = 0, 0, time.time(), 0.0
    proc_hist = []
    t_last_frame, fps, fps_warned = None, 0.0, False
    try:
        while not quit_evt.is_set():
            got = src.read(last_id)
            if got is None:
                if not src.live:
                    break
                print("[main] 1 秒沒有新畫面")
                pilot.stop("鏡頭沒有畫面")
                sender.set(0, 0, 0)
                continue
            last_id, t_frame, frame = got
            if t_last_frame is not None and t_frame > t_last_frame:
                inst = 1.0 / (t_frame - t_last_frame)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            t_last_frame = t_frame
            if src.live and n > 30 and fps < 15 and not fps_warned:
                print(f"⚠ 鏡頭只有 {fps:.1f} fps（應該要 ≥ 25）：多半是曝光時間太長，"
                      "用 cam_tuner.py 把 exposure_time_absolute 調到 330 以下。車子會來不及轉彎。")
                fps_warned = True
            if args.record:
                if raw_writer is None:
                    raw_path = log_path[:-4] + ".mp4"
                    raw_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                 (frame.shape[1], frame.shape[0]))
                    print(f"錄影中：{raw_path}")
                raw_writer.write(frame)
            ir = imu.read() if imu else None
            pilot.pitch = ir[3] if ir else None
            pilot.imu_yaw = ir[1] if ir else None
            t0 = time.time()
            fwa, bwl, bwr = pilot.step(frame, t_frame, sender.sta_age())
            sender.set(fwa, bwl, bwr)
            proc_ms = (time.time() - t0) * 1000
            proc_hist.append(proc_ms)
            res = pilot.res
            wlog.writerow({"t": round(t_frame, 3), "state": pilot.state, "confidence": res.confidence,
                           "offset": round(res.offset, 4), "heading": round(res.heading, 2),
                           "curvature": round(res.curvature, 3), "shape": res.shape, "slope": res.slope,
                           "slope_deg": None if res.slope_deg is None else round(res.slope_deg, 1),
                           "lane_width": None if res.lane_width is None else round(res.lane_width, 3),
                           "fwa": round(fwa, 1), "bwl": round(bwl, 1), "bwr": round(bwr, 1),
                           "proc_ms": round(proc_ms, 1),
                           "latency_ms": round((time.time() - t_frame) * 1000, 1) if src.live else None,
                           "fps": round(fps, 1),
                           "x_near": round(res.x_range[0], 3) if res.center is not None else None,
                           "corner_mode": pilot.ctl.corner.mode,
                           "corner": "" if not res.corner else f"{res.corner[0]}:{res.corner[1]:.2f}:{res.corner[2]:+.2f}",
                           "corner_dbg": res.corner_dbg,
                           "motion": round(pilot.stall.motion, 2), "boost": round(pilot.stall.extra, 1),
                           "hill": pilot.hill.mode, "junction": (pilot.junc.mode or "") + ("*" if pilot.junc.cam_used else ""),
                           "bias": round(pilot.ctl.bias, 3), "yaw": round(pilot.yaw.value, 1),
                           "imu_tilt": round(ir[0], 1) if ir else None,
                           "imu_pitch": round(ir[3], 1) if ir else None, "imu_yaw": round(ir[1], 1) if ir else None,
                           "imu_acc": "/".join(f"{v:+.2f}" for v in ir[2]) if ir else None,
                           "dash_y": None if pilot.junc.dash_seen is None else round(pilot.junc.dash_seen, 3),
                           "tunnel": pilot.tunnel.mode})
            n += 1
            now = time.time()
            if args.save_video or (stream and now - t_stream >= 1 / dbg["stream_hz"]):
                vis = pilot.overlay()
                if stream:
                    stream.push(vis)
                    t_stream = now
                if args.save_video:
                    if writer is None:
                        vfps = getattr(src, "fps", 30.0)
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), vfps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
            if now - t_report >= 1.0 and src.live:
                print(f"{pilot.state:<4} 線{res.confidence:.1f} 偏{res.offset * 100:+5.1f}cm "
                      f"曲率{res.curvature:+.2f} {res.shape}/{res.slope} → fwa {fwa:+4.0f} "
                      f"速度 {bwl:3.0f}/{bwr:3.0f}  {fps:4.1f} fps  處理 {np.mean(proc_hist[-30:]):.1f} ms"
                      + (f"  {pilot.reason}" if pilot.reason else ""))
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        src.close()
        if imu:
            imu.close()
        flog.close()
        if writer:
            writer.release()
        if raw_writer:
            raw_writer.release()
    if proc_hist:
        print(f"共 {n} 張，辨識+控制平均 {np.mean(proc_hist):.1f} ms（最大 {np.max(proc_hist):.1f}）")
    print(f"紀錄檔：{log_path}")


if __name__ == "__main__":
    main()
