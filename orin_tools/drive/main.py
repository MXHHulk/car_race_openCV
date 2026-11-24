#!/usr/bin/env python3
"""新版循線主程式（2026-11-17 重寫）。放在賽道任何位置都能出發，行為見 driver.py。

  python3 webctl.py                      # 網頁服務（手機開 http://<orin-ip>:8000/，可啟動 / 關閉主程式）
  python3 main.py                        # 只開主程式：終端機 Enter 出發 / 停車，q 離開
  python3 main.py --dry-run --source x.mp4 --go   # 電腦上用錄影測（不開序列埠）
  python3 tools/calibrate.py             # 鏡頭角度動過就重跑（車子擺在直線正中間）
"""
import argparse
import csv
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
from driver import Driver, IDLE, LOST, RUN, STOP  # noqa: E402

LOG_KEYS = ["t", "state", "mode", "why", "conf", "offset", "heading", "far", "lane_w", "sev", "fwa", "bwl", "bwr",
            "motion", "boost", "bright", "dark", "dash", "pitch", "yaw", "proc_ms", "fps", "reason", "map_s", "map_sd", "obs", "obs_raw"]


class Recorder:
    """一趟的紀錄：CSV（每張一列）+ 原始畫面錄影，檔名 run_YYYYmmdd_HHMMSS.*。"""

    def __init__(self, log_dir):
        self.dir = log_dir
        self.path = None
        self.f = self.w = self.vw = None

    @property
    def is_open(self):
        return self.f is not None

    def open(self):
        self.close()
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, time.strftime("run_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, LOG_KEYS)
        self.w.writeheader()
        print(f"紀錄中：{self.path}")

    def frame(self, frame):
        if not self.is_open:
            return
        if self.vw is None:
            self.vw = cv2.VideoWriter(self.path[:-4] + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                      (frame.shape[1], frame.shape[0]))
        self.vw.write(frame)

    def row(self, d):
        if self.is_open:
            self.w.writerow(d)

    def close(self):
        # 關檔時強制寫進硬碟：11-22 21:48:55 斷電重開後 run_20261122_214145.mp4（21:42 寫完）不見了
        if self.f is not None:
            try:
                self.f.flush()
                os.fsync(self.f.fileno())
            except OSError:
                pass
            self.f.close()
            print(f"紀錄檔：{self.path}")
        if self.vw is not None:
            self.vw.release()
            try:
                os.sync()
            except (OSError, AttributeError):
                pass
        self.f = self.w = self.vw = None


def keyboard(drv, quit_evt):
    while not quit_evt.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip().lower() == "q":
            quit_evt.set()
        else:
            drv.toggle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--source", help="影片檔（不給就用鏡頭）")
    ap.add_argument("--dry-run", action="store_true", help="不開序列埠")
    ap.add_argument("--go", action="store_true", help="啟動後直接出發")
    ap.add_argument("--record", action="store_true", help="沒開網頁時也寫紀錄")
    ap.add_argument("--race", action="store_true", help="等中控台綠燈才出發")
    ap.add_argument("--save-video", help="把疊圖存成影片（離線看辨識結果）")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(args.config))
    cc, lk = cfg["camera"], cfg["link"]

    src = VideoFile(args.source) if args.source else Camera(cc["device"], cc["width"], cc["height"], cc["fps"],
                                                          cc.get("v4l2_ctrls"))
    drv = Driver(cfg, base)
    from link import Sender
    sender = Sender(lk["port"], lk["baud"], lk["hz"], dry_run=args.dry_run)
    imu = None
    ic = cfg.get("imu") or {}
    if ic.get("enable") and src.live:
        try:
            from imu import IMU
            imu = IMU(ic.get("bus", 1), ic.get("addr", 0x68), forward=ic.get("forward", "+x"),
                      calib_path=os.path.join(HERE, "imu_calib.json"))
        except OSError as e:
            print(f"⚠ IMU 開不起來（{e}）：上坡 / 下坡 / 虛線左轉的角度判斷會失效")

    rec = Recorder(os.path.join(base, cfg["log"]["dir"]))
    wc = cfg.get("web") or {}
    use_web = wc.get("enable") and src.live
    race = None
    if src.live:
        from race_go import RaceStart
        race = RaceStart(drv.start, drv.stop)
    web = None
    if use_web:
        from webui import WebUI
        web = WebUI(int(wc.get("port", 8001)), drv, race, record=wc.get("record", True), race_mode=args.race,
                    host=wc.get("host", "127.0.0.1"))
        web.full_mps = float(wc.get("full_mps", 1.57))
        print(f"控制埠：http://{wc.get('host', '127.0.0.1')}:{wc.get('port', 8001)}/（手機開 webctl.py 的網頁）")
    elif args.record or not src.live:
        rec.open()
    quit_evt = threading.Event()
    if args.go:
        drv.start()
        if web:
            web.session = True
    elif args.race and race is not None:
        race.arm()
    if src.live and sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(drv, quit_evt), daemon=True).start()
        print("按 Enter 出發 / 停車，q 再按 Enter 離開")

    last_id, n, t_report, t_stream, t_idle = 0, 0, time.time(), 0.0, None
    fps, t_last, proc = 0.0, None, []
    writer = None
    try:
        while not quit_evt.is_set():
            got = src.read(last_id)
            if got is None:
                if not src.live:
                    break
                drv.stop("鏡頭沒有畫面")
                sender.set(0, 0, 0)
                continue
            last_id, t_frame, frame = got
            if t_last is not None and t_frame > t_last:
                fps = 1 / (t_frame - t_last) if fps == 0 else 0.9 * fps + 0.1 / (t_frame - t_last)
            t_last = t_frame
            if web:
                if web.running() and not web.session:
                    web.session = True
                if web.session and web.record and not rec.is_open:
                    rec.open()
                if web.session and not web.running():
                    t_idle = t_idle or time.time()
                    if time.time() - t_idle > 1.5:
                        web.session = False
                        rec.close()
                else:
                    t_idle = None
            rec.frame(frame)
            ir = imu.read() if imu else None
            pitch, yaw = (ir[3], ir[1]) if ir else (None, None)
            t0 = time.time()
            fwa, bwl, bwr = drv.step(frame, t_frame, pitch, yaw, sender.sta_age())
            sender.set(fwa, bwl, bwr)
            pm = (time.time() - t0) * 1000
            proc.append(pm)
            s = drv.s
            rec.row({"t": round(t_frame, 3), "state": drv.state, "mode": drv.mode, "why": drv.why, "conf": s.conf,
                     "offset": round(s.offset, 3), "heading": round(s.heading, 1), "far": round(s.far, 2),
                     "lane_w": None if s.lane_w is None else round(s.lane_w, 3), "sev": round(drv.sev, 1),
                     "fwa": round(fwa, 1), "bwl": round(bwl, 1), "bwr": round(bwr, 1),
                     "motion": round(s.motion, 2), "boost": round(drv.boost, 1), "bright": round(s.bright, 1),
                     "dark": int(s.dark), "dash": s.dash, "pitch": None if pitch is None else round(pitch, 1),
                     "yaw": None if yaw is None else round(yaw, 1), "proc_ms": round(pm, 1), "fps": round(fps, 1),
                     "reason": drv.reason,
                     "map_s": None if drv.loc is None or drv.loc.s is None else round(drv.loc.s, 2),
                     "map_sd": None if drv.loc is None else round(drv.loc.sd, 2),
                     "obs": None if drv.obs_d is None else round(drv.obs_d, 2),
                     "obs_raw": None if drv.obs_raw is None else round(drv.obs_raw, 2)})
            n += 1
            if web is not None and web.want_raw():
                web.push_raw(frame)      # tools/car_rec.py 錄其他車：原始畫面（最近 2 秒有人要才編碼）
            now = time.time()
            want_vis = args.save_video or (web and now - t_stream >= 1 / cfg["log"]["stream_hz"])
            if want_vis:
                txt = f"{drv.state} {drv.why} fwa={fwa:+.0f} L={bwl:.0f} R={bwr:.0f} lines={s.conf} far={s.far:.1f}"
                vis = drv.eye.draw(frame, s, drv.target, txt)
                if web:
                    web.push(vis, fwa, bwl, bwr, fps, (os.path.basename(rec.path) + ("" if rec.is_open else "（已存）")) if rec.path else "", drv.v, drv.why)
                    t_stream = now
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
            if src.live and now - t_report >= 1.0:
                print(f"{drv.state:<4} {drv.why:<10} 線{s.conf} 偏{s.offset * 100:+5.1f}cm 看{s.far:.1f}m "
                      f"fwa {fwa:+4.0f} 速度 {bwl:3.0f}/{bwr:3.0f} 俯仰 {pitch if pitch is not None else 0:+.0f} "
                      f"{fps:4.1f} fps {np.mean(proc[-30:]):.1f} ms" + (f"  {drv.reason}" if drv.reason else ""))
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        src.close()
        if imu:
            imu.close()
        rec.close()
        if web:
            web.close()
        if writer:
            writer.release()
    if proc:
        print(f"共 {n} 張，辨識+控制平均 {np.mean(proc):.1f} ms（最大 {np.max(proc):.1f}）")


if __name__ == "__main__":
    main()
