#!/usr/bin/env python3
"""離線回放：拿錄好的 run_*.mp4 + 同名 CSV（取俯仰、車頭角）餵給新程式，看它每段會怎麼判斷。
車子實際走的路線是錄影當時的，所以只能看「辨識 + 決策」，不是閉迴路模擬。

  python tools/replay.py ../logs/2026-11-07/run_20261107_101419.mp4 [--start 0] [--end 60] [--out-video out.mp4]
"""
import argparse
import collections
import csv
import io
import os
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from driver import Driver  # noqa: E402


def load_csv(path):
    raw = open(path, "rb").read().split(b"\x00")[0].decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    num = lambda r, *ks: next((float(r[k]) for k in ks if r.get(k) not in (None, "")), None)
    return [(num(r, "t"), num(r, "imu_pitch", "pitch"), num(r, "imu_yaw", "yaw")) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--calib", default=None, help="用別的校正檔（舊鏡頭角度的錄影要用當時的）")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--out-video", dest="out_video", help="輸出疊圖影片（不能和輸入同一個檔）")
    ap.add_argument("--every", type=float, default=1.0, help="每幾秒印一行")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.calib:
        cfg["geometry"]["calib"] = os.path.abspath(args.calib)
    if args.out_video and os.path.abspath(args.out_video) == os.path.abspath(args.video):
        sys.exit("--out-video 不能和輸入影片同一個檔")
    rows = load_csv(args.video[:-4] + ".csv")
    drv = Driver(cfg, HERE)
    cap = cv2.VideoCapture(args.video)
    t0 = rows[0][0]
    writer = None
    stats = collections.Counter()
    whys = collections.Counter()
    last_print = -1e9
    events = []
    prev_mode = drv.mode
    bad = 0
    for i, (t, pitch, yaw) in enumerate(rows):
        ok, frame = cap.read()
        if not ok:                              # 車上錄的 mp4 常常連續讀會壞掉，跳到這一張重讀可以
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
        if not ok:
            bad += 1
            if bad > 300:
                break
            continue
        rel = t - t0
        if rel < args.start:
            drv.eye.process(frame)
            continue
        if rel > args.end:
            break
        if drv.state not in ("RUN", "LOST"):
            drv.state = "RUN"                   # 回放：一直當作在跑（看決策，不管會不會停）
            drv.t_go = drv.t_go or t
        fwa, bwl, bwr = drv.step(frame, t, pitch, yaw, 0.0)
        s = drv.s
        stats[s.conf] += 1
        whys[drv.why.split("/")[0]] += 1
        if drv.mode != prev_mode:
            events.append(f"{rel:6.1f}s {prev_mode} → {drv.mode}")
            prev_mode = drv.mode
        if rel - last_print >= args.every:
            print(f"{rel:6.1f}s 線{s.conf} 偏{s.offset * 100:+5.1f}cm 向{s.heading:+5.1f}° 看{s.far:3.1f}m "
                  f"彎{drv.sev:4.1f} fwa{fwa:+5.1f} v{bwl:5.1f}/{bwr:5.1f} 俯仰{pitch if pitch is not None else 0:+5.1f} "
                  f"亮{s.bright:3.0f} 虛{s.dash}  {drv.why}")
            last_print = rel
        if args.out_video:
            vis = drv.eye.draw(frame, s, drv.target, f"{rel:5.1f}s {drv.why} fwa={fwa:+.0f} v={bwl:.0f}/{bwr:.0f}")
            if writer is None:
                writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
    if writer:
        writer.release()
    n = sum(stats.values()) or 1
    print(f"\n兩條線 {100 * stats[2] / n:.0f}%  一條 {100 * stats[1] / n:.0f}%  沒有 {100 * stats[0] / n:.0f}%")
    print("速度原因：", ", ".join(f"{k} {100 * c / n:.0f}%" for k, c in whys.most_common()))
    for e in events:
        print("事件：", e)


if __name__ == "__main__":
    main()
