#!/usr/bin/env python3
"""鏡頭左右朝向檢查：車子擺在直線正中間、車頭對齊車道，一邊跑這支一邊轉鏡頭，
讓「左右偏」接近 0°、兩條線都抓到。

2026-11-09：換透明前板後，車子擺正卻只抓得到左線（起步直線兩條線比例 15–36%，11-07 是 77–85%），
右線在畫面上很早就跑出右邊 → 鏡頭往左偏了。

  python3 tools/cam_aim.py            # 每 0.5 秒印一行，Ctrl+C 結束（要先關主程式，鏡頭才不會被佔用）
"""
import math
import os
import statistics as st
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.join(os.path.dirname(HERE), "lane")
sys.path.insert(0, LANE)
sys.path.insert(1, os.path.dirname(HERE))
from camera import Camera  # noqa: E402
from vision import LaneDetector  # noqa: E402
import lane_view_check as lvc  # noqa: E402


def main():
    cfg = yaml.safe_load(open(os.path.join(LANE, "config.yaml"), encoding="utf-8"))
    cc = cfg["camera"]
    cam = Camera(cc["device"], cc["width"], cc["height"], cc["fps"], cc.get("v4l2_ctrls"))
    det = LaneDetector(cfg, LANE)
    f_px = (lvc.W / 2) / math.tan(math.radians(cfg["geometry"]["hfov"] / 2))
    print("車子擺在直線正中間、車頭對齊車道。目標：兩條線 9 成以上、左右偏 0° ±3°、地平線 ±5 列內不變")
    last, buf = 0, []
    t_print = time.time()
    try:
        while True:
            got = cam.read(last)
            if got is None:
                continue
            last, _, frame = got
            r = lvc.analyse(frame, det.acfg, None)
            vp = r.get("vp_near")
            buf.append((r["n_lines"], vp))
            if time.time() - t_print >= 0.5:
                two = sum(n == 2 for n, _ in buf) / len(buf)
                xs = [v[0] for _, v in buf if v]
                ys = [v[1] for _, v in buf if v]
                if xs:
                    yaw = math.degrees(math.atan((st.median(xs) - lvc.W / 2) / f_px))
                    hint = "往右轉鏡頭" if yaw > 3 else ("往左轉鏡頭" if yaw < -3 else "OK")
                    print(f"兩條線 {two * 100:3.0f}%  左右偏 {yaw:+5.1f}°（{hint}）  地平線 {st.median(ys):5.1f} 列"
                          f"（11-07 約 34–40）")
                else:
                    print(f"兩條線 {two * 100:3.0f}%  抓不到兩條線，算不出朝向（右線可能整條在畫面外 → 往右轉鏡頭）")
                buf, t_print = [], time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
