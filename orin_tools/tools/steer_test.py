#!/usr/bin/env python3
"""轉向能力測試：固定前輪角度和速度往前開一小段，用 IMU 量車頭實際轉多快，和狀況好時比較。

2026-11-04 跑不完一圈時，紀錄裡前輪 36–45° 的轉向效果只剩 2026-11-03 的 0.6–0.7（小角度還正常），
用這支在平地上直接量：左右分開、角度分開，不受鏡頭影響。

  python3 tools/steer_test.py                     # 預設 ±15、±25、±35 度，速度 30%，每次 1.5 秒
  python3 tools/steer_test.py --angles 35 -35 --speed 30

車子放在空地上（前方和左右各留 1 m），每次跑完會停，等你按 Enter 把車子擺回去再跑下一個角度。
比例 = 實際轉速 ÷（1.5 × 速度% × tan 前輪）：2026-11-03 狀況好時各角度都在 0.9–1.05。
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane"))
from imu import IMU  # noqa: E402
from motor_cmd import BMduinoLink  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=float, nargs="+", default=[15, -15, 25, -25, 35, -35])
    ap.add_argument("--speed", type=float, default=30)
    ap.add_argument("--time", type=float, default=1.5, help="每個角度往前開幾秒")
    ap.add_argument("--hz", type=float, default=20)
    args = ap.parse_args()
    link = BMduinoLink()
    imu = IMU()
    print("IMU 校正中，車子不要動…")
    while imu.read() is None:
        time.sleep(0.1)
    results = []
    try:
        for a in args.angles:
            input(f"\n把車子擺好（前輪 {a:+.0f}°、速度 {args.speed:.0f}%），按 Enter 開始…")
            # 先打好前輪再出發，避免算到前輪還在轉的那段
            t_end = time.time() + 0.5
            while time.time() < t_end:
                link.send(a, 0, 0)
                link.poll()
                time.sleep(1 / args.hz)
            samples = []
            t0 = time.time()
            while time.time() - t0 < args.time:
                link.send(a, args.speed, args.speed)
                link.poll()
                r = imu.read()
                if r is not None:
                    samples.append((time.time(), r[1]))
                time.sleep(1 / args.hz)
            link.stop()
            # 只算後半段（起步加速完）
            half = [s for s in samples if s[0] - t0 >= args.time / 2]
            rate = (half[-1][1] - half[0][1]) / (half[-1][0] - half[0][0]) if len(half) > 2 else float("nan")
            pred = 1.5 * args.speed * math.tan(math.radians(abs(a)))
            ratio = abs(rate) / pred if pred else float("nan")
            side = "右轉" if a > 0 else "左轉"
            ok = (rate < 0) == (a > 0)        # 右轉時車頭角（左正）應該變小
            print(f"前輪 {a:+.0f}°（{side}）：車頭 {rate:+.1f}°/s，比例 {ratio:.2f}" + ("" if ok else "  ⚠ 方向不對"))
            results.append((a, rate, ratio))
    finally:
        link.close()
        imu.close()
    print("\n角度    車頭轉速   比例（狀況好時 0.9–1.05）")
    for a, rate, ratio in results:
        print(f"{a:+4.0f}°  {rate:+7.1f}°/s   {ratio:.2f}")


if __name__ == "__main__":
    main()
