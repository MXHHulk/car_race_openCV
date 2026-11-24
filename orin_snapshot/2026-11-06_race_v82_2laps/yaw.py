"""用畫面上方背景的左右位移估車頭轉了幾度（沒有 IMU 時用）。

固定時間左轉每趟結果不同：左轉本身都轉約 85°，但進路口前車頭已經歪了 −4 ~ +25°
（run_20261030_203404 轉過頭、204115 轉不夠）。以坡道上的平均方向為基準量總共轉了幾度：
  左轉結束後 1 s：轉不夠 71–79°，成功 94–98°，轉過頭 110°
"""
from collections import deque

import cv2
import numpy as np

ROWS = 110             # 只看畫面上方（背景，不含賽道）
W, H = 320, 55         # 縮小後做相位相關，一張不到 1 ms
MAX_DX = 30            # 一張超過這麼多像素當成雜訊（手伸進畫面、畫面全黑）
HFOV = 70.0            # 度


class YawEstimator:
    def __init__(self):
        self.value = 0.0       # 累積轉角（度），正 = 往左
        self.prev = None
        self.hist = deque()    # (t, value)

    def update(self, frame, t):
        g = np.float32(cv2.cvtColor(cv2.resize(frame[:ROWS], (W, H)), cv2.COLOR_BGR2GRAY))
        if self.prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(self.prev, g)
            if abs(dx) < MAX_DX:
                self.value += dx / W * HFOV
        self.prev = g
        self.hist.append((t, self.value))
        while self.hist and t - self.hist[0][0] > 3.0:
            self.hist.popleft()
        return self.value

    def rate(self, t, span=0.1):
        """最近 span 秒的轉速（°/s，左正）。"""
        old = [(tt, v) for tt, v in self.hist if t - tt >= span]
        if not old:
            return 0.0
        tt, v = old[-1]
        return (self.value - v) / max(1e-3, t - tt)

    def ref(self, t):
        """最近 3 s 到 0.5 s 前的平均方向：坡道上車頭會左右晃，取平均才是坡道本身的方向。"""
        old = [v for tt, v in self.hist if t - tt >= 0.5]
        return float(np.mean(old)) if old else self.value


class ImuYaw(YawEstimator):
    """和 YawEstimator 一樣的介面，但角度來自 IMU 陀螺儀（imu.py）。
    畫面估的會偏：run_20261103_185435、190800 對準下坡時畫面算 75–78°，陀螺儀只有 69–73°，
    車頭其實偏右、從平台右側掉下去；成功的 184744 兩者都是 80°。"""

    def update(self, value, t):
        self.value = value
        self.hist.append((t, value))
        while self.hist and t - self.hist[0][0] > 3.0:
            self.hist.popleft()
        return value
