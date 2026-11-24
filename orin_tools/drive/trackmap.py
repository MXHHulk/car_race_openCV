"""賽道地圖 + 定位（2026-11-21：使用者改成背地圖跑；出發一定在起點或起點後的大直線）。

地圖（track_map.json，tools/build_map.py 用多圈紀錄建）：
  一圈從上坡開始 s = 0 算到下一次上坡，長 L m；每 0.1 m 的車頭角 psi(s)（相對上坡開始，一圈約 −721°）、
  坡頂、虛線左轉、下坡、暗處、岔路（位置 + 走哪邊）、出發區間。

定位（Localizer）：一維直方圖濾波，車子在這圈的哪個 s。
  預測：用指令速度估的行駛距離往前推（和建地圖同一套估法，偏差互相抵銷）
  修正：最近 win m 內陀螺儀量到的車頭轉了多少，和地圖上每個 s 的「前 win m 轉了多少」比
  錨點：上坡開始、坡頂、虛線左轉開始、下坡開始（IMU / 動作很確定）直接把位置拉過去
"""
import json
import math
import os

import numpy as np


def odo_speed(cmd_avg, pitch, full_mps, down_ms=0.6, down_pitch=7.0):
    """行駛速度估計（m/s）：平均輪速 % × full_mps；下坡（反打 / 滑行，指令 ≤ 0 車子還是往下走）用固定 down_ms。"""
    if pitch is not None and pitch <= -down_pitch:
        return down_ms
    return max(0.0, cmd_avg) / 100.0 * full_mps


class TrackMap:
    def __init__(self, path):
        m = json.load(open(path, encoding="utf-8"))
        self.m = m
        self.L, self.ds = float(m["L"]), float(m["ds"])
        self.psi = np.array(m["psi"], float)
        self.n = len(self.psi)
        self.total = float(self.psi[-1] + (self.psi[-1] - self.psi[-2]))   # 一圈總轉（約 −721°）
        k = np.gradient(self.psi, self.ds)                                    # 曲率（°/m）
        w = max(1, int(0.3 / self.ds))
        self.kappa = np.convolve(np.concatenate([k[-w:], k, k[:w]]), np.ones(2 * w + 1) / (2 * w + 1), "same")[w:-w]
        self.forks = m.get("forks", [])
        self.dark = m.get("dark", [])

    def wrap(self, s):
        return s % self.L

    def ahead(self, s_from, s):
        """s 在 s_from 前方多遠（0 ~ L）。"""
        return (s - s_from) % self.L

    def near(self, s, target, before, after):
        """s 在 target 前 before m 到後 after m 之間。"""
        d = (s - target + self.L / 2) % self.L - self.L / 2
        return -before <= d <= after

    def psi_shift(self, w_cells):
        """每個 s 的「前 w 格轉了多少」= psi(s) − psi(s − w)（跨圈時加一圈總轉）。"""
        prev = np.roll(self.psi, w_cells)
        prev[:w_cells] -= self.total
        return self.psi - prev

    def max_kappa_ahead(self, s, dist):
        i0 = int(self.wrap(s) / self.ds)
        n = max(1, int(dist / self.ds))
        idx = (i0 + np.arange(n)) % self.n
        return float(np.max(np.abs(self.kappa[idx])))


class Localizer:
    def __init__(self, tmap, cfg):
        self.map, self.c = tmap, cfg
        self.p = None
        self.carry = 0.0
        self.hist = []              # 最近的 (ds, dyaw)
        self.yaw_prev = None
        self.s, self.sd, self.ok = None, 99.0, False
        self.wc = max(2, int(cfg.get("win_m", 1.5) / tmap.ds))
        self.dpsi = tmap.psi_shift(self.wc)

    def reset(self, lo=None, hi=None):
        """出發：起點或起點後的大直線（地圖 start 區間）均勻分布。"""
        m = self.map
        lo, hi = (m.m.get("start", [0, m.L]) if lo is None else (lo, hi))
        s = np.arange(m.n) * m.ds
        if hi - lo >= m.L - m.ds:
            # 整圈：(hi - lo) % L 會變成 0，只剩 s = lo 一格（11-22 144642：定位一出發就「確定」在 0 m、sd 0，實際在 ~33 m）
            self.p = np.ones(m.n)
        else:
            self.p = (((s - lo) % m.L) <= ((hi - lo) % m.L)).astype(float)
        self.p /= self.p.sum()
        self.carry, self.hist, self.yaw_prev = 0.0, [], None
        self._estimate()

    def _blur(self, sigma_m):
        k = max(1, int(3 * sigma_m / self.map.ds))
        x = np.arange(-k, k + 1) * self.map.ds
        g = np.exp(-0.5 * (x / max(1e-3, sigma_m)) ** 2)
        g /= g.sum()
        pp = np.concatenate([self.p[-k:], self.p, self.p[:k]])
        self.p = np.convolve(pp, g, "same")[k:-k]

    def anchor(self, s_at, sigma=0.6, gate=None):
        """確定的事件（上坡開始、虛線左轉、下坡開始，地圖上各只有一個）：位置直接設到 s_at 附近。
        但定位本來就很確定、又差超過 gate m = 這個事件是誤觸發（車子被拿起來、顛簸），不理。回傳有沒有採用。"""
        if self.p is None:
            return False
        m = self.map
        gate = self.c.get("anchor_gate", 6.0) if gate is None else gate
        if self.ok and abs((self.s - s_at + m.L / 2) % m.L - m.L / 2) > gate:
            return False
        s = np.arange(m.n) * m.ds
        d = (s - s_at + m.L / 2) % m.L - m.L / 2
        self.p = np.exp(-0.5 * (d / sigma) ** 2)
        self.p /= self.p.sum()
        self.carry = 0.0
        self._estimate()
        return True

    def measure_near(self, points, sigma=1.0, floor=0.03):
        """軟性的位置量測：現在在這幾個位置之一附近（例：看到 V = 在某個岔路附近，不知道是哪一個）。"""
        if self.p is None or not points:
            return
        m = self.map
        s = np.arange(m.n) * m.ds
        like = np.zeros(m.n)
        for pt in points:
            d = (s - pt + m.L / 2) % m.L - m.L / 2
            like = np.maximum(like, np.exp(-0.5 * (d / sigma) ** 2))
        self.p = self.p * (floor + like)
        self.p /= self.p.sum()
        self._estimate()

    def step(self, dt, v_ms, yaw):
        if self.p is None:
            return
        m, c = self.map, self.c
        ds = v_ms * dt
        # 預測：往前推 ds（整數格滾動，餘數累積），再依距離模糊
        self.carry += ds / m.ds
        sh = int(self.carry)
        if sh:
            self.p = np.roll(self.p, sh)
            self.carry -= sh
        self._blur(c.get("odo_sigma0", 0.004) + c.get("odo_rel", 0.06) * ds)
        # 修正：最近 win m 內的車頭變化 vs 地圖
        if yaw is not None:
            dy = 0.0 if self.yaw_prev is None else yaw - self.yaw_prev
            self.yaw_prev = yaw
            self.hist.append((ds, dy))
            tot, dsum, k = 0.0, 0.0, len(self.hist)
            while k > 0 and dsum < self.wc * m.ds:
                k -= 1
                dsum += self.hist[k][0]
                tot += self.hist[k][1]
            self.hist = self.hist[k:]
            if dsum >= 0.8 * self.wc * m.ds and ds > 0:
                sig = c.get("psi_sigma", 15.0)
                like = np.exp(-0.5 * ((self.dpsi - tot) / sig) ** 2)
                self.p = self.p * (c.get("like_floor", 0.02) + like) ** c.get("like_pow", 0.15)
                self.p /= self.p.sum()
        self._estimate()

    def _estimate(self):
        m = self.map
        i = int(np.argmax(self.p))
        k = int(1.0 / m.ds)
        idx = (i + np.arange(-k, k + 1)) % m.n
        w = self.p[idx]
        mass = float(w.sum())
        x = np.arange(-k, k + 1) * m.ds
        mu = float((w * x).sum() / max(1e-9, mass))
        self.s = m.wrap(i * m.ds + mu)
        self.sd = float(math.sqrt(max(0.0, (w * (x - mu) ** 2).sum() / max(1e-9, mass))))
        self.ok = mass >= self.c.get("ok_mass", 0.8) and self.sd <= self.c.get("ok_sd", 0.5)


def load(base_dir, cfg):
    path = os.path.join(base_dir, cfg.get("file", "track_map.json"))
    if not cfg.get("enable") or not os.path.exists(path):
        return None, None
    tm = TrackMap(path)
    return tm, Localizer(tm, cfg)
