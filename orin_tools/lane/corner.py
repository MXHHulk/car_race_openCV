"""直角彎：走「外 → 內 → 外」的大弧線，代替沿中線硬轉。

沿中線過直角彎，半徑只有半個路寬（約 0.27 m），比車子最小迴轉半徑還小，轉不過去。
改成：看到彎道先往外側靠 → 內角快到時開始走一段固定半徑的圓弧、擦過內角 → 轉完 90° 再接回循線。

流程（mode）：
  idle      平常循線
  approach  看到內角（vision 的 res.corner），目標往外側偏 out_offset；每張記下內角距離，用來估車速
  wait      內角進到鏡頭近處盲區看不到了，用估出的車速推算它還有多遠
  turn      內角剩 trigger 公尺時開始轉：固定前輪角度走圓弧，轉完 90° 的時間到了就回到 idle

圓弧半徑在兩個限制中間取中點（車子後輪軸中心走的圓，起點在目前位置、方向沿著車道）：
  不切到內角：R ≥ (x² + a² − m²) / (2(a − m))     x = 內角在前方幾 m、a = 內角在側邊幾 m、m = 車寬一半 + 線寬一半
  不壓到出彎的外側線：R ≤ x + 路寬 − m
"""
import math

import numpy as np

APPROACH_MAX = 1.2   # 內角在這個距離內才開始處理（m）
EXIT_HEADING = 12.0  # 出彎車道和車頭夾角小於這個（度）才算轉完
LOST_ABORT = 0.8    # 推算 / 轉彎中完全看不到線超過幾秒就放棄（交回 main 的「找不到線 → 停車」）


class CornerPlanner:
    def __init__(self, cfg):
        c = cfg["control"].get("corner") or {}
        g = cfg["geometry"]
        self.on = bool(c.get("enable", False))
        self.out = c.get("out_offset", 0.10)          # 入彎前往外側靠多少（m）
        self.trigger = c.get("trigger", 0.30)         # 內角剩多遠開始轉（m，後輪軸到內角）
        self.v_default = c.get("v_default", 0.4)      # 估不出車速時用的值（m/s）
        self.max_wait = c.get("max_wait", 1.5)        # 內角看不到後最多推算幾秒
        self.m = g.get("car_width", 0.30) / 2 + c.get("line_half", 0.025)
        self.W = g["lane_width"]
        # 等效軸距：車子實際轉得多急（打滑、連桿都算進去），用 tools/turn_test.py 量。
        # 和一般循線用的 geometry.wheelbase（實際軸距）分開：等效值比實際大很多，拿去給 Pure Pursuit 用會轉向過猛。
        self.L = c.get("wheelbase", g["wheelbase"])
        self.fwa_lim = cfg["control"]["fwa_limit"]
        self.info = ""
        self.reset()

    def reset(self):
        self.mode, self.side, self.hist = "idle", None, []
        self.t0 = self.T = self.fwa = self.v = None
        self.t_line = None

    @property
    def active(self):
        """入彎、推算、轉彎中：鏡頭看不到線是正常的，不要因此停車（推算最多 max_wait 秒就會回到 idle）。"""
        return self.mode in ("approach", "wait", "turn")

    def _speed(self):
        """內角距離隨時間變短的速率 = 車速。至少要 3 張畫面。"""
        h = self.hist[-6:]                      # 只看最近幾張：入彎時會減速，太舊的會把車速估高
        if len(h) < 3:
            return None
        t = np.array([p[0] for p in h])
        x = np.array([p[1] for p in h])
        if np.ptp(t) < 0.05:
            return None
        v = -np.polyfit(t - t[0], x, 1)[0]
        return v if 0.05 < v < 3.0 else None

    def _start(self, t, x, a):
        m = self.m
        r_apex = (x * x + a * a - m * m) / (2 * (a - m)) if a > m + 0.01 else float("inf")
        r_exit = x + self.W - m
        R = (r_apex + r_exit) / 2 if math.isfinite(r_apex) else r_exit
        r_min = self.L / math.tan(math.radians(self.fwa_lim))
        R = max(R, r_min)
        v = self.v or self._speed() or self.v_default
        self.mode, self.t0, self.v = "turn", t, v
        self.T = (math.pi / 2) * R / v
        deg = math.degrees(math.atan(self.L / R))
        self.fwa = -deg if self.side == "left" else deg     # 協議：負左正右
        self.info = (f"開始轉{self.side}：內角 {x:.2f} m 前、{a:.2f} m 側；半徑 {R:.2f} m"
                     f"（可用 {r_apex:.2f}~{r_exit:.2f}）、前輪 {self.fwa:+.0f}°、車速 {v:.2f} m/s、預計 {self.T:.1f} s")
        print(f"[corner] {self.info}")
        return self.fwa, 0.0

    def update(self, res, t):
        """回傳 (強制前輪角度或 None, 中線橫向偏移 m)。偏移正值 = 往左。"""
        if not self.on or t is None:
            return None, 0.0
        if res.confidence > 0 or self.t_line is None:
            self.t_line = t
        if self.mode in ("wait", "turn") and t - self.t_line > LOST_ABORT:
            print(f"[corner] {LOST_ABORT} 秒完全看不到線，放棄轉彎")
            self.reset()
            return None, 0.0
        c = res.corner
        if self.mode == "approach" and c is None and res.corner_weak and res.corner_weak[0] == self.side:
            c = res.corner_weak          # 已經在入彎：內角太近、內側線剩一小段判斷不了直不直，只看同一邊是否較短
        if self.mode in ("idle", "approach"):
            if c is not None and c[1] < APPROACH_MAX:
                if c[0] != self.side:
                    self.hist = []
                self.mode, self.side = "approach", c[0]
                self.hist = (self.hist + [(t, c[1], abs(c[2]))])[-12:]
                if c[1] <= self.trigger:
                    return self._start(t, c[1], abs(c[2]))
            elif self.mode == "approach":
                # 內角進了近處盲區：用車速往前推算
                self.v = self._speed()
                self.mode = "wait"
        if self.mode == "wait":
            t_seen, x_seen, a = self.hist[-1]
            v = self.v or self.v_default
            x_now = x_seen - v * (t - t_seen)
            if x_now <= self.trigger:
                return self._start(t, max(x_now, 0.05), a)
            if t - t_seen > self.max_wait:
                self.reset()
        if self.mode == "turn":
            # 什麼時候轉完：不能只看時間（車速估不準就轉不足或轉過頭）。
            # 時間過了 70% 後，鏡頭看到出彎的車道已經接近正前方才結束；最多等到 1.6 倍時間。
            el = t - self.t0
            aligned = res.center is not None and abs(res.heading) <= EXIT_HEADING
            if (el >= 0.7 * self.T and aligned) or el >= 1.6 * self.T:
                print(f"[corner] 轉完（{el:.1f} s，預計 {self.T:.1f} s），回到循線")
                self.reset()
                return None, 0.0
            return self.fwa, 0.0
        if self.mode in ("approach", "wait"):
            return None, (-self.out if self.side == "left" else self.out)   # 左彎外側在右（y 負）
        return None, 0.0
