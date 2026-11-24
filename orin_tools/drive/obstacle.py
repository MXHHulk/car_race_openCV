"""前方車輛 / 障礙物偵測（只用鏡頭）。11-23 依使用者錄的 4 台車（1.0 m、同向正前方，各 10 張）和沒車的跑車影片做的。

想法：別台車在我們車道裡是一塊「比路面暗（或很亮）的東西」，而且兩旁（車道內、車子旁邊）還看得到路面；
  隧道 / 暗處是整片變暗，兩旁也一起暗 → 不算；路面盡頭（坡頂、平台邊）車道線也在那裡結束 → 不算。

320×240 灰階，從近往遠一列一列看：
  中帶：車道中線 ± core_w（m）  側帶：中線 ±[side_in, side_out]（m），左右各一條
  路面參考 ref：近處 n_init 列起算，之後只用「很像路面」（差 ≤ upd_tol）的列更新（跟著鏡頭暗角：越遠越亮）
  一列像車：中帶中位數 < ref − drop（暗）或 > ref + bright（很亮），而且左或右側帶和 ref 差 ≤ side_tol
  連續 min_rows 列像車 → 第一列的地面距離
  只找到「車道線看得到的距離 − far_margin」（車比車道窄，車後面兩旁還看得到線；路面盡頭線就斷了）
驗證（11-23）：40 張有車偵測到 39 張，量到 0.98–1.09 m（標 1.0 m）；沒車影片單張誤判 2.3–2.7%，幾乎都在坡道區。
"""
import numpy as np
import cv2


class ObstacleDetector:
    def __init__(self, eye, cfg=None):
        self.eye = eye
        c = cfg or {}
        self.core_w = c.get("core_w", 0.08)
        self.side_in = c.get("side_in", 0.14)
        self.side_out = c.get("side_out", 0.20)
        self.x_near = c.get("x_near", 0.30)
        self.x_far = c.get("x_far", 1.6)
        self.n_init = int(c.get("n_init", 6))
        self.drop = c.get("drop", 14)
        self.bright = c.get("bright", 45)
        self.side_tol = c.get("side_tol", 10)
        self.min_rows = int(c.get("min_rows", 3))
        self.ema = c.get("ema", 0.3)
        self.need_lane = bool(c.get("need_lane", 1))
        self.upd_tol = c.get("upd_tol", 6)
        self.far_margin = c.get("far_margin", 0.25)

    def process(self, gray, center=None, far=None):
        e = self.eye
        H, W = gray.shape
        if self.need_lane and (center is None or len(center) < 2):
            return None, {"why": "沒有車道"}
        x_far = self.x_far if far is None else min(self.x_far, far - self.far_margin)
        if x_far <= self.x_near + 0.05:
            return None, {"why": "車道看得太近"}
        _, vn = e.to_image(np.array([self.x_near]), np.array([0.0]))
        _, vf = e.to_image(np.array([x_far]), np.array([0.0]))
        v0, v1 = min(H - 1, int(vn[0])), max(0, int(vf[0]))
        if v0 <= v1:
            return None, {"why": "範圍太小"}
        vs = np.arange(v0, v1 - 1, -1)
        xs = e.h * e.f / np.maximum(vs - e.vh, 1e-3)
        cy = np.zeros_like(xs) if center is None else np.interp(xs, center[:, 0], center[:, 1])
        # 一次算好三條帶子在每一列的左右欄（向量化：一列一列呼叫座標轉換要 22 ms）
        edges = [cy - self.core_w, cy + self.core_w, cy + self.side_in, cy + self.side_out,
                 cy - self.side_out, cy - self.side_in]
        us = [np.round(e.to_image(xs, y)[0]).astype(int) for y in edges]
        off = max(0, v1 - 2)
        g = cv2.medianBlur(np.ascontiguousarray(gray[off:v0 + 3]), 5)
        # 每條帶子每一列的平均（先中值濾波過，平均也穩）：用累加和一次算完（每列 np.median 要 12 ms）
        cs = np.concatenate([np.zeros((g.shape[0], 1)), np.cumsum(g, axis=1, dtype=np.float64)], axis=1)
        ridx = vs - off

        def band(ua, ub):
            a = np.clip(np.minimum(ua, ub), 0, W - 1)
            b = np.clip(np.maximum(ua, ub), 0, W - 1)
            n = b - a + 1
            m = (cs[ridx, b + 1] - cs[ridx, a]) / np.maximum(n, 1)
            return np.where(n >= 2, m, np.nan)

        C, Lb, Rb = band(us[0], us[1]), band(us[2], us[3]), band(us[4], us[5])

        ref, run, first, rows = None, 0, None, []
        for k in range(len(vs)):
            c = C[k]
            if np.isnan(c):
                continue
            c = float(c)
            l = None if np.isnan(Lb[k]) else float(Lb[k])
            r = None if np.isnan(Rb[k]) else float(Rb[k])
            x = float(xs[k])
            if ref is None or k < self.n_init:
                ref = c if ref is None else (1 - self.ema) * ref + self.ema * c
                rows.append((x, c, l, r, 0)); continue
            odd = (c < ref - self.drop) or (c > ref + self.bright)
            side_ok = any(s is not None and abs(s - ref) <= self.side_tol for s in (l, r))
            car = odd and side_ok
            rows.append((x, c, l, r, int(car)))
            if car:
                run += 1
                if run == 1:
                    first = x
                if run >= self.min_rows:
                    return first, {"rows": rows, "ref": ref}
            else:
                run = 0
                if abs(c - ref) <= self.upd_tol:
                    ref = (1 - self.ema) * ref + self.ema * c
        return None, {"rows": rows, "ref": ref}
