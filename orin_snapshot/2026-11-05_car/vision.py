"""辨識：找左右白線（沿用 lane_view_check.py 驗證過的演算法），換成車身座標的中線。

車身座標：原點在後輪軸中心，x 往前、y 往左，單位公尺。
不用四點鳥瞰校正：有了地平線位置（基準檔）、鏡頭高度和視角，就能把畫面上任一點換到地面。

輸出 LaneResult：
  offset     車子偏離中線多少 m，正值 = 車子偏右（中線在車子左邊）
  heading    中線方向相對車頭的角度（度），正值 = 路往左偏
  curvature  前方中線曲率 1/m，正值 = 左彎
  confidence 兩條線 = 1、一條 = 0.5、沒有 = 0
  center     中線擬合 y = c0·x² + c1·x + c2（車身座標）；None 代表沒有
"""
import json
import math
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lane_view_check as lvc  # noqa: E402

W, H = lvc.W, lvc.H


@dataclass
class LaneResult:
    confidence: float = 0.0
    offset: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    center: np.ndarray = None
    x_range: tuple = (0.0, 0.0)       # 中線在 x 方向看得到的範圍（m）
    shape: str = "none"               # straight / left / right / none
    slope: str = "unknown"            # flat / uphill / downhill / unknown
    slope_deg: float = None
    lane_width: float = None          # 這一張量到的路寬（m）
    corner: tuple = None              # 前方直角彎：(轉向 "left"/"right", 內角 x, 內角 y)；沒有就 None
    corner_weak: tuple = None         # 同上，但只看「內側線比較短」，不檢查直不直（已經在處理直角彎時用來持續追蹤）
    corner_dbg: str = ""              # 除錯用：左右線最遠距離、是否夠直，寫進紀錄檔
    pts: np.ndarray = field(default=None, repr=False)  # 中線取樣點 (N, 2)
    raw: dict = field(default=None, repr=False)        # lane_view_check.analyse 的結果（畫圖用）


class GroundMap:
    """畫面座標（320×240）↔ 車身座標，假設路面是平的。"""

    def __init__(self, v_h, cam_height, hfov, cam_to_rear):
        self.f = (W / 2) / math.tan(math.radians(hfov / 2))
        self.cx, self.cy = W / 2, H / 2
        self.v_h = v_h
        self.p = math.atan((self.cy - v_h) / self.f)  # 鏡頭俯角
        self.hc, self.dx = cam_height, cam_to_rear

    def to_vehicle(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        cp, sp = math.cos(self.p), math.sin(self.p)
        zc = self.f * self.hc / ((v - self.v_h) * cp)
        X = (u - self.cx) * zc / self.f
        D = (zc - self.hc * sp) / cp
        return D + self.dx, -X

    def to_image(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        cp, sp = math.cos(self.p), math.sin(self.p)
        D, X = x - self.dx, -y
        zc = D * cp + self.hc * sp
        zc = np.where(zc > 1e-3, zc, np.nan)
        return self.cx + self.f * X / zc, self.cy + self.f * (self.hc * cp - D * sp) / zc


class LaneDetector:
    def __init__(self, cfg, base_dir="."):
        g, v = cfg["geometry"], cfg["vision"]
        self.geo = g
        self.base = None
        bpath = v.get("baseline")
        if bpath:
            bpath = os.path.join(base_dir, bpath) if not os.path.isabs(bpath) else bpath
            if os.path.exists(bpath):
                with open(bpath, encoding="utf-8") as f:
                    self.base = json.load(f)
                print(f"[vision] 已載入基準 {bpath}（地平線 y={self.base['h_near']:.1f}）")
            else:
                print(f"[vision] 找不到基準檔 {bpath}，改用每張畫面自己估的地平線（較不穩）。"
                      "請先用 lane_view_check.py --save-baseline 建立。")
        b = self.base or {}
        self.acfg = SimpleNamespace(
            roi_top=v["roi_top"] if v.get("roi_top") is not None else b.get("roi_top", 0.35),
            roi_bottom=v["roi_bottom"] if v.get("roi_bottom") is not None else b.get("roi_bottom", 1.0),
            thresh=v["thresh"], min_thresh=v["min_thresh"], hfov=g["hfov"], curve_px=v["curve_px"],
            slope_deg=v["slope_deg"], min_span=v["min_span"], cutoff_w=v["cutoff_w"],
            nwin=v["nwin"], margin=v["margin"], minpix=v["minpix"])
        self.smooth = v["smooth"]
        self.lane_w = g["lane_width"]
        self._coef = None
        self._last_center = None  # 最近一次成功的中線，看不到線也不清掉：只剩一條線時用來判斷它是左線還是右線
        self.only_left = False    # 只沿左線走（坡頂平台：右邊有虛線、藍膠帶、U 形框會干擾）；由 main.py 的爬坡模式切換
        self.w_recent = None      # 最近兩條平行線量到的路寬（不管合不合理）：只沿左線走時用它的一半
        self._y_lr = None         # 上一張左線、右線在近處的橫向位置（m）：只剩一條線時用來判斷是哪一條
        self._prev = None         # 上一張的結果（爬坡模式擋偏移突然跳動用）
        self._n_jump = 0
        self._v_h = b.get("h_near")
        self.gmap = self._make_map(self._v_h) if self._v_h is not None else None

    def _make_map(self, v_h):
        g = self.geo
        return GroundMap(v_h, g["cam_height"], g["hfov"], g["cam_to_rear_axle"])

    def process(self, bgr):
        r = lvc.analyse(bgr, self.acfg, self.base)
        self._bin = r["_bin"]
        res = LaneResult(confidence=r["confidence"], shape=r["shape"], slope=r["slope"],
                         slope_deg=r["slope_deg"], raw=r)
        if self.base is None and r["h_near"] is not None and r["h_near"] < r["roi_top"]:
            # 沒有基準：用近段消失點慢慢更新地平線
            self._v_h = r["h_near"] if self._v_h is None else 0.9 * self._v_h + 0.1 * r["h_near"]
            self.gmap = self._make_map(self._v_h)
        if self.gmap is None or r["n_lines"] == 0:
            self._coef = None
            return res
        pts = self._centerline(r["_left"], r["_right"], res)
        if pts is None or len(pts) < 6:
            res.confidence = 0.0
            self._coef = None
            return res
        x, y = pts[:, 0], pts[:, 1]
        if self._jump(pts, res):
            return res
        deg = 2 if np.ptp(x) > 0.15 else 1
        coef = np.polyfit(x, y, deg)
        if deg == 1:
            coef = np.r_[0.0, coef]
        if self._coef is not None and self.smooth > 0:
            coef = self.smooth * self._coef + (1 - self.smooth) * coef
        self._coef = self._last_center = coef
        a, b, c = coef
        x0 = float(x.min())
        res.center, res.pts, res.x_range = coef, pts, (x0, float(x.max()))
        res.offset = float(np.polyval(coef, x0))
        slope = 2 * a * x0 + b
        res.heading = math.degrees(math.atan(slope))
        res.curvature = float(2 * a / (1 + slope ** 2) ** 1.5)
        self._prev = (coef, pts, res.x_range, res.offset, res.heading, res.curvature)
        return res

    def _jump(self, pts, res):
        """爬坡模式：這張的中線和上一張差太多（車子不可能一張畫面就橫移 JUMP_M）→ 沿用上一張的結果。
        坡道上地面不是平的，換算成公尺常常一張就跳 30–40 cm（run_20261029_181936 49.3 s：+5 → +41 cm），
        前輪跟著打到底、加速一推就衝出坡頂。連續 JUMP_N 張都一樣才當成是真的。"""
        if not self.only_left or self._prev is None:
            self._n_jump = 0
            return False
        coef, ppts, xr, off, hd, curv = self._prev
        x0 = float(pts[:, 0].min())
        raw = float(np.median(pts[pts[:, 0] < x0 + 0.2, 1]))       # 這張近處的中線位置（還沒平滑）
        if abs(raw - float(np.polyval(coef, x0))) <= JUMP_M or self._n_jump >= JUMP_N:
            self._n_jump = 0
            return False
        self._n_jump += 1
        res.center, res.pts, res.x_range = coef, ppts, xr
        res.offset, res.heading, res.curvature = off, hd, curv
        return True

    def _line_pts(self, ln):
        rows, xs = lvc._row_centers(ln)
        ok = rows > self.gmap.v_h + 4
        if ok.sum() < 5:
            return None
        x, y = self.gmap.to_vehicle(xs[ok], rows[ok])
        good = (x > 0) & (x < 5)
        return np.stack([x[good], y[good]], 1) if good.sum() >= 5 else None

    def _corner(self, L, R):
        """直角彎：內側線在轉角就斷了，外側線還會往前延伸。比較兩條線看得到的最遠距離，
        短很多的那條是內側線，它的最遠點就是內角。回傳 (轉向, 內角 x, 內角 y) 或 None。"""
        if L is None or R is None:
            return None, None
        (lx, ly), (rx, ry) = _far_end(L), _far_end(R)
        if rx - lx > CORNER_GAP:
            weak = ("left", lx, ly)
            return (weak if _straight(L) and self._sideways("left", lx, ly) else None), weak
        if lx - rx > CORNER_GAP:
            weak = ("right", rx, ry)
            return (weak if _straight(R) and self._sideways("right", rx, ry) else None), weak
        return None, None

    def _sideways(self, side, x, y):
        """真的直角彎：內側線在轉角會往轉彎方向「橫著」接下去（和車道垂直、距離不變，在畫面上是水平的）。
        排除兩種長得很像的情況：
          賽道盡頭：轉角後面什麼都沒有 → 找不到白線
          岔路（「>」形）：另一條線斜斜地往車子這邊回來 → 在畫面上往下斜
        做法：從內角往轉彎那一側，每隔 4 個像素看那一欄的白點在第幾列，連起來要夠長、而且接近水平。"""
        u, v = self.gmap.to_image(x, y)
        if not (np.isfinite(u) and np.isfinite(v)):
            return False
        u, v = int(round(float(u))), int(round(float(v)))
        b = self._bin
        d = -1 if side == "left" else 1
        r0, r1 = max(0, v - SIDE_UP), min(b.shape[0], v + SIDE_DOWN)
        offs = list(range(SIDE_NEAR, SIDE_FAR + 1, 4))
        pts = []
        for off in offs:
            c = u + d * off
            if not 0 <= c < b.shape[1]:
                break
            rows = np.nonzero(b[r0:r1, c])[0]
            if len(rows):
                pts.append((off, float(rows.mean())))
        if len(pts) < SIDE_FILL * len(offs):              # 白線不夠長（或根本沒有）→ 賽道盡頭
            return False
        p = np.array(pts)
        slope = np.polyfit(p[:, 0], p[:, 1], 1)[0]        # 每往旁邊 1 像素，白線往下幾列
        return abs(slope) <= SIDE_SLOPE

    def _centerline(self, left, right, res):
        L = self._line_pts(left) if left is not None else None
        R = self._line_pts(right) if right is not None else None
        res.corner, res.corner_weak = self._corner(L, R)
        res.corner_dbg = _corner_dbg(L, R)
        parallel = L is not None and R is not None and self._track_width(L, R)
        long_l = L is not None and np.ptp(L[:, 0]) >= LEFT_ONLY_SPAN
        if self.only_left and L is not None and R is not None and not parallel and long_l:
            # 爬坡模式：右線和左線不平行（右邊的虛線 / 膠帶 / U 形框）就只用左線，和它保持 w_recent 的一半。
            # 平行時照常兩條線置中：坡道底部近處兩條線會往內彎成 V 形，只用左線平移會放大誤差
            # （run_20261028_190458 46 s：車子在正中間，只用左線算出偏 +29 cm）。只剩一條線時照常用連續性判斷左右：
            # 坡頂平台的左邊界會往左彎出畫面，那時候剩下的那條是右線（run_20261028_183945 52.8 s），硬當左線會往右衝
            R = None
        if L is not None and R is not None:
            d2 = ((L[:, None, :] - R[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(1)
            ok = (j > 0) & (j < len(R) - 1)
            w = np.sqrt(d2[np.arange(len(L)), j])[ok] if ok.sum() >= 5 else np.zeros(1)
            apart = right["x_bot"] - left["x_bot"] > W / 2
            # 近處也量一次：粗線被拆成兩條時，路寬剛好落在 0.5 倍門檻上，會被當成兩條線（run_20261031_154631 150.9 s）
            close = _near_y(L) - _near_y(R) < 0.6 * self.geo["lane_width"]
            if (np.median(w) < 0.5 * self.geo["lane_width"] or close) and not apart:
                # 畫面上分在兩側（相距超過半個畫面寬）就一定是兩條線：坡道上量到的路寬會偏窄（0.26–0.29 m），
                # run_20261028_190458 46.9–51.5 s 兩條線清清楚楚，卻被當成同一條、只留一條又判錯邊，前輪打右到底 4.5 秒
                # 路寬不合理（兩條「線」其實是同一條）：只留像素多的那條，當成單線處理
                keep, pts = (left, L) if left["n"] >= right["n"] else (right, R)
                L, R = (pts, None) if keep["x_bot"] < W / 2 else (None, pts)
                res.confidence = 0.5
            elif ok.sum() >= 5:
                res.lane_width = float(np.median(w))
                # 量到的路寬合理才拿來更新（只看到一條線時要用）
                if 0.6 * self.geo["lane_width"] < res.lane_width < 1.5 * self.geo["lane_width"]:
                    self.lane_w = 0.95 * self.lane_w + 0.05 * res.lane_width
                self._y_lr = (_near_y(L), _near_y(R))
                return (L[ok] + R[j[ok]]) / 2
        one, side = (L, -1) if L is not None else (R, 1)   # 左線往右推（y 減少）、右線往左推
        if one is None:
            return None
        half = self.lane_w / 2
        if self.only_left and self.w_recent is not None:
            half = self.w_recent / 2
        y_near = _near_y(one)
        if self._y_lr is not None:
            # 急彎時外側線會滑到畫面另一半，用「畫面左半 = 左線」會判反、方向打錯。線的位置每張之間是連續的，
            # 看它比較靠近上一張的左線還是右線。不能用「在上一次中線哪一邊」：一條粗線被拆成兩條時中線跟著跳，
            # 左線就被判成右線，偏移從 16 cm 跳到 56 cm，前輪往左開出賽道（run_20261031_154631 第二圈 151 s）
            side = -1 if abs(y_near - self._y_lr[0]) <= abs(y_near - self._y_lr[1]) else 1
        elif self._last_center is not None:
            rel = one[:, 1] - np.polyval(self._last_center, one[:, 0])
            side = -1 if np.median(rel) > 0 else 1
        self._y_lr = (y_near, y_near - 2 * half) if side < 0 else (y_near + 2 * half, y_near)
        x, y = one[:, 0], one[:, 1]
        order = np.argsort(x)
        x, y = x[order], y[order]
        k = np.polyfit(x, y, 2 if np.ptp(x) > 0.15 else 1)
        dy = np.polyval(np.polyder(k), x)
        n = np.stack([-dy, np.ones_like(dy)], 1) / np.sqrt(1 + dy ** 2)[:, None]  # 左法向量
        return np.stack([x, y], 1) + side * n * half

    def _track_width(self, L, R):
        """兩條線平行就回傳 True，並更新 w_recent。坡道上量到的路寬只有 0.29 m（平地 0.55，run_20261028_183945 47–50 s），
        原因不明，但兩條線置中在坡道上爬得好好的；只沿左線走時如果固定用 lane_width 的一半會貼到右線。
        只在兩條線平行時更新（坡頂平台右邊的 U 形框、膠帶通常不平行）。"""
        xs = np.linspace(max(L[:, 0].min(), R[:, 0].min()), min(L[:, 0].max(), R[:, 0].max()), 5)
        if xs[-1] - xs[0] < 0.15:
            return False
        kl = np.polyfit(L[:, 0], L[:, 1], 1)
        kr = np.polyfit(R[:, 0], R[:, 1], 1)
        if abs(math.degrees(math.atan(kl[0]) - math.atan(kr[0]))) > 12:
            return False
        w = float(np.median(np.polyval(kl, xs) - np.polyval(kr, xs))) * math.cos(math.atan(kl[0]))
        if 0.2 < w < 0.8:
            self.w_recent = w if self.w_recent is None else 0.5 * self.w_recent + 0.5 * w   # 坡頂路寬 0.29 → 0.58 要跟得上
            return True
        return False


def _near_y(pts):
    """線在近處（最近 40% 的點）的橫向位置（m）。"""
    return float(np.median(pts[pts[:, 0] <= np.percentile(pts[:, 0], 40), 1]))


JUMP_M, JUMP_N = 0.12, 6   # 爬坡模式：中線一張跳超過 0.12 m 先不信，連續 6 張（0.2 s）才接受
LEFT_ONLY_SPAN = 0.5  # 只沿左線走時，左線至少要看到這麼長（m）；太短又斜的一小段平移出去誤差很大
                     # （run_20261029_181936 49.3 s：左線只剩 0.40–0.69 m，算出偏 +43 cm，坡頂又往左衝出去）
CORNER_GAP = 0.30   # 內側線比外側線短這麼多（m）才算直角彎
CORNER_KMAX = 0.8   # 直角彎的內側線在轉角前是直的；曲率超過這個（1/m）代表是一般彎道，不算
SIDE_NEAR, SIDE_FAR = 8, 60   # 從內角往轉彎方向，檢查畫面上第幾到第幾個像素（320×240）
SIDE_UP, SIDE_DOWN = 15, 40   # 內角往上、往下各看幾列（實測真直角彎的橫線在內角下方 13–25 列）
SIDE_FILL = 0.6               # 檢查的欄位裡至少這個比例找得到白線
SIDE_SLOPE = 0.35             # 橫線斜率上限（列/像素）：實測直角彎 −0.21～+0.23，岔路「>」約 +0.48


def _straight(P):
    """一般彎道的內側線會慢慢彎，直角彎的內側線到轉角前都是直的。
    最遠的 12 cm 不算：那段已經在轉角上、開始往旁邊轉，會讓直線看起來像彎的。"""
    keep = P[:, 0] < P[:, 0].max() - 0.12
    x, y = P[keep, 0], P[keep, 1]
    if len(x) < 6 or np.ptp(x) < 0.15:
        return False
    a = np.polyfit(x, y, 2)[0]
    return abs(2 * a) < CORNER_KMAX


def _corner_dbg(L, R):
    f = lambda P: "-" if P is None else f"{P[:, 0].max():.2f}{'直' if _straight(P) else '彎'}"
    return f"L{f(L)} R{f(R)}"


def _far_end(P):
    i = int(np.argmax(P[:, 0]))
    return float(P[i, 0]), float(P[i, 1])


def target_point(res, ld):
    """中線上離後輪軸 ld 公尺的點；看不到那麼遠就取最遠的可見點。"""
    if res.center is None:
        return None
    x0, x1 = res.x_range
    xs = np.linspace(max(0.0, x0), max(x1, x0 + 0.01), 200)
    ys = np.polyval(res.center, xs)
    d = np.hypot(xs, ys)
    i = int(np.argmax(d >= ld)) if (d >= ld).any() else len(xs) - 1
    return float(xs[i]), float(ys[i])
