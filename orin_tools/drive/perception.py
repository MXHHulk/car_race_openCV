"""看畫面：車道線 → 地面座標 → 中線；另外看前方橫越的虛線、暗處、畫面移動量。

不記任何賽道位置，每張畫面只用自己看到的東西（加上上一張的左右線位置，分辨只剩一條線時是哪一邊）。

座標：地面 x 往前、y 往左（m），原點在鏡頭正下方。平地假設：
  像素 (u, v)（工作解析度）→ x = h·f / (v − v_h)，y = −(u − c_x)·x / f
v_h（地平線列）、c_x（正前方那一欄）由 tools/calibrate.py 在直線上量出來存成 calib.json。
"""
import collections
import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Line:
    pts: np.ndarray            # (N, 2) 地面座標，由近到遠
    length: float
    near_y: float              # 最近一點的 y
    width: float = 0.0         # 地面上的平均寬度（m）＝ 面積 / 長度
    is_dash: bool = False      # 沿行進方向的虛線串成的線


@dataclass
class Sense:
    left: Line = None
    right: Line = None
    center: np.ndarray = None  # (N, 2) 中線，由近到遠；None = 看不到線
    conf: int = 0              # 看到幾條線（0 / 1 / 2）
    offset: float = 0.0        # 中線在最近處的 y（正 = 中線在車子左邊 → 車子偏右）
    heading: float = 0.0       # 中線近處方向（度，正 = 往左）
    far: float = 0.0           # 中線看到多遠（m）
    lane_w: float = None       # 兩條線都有時量到的路寬
    dark: bool = False
    bright: float = 0.0
    dash: int = 0              # 前方橫越虛線排成一線的塊數
    dash_x: float = None       # 那條虛線在車子正前方多遠（m，從鏡頭算）
    far_bright: float = 255.0  # ROI 遠處（上 1/3）的平均亮度
    dark_k: float = 1.0        # 暗處 / 隧道口門檻倍率（場地整體變暗時 < 1）
    vp_row: float = None       # 兩條線（近處 0.9 m）在畫面上的交點列（320×240）；離地平線很遠 = 前面的路和車子不在同一個平面（坡底）
    fork: bool = False         # 前方有 Y 字岔路的 V 分隔線
    fork_tip_y: float = None   # V 尖端在車子左右哪裡（m，左正）
    motion: float = 0.0
    thr: float = 0.0
    dbg: dict = field(default_factory=dict)


class Perception:
    def __init__(self, cfg, base_dir="."):
        self.cfg = cfg
        g, v = cfg["geometry"], cfg["vision"]
        self.W, self.H = v["work_w"], v["work_h"]
        self.f = (self.W / 2) / math.tan(math.radians(g["hfov"] / 2))
        self.h = g["cam_height"]
        self.vh, self.cx = self.H * 0.12, self.W / 2      # 沒校正時的預設
        path = os.path.join(base_dir, g.get("calib", "calib.json"))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fp:
                c = json.load(fp)
            self.vh, self.cx = c["horizon"] * self.H / c.get("h", self.H), c["center"] * self.W / c.get("w", self.W)
            self.h = c.get("cam_height", self.h)
            print(f"[vision] 校正檔 {path}：地平線第 {self.vh:.1f} 列、中心第 {self.cx:.1f} 欄、鏡頭高 {self.h:.3f} m")
        else:
            print(f"[vision] ⚠ 沒有校正檔 {path}，用預設地平線（請在直線上跑 tools/calibrate.py）")
        self.lane_w = g["lane_width"]
        self.v = v
        self.prev_off = 0.0           # 上一張中線偏移
        self.prev_l = self.prev_r = None   # 上一張左 / 右線最近處的 y（只剩一條線又在中間時分邊用）
        self.trk_l = self.trk_r = (None, 0)   # 真的看到的左 / 右線：(最近處 y, 第幾張)
        self.frame_n = 0
        self._v_seen = False
        self._v_right = self._v_left = self._v_tip_y = None
        self._v_size = 0.0
        self.fork_side = None          # Driver 設定：岔路口走 "left" / "right"（None = 一般循線）
        self.post_turn = False         # Driver 設定：虛線左轉剛結束
        self.dash_left = False         # Driver 設定：虛線左轉完到下坡前，虛線一律當左線（沿虛線右側走）
        self.dash_side = cfg["junction"].get("dash_side", "right")   # 虛線在車子哪一邊（right = 車走虛線左側）
        self.slope = False             # Driver 設定：在坡上 / 剛到坡頂、坡底
        self.slope_down = False        # Driver 設定：下坡中 / 坡底（俯仰 ≤ −foot_pitch）
        self.last_side = None          # 只剩一條線時上一張判成哪一邊
        self.prev_gray = None
        self.n_dark = 0
        self.bright_hist = collections.deque(maxlen=int(self.v.get("bright_ref_n", 180)))
        self.dark_k = 1.0

    # ------------------------------------------------------------------ 座標
    def to_ground(self, u, v):
        x = self.h * self.f / np.maximum(v - self.vh, 1e-3)
        return x, -(u - self.cx) * x / self.f

    def to_image(self, x, y):
        x = np.maximum(np.asarray(x, dtype=float), 1e-3)
        return self.cx - np.asarray(y) * self.f / x, self.vh + self.h * self.f / x

    def roi(self):
        top = int(max(0, self.vh + self.v["roi_gap"]))
        bot = int(self.v["roi_bottom"] * self.H)
        return top, max(top + 10, bot)

    # ------------------------------------------------------------------ 主流程
    def process(self, frame):
        s = Sense()
        self.frame_n += 1
        small = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        s.motion = self._motion(frame)
        top, bot = self.roi()
        roi = cv2.medianBlur(gray[top:bot], 3)
        s.bright = float(roi.mean())
        s.far_bright = float(roi[: max(1, roi.shape[0] // 3)].mean())   # ROI 上面 1/3（遠處）的亮度：隧道口先變暗
        # 暗處 / 隧道口門檻跟著場地亮度：最近 bright_ref_n 張亮度的 75 百分位當「正常亮度」，
        #  低於 bright_ref_nom 時門檻按比例降低（只降不升，昨晚那種亮度維持原門檻）
        #  11-23 105609 / 105632：場地亮度中位 45（11-22 221554 是 65），一般彎道亮度 36–47 剛好在門檻 40 / 44 / 50 附近，
        #  14% 的畫面被判成隧道口、25–30% 在「暗處維持方向」，第一個右彎前輪被鎖成 −2 ~ −7°（往左），沒轉
        self.bright_hist.append(s.bright)
        nom = self.v.get("bright_ref_nom", 0)
        if nom > 0 and len(self.bright_hist) >= 15:
            ref = float(np.percentile(self.bright_hist, 75))
            self.dark_k = max(self.v.get("bright_k_min", 0.5), min(1.0, ref / nom))
        s.dark_k = self.dark_k
        self.n_dark = self.n_dark + 1 if s.bright < self.v["dark"] * self.dark_k else 0
        s.dark = self.n_dark >= self.v["dark_n"]
        thr, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 門檻下限跟著畫面亮度：隧道裡白線只有 90–100，固定 110 會整條不見（140849 35 s）
        lo = min(self.v["min_thr"], max(self.v["dark_min_thr"], 2.2 * float(np.median(roi))))
        thr = float(min(self.v["max_thr"], max(lo, thr)))
        self._v_right = self._v_left = self._v_tip_y = None
        self._v_size = 0.0
        s.thr = thr
        binary = (roi > thr)
        # 白線是中性的白；坡道旁的米色木板也很亮但偏黃（飽和度高）：180923 坡上真的右線和木板連成一塊被認錯
        smax = self.v.get("sat_max")
        if smax:
            sat = cv2.cvtColor(small[top:bot], cv2.COLOR_BGR2HSV)[..., 1]
            # 只對很亮的像素要求低飽和度：隧道裡白線暗、顏色雜訊大（飽和度 31–55），190231 整條右線被濾掉
            binary &= (sat < smax) | (roi < self.v.get("sat_gray", 120))
        binary = binary.astype(np.uint8)
        lines = self._lines(binary, top)
        s.fork, s.fork_tip_y = self._v_seen, self._v_tip_y
        self._v_seen = False
        if self.fork_side:
            self._assign_fork(s, lines, self.fork_side)
        elif self.dash_left and any(l.is_dash for l in lines):
            self._assign_dash_left(s, lines)
        else:
            self._assign(s, lines)
        self._center(s)
        self.dash_x = None
        s.dash = self._dash(frame)
        s.dash_x = self.dash_x
        s.dbg = {"top": top, "bot": bot, "cands": lines}
        return s

    def _motion(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60), interpolation=cv2.INTER_AREA)
        g = g[15:45].astype(np.float32)
        m = 0.0 if self.prev_gray is None else float(np.mean(np.abs(g - self.prev_gray)))
        self.prev_gray = g
        return m

    def _lines(self, binary, top):
        """白色元件 → 地面上的折線（由近到遠）。太短、太遠的不要。"""
        n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = []
        blobs = []
        max_r = self.v["max_range"]
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 15:
                continue
            ys, xs = np.nonzero(lab == i)
            rows = st[i, cv2.CC_STAT_HEIGHT]
            xt = self._is_v(ys, xs, rows)
            if xt is not None and not self._v_ground_ok(xs, ys, xt, top):
                xt = None
            if xt is not None:
                # Y 字岔路的 V 分隔線：不當車道線。左右臂都留著：走右邊時右臂是那條路的左線，走左邊時左臂是右線
                self._v_seen = True
                rx = xs >= xt
                ra = self._make_line(xs[rx], ys[rx], top, rows)
                la = self._make_line(xs[~rx], ys[~rx], top, rows)
                size = (ra.length if ra else 0) + (la.length if la else 0)
                if size > self._v_size:
                    self._v_size, self._v_right, self._v_left = size, ra, la
                    y0 = ys.max()
                    gx, gy = self.to_ground(np.array([float(xt)]), np.array([float(y0 + top)]))
                    self._v_tip_y = float(gy[0])
                continue
            ln = self._make_line(xs, ys, top, rows)
            if ln is None or ln.length < 0.45:
                # 短的白塊：可能是沿著車道排的虛線（平台那段左邊界是虛線），留著串成一條線
                gx, gy = self.to_ground(xs.astype(float), ys.astype(float) + top)
                rr = np.hypot(gx, gy)
                if rr.min() <= self.v["max_range"]:
                    ext = float(np.hypot(gx.max() - gx.min(), gy.max() - gy.min()))
                    if ext <= 0.5:
                        blobs.append((float(gx.mean()), float(gy.mean())))
            # 地面上太寬的白色區塊 = 牆、柱子、白板（不在地上，投影出來很寬）：不當車道線
            # 202536 撞柱：白柱子的底邊被當成右線，和左線配成一對，中線被拉往柱子
            if ln is not None and ln.width <= self.v.get("max_line_w", 1e9):
                out.append(ln)
        dl = self._dash_line(blobs)
        if dl is not None:
            out.append(dl)
        out.sort(key=lambda l: -l.length)
        return out[:6]

    def _dash_line(self, blobs):
        """沿著行進方向排成一直線的虛線塊 → 一條車道線（190907 平台左邊界是虛線，只看右實線、照標準路寬算中線偏右，
        車被帶去貼右邊）。至少 dashline_n 塊、方向和車頭差 dashline_deg 以內、塊之間距離 ≤ dashline_gap。"""
        v = self.v
        n_min = v.get("dashline_n", 3)
        if len(blobs) < n_min:
            return None
        c = np.array(blobs)
        best = None
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                d = c[j] - c[i]
                L = float(np.hypot(*d))
                if L < 0.15:
                    continue
                u = d / L
                if abs(math.degrees(math.atan2(u[1], u[0]))) > v["dashline_deg"] and abs(math.degrees(math.atan2(-u[1], -u[0]))) > v["dashline_deg"]:
                    continue
                perp = np.abs((c[:, 0] - c[i, 0]) * u[1] - (c[:, 1] - c[i, 1]) * u[0])
                m = c[perp <= v["dashline_tol"]]
                if len(m) < n_min:
                    continue
                m = m[np.argsort(np.hypot(m[:, 0], m[:, 1]))]
                gaps = np.hypot(*np.diff(m, axis=0).T)
                if gaps.max() > v["dashline_gap"]:
                    continue
                span = float(gaps.sum())
                score = len(m) + span
                if best is None or score > best[0]:
                    best = (score, m, span)
        if best is None:
            return None
        _, m, span = best
        return Line(pts=m.astype(float), length=span, near_y=float(m[0, 1]), width=0.05, is_dash=True)

    def _make_line(self, xs, ys, top, rows):
        """一群白色像素 → 地面上的折線（由近到遠）；太短、太遠回傳 None。"""
        max_r = self.v["max_range"]
        if True:
            gx, gy = self.to_ground(xs.astype(float), ys.astype(float) + top)
            r = np.hypot(gx, gy)
            keep = r <= max_r
            if keep.sum() < 8:
                return None
            gx, gy, r = gx[keep], gy[keep], r[keep]
            # 每個像素在地面上的面積 ≈ x³ / (h·f²)
            area = float(np.sum(gx ** 3)) / (self.h * self.f ** 2)
            # 依距離分段平均（每 4 cm 一點）：彎過來的線也排得出順序
            k = np.floor(r / 0.04).astype(int)
            uk, inv = np.unique(k, return_inverse=True)
            cnt = np.bincount(inv)
            px = np.bincount(inv, gx) / cnt
            py = np.bincount(inv, gy) / cnt
            pts = np.stack([px, py], 1)
            if len(pts) < 3:
                return None
            seg = np.hypot(*np.diff(pts, axis=0).T)
            length = float(seg.sum())
            if rows < self.v["min_rows"] and length < self.v["min_len"]:
                return None
            # 很寬很扁、又很短的（虛線、雜點）不當車道線
            if length < 0.12:
                return None
            pts = self._cut_kink(pts)
            if len(pts) < 3:
                return None
            length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
            if length < 0.12:
                return None
            return Line(pts=pts, length=length, near_y=float(pts[0, 1]), width=area / max(length, 1e-3))

    def _assign(self, s, lines):
        """挑左線 / 右線。兩條：近處相距約一個路寬的那一對。一條：在上一張中線的左邊就是左線。"""
        w = self.lane_w
        if self.post_turn:
            # 虛線左轉剛結束：橫過前方的線（斜超過 post_turn_deg）不是要跟的車道（180435：虛線塊、前方橫的實線被當車道線）
            lim = self.v["post_turn_deg"]
            lines = [l for l in lines if abs(math.degrees(math.atan2(*(l.pts[min(len(l.pts) - 1, 6)] - l.pts[0])[::-1]))) <= lim]
        # 坡上 / 坡頂：只用從車子很近處開始的線（坡頂後面的柱子、桿子投影出來起點在 0.5–0.9 m；190544 被當成左右線）
        near_max = self.v["near_max_slope"] if self.slope else self.v["near_max"]

        def near_ok(l):
            d = np.hypot(*l.pts[0])
            if d <= near_max:
                return True
            if self.slope and d <= self.v["near_max"]:
                # 線從畫面左右邊緣切出去（車頭歪著進坡時常見）：最近的一點在邊緣上，距離本來就遠，不是柱子
                #  （195215 車頭歪 24° 上坡，左線最近點在畫面邊緣 0.70 m，被 0.5 m 擋掉，坡上 2.5 s 沒線停車）
                u, _ = self.to_image(np.array([l.pts[0][0]]), np.array([l.pts[0][1]]))
                return bool(np.isfinite(u[0]) and (u[0] <= 4 or u[0] >= self.W - 5))
            return False
        if self.slope:
            lines = [l for l in lines if near_ok(l)]
        near = [l for l in lines if near_ok(l)]   # 要從車子附近開始的線
        best = None
        for i, a in enumerate(near):
            for b in near[i + 1:]:
                # 比較兩條線最近處附近（同一個 x）的 y
                x0 = max(a.pts[0, 0], b.pts[0, 0])
                ya, yb = np.interp(x0, a.pts[:, 0], a.pts[:, 1]), np.interp(x0, b.pts[:, 0], b.pts[:, 1])
                gap = abs(ya - yb)
                # 兩條線要夾著車子（左線在左、右線在右）：不然會和隔壁那段賽道的線配成一對（140849 25 s）
                xn = min(a.pts[0, 0], b.pts[0, 0])        # 在比較近的那條線的最近處看（140849 25 s 在遠處看會誤判夾著）
                yna, ynb = np.interp(xn, a.pts[:, 0], a.pts[:, 1]), np.interp(xn, b.pts[:, 0], b.pts[:, 1])
                hi, lo = max(yna, ynb), min(yna, ynb)
                straddle = hi > -self.v["side_band"] and lo < self.v["side_band"]
                # 和追蹤一致：剛剛還是左線的那條不能變成右線（194747 車子壓在外側線上，和隔壁賽道的線配成一對，逆向）
                L_, R_ = (a, b) if yna > ynb else (b, a)
                # 只有這一對的中線離車子很遠（車子不在這個車道裡）時才用追蹤擋；
                # 中線在車子附近的一對一定可信（195124 追蹤一直錯、兩條線都看到也配不起來）
                if abs((yna + ynb) / 2) > 0.35 * w and self._conflict(L_, R_):
                    continue
                # 兩條都要夠長：虛線塊（每塊 0.15–0.25 m）不能配成一對
                long_ok = min(a.length, b.length) >= self.v["pair_min_len"]
                # 路寬下限：平地 pair_min_rel × w（坡上透視會量窄，維持 0.6）
                #  11-22 221554 三次坡底都把左邊別段賽道的弧線和車道左線配成一對，路寬只有 0.31–0.35 m（真的 ≈ 0.50），
                #  左線被當右線、單線追蹤沿用，偏移 +38 cm，前輪只打 +9°，第 3 次衝出右彎外側
                #  下坡 / 坡底不放寬（11-23 110643 兩次坡底：左邊別段弧線 y +0.23 和車道左線 y −0.05 配成一對，
                #  路寬 0.30–0.33 m，第 1442–1451、3134–3143 張；左線被當右線，前輪只打 +3 ~ +5°。
                #  下坡正確的一對量到 ≥ 0.5 m：221554 下坡 91 張 / 坡底 171 張沒有一張 < 0.375 m；量窄只發生在上坡）
                up_slope = self.slope and not (self.slope_down and self.v.get("pair_down_strict", False))
                lo_rel = 0.6 if up_slope else self.v.get("pair_min_rel", 0.6)
                if lo_rel * w <= gap <= 1.5 * w and abs(a.pts[0, 0] - b.pts[0, 0]) < 0.5 and straddle and long_ok:
                    # 長的、路寬對的、中線離車子近的（車子本來就在車道裡）
                    score = a.length + b.length - 2 * abs(gap - w) - 1.5 * abs((ya + yb) / 2)
                    if best is None or score > best[0]:
                        best = (score, a, b) if ya > yb else (score, b, a)
        if best:
            s.left, s.right, s.conf = best[1], best[2], 2
            x0 = max(s.left.pts[0, 0], s.right.pts[0, 0])
            gap = float(np.interp(x0, s.left.pts[:, 0], s.left.pts[:, 1]) - np.interp(x0, s.right.pts[:, 0], s.right.pts[:, 1]))
            s.lane_w = gap
            self.lane_w = 0.95 * self.lane_w + 0.05 * min(1.3 * w, max(0.8 * w, gap))
        elif near or lines:
            # 一條線：挑最近開始的那條（夠長的話）。哪一邊看它最近那一點在車子的左邊還右邊；
            # 太靠中間（±side_band）才用上一張的左右線位置判斷。
            # （彎道裡只看到外側線時，舊程式用中線位置判斷常常判反，中線算到賽道外，車子原地繞圈）
            # 附近沒有夠長的線：用稍遠一點、夠長的線（不用附近的短虛線塊；180435 26.55 s 選了 0.13 m 的虛線塊）
            far_long = [l for l in lines if l.length >= 0.5 and np.hypot(*l.pts[0]) <= self.v["near_max"] + 1.0]
            pool = [l for l in near if l.length >= 0.3] or far_long or near or lines
            ln = min(pool, key=lambda l: np.hypot(*l.pts[0]))
            band = self.v["side_band"]
            # 先看追蹤：剛剛（track_n 張內）看過的左 / 右線，位置連續（差 track_dy 以內）就還是那一條。
            # 194321 右彎跑寬、車子跑出左線外，左線跑到車子右邊，只看「在車子哪一邊」會判成右線、前輪回正衝出去
            tr = [(abs(ln.near_y - y), side) for side, (y, n) in (("l", self.trk_l), ("r", self.trk_r))
                  if y is not None and self.frame_n - n <= self.v["track_n"]]
            tr = [x for x in tr if x[0] <= self.v["track_dy"]]
            # 線就在車子正下方（騎在線上）：看線往哪邊彎出去。往右彎出去 = 車在它左邊外側 → 它是左線；往左 → 右線。
            # 202605 右彎車騎在左線上，追蹤判成右線、一直往左修，整條路線嚴重偏左
            j = min(len(ln.pts) - 1, max(1, int(np.searchsorted(np.hypot(*(ln.pts - ln.pts[0]).T), 0.4))))
            ang = math.degrees(math.atan2(ln.pts[j, 1] - ln.pts[0, 1], ln.pts[j, 0] - ln.pts[0, 0]))
            if abs(ln.near_y) < self.v["under_band"] and abs(ang) >= self.v["under_deg"]:
                left = ang < 0
            elif tr and abs(ln.near_y) < 0.6 * self.lane_w:
                if len(tr) == 2 and abs(tr[0][0] - tr[1][0]) < 0.05 and self.last_side is not None:
                    # 左右線的追蹤位置幾乎重疊（210415：一張誤判成右線後，兩邊紀錄只差 1 cm，之後一直挑錯邊）：沿用上一張
                    left = self.last_side == "l"
                else:
                    left = min(tr)[1] == "l"
            elif ln.near_y > band:
                left = True
            elif ln.near_y < -band:
                left = False
            elif self.prev_l is not None and self.prev_r is not None:
                left = abs(ln.near_y - self.prev_l) < abs(ln.near_y - self.prev_r)
            else:
                left = ln.near_y > 0
            if left:
                s.left = ln
            else:
                s.right = ln
            self.last_side = "l" if left else "r"
            s.conf = 1
        if s.left is not None:
            self.trk_l = (s.left.near_y, self.frame_n)
        if s.right is not None:
            self.trk_r = (s.right.near_y, self.frame_n)
        self.prev_l = s.left.near_y if s.left is not None else (self.prev_l if s.right is None else s.right.near_y + self.lane_w)
        self.prev_r = s.right.near_y if s.right is not None else (self.prev_r if s.left is None else s.left.near_y - self.lane_w)

    def _is_v(self, ys, xs, rows):
        """Y 字岔路中間的 V 字分隔線：最下面（離車最近）是尖端，往上 v_dy 列分成左右兩段、尖端在兩段中間。
        195848 17–19 s、30–31 s 的岔路是這個形狀（車子從 Y 的下面開過來）。"""
        if rows < self.v["v_min_rows"]:
            return None
        y0 = ys.max()
        if y0 >= self.roi()[1] - self.roi()[0] - 2:     # 尖端被畫面下緣切掉：看不出是不是 V
            return None
        xt = xs[ys == y0].mean()
        # 從最下面往上逐列掃：找兩臂剛分開的那一列（開口中點 = 尖端位置），再往上開口要越來越大。
        #  原本要「尖端在兩臂中間」、只看 0.35 / 0.7 兩個高度：173712 斜著看的 V 右臂幾乎垂直，
        #  最下面的點在右臂底部，一直不過，整個岔路沒認到。真假 V 的地面形狀再由 _v_ground_ok 檢查
        def split(dy):
            sel = np.sort(xs[ys == y0 - dy])
            if len(sel) < 2:
                return None
            gaps = np.flatnonzero(np.diff(sel) > 3)
            if len(gaps) != 1:
                return None
            return sel[gaps[0]], sel[gaps[0] + 1]
        lo = None
        for dy in range(2, int(0.7 * rows) + 1):
            sp = split(dy)
            if sp is not None and sp[1] - sp[0] >= self.v["v_gap"]:
                lo = (dy, sp)
                break
            if sp is None and dy > self.v["v_dy"] and len(xs[ys == y0 - dy]) == 0:
                return None
        if lo is None:
            return None
        g0 = lo[1][1] - lo[1][0]
        ok = 0
        for dy in range(lo[0] + self.v["v_dy"], int(0.95 * rows) + 1):
            sp = split(dy)
            if sp is not None and sp[1] - sp[0] >= max(2 * self.v["v_gap"], g0 + 4):
                ok += 1
        if ok < max(3, int(0.15 * rows)):
            return None
        return (lo[1][0] + lo[1][1]) / 2.0

    def _v_ground_ok(self, xs, ys, xt, top):
        """V 在地面上要像真的 Y 字分隔線（11-21 164610 岔路誤判）：
        1. 最遠不超過 v_far_max m：真的 V 最遠 1.9–2.5 m（很近時 ≤ 4.1）；誤判的一路延伸到地平線 7.27 m
           （從平台往下看到下層賽道的線、坡道木邊）
        2. 兩臂在地面上要張開 ≥ v_open_deg 度：真的 V 26–31°；誤判的 2–7°（兩條幾乎平行的線黏在一起）"""
        gx, gy = self.to_ground(xs.astype(float), ys.astype(float) + top)
        if not np.isfinite(gx).all() or gx.max() > self.v["v_far_max"]:
            return False
        rx = xs >= xt
        angs = []
        for m in (rx, ~rx):
            if m.sum() < 5 or np.ptp(gx[m]) < 0.05:
                return False
            angs.append(math.degrees(math.atan(np.polyfit(gx[m], gy[m], 1)[0])))
        return abs(angs[0] - angs[1]) >= self.v["v_open_deg"]

    def _assign_dash_left(self, s, lines):
        """虛線左轉完的平台：使用者規定走虛線的哪一側（dash_side）。
        dash_side = right：虛線在車子右邊（車走虛線左側，11-21 202028 後使用者要求）→ 虛線當右線，左線找虛線左邊約一個車道寬的實線
        dash_side = left ：虛線在車子左邊（車走虛線右側）"""
        w = self.lane_w
        reach = self.v["near_max"] + 0.5
        dl = max((l for l in lines if l.is_dash), key=lambda l: l.length)
        if self.dash_side == "right":
            # 緊貼虛線左側走：不找左線（左邊那條車道通到平台左緣，203112 / 203244 一路開出平台），
            #  中線 = 虛線往左 dash_keep m；下坡口就在虛線正前方（好的圈都沿著虛線下坡）
            s.right, s.conf = dl, 1
            s.left = None
            self.trk_r = (dl.near_y, self.frame_n)
        else:
            s.left, s.conf = dl, 1
            other = [l for l in lines if not l.is_dash and 0.6 * w <= dl.near_y - l.near_y <= 1.5 * w
                     and np.hypot(*l.pts[0]) <= reach]
            if other:
                s.right, s.conf = max(other, key=lambda l: l.length), 2
                self.trk_r = (s.right.near_y, self.frame_n)
            self.trk_l = (dl.near_y, self.frame_n)

    def _assign_fork(self, s, lines, side):
        """岔路口：走右邊 → V 的右臂當左線，右臂右邊約一個路寬的線當右線；走左邊 → 左右相反。
        這一張沒看到 V：照一般方式配（追蹤會記得 V 臂的位置）。"""
        w = self.lane_w
        reach = self.v["near_max"] + 0.5
        if side == "right" and self._v_right is not None:
            arm = self._v_right
            s.left, s.conf = arm, 1
            other = [l for l in lines if 0.6 * w <= arm.near_y - l.near_y <= 1.5 * w and np.hypot(*l.pts[0]) <= reach]
            if other:
                s.right, s.conf = max(other, key=lambda l: l.length), 2
                self.trk_r = (s.right.near_y, self.frame_n)
            self.trk_l = (arm.near_y, self.frame_n)
        elif side == "left" and self._v_left is not None:
            arm = self._v_left
            s.right, s.conf = arm, 1
            other = [l for l in lines if 0.6 * w <= l.near_y - arm.near_y <= 1.5 * w and np.hypot(*l.pts[0]) <= reach]
            if other:
                s.left, s.conf = max(other, key=lambda l: l.length), 2
                self.trk_l = (s.left.near_y, self.frame_n)
            self.trk_r = (arm.near_y, self.frame_n)
        else:
            self._assign(s, lines)
            # 還沒看到 V、但已經知道要走哪邊（地圖）：路面在岔路前變寬時，只貼著要走那一邊的外側線（離它半個路寬），
            #  不要走變寬路面的中線（中線正對 V 尖端，最後落在哪一側只差幾公分）。
            #  11-22 195822 30.4 m 岔路（走左）：第 2 圈到 V 時尖端在車子右邊 19–20 cm，順順走左；
            #  第 3 圈尖端在車子左邊 15 cm，照地圖硬往左切過 V 尖端，前輪 −44°、後輪 12/98，甩出去
            if s.conf == 2 and s.left is not None and s.right is not None:
                gap = s.left.near_y - s.right.near_y
                if gap > self.v.get("fork_widen", 1.15) * w:
                    if side == "left":
                        s.right, s.conf = None, 1
                    else:
                        s.left, s.conf = None, 1

    def _cut_kink(self, pts):
        """線在某一點突然轉彎（岔路：一邊的線轉進分支）就只留近的那一段。
        195124 岔路口左線往左轉 60°，中線跟著往左拉，車子沒有直走。一般彎道 0.2 m 內轉不到 kink_deg。"""
        w = self.v["kink_win"]
        d = np.hypot(*np.diff(pts, axis=0).T)
        s = np.r_[0.0, np.cumsum(d)]
        for i in range(1, len(pts) - 1):
            a = np.searchsorted(s, s[i] - w)
            b = np.searchsorted(s, s[i] + w)
            if a >= i or b >= len(pts) or b <= i:
                continue
            v1, v2 = pts[i] - pts[a], pts[b] - pts[i]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            if c < np.cos(np.radians(self.v["kink_deg"])):
                return pts[:i + 1]
        return pts

    def reset_track(self):
        """出發時清掉左右線追蹤（待命時車子被拿著走，追蹤到的東西不能用）。"""
        self.trk_l = self.trk_r = (None, 0)
        self.prev_l = self.prev_r = None

    def _conflict(self, L, R):
        """這一對和最近追蹤到的左 / 右線對不上（左線跑去當右線、或右線跑去當左線）。"""
        dy, nmax = self.v["track_dy"], self.v["track_n"]
        yl, nl = self.trk_l
        yr, nr = self.trk_r
        if yl is not None and self.frame_n - nl <= nmax:
            if abs(R.near_y - yl) <= dy and abs(L.near_y - yl) > dy:
                return True
        if yr is not None and self.frame_n - nr <= nmax:
            if abs(L.near_y - yr) <= dy and abs(R.near_y - yr) > dy:
                return True
        return False

    def _shift(self, pts, side):
        """把線往路中間平移半個路寬（沿法線）。side = +1：左線往右，−1：右線往左。"""
        d = np.gradient(pts, axis=0)
        n = np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6
        t = d / n
        right = np.stack([t[:, 1], -t[:, 0]], 1)          # 前進方向的右手邊
        return pts + side * right * (self.lane_w / 2)

    def _center(self, s):
        parts = []
        if s.left is not None:
            parts.append(self._shift(s.left.pts, +1))
        if s.right is not None:
            if self.dash_left and self.dash_side == "right" and getattr(s.right, "is_dash", False):
                keep = self.v.get("dash_keep", 0.15)
                d = np.gradient(s.right.pts, axis=0)
                n = np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6
                tt = d / n
                parts = [s.right.pts - np.stack([tt[:, 1], -tt[:, 0]], 1) * keep]
            else:
                parts.append(self._shift(s.right.pts, -1))
        if not parts:
            return
        c = self._merge(parts)
        if c is None:
            return
        s.center = c
        s.offset = float(c[0, 1])
        j = min(len(c) - 1, max(1, int(np.searchsorted(np.hypot(*(c - c[0]).T), 0.3))))
        s.heading = float(math.degrees(math.atan2(c[j, 1] - c[0, 1], c[j, 0] - c[0, 0])))
        s.vp_row = self._vp_row(s)
        s.far = float(np.hypot(c[-1, 0], c[-1, 1]))
        self.prev_off = s.offset

    def _vp_row(self, s, near=0.9):
        """左右線近處在畫面上延長後的交點列；只有一條線回傳 None。"""
        if s.left is None or s.right is None:
            return None
        fits = []
        for l in (s.left, s.right):
            q = l.pts[np.hypot(*l.pts.T) <= near]
            if len(q) < 3:
                q = l.pts[:4]
            u, v = self.to_image(q[:, 0], q[:, 1])
            if np.ptp(v) < 3:
                return None
            fits.append(np.polyfit(v, u, 1))
        (a1, b1), (a2, b2) = fits
        if abs(a1 - a2) < 1e-3:
            return None
        return float((b2 - b1) / (a1 - a2))

    @staticmethod
    def _resample(p, step=0.05):
        """沿線每 step m 取一點（由近到遠）。"""
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        if d[-1] < step:
            return p[:1]
        u = np.arange(0.0, d[-1] + 1e-6, step)
        return np.stack([np.interp(u, d, p[:, 0]), np.interp(u, d, p[:, 1])], 1)

    def _merge(self, parts):
        """兩條線平移後的中線合成一條：重疊的地方平均；只剩一條的那段接著延伸、對齊重疊段尾端（不跳）。
        原本把兩條線的點混在一起依距離分組，兩條不完全平行時中線鋸齒狀（195124 岔路）。"""
        parts = [self._resample(q[q[:, 0] > 0.05]) for q in parts if (q[:, 0] > 0.05).sum() >= 2]
        parts = [q for q in parts if len(q) >= 2]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        a, b = sorted(parts, key=len)                    # a 短、b 長
        n = len(a)
        # 以「離車子的距離」對齊：兩條各自從最近點起算，同一個 index 大約是同一個距離
        m = (a + b[:n]) / 2
        if len(b) > n:
            ext = b[n:] + (m[-1] - b[n - 1])             # 長的那條接著走，平移到和平均的尾端接上
            m = np.concatenate([m, ext])
        return m

    # ------------------------------------------------------------------ 虛線
    def _dash(self, frame):
        """前方橫越的虛線（路口）：至少 dash_min 塊虛線方塊排成一直線，而且這條線在地面上是「橫的」
        （和車頭方向夾角 ≥ dash_cross_deg），回傳塊數（不夠回傳 0~1）。
        原本要「同一列、左右分開」：鏡頭往下壓、車子斜著接近時（201231 28 s）虛線在畫面上是斜的，一直只算到 1 塊。
        沿著車道的虛線（左轉後那段直線）在地面上是直的，不算。"""
        H, W = frame.shape[:2]
        vh = self.vh * H / self.H
        t0 = int(max(0, vh + self.v["dash_top"]))
        t1 = int(min(self.v["roi_bottom"] * H, vh + self.v["dash_bottom"]))
        if t1 - t0 < 40:
            return 0
        roi = frame[t0:t1]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, lab, st, cen = cv2.connectedComponentsWithStats((g > self.v["dash_thr"]).astype(np.uint8), 8)
        hgt = t1 - t0
        idx = [i for i in range(1, len(st)) if 12 <= st[i, 2] <= 240 and 5 <= st[i, 3] <= 90
               and st[i, 4] >= 0.35 * st[i, 2] * st[i, 3] and st[i, 4] >= 40
               and st[i, 1] > 2 and st[i, 1] + st[i, 3] < hgt - 2 and st[i, 0] > 2 and st[i, 0] + st[i, 2] < W - 2]
        if len(idx) < 2:
            return len(idx)
        sx, sy = self.W / W, self.H / H
        sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1]
        # 每一塊都要像「黑色地墊上的白色膠帶小方塊」，不然不算（11-21 155804：電池標籤、紙張被當成虛線）
        #   地面寬 ≤ dash_max_m、沿車頭方向長 ≤ dash_max_len（真的虛線 ≤ 0.12 / 0.31 m；誤判的 0.2 / 0.33–0.55）
        #   顏色飽和度 ≤ dash_sat_max（真的 5–13；誤判的多半 15–32）
        #   周圍一圈的亮度中位數 ≤ dash_ring_max（真的 53–118 = 黑地墊；誤判的 117–149）
        pad, keep = 6, []
        for i in idx:
            x0, y0, w_, h_ = st[i, :4]
            cx, cy = self.to_ground(np.array([x0, x0 + w_]) * sx, (np.array([y0 + h_, y0 + h_]) + t0) * sy)
            fx, _ = self.to_ground(np.array([x0 + w_ / 2] * 2) * sx, (np.array([y0, y0 + h_]) + t0) * sy)
            if abs(cy[0] - cy[1]) > self.v["dash_max_m"] or abs(fx[0] - fx[1]) > self.v["dash_max_len"]:
                continue
            m = lab[y0:y0 + h_, x0:x0 + w_] == i
            if sat[y0:y0 + h_, x0:x0 + w_][m].mean() > self.v["dash_sat_max"]:
                continue
            ya, yb, xa, xb = max(0, y0 - pad), min(hgt, y0 + h_ + pad), max(0, x0 - pad), min(W, x0 + w_ + pad)
            ring = g[ya:yb, xa:xb][lab[ya:yb, xa:xb] != i]
            if ring.size and np.median(ring) > self.v["dash_ring_max"]:
                continue
            keep.append(i)
        if len(keep) < 2:
            return len(keep)
        u = np.array([cen[i][0] for i in keep]) * sx
        v = (np.array([cen[i][1] for i in keep]) + t0) * sy
        gx, gy = self.to_ground(u, v)
        n = len(gx)
        if n < 2:
            return n
        best, self.dash_x = 1, None
        tol, min_ang = self.v["dash_tol"], np.radians(self.v["dash_cross_deg"])
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = gx[j] - gx[i], gy[j] - gy[i]
                L = np.hypot(dx, dy)
                if L < 0.08 or L > 1.2:
                    continue
                if abs(np.arctan2(abs(dx), abs(dy))) > np.pi / 2 - min_ang:   # 線太「直」（沿著車道）
                    continue
                d = np.abs((gx - gx[i]) * dy - (gy - gy[i]) * dx) / L
                k = int((d <= tol).sum())
                if k > best:
                    sel = d <= tol
                    if not self._dash_gaps_dark(g, t0, u[sel] / sx, v[sel] / sy):
                        continue
                    best = k
                    if os.environ.get("DASH_DEBUG_MEM"):
                        self._dbg_mem = list(zip((u[sel] / sx).tolist(), (v[sel] / sy).tolist()))
                    # 這條虛線在車子正前方（y = 0）離鏡頭多遠
                    self.dash_x = float(gx[i] - gy[i] * dx / dy) if abs(dy) > 1e-6 else float(gx[i])
        return best

    def _dash_gaps_dark(self, g, t0, us, vs):
        """虛線塊和塊之間要是黑地墊：沿著相鄰兩塊中心連線取樣（扣掉亮的塊本身），中位數不能比整個範圍亮太多。
        11-19 181909：彎道實線上的幾個反光點排成斜線被當成虛線，點和點之間是灰白的實線（中位數 134–141，
        整個範圍 37）；真的虛線之間是黑地墊（55–94，整個範圍 51–57）。"""
        o = np.argsort(us)
        us, vs = np.asarray(us)[o], np.asarray(vs)[o]
        base = float(np.median(g))
        thr = self.v["dash_thr"]
        for u0, v0, u1, v1 in zip(us[:-1], vs[:-1], us[1:], vs[1:]):
            n = max(2, int(np.hypot(u1 - u0, v1 - v0)))
            uu = np.clip(np.linspace(u0, u1, n).astype(int), 0, g.shape[1] - 1)
            vv = np.clip(np.linspace(v0, v1, n).astype(int) - t0, 0, g.shape[0] - 1)
            val = g[vv, uu]
            val = val[val <= thr]
            if len(val) and np.median(val) - base > self.v["dash_gap_rel"]:
                return False
        return True

    # ------------------------------------------------------------------ 疊圖
    def draw(self, frame, s, target=None, text=""):
        vis = cv2.resize(frame, (self.W * 2, self.H * 2))
        k = 2.0
        top, bot = s.dbg.get("top", 0), s.dbg.get("bot", self.H)
        cv2.line(vis, (0, int(top * k)), (vis.shape[1], int(top * k)), (0, 200, 255), 1)
        cv2.line(vis, (0, int(bot * k)), (vis.shape[1], int(bot * k)), (0, 200, 255), 1)

        def poly(pts, col, th=2):
            if pts is None or len(pts) < 2:
                return
            u, v = self.to_image(pts[:, 0], pts[:, 1])
            p = np.stack([u * k, v * k], 1)
            p = p[np.isfinite(p).all(1)].astype(np.int32)
            cv2.polylines(vis, [p], False, col, th)
        for c in s.dbg.get("cands", []):
            poly(c.pts, (90, 90, 90), 1)
        poly(s.left.pts if s.left else None, (80, 230, 80))
        poly(s.right.pts if s.right else None, (230, 80, 230))
        poly(s.center, (0, 165, 255))
        if target is not None:
            u, v = self.to_image(target[0], target[1])
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(vis, (int(u * k), int(v * k)), 6, (0, 0, 255), -1)
        if text:
            cv2.putText(vis, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        return vis
