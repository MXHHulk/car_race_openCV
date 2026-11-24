"""控制：Pure Pursuit 算前輪轉角，速度規劃算後輪速度。"""
import math
from collections import deque

from corner import CornerPlanner
from vision import target_point


class Controller:
    def __init__(self, cfg):
        self.c, self.g = cfg["control"], cfg["geometry"]
        self.v = 0.0          # 目前速度命令 %（有加速度限制）
        self.fwa = 0.0
        self.target = None
        self.t_up = self.t_down = -1e9   # 最近一次看到上坡 / 下坡的時間
        self.integ = 0.0                  # 偏移的累積（m·s），彎道持續偏同一邊時用來多轉一點
        self.hold_i = False               # True = 積分暫停累積（車子卡住不動時，main.py 設定）
        self.t_steer = None
        self.bias, self.t_bias = 0.0, None   # 右彎往內側（右）靠的量（m，負 = 右）
        self.left_free = False               # True = 這段不限制左修（下坡到底要左轉，main.py 設定）
        self.no_bias = False                 # True = 不往內側靠（爬坡模式：坡道是直的，main.py 設定）
        self.fwa_lp = 0.0                    # 速度規劃用的 |前輪角度|（平滑過）
        self.fwa_cv = 0.0                    # 彎道裡的 |前輪角度|（平滑較久，代表整個彎的彎度）
        self._pt, self._t_curve = 0.0, None  # 速度規劃的時間、這次進彎的時間（None = 在直線上）
        self.straight = 0.0                  # 1 = 直線、0 = 彎道（_profile_speed 算）
        self.y_extra = 0.0                   # 目標點橫向多偏多少（m，左正；main.py 設定，隧道段往左靠）
        self.ld_extra = 0.0                  # 前視距離多加多少 m（main.py 設定：下坡前看遠一點，比較不晃）
        self.yaw_rate = None                 # IMU 車頭轉速（°/s，左正），main.py 設定；None = 沒有 IMU
        self._pv = deque(maxlen=5)           # 遠處預看的前輪角（最近幾張），速度規劃提早減速用
        self.preview = 0.0                   # 平滑後的預看前輪角（度）
        self.corner = CornerPlanner(cfg)

    @property
    def corner_active(self):
        return self.corner.active

    def steer(self, res, speed_pct, t=None):
        """t：目前時間（s）。有給才會處理直角彎（corner.py）；待命時不給。"""
        c = self.c
        forced, y_off = self.corner.update(res, t)
        if forced is not None:                    # 直角彎轉彎中：固定前輪角度走圓弧
            self.target = None
            self.fwa = max(-c["fwa_limit"], min(c["fwa_limit"], forced))
            return self.fwa
        ld = c["lookahead"] + c["lookahead_per_speed"] * speed_pct / 100.0 + self.ld_extra
        # 直線跑快時前視太近，車子會左右蛇行：週期 2 s、車頭 ±8°、前輪 ±15°（run_20261101_165649 48–54 s）。
        # 直線看遠一點；彎道不加，不然會切到內側
        ld += (c.get("speed_profile") or {}).get("straight_ld", 0.0) * self.straight
        self.target = target_point(res, ld)
        if self.target is None:
            return None
        if y_off:                                 # 直角彎入彎前：目標往外側偏
            self.target = (self.target[0], self.target[1] + y_off)
        elif not self.corner_active:
            self.target = (self.target[0], self.target[1] + self._inner_bias(res, t) + self.y_extra)
        x, y = self.target
        alpha = math.atan2(y, x)
        dist = max(math.hypot(x, y), 1e-3)
        L = self.g["wheelbase"]
        delta = math.atan(2 * L * math.sin(alpha) / dist)  # 正 = 左轉
        # 預看：同樣用 Pure Pursuit 算「開到遠處 preview_ld 那一點要打多少前輪」，代表前方的彎有多急。
        # 現在的速度只看已經打了多少前輪，前輪打大才減速就太晚了（run_20261104_203057、203658 都是
        # 前輪打到底才發現轉不過去）。圓弧上 Pure Pursuit 的角度和距離無關（≈ atan(L × 曲率)），可以直接比
        sp = c.get("speed_profile") or {}
        pv = sp.get("preview_ld", 0.0)
        if pv:
            tp = target_point(res, pv)
            if tp is not None and math.hypot(*tp) >= 0.8 * pv:
                d2 = max(math.hypot(*tp), 1e-3)
                self._pv.append(abs(math.degrees(math.atan(2 * L * math.sin(math.atan2(tp[1], tp[0])) / d2))))
        # 預看也看畫面的曲率：遠處的點少，二次擬合把剛出現的彎平滑掉，上面那個角度沒有比前輪早；
        # 曲率卻比前輪早 0.3–0.5 s 變大（203057 40.8 s 前輪 +1° 時曲率已經 −0.7 ~ −1.1）。
        # 換成「這個彎要打幾度」：前輪 ≈ atan(preview_k × 曲率)，彎道裡曲率 0.77 時前輪約 34° → 0.87
        pk = sp.get("preview_k", 0.0)
        if pk and res.center is not None:
            self._pv.append(abs(math.degrees(math.atan(pk * abs(res.curvature)))))
        # 曲率前饋：鏡頭近處有盲區，目標點常落在可見範圍的最前緣，彎道還沒進到計算裡。
        # 直接把目標點處中線的曲率換成轉角加上去，看到彎就開始轉。k_ff = 0 等於純 Pure Pursuit。
        # 注意：2026-10-28 模擬中，入彎時曲率估計會跳號，開 0.5 以上反而打錯方向衝出去，預設關閉。
        k_ff = c.get("k_ff", 0.0)
        if k_ff and res.center is not None:
            a, b, _ = res.center
            s = 2 * a * x + b
            kappa = 2 * a / (1 + s * s) ** 1.5
            delta += k_ff * math.atan(L * kappa)
        # 爬坡模式不加置中修正（climb_trim）：早上 fwa_trim 0 時坡道 5/6 次都過；補 6° 後出彎接坡車頭一直往右轉，
        # 134322、134620 坡上偏右 15–24 cm 撞右側木板 / 掉下坡道
        trim = c.get("climb_trim", c["fwa_trim"]) if self.no_bias else c["fwa_trim"]
        fwa = -math.degrees(delta) + trim                                   # 協議：負左正右
        fwa += self._integral(res, t)
        # 直線蛇行：車頭以 2.4 s 週期左右擺 ±6°、前輪 ±10°（run_20261103_192609 10–15 s）。
        # 前輪和舵機有延遲，偏移修回中間時車頭已經轉過頭。用陀螺儀的車頭轉速當阻尼：
        # 車頭正在往左轉 → 前輪先往右回一點（fwa 正）。只在直線用，彎道裡轉速是該有的
        yd = (c.get("speed_profile") or {}).get("yaw_damp", 0.0)
        if yd and self.yaw_rate is not None:
            fwa += yd * self.yaw_rate * self.straight
        # 右彎中不要往左大修：進坡前右線（內側）跑到正前方時前輪打左 −32°，車頭修過頭，
        # 之後打右 +44° 也救不回來，貼上左邊線撞牆（run_20261030_174120 61.6–63 s）。寧可稍微壓到內側線
        ib = c.get("inner_bias") or {}
        # 明顯是左彎時不限制：靠右量退得慢，速度一快就轉不過右接左的 S 彎（模擬 s_rl）
        if (ib.get("enable", False) and not self.left_free and self.bias <= -ib.get("left_from", 0.02)
                and res.curvature < ib.get("left_curv", 0.5)):
            fwa = max(fwa, -ib.get("left_max", 15))
        # 轉向放大（steer_gain）：舵機 / 輪胎變弱時，實際轉的只有指令的一部分。
        # steer_test 2026-11-05：±15–25° 只有狀況好時的 0.68–0.79（左右平均），放大指令補回來；置中修正不放大。
        # 打到底的角度補不回來（機構極限），只對小、中角度有用
        # 只放大超過 steer_from 度的部分：直線上的小修正不變（整體放大或「依直線程度放大」模擬直線擺動 10 → 26–28 cm，
        # 角度一大就被放大、越修越大），彎道裡的大角度才補回來
        g, d0 = c.get("steer_gain", 1.0), c.get("steer_from", 12.0)
        # 整體轉向倍率（steer_scale）：換新電池後舵機有力、轉得到位，同樣指令車頭轉得比舊電池多約 1.4 倍
        # （前輪打到底車頭轉速 70 vs 50°/s），出彎轉過頭、直線左右甩越來越大（172251、172455，172455 速度 ×0.8 也一樣）。
        # 置中修正不縮
        # 左右分開（steer_scale_r / _l）：滿電電池 steer_test 2026-11-06 17:3x 右轉比例 1.96–2.65（±25/35° 車頭轉速是左轉的 2 倍），
        # 左轉 0.74–1.15 正常
        ss = c.get("steer_scale", 1.0)
        ss = c.get("steer_scale_r", ss) if fwa > trim else c.get("steer_scale_l", ss)
        if ss != 1.0:
            fwa = trim + ss * (fwa - trim)
        e = fwa - trim
        if g != 1.0 and abs(e) > d0:
            fwa = trim + math.copysign(d0 + g * (abs(e) - d0), e)
        self.fwa = max(-c["fwa_limit"], min(c["fwa_limit"], fwa))
        return self.fwa

    def _inner_bias(self, res, t):
        """右彎時目標往右（內側）偏，最多 inner_max m（control.inner_bias）。
        彎道上坡前是一段長右彎，車子一直走在左邊（外側）10–20 cm，進坡時前輪打到底也轉不進去，
        第二圈貼著左邊撞牆（run_20261029_201418 133–140 s）。
        右彎一出現就往右靠（快）、曲率回到 0 時慢慢回中（慢）：進坡前曲率會短暫跳成左彎，不要因此馬上回中。"""
        c = self.c.get("inner_bias") or {}
        if not c.get("enable", False) or t is None:
            return 0.0
        # 坡道上的曲率是雜訊，一直靠右 3–5 cm（坡道上換算的距離只有實際一半），前輪長期 +10°，
        # 快到坡頂時頂上右側（run_20261030_212836、213126、214128、214333）
        if self.no_bias:
            self.bias, self.t_bias = 0.0, t
            return 0.0
        want = -min(c.get("inner_max", 0.06), c.get("k", 0.06) * max(0.0, -res.curvature))
        # 直線上不靠：曲率雜訊會一直觸發，隧道前直線還靠右 2–4 cm，加上蛇行就壓到右線（run_20261103_202846–204141）
        if self.straight >= 0.8:
            want = 0.0
        dt = 0.0 if self.t_bias is None else min(0.2, max(0.0, t - self.t_bias))
        self.t_bias = t
        tau = c.get("attack_s", 0.3) if want < self.bias else c.get("release_s", 1.5)
        self.bias += (want - self.bias) * min(1.0, dt / tau)
        return self.bias

    def _integral(self, res, t):
        """偏移的積分修正（I 控制）。前輪會打滑，實際轉的比指令少，Pure Pursuit 只看「現在偏多少」，
        長彎道裡車子會一直穩定偏外側（2026-10-28 實車：每段長右彎都偏外側 10–13 cm）。
        持續偏同一邊就把偏移累積起來、逐漸多轉一點；慢慢衰減（leak），避免出彎後還一直轉。"""
        ki = self.c.get("k_i", 0.0)
        if not ki or t is None:
            return 0.0
        dt = 0.0 if self.t_steer is None else min(0.2, max(0.0, t - self.t_steer))
        self.t_steer = t
        if not self.hold_i:
            # 卡住時不累積：車子沒在動，偏移不會變，積分只會一直往同一邊加。
            # run_20261029_180918 46.8–50.4 s 卡在坡道上，前輪從 −13° 慢慢加到 −25°，加速一推動就往左衝出坡頂
            # 出彎後積分還留著（彎道裡累積的），直線上照 i_leak_s 3 s 慢慢退會一直往同一邊推、出彎甩過頭；
            # 直線上改用 i_leak_straight_s 快點退掉（沒設就和原本一樣）
            tau = self.c.get("i_leak_s", 3.0)
            ts_ = self.c.get("i_leak_straight_s")
            if ts_:
                tau = tau + (ts_ - tau) * self.straight
            leak = math.exp(-dt / tau)
            # 積分的是「離目標多遠」：有往內側靠（bias）時以靠過去的位置為準，不然積分會把車拉回中線
            # 直線上不累積（i_curve_only）：全速直線時積分讓車子左右擺（模擬 100%：k_i 60 擺 9.8 cm、0 → 6.2 cm）。
            # 積分是給長彎道偏外側用的，直線偏移交給 Pure Pursuit 和 fwa_trim
            w = (1.0 - self.straight) if self.c.get("i_curve_only", False) else 1.0
            self.integ = self.integ * leak + (res.offset + self.bias + self.y_extra) * dt * w
        lim = self.c.get("i_max", 15.0)
        return max(-lim, min(lim, -ki * self.integ))   # offset 正 = 車子偏右 → 要往左（fwa 負）

    def speed(self, res, dt, t):
        """t：目前時間（s），坡道判斷要用。

        鏡頭只看得到「相對車身」的坡度變化：爬坡途中看到坡頂平台會判成下坡、下坡途中看到坡底會判成上坡。
        沒有 IMU 時用時間判斷：剛看過上坡（slope_memory 秒內）又看到下坡 → 是坡頂，不減速；
        剛看過下坡又看到上坡 → 是坡底，不加速。
        """
        c = self.c
        p = c.get("speed_profile") or {}
        if p.get("enable", False):
            return self.ramp(self._profile_speed(p, dt), dt)
        v = c["v_max"] * (1 - c["k_curv"] * abs(res.curvature))
        v *= 0.5 + 0.5 * res.confidence
        mem = c.get("slope_memory", 3.0)
        if res.slope == "uphill":
            if t - self.t_down > mem:
                v += c["uphill_boost"]
            self.t_up = t
        elif res.slope == "downhill":
            if t - self.t_up > mem:
                v *= c["downhill_factor"]
            self.t_down = t
        v = max(c["v_min"], min(c["v_max"] + c["uphill_boost"], v))
        return self.ramp(v, dt)

    def _profile_speed(self, p, dt):
        """依前輪角度決定速度：直線快、彎道慢（control.speed_profile）。
        舊算法看曲率和信心度，直線上一樣常掉到 24–30%：只看到一條線 ×0.75、曲率雜訊、下坡誤判 ×0.6
        （run_20261031_141947 平均只有 32%）。前輪角度是車子真正在轉的量，雜訊小得多。
        前輪打大時馬上減速、回正後慢慢加速。"""
        a = abs(self.fwa)
        tau = p.get("attack_s", 0.1) if a > self.fwa_lp else p.get("release_s", 0.6)
        self.fwa_lp += (a - self.fwa_lp) * min(1.0, dt / tau)
        lo, hi = p.get("fwa_low", 8.0), p.get("fwa_high", 24.0)
        s = self.straight = max(0.0, min(1.0, (hi - self.fwa_lp) / (hi - lo)))
        vs, vc = p.get("v_straight", 40.0), p.get("v_curve", 30.0)
        slow = getattr(self, "curve_slow", False)
        if slow:
            vc = p.get("v_curve_slow", vc)          # 隧道前後的彎道維持穩定版 1 的速度（main.py 設定 curve_slow）
        load = p.get("lat_load", 0)
        if not load:
            # 舊算法也要照 v_sat：前輪快打到底時降到 v_sat。沒這行時進坡前右彎（打到底）是 v_curve 38%，
            # 早上 lat_load 版是 v_sat 34%：134752、134907 進坡卡左護條（車速 38），早上 34% 的五趟都過
            v = vc + (vs - vc) * s
            if self.fwa_lp >= p.get("sat_deg", 99):
                v = min(v, p.get("v_sat_slow", p.get("v_sat", vc)) if slow else p.get("v_sat", vc))
            return v
        # 依彎道緩急決定速度：車頭轉速 ≈ yaw_gain × 速度% × tan(前輪)（run_20261103_200840、210400 擬合，
        # 相關 0.9，前輪 18–45° 實際/預測都在 1 左右）。速度 × 轉速（和側向力成正比）固定在 lat_load：
        # 速度 = √(lat_load / (yaw_gain × tan 前輪))，彎越緩越快。原本前輪超過 24° 一律 v_curve，
        # 轉速 23°/s 的緩彎和 46°/s 的急彎都是 37%。急彎（約 34° 以上）仍是 v_curve
        # 彎道裡前輪會來回修（195914 大彎 12 ↔ 41°），照 fwa_lp 算速度就 40 ↔ 60% 一直上下：
        # 改用平滑較久的 fwa_cv 代表整個彎的彎度，彎中速度均勻；安全靠下面的實際轉速限速
        self.fwa_cv += (a - self.fwa_cv) * min(1.0, dt / p.get("curve_tau", 0.5))
        self._pt += dt
        if self.fwa_lp > lo and self._t_curve is None:
            self._t_curve = self._pt
        elif self.fwa_lp <= lo:
            self._t_curve = None
        tan_d = math.tan(math.radians(max(self.fwa_cv, 1.0)))
        v = min(math.sqrt(load / (p.get("yaw_gain", 1.5) * tan_d)), p.get("v_cmax", vs))
        # 前輪 fwa_low 以下是直線（只是左右小修），全速；往上 blend_deg 度內慢慢過渡到彎道速度，不要一下掉很多
        w = max(0.0, min(1.0, (self.fwa_lp - lo) / p.get("blend_deg", 4.0)))
        v = vs + (min(v, vs) - vs) * w
        # 進彎（剛從直線打出去 entry_s 秒內）照舊算法早點減速：前輪是慢慢打上去的（+5 → +23° 約 0.5 s），
        # 用側向負載算會在 12–23° 還維持 50–55%，接急彎衝太快（模擬 mixed）。
        # 只看剛進彎：彎中每次往大修都當成進彎，速度就掉回 v_curve（195914 彎裡 40 ↔ 60%）
        if self._t_curve is not None and self._pt - self._t_curve < p.get("entry_s", 0.6):
            # 進彎時前輪到 entry_high 就降到 v_curve（比 fwa_high 早）：直線 75% 衝進右彎，
            # 前輪 +21° 時還有 56%，開到外側線上（run_20261104_203057 41.8 s）
            eh = p.get("entry_high", hi)
            se = max(0.0, min(1.0, (eh - self.fwa_lp) / max(1e-3, eh - lo)))
            v = min(v, vc + (vs - vc) * se)
        # 前輪快打到底 = 彎太急、轉向到極限、車子開始往外滑：側向負載的算法前提（車頭轉速和前輪成正比）不成立，
        # 照平滑前輪角還會算出 50%，直接降到 v_sat（203057 進坡彎 +44° 時 46–51%，貼到左側木條）
        if self.fwa_lp >= p.get("sat_deg", 99):
            v = min(v, p.get("v_sat", vc))
        # 實際轉速比前輪算的大（打滑、前輪還沒反應）時也照實際的限速
        if self.yaw_rate is not None and abs(self.yaw_rate) > 1.0:
            v = min(v, max(vc, p.get("lat_over", 1.3) * load / abs(self.yaw_rate)))
        # 前方的彎：遠處預看的前輪角也照同樣的側向負載算速度，取比較慢的，進彎前就先減速。
        # 遠處雜訊多：取最近 5 張中位數，變大馬上跟（要減速）、變小慢慢放（出彎前不要先加速）
        use_pv = p.get("preview_ld", 0.0) or p.get("preview_k", 0.0)
        if self._pv and use_pv:
            pf = sorted(self._pv)[len(self._pv) // 2]
            tau = 0.05 if pf > self.preview else p.get("preview_release_s", 0.4)
            self.preview += (pf - self.preview) * min(1.0, dt / tau)
            if self.preview > lo:
                tp_ = math.tan(math.radians(self.preview))
                vp = min(math.sqrt(load / (p.get("yaw_gain", 1.5) * tp_)), p.get("v_cmax", vs))
                wp = max(0.0, min(1.0, (self.preview - lo) / p.get("blend_deg", 4.0)))
                vp = vs + (min(vp, vs) - vs) * wp
                if self.preview >= p.get("sat_deg", 99):
                    vp = min(vp, p.get("v_sat", vc))
                v = min(v, max(min(vc, p.get("v_sat", vc)), vp))
        floor = vc
        if self.fwa_lp >= p.get("sat_deg", 99) or (use_pv and self.preview >= p.get("sat_deg", 99)):
            floor = min(vc, p.get("v_sat", vc))      # 轉向到極限時可以低於 v_curve
        return max(floor, min(vs, v))

    def ramp(self, v_target, dt):
        """加速有上限，減速立即生效。"""
        if v_target > self.v:
            self.v = min(v_target, self.v + self.c["accel_per_s"] * dt)
        else:
            self.v = v_target
        return self.v

    def wheels(self, v, fwa):
        """回傳 (bwl, bwr)。開啟差速時前輪轉越多，外側輪越快、內側輪越慢，幫忙轉急彎。

        diff_gain：1 = 打到底時外側 2 倍、內側停；> 1 內側會反轉。和 motor_cmd.py 的 diff_split() 同一個公式，
        用 `motor_cmd.py --teleop` 手動試出能過彎的增益後直接填進來。
        """
        if not self.c["differential"] or v == 0:
            return v, v
        g = self.c.get("diff_gain", 1.0)
        # 只有大轉彎才用差速：直線上的小修正也加差速，前輪一左一右時後輪也跟著一推一拉，會越晃越大。
        # 前輪超過 diff_from 度才開始，打到 fwa_limit 時差速最大（和原本一樣）。
        d0, lim = self.c.get("diff_from", 25.0), self.c["fwa_limit"]
        r = max(0.0, min(1.0, (abs(fwa) - d0) / max(1e-3, lim - d0)))
        r = r if fwa > 0 else -r                                 # fwa 正 = 右轉，右輪是內側
        clip = lambda u: max(-100.0, min(100.0, u))
        return clip(v * (1 + g * r)), clip(v * (1 - g * r))
