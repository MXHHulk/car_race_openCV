"""岔路接隧道的直線段（control.tunnel）：限速、目標往左偏一點，方向完全照鏡頭。

拱門入口比前面那段直線偏左，車道在拱門前往左拐一下（183147 順利通過時車頭往左轉了約 28° 再轉回來）。
接近拱門時左線出了畫面、只剩右線，鏡頭算出「偏右、前方往左」是對的：
  - 190038（race）以 65–87% 衝進來，前輪打到 −40° 也來不及，撞上拱門右腳
  - 191047、191347 第一版這裡「只剩一條線時不准往左」（以為是誤判），車子照直線方向直直撞上右腳
所以這裡不干涉方向，只做兩件事：
  - 速度最多 v（183147 66% 修得過、但甩到 ±30°）
  - 目標點往左偏 y_bias m（Controller.y_extra），進拱門前就先靠左，離右腳遠一點、左拐的量也小一點

哪一段：車頭角回到出發時的方向（每圈 −360°，順時針一圈）。出發 after_s 秒後，
車頭角離「出發方向 − 360°×圈數 + yaw_offset」window 度以內、維持 min_s 秒。
注意：出發位置和方向要和 2026-11-05 一樣（起跑線，隧道在一圈的最後）；換起點要改 yaw_offset。
"""


import cv2


class TunnelGuard:
    def __init__(self, cfg):
        c = (cfg["control"].get("tunnel") or {})
        self.on = bool(c.get("enable", False))
        self.after_s = c.get("after_s", 15.0)      # 出發後幾秒才開始找（起跑那段直線方向一樣，不算）
        self.window = c.get("window", 25.0)        # 車頭角離隧道方向幾度以內
        self.min_s = c.get("min_s", 0.6)           # 維持多久才算進入（彎道掃過去不算）
        self.yaw_offset = c.get("yaw_offset", 0.0)  # 隧道段方向 = 出發方向 + 這個（度，左正）
        # 第幾圈起才認（0 = 出發那段就算）：135819、141618 從隧道前的岔路直線出發，出發方向就是隧道方向，
        # 原本要等繞一圈（−360°）才認，第一次進隧道沒限速，65% 撞拱門右腳
        self.min_lap = c.get("min_lap", 1)
        self.lap_need = self.min_lap                # 對準路口之後一律要 −360°（第 1 圈）：0 會把下坡後同方向那段也認成隧道
        self.junc_offset = c.get("junc_offset", None)   # 隧道方向 − 路口左轉開始的車頭（度），None = 不用
        self.v = c.get("v", 50)                    # 隧道裡（畫面暗）最快幾 %
        self.v_zone = c.get("v_zone", self.v)      # 用車頭方向認的岔路 / 隧道段（暗之前）最快幾 %
        self.y_bias = c.get("y_bias", 0.06)         # 目標往左偏幾 m
        # 前視多加幾 m（負 = 縮短）。原本 +0.2 想提早看到拱門前往左拐，結果同樣偏 18 cm 前輪只算出 −9°（短的話 −22°），
        # 194348、194020 都貼上隧道右牆（偏右 20–29 cm），194348 卡住。改成縮短，反應快一點
        self.ld = c.get("ld", -0.1)
        self.max_rate = c.get("max_rate", 20.0)    # 車頭轉速（°/s）超過這個不算進入（在彎道裡）
        self.enter = c.get("enter", 15.0)          # 進入時車頭要離隧道方向幾度以內（進入後放寬到 window）
        # 用畫面亮度認隧道（和起點無關）：195533 換了起點，用車頭方向認不到，隧道段沒啟動，
        # 65% 出隧道進 S 彎開出賽道。畫面中段（640×480 的第 150–330 列）平均亮度：隧道裡 14–32，
        # 其他地方幾乎都在 40 以上（另外出發後約 13 s 有一處也會暗 1 秒左右，一起限速）
        self.dark = c.get("dark", 40)              # 亮度低於這個算暗，0 = 不用
        self.dark_n = c.get("dark_n", 3)           # 連續幾張才算
        self.after_dark_s = c.get("after_dark_s", 2.5)   # 變亮後再維持幾秒（出隧道接 S 彎）
        self.v_after = c.get("v_after", 45)
        self.reset()

    def reset(self):
        self.yaw0 = self.t_start = self.t_in = None
        self.prev = None                            # (t, yaw) 上一張，算車頭轉速
        self.n_dark, self.t_dark = 0, None          # 連續幾張暗、最後一次暗的時間
        self.in_dark = False
        self.zone_dark = False                      # 這次岔路 / 隧道段已經暗過（進過隧道）
        self.active = False
        self.mode = ""                              # 寫紀錄用："" / "on"

    @property
    def ld_extra(self):
        """給 Controller：前視多加多少（m）。"""
        return self.ld if self.in_dark else 0.0    # 只在隧道裡縮短：岔路直線也縮短會左右晃

    def _dark_update(self, t, frame):
        """回傳 (隧道裡, 剛出隧道)。"""
        if not self.dark or frame is None:
            return False, False
        m = float(cv2.cvtColor(frame[150:330:4, ::4], cv2.COLOR_BGR2GRAY).mean())
        self.n_dark = self.n_dark + 1 if m < self.dark else 0
        if self.n_dark >= self.dark_n:
            if not self.in_dark:
                print(f"[tunnel] 畫面變暗（{m:.0f}）：隧道，限速 {self.v}%")
            self.in_dark, self.t_dark = True, t
        elif self.in_dark and self.n_dark == 0:
            self.in_dark = False
            print(f"[tunnel] 出隧道：{self.after_dark_s:.1f} 秒內最快 {self.v_after}%")
        after = not self.in_dark and self.t_dark is not None and t - self.t_dark < self.after_dark_s
        return self.in_dark, after

    @property
    def y_extra(self):
        """給 Controller：目標點橫向多偏多少（m，左正）。"""
        # 只在進隧道前（岔路段）往左偏：隧道裡也偏會貼左牆（193243）。
        # 130740 岔路段一路偏右 15–17 cm，到拱門往左拐時又被甩到右邊壓線
        # 進過隧道（暗過一次）就不再偏：132419 出隧道後還在這段，y_bias 0.13 把車子推到左邊 13–19 cm
        return self.y_bias if (self.active and not self.in_dark and not self.zone_dark) else 0.0

    def anchor(self, t, turn_yaw):
        """坡頂路口開始左轉時呼叫：用路口重新對準隧道方向（和起點無關）。
        21 趟紀錄：隧道的車頭方向 = 路口左轉開始時的車頭 + junc_offset（−112°，±8°，mod 360）。
        從別的地方出發時（135819 隧道在出發後 8 s，起點方向對不到），第二圈以後還是認得到。"""
        if self.junc_offset is None or turn_yaw is None:
            return
        self.yaw0 = turn_yaw + self.junc_offset - self.yaw_offset
        self.lap_need = 1
        if self.t_start is None:
            self.t_start = t
        self.t_start = min(self.t_start, t - self.after_s)   # 對準後下一段就可以認（不用再等 after_s）
        self.t_in, self.active = None, False

    def start(self, t, yaw):
        """出發時呼叫：記下出發方向。"""
        self.lap_need = self.min_lap
        self.yaw0, self.t_start, self.t_in, self.active = yaw, t, None, False

    def update(self, t, yaw, v, running, frame=None, skip_dark=False):
        """回傳限速後的速度。不在隧道段時原樣回傳。
        skip_dark：這段不用亮度認隧道（下坡後那幾處暗的地方，見 main.py）；已經用車頭方向認出隧道段時不跳過。"""
        self.mode = ""
        use_dark = running and self.on and not (skip_dark and not self.active)
        dark, after = self._dark_update(t, frame) if use_dark else (False, False)
        if not use_dark:
            self.in_dark, self.n_dark = False, 0
            if skip_dark:
                self.t_dark = None              # 下坡後的暗處也不要觸發「出隧道後限速」
        v = self._heading_update(t, yaw, v, running)
        if dark and self.active:
            self.zone_dark = True
        if dark:
            self.mode = "dark"
            v = min(v, self.v)
        elif after and not self.active:
            self.mode = "after"
            v = min(v, self.v_after)
        return v

    def _heading_update(self, t, yaw, v, running):
        if self.yaw0 is None and yaw is not None and self.t_start is not None and t - self.t_start < 3.0:
            self.yaw0 = yaw                         # 出發那一刻 IMU 還沒讀到（192130）：用第一個讀到的值
        if not self.on or not running or yaw is None or self.yaw0 is None:
            self.active = False
            return v
        rate = 0.0
        if self.prev is not None and t > self.prev[0]:
            rate = (yaw - self.prev[1]) / (t - self.prev[0])
        self.prev = (t, yaw)
        rel = yaw - self.yaw0 - self.yaw_offset
        laps = round(-rel / 360.0)
        err = rel + 360.0 * laps                    # 離隧道方向幾度（左正）
        inside = laps >= self.lap_need and abs(err) <= self.window and t - self.t_start >= self.after_s
        if not inside:
            if self.active:
                print(f"[tunnel] 離開隧道段（車頭 {err:+.0f}°）")
            self.t_in, self.active = None, False
            return v
        # 要進入時車頭要幾乎不轉（直線）：193243 起點換了，第一圈 21 s 一個大彎掃過這個方向也開了 2 秒。
        # 已經進入後不管轉速（拱門前要往左拐）
        if not self.active and (abs(rate) > self.max_rate or abs(err) > self.enter):
            self.t_in = None
            return v
        self.t_in = t if self.t_in is None else self.t_in
        # 從岔路直線出發（min_lap 0）：出發頭 1.5 秒內方向對就立刻啟動，不等 min_s。
        # 164347 出發後 1.9 s 就到拱門，等 min_s 到 1.14 s 才啟動，前面以 50% 偏右 16 cm 衝過去，
        # 拱門前硬往左打 −31°，進隧道又甩到左邊、再往右打 +23° 撞上隧道右牆卡住
        quick = self.lap_need == 0 and laps == 0 and t - self.t_start < 1.5
        if not self.active and (quick or t - self.t_in >= self.min_s):
            self.active, self.zone_dark = True, False
            print(f"[tunnel] 進入岔路 / 隧道段（第 {laps} 圈，車頭 {err:+.0f}°）：限速 {self.v}%、目標往左 {self.y_bias * 100:.0f} cm")
        if not self.active:
            return v
        self.mode = "on"
        return min(v, self.v_zone)
