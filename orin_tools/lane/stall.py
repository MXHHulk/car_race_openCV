"""卡住偵測：車子在行駛中、鏡頭畫面卻幾乎不動，就是卡住了（上坡推不動、輪子頂到東西）。

判斷方法：每張畫面縮成 80×60 灰階，取中間一帶，和上一張比平均差多少（motion）。
實測（2026-10-28 正式賽道 run_20261028_170420）：
  待命 0.32、卡在坡道 0.29；行駛中 1.5–1.8、隧道 3.4，0.5 秒平均最低也有 0.99。
用最近 0.5 秒的中位數判斷，有人走過畫面或燈閃一下不會誤判。

卡住時：每秒加 boost_per_s 的速度，最多加到 boost_max；同時關掉後輪差速，兩輪一起出力往前推。
推動之後不要馬上退：畫面變化在 thresh～release 之間（有在動但很慢 = 還在爬坡）就維持加速，
超過 release（恢復正常速度）才把加上去的速度退掉。
  2026-10-28 run_20261028_181425：彎道上坡時車子一動就把加速退掉，回到 24% 又卡住，來回好幾次。
  坡道上剛推動時畫面變化 0.95–1.27，正常行駛 1.4–2.0，所以 release 取 1.4。
  run_20261028_182916：坡道入口的突起，車子一衝過去畫面變化會瞬間跳到 2–4，被當成「恢復正常」就退掉加速，
  又卡住。所以要連續 release_s 秒都超過 release 才開始退，而且退得比加得慢。
  run_20261028_183945：長坡道上還是走走停停（推動 → 退掉 → 又卡住，循環 5 次）。
  坡道需要「持續」的力量：卡住後推動時的加速量，先保底 floor_ratio（70%）維持 floor_s 秒，之後才往下退。
"""
from collections import deque

import cv2
import numpy as np


class StallGuard:
    def __init__(self, cfg):
        c = cfg["control"].get("stall") or {}
        self.on = bool(c.get("enable", True))
        self.thresh = c.get("thresh", 0.6)        # 畫面變化低於這個算沒在動
        self.hold = c.get("hold_s", 0.5)          # 持續多久才算卡住（s）
        self.rate = c.get("boost_per_s", 25)      # 卡住時每秒加多少速度 %
        self.release_s = c.get("release_s", 1.0)  # 畫面變化要連續這麼久都正常，才開始退加速（s）
        self.drop = c.get("drop_per_s", 15)       # 退加速時每秒退多少 %
        self.floor_ratio = c.get("floor_ratio", 0.7)  # 推動後保底：卡住時加到的最大值 × 這個比例
        self.floor_s = c.get("floor_s", 3.0)          # 保底維持幾秒（從最後一次卡住算起）
        self.peak = 0.0
        self.t_stall = None
        self.max = c.get("boost_max", 45)         # 最多加多少 %
        # 坡道入口提早判斷：前輪打很大、差速讓內輪幾乎停住，一上坡就只剩外輪在推，車子停下來，
        # 等 0.5 秒中位數 + hold_s 才加速要 1–4 秒（run_20261031_141947 69.7–74 s）。
        # 前輪超過 quick_fwa、最近 5 張畫面變化低於 quick_m 就馬上關差速、加 quick_boost。0 = 不用
        self.quick_fwa = c.get("quick_fwa", 0)
        self.quick_m = c.get("quick_m", 1.6)
        self.quick_boost = c.get("quick_boost", 15)
        # 前輪打到快底時再提早一點：畫面變化一從 7–12 掉下來就動作，不等車子停（最近 3 張）。
        # 只用這條不夠：有時進坡前輪只有 32°（run_20261101_141924），所以兩條都留。平地彎道畫面變化約 2，
        # 但前輪不會到 40°；門檻放到 38 會在彎道誤動作（run_20261031_165533 47 s）
        self.early_fwa = c.get("early_fwa", 0)
        self.early_m = c.get("early_m", 4.0)
        # IMU 俯仰角超過這個（度，上坡正）而且車子開始變慢（最近 3 張 < early_m）就加速，0 = 不用。
        # 前輪不一定打很大：run_20261103_185435 進坡時前輪只有 +22°，quick / early 都沒觸發，
        # 36% 在坡上停了 1.3 秒才等到一般卡住加速。俯仰角 52.2 s 就超過 9°，車子 53.3 s 才停
        self.ramp_pitch = c.get("ramp_pitch", 0)
        # 每段上坡只觸發一次：坡道上畫面變化本來就只有 2–4，一直觸發會讓加速整段退不掉，
        # 路口判斷（加速退掉後 free_s 秒）跟著延後、錯過路口。俯仰角回到一半以下才能再觸發
        self.ramp_armed = True
        self.release = c.get("release", 1.4)      # 畫面變化超過這個（恢復正常速度）才開始退掉加速
        self.prev = None
        self.hist = deque(maxlen=15)              # 約 0.5 秒
        self.extra = 0.0
        self.t_low = self.t_last = self.t_high = None
        self.motion = 0.0
        self.med = 0.0                            # 最近 0.5 秒畫面變化的中位數（hilltop.py 也用）

    @property
    def active(self):
        return self.extra > 0

    def _motion(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60), interpolation=cv2.INTER_AREA)
        g = g[15:45].astype(np.float32)           # 中間一帶：避開天花板和畫面下方的車身
        m = 0.0 if self.prev is None else float(np.mean(np.abs(g - self.prev)))
        self.prev = g
        return m

    def update(self, frame, t, driving, fwa=0.0, pitch=None):
        """driving：現在有在下速度指令。fwa：目前前輪角度。pitch：IMU 俯仰角（度），None = 沒有 IMU。
        回傳要額外加的速度 %。"""
        self.motion = self._motion(frame)
        self.hist.append(self.motion)
        dt = 0.0 if self.t_last is None else max(0.0, t - self.t_last)
        self.t_last = t
        if not self.on or not driving:
            self.extra, self.t_low, self.t_high, self.peak, self.t_stall = 0.0, None, None, 0.0, None
            return 0.0
        med = self.med = float(np.median(self.hist))
        stalled = False
        if len(self.hist) == self.hist.maxlen and med < self.thresh:
            self.t_low = t if self.t_low is None else self.t_low
            stalled = t - self.t_low >= self.hold
        else:
            self.t_low = None
        was = self.active
        if med > self.release:
            self.t_high = t if self.t_high is None else self.t_high
        else:
            self.t_high = None
        if stalled:
            self.extra = min(self.max, self.extra + self.rate * dt)
            self.peak = max(self.peak, self.extra)
            self.t_stall = t
        elif self.t_high is not None and t - self.t_high >= self.release_s:   # 連續正常一段時間：退加速
            floor = 0.0
            if self.t_stall is not None and t - self.t_stall < self.floor_s:
                floor = self.peak * self.floor_ratio      # 剛推動不久：保底，不要一下就退到推不動
            self.extra = max(floor, self.extra - self.drop * dt)
        recent = list(self.hist)[-5:]
        # 有 IMU 時要真的開始上坡才算：平地急彎前輪也會到 40°，誤觸發後差速關掉、保底維持 3 s，
        # 車子進坡那個彎轉不過去（run_20261103_200219 28.0 s 俯仰 −0.4° 觸發，騎上坡道左側木條）
        sloped = pitch is None or not self.ramp_pitch or pitch >= self.ramp_pitch / 2
        quick = (sloped and self.quick_fwa and abs(fwa) >= self.quick_fwa and len(recent) == 5
                 and np.median(recent) < self.quick_m)
        early = (sloped and self.early_fwa and abs(fwa) >= self.early_fwa and len(recent) == 5
                 and np.median(recent[-3:]) < self.early_m)
        if pitch is not None and pitch < self.ramp_pitch / 2:
            self.ramp_armed = True
        # 前輪已經回到 quick_fwa 以內也觸發：不用等車子慢下來（run_20261103_191003 俯仰 50.07 s 過 9°，
        # 等到 50.9 s 畫面變化才掉到 4 以下，中間 36% 有停頓感）。前輪還打很大時先保留差速把彎轉完
        ramp = (self.ramp_pitch and self.ramp_armed and pitch is not None and pitch >= self.ramp_pitch
                and len(recent) == 5 and (np.median(recent[-3:]) < self.early_m
                                          or (self.quick_fwa and abs(fwa) < self.quick_fwa)))
        if ramp:
            self.ramp_armed = False
        if quick or early or ramp:
            self.extra = max(self.extra, self.quick_boost)
            self.peak = max(self.peak, self.extra)
            self.t_stall = t
        if self.extra == 0.0:
            self.peak = 0.0
        # 其他（有在動但很慢，或只是往前衝了一下）：維持目前的加速
        if self.active and not was:
            print(f"[stall] 畫面不動（變化 {med:.2f}），車子可能卡住：加速並關掉差速")
        elif was and not self.active:
            print("[stall] 車子重新動了，恢復正常速度")
        return self.extra
