"""開車：每張畫面 step() 一次，回傳 (前輪角度, 左後輪 %, 右後輪 %)。不碰硬體，離線回放也用它。

方向：Pure Pursuit 追中線上前視 ld 的點。前輪角度正 = 往右（和韌體一樣）。
速度：照下面的順序決定，最後寫在 self.why（紀錄、畫面上看得到為什麼這個速度）
  1. 進彎減速    「彎的程度」= max(現在要打的角度, 前方 preview m 那點要打的角度) → v_curve / v_sat
  2. 大直線加速  兩條線都看到、看得到 straight_far 以上、又直 → v_straight（只看到一條線或看不遠 → v_mid）
  3. 上坡提速    俯仰 ≥ up_pitch → 至少 up_v；畫面不動（卡住）再慢慢加推力
  4. 虛線左轉    平地上連續 n 張看到前方橫越的虛線 → 前輪打 fwa、車頭轉 yaw 度後回到循線
  5. 下坡反打    俯仰 ≤ −down_pitch → 一路反打後輪（越快打越多，down_brake_min ~ down_brake）、太慢 down_v 推；卡住推到 down_stuck_v
另外：暗處（隧道）限 v_dark、前視縮短；看不到線維持方向、慢速，太久停車。
"""
import collections
import cv2
import math
import os
import statistics

from perception import Perception
from obstacle import ObstacleDetector
import trackmap
import numpy as np

IDLE, RUN, LOST, STOP = "IDLE", "RUN", "LOST", "STOP"


class Driver:
    def __init__(self, cfg, base_dir="."):
        self.cfg = cfg
        self.eye = Perception(cfg, base_dir)
        # 前車偵測（11-23 使用者要求：跟車、太近就停，只用鏡頭）。obstacle.act false = 只記錄（CSV obs / obs_raw），不影響車速
        self.oc = cfg.get("obstacle") or {}
        self.obs_det = ObstacleDetector(self.eye, self.oc) if self.oc.get("enable", False) else None
        self.obs_hist = collections.deque(maxlen=int(self.oc.get("win", 6)))
        self.obs_raw = self.obs_d = None
        self.obs_on = False
        self.g, self.sc, self.sp, self.sl, self.jc = (cfg["geometry"], cfg["steer"], cfg["speed"],
                                                      cfg["slope"], cfg["junction"])
        self.state, self.reason = IDLE, ""
        self.s = None
        self.cmd = (0.0, 0.0, 0.0)
        # 背地圖（11-21 使用者改成背地圖跑；出發在起點或起點後的大直線）
        self.mc = cfg.get("map") or {}
        self.tmap, self.loc = trackmap.load(base_dir, self.mc)
        if self.tmap is not None:
            print(f"[driver] 地圖：圈長 {self.tmap.L:.1f} m、{self.tmap.m.get('laps')} 圈平均、岔路 {self.tmap.forks}")
        self.map_dash_lap = False     # 這一圈虛線左轉做過了沒（上坡開始時重設）
        self.map_lap = False          # 出發後上坡有對到地圖（之後才照地圖認虛線 / 漏判左轉）
        self.map_fork_trust = False   # 出發後看過 2 次 V、定位確定（不用等上坡就開始用地圖）
        self.n_vseen = 0
        self.odo = 0.0                # 出發後的行駛距離（m，和定位同一套估法）
        self.odo_vlast = None         # 上一個 V 出現時的行駛距離
        self.lm_ev = []               # 最近看到的地標（種類, 行駛距離）
        self.t_frozen = None
        self.was_dark = False
        self.lm = None
        self.fork_anchored = False    # 這段 V 已經拿來校正過定位
        self.fork_by_map = False      # 現在的岔路方向是地圖暫定的（影像一確認 V 就照尖端規則重新決定）
        self.fwa = self.sc["fwa_trim"]
        self.v = 0.0
        self.why = ""
        self.mode = "follow"          # follow / turn
        self.target = None
        self.t_prev = None
        self.t_go = self.t_lost = None
        self.n_up = self.n_down = self.n_dash = 0
        self.t_still = None
        self.boost = 0.0
        self.brake_until = 0.0
        self.t_turn = self.yaw_turn0 = None
        self.t_turn_done = -1e9
        self.sev = 0.0
        self.pitch = None
        self.n_fork = 0
        self.fork_until = -1e9
        self.fork_side = None
        self.n_moving = 0
        self.t_dash_seen = None
        self.safe = cfg.get("safety") or {}
        self.elevated = False         # 在高處（上完坡、還沒下坡）：坡道、平台
        self.dtrk = None              # 前方橫越虛線的追蹤（_dash_track）
        self.full_mps = float((cfg.get("web") or {}).get("full_mps", 1.57))   # 100% 車速（m/s），預測虛線距離用
        self.h_ref = None             # 最後一張可信的 (車道方向, 車頭角, 時間)
        self.n_glitch = 0
        self.t_up_end = -1e9          # 上坡結束（到坡頂）的時間
        self.t_down_end = -1e9        # 下坡結束（到坡底）的時間
        self.t_down_start = -1e9      # 下坡開始的時間
        self.t_crest_still = None     # 坡頂邊緣畫面開始不動的時間
        self.down_cmd = 0.0           # 下坡滑行模式的反打量（低通）
        self.up_latch = False         # 上坡遲滯（坡上顛簸不要當成坡頂）
        self.n_below_up = 0
        self.off_trk = None           # 坡上 / 坡頂追蹤的偏移（單線認錯邊修正用）
        self.off_fixed = False
        self.pc_v = None
        self.ramp_dir_last = None
        self.exit_ref = None          # 平台出口維持的車道方向（陀螺儀）
        self.exit_last = None         # (最後的出口方向, 時間)
        self.n_exit_wrong = 0         # 平台出口：連續幾張在虛線右邊（偏移 ≥ exit_wrong_off）
        self.up_pend = None           # 上坡起點待確認（記下當時的行駛距離）
        self.laps, self.t_lap = [], None   # 單圈秒數（虛線左轉到下一次虛線左轉）
        self.down_ref = None          # 下坡維持的方向
        self.was_down_hold = False              # 坡頂後的限速（緩慢提速）
        self.cresting = False
        self.n_up_off = 0
        self.far_hist = collections.deque()   # 遠處亮度（隧道口：亮度一直往下掉）
        self.was_down = False
        self.t_up0 = None             # 這次上坡開始的時間
        self.ramp_pts = []            # 剛上坡時量到的坡道方向（車頭角 + 車道方向）
        self.ramp_dir = None          # 這次坡道的方向（只用在這一段坡，不記賽道）
        self.ramp_d = None            # 坡上想往左轉幾度（給後輪差速用）
        self.down_d = None            # 下坡想往左轉幾度（給後輪差速用）
        self.ramp_i = 0.0             # 坡上後輪差速的積分（%）
        self.t_steep = None           # 俯仰到 ramp_learn_pitch 的時間
        self.up_mis = False           # 坡底還沒對正
        self.center_c = 1.0           # 居中程度 0–1
        self.trim_i = 0.0             # 平地直線自動修正的前輪置中角（度，負 = 往左）
        self.diff_trim = self.sc["diff_trim0"]   # 平地後輪差速補償（右輪多 diff_trim·v %，正 = 往左帶）
        self.yaw_prev = None
        self.yaw_rate = 0.0           # 車頭轉速（度/秒，左正），低通
        self.was_up = False

    # ------------------------------------------------------------------ 狀態
    def lap_info(self):
        """網頁用：完整圈數、上一圈、平均、最快（秒）。"""
        L = self.laps
        return {"laps": len(L), "lap_last": L[-1] if L else None,
                "lap_avg": sum(L) / len(L) if L else None, "lap_best": min(L) if L else None}

    def start(self):
        if self.state in (IDLE, STOP):
            self.state, self.reason = RUN, ""
            self.laps, self.t_lap = [], None          # 每趟重新計圈
            self.t_go = None
            self.mode = "follow"
            self.v, self.boost = 0.0, 0.0
            self.trim_i = 0.0
            self.diff_trim = self.sc["diff_trim0"]
            self.eye.reset_track()
            if self.loc is not None:
                if self.mc.get("start_anywhere"):
                    self.loc.reset(0.0, self.tmap.L)      # 任何地方出發：整圈都有可能
                else:
                    self.loc.reset()
                self.map_dash_lap = self.map_lap = self.map_fork_trust = False
                self.n_vseen = 0
                self.odo, self.odo_vlast = 0.0, None
                self.up_pend = None
                self.lm_ev = []
            print("[driver] 出發")

    def stop(self, reason="手動停止"):
        if self.state in (RUN, LOST):
            print(f"[driver] 停止：{reason}")
        self.state, self.reason = STOP, reason

    def toggle(self):
        if self.state in (RUN, LOST):
            self.stop()
        elif self.state == STOP:
            self.state, self.reason = IDLE, ""
        else:
            self.start()

    # ------------------------------------------------------------------ 每張畫面
    def step(self, frame, t, pitch=None, yaw=None, sta_age=0.0):
        sp, sl, sc = self.sp, self.sl, self.sc
        dt = 0.0 if self.t_prev is None else max(0.0, min(0.2, t - self.t_prev))
        self.t_prev = t
        s = self.s = self.eye.process(frame)
        self.pitch = pitch

        self.n_up = self.n_up + 1 if (pitch is not None and pitch >= sl["up_pitch"]) else 0
        self.n_down = self.n_down + 1 if (pitch is not None and pitch <= -sl["down_pitch"]) else 0
        up, down = self.n_up >= sl["pitch_n"], self.n_down >= sl.get("down_n", sl["pitch_n"])
        # 上坡遲滯：一旦算上坡，要俯仰連續 up_off_n 張 < up_off_pitch 才算到坡頂
        #  （11-21 165731：坡上顛簸俯仰 13 → 5.4 → 8 → 5.2 來回跳，一張 < 9° 就被當成坡頂，速度壓到 32% 停在坡上）
        if pitch is not None and pitch < sl.get("up_off_pitch", sl["up_pitch"]):
            self.n_up_off += 1
        else:
            self.n_up_off = 0
        # 過坡頂中：還在上坡遲滯裡、但俯仰已經連續幾張 < up_pitch（真的在過坡頂，不是單張顛簸）
        self.n_below_up = self.n_below_up + 1 if (pitch is not None and pitch < sl["up_pitch"]) else 0
        if up:
            self.up_latch = True
        elif self.up_latch and (pitch is None or self.n_up_off >= sl.get("up_off_n", 1) or down):
            self.up_latch = False
        up_start = self.n_up == sl["pitch_n"]
        down_start = self.n_down == sl.get("down_n", sl["pitch_n"])
        up = up or self.up_latch
        self.cresting = (up and self.n_below_up >= sl.get("crest_n", 3)
                         and self.t_up0 is not None and t - self.t_up0 >= sl.get("crest_min_s", 2.0))
        if self.cresting:
            self.boost = 0.0          # 坡頂：卡住推力歸零（190823 殘留推力 +20 以 90/70 衝過坡頂，前輪懸空偏掉）
        slope = up or down
        mode_prev = self.mode
        self._map_step(t, pitch, yaw, up_start, down_start)
        mp = self.map_ok()

        # 狀態
        if self.state in (RUN, LOST) and sta_age > self.cfg["link"]["sta_timeout"]:
            self.stop(f"超過 {self.cfg['link']['sta_timeout']} 秒沒收到 STA")
        if self.state == RUN and s.conf == 0 and self.mode == "follow":
            self.state, self.t_lost = LOST, t
        elif self.state == LOST:
            if s.conf > 0:
                self.state = RUN
            elif t - self.t_lost > (2.5 if (slope or s.dark) else sp["lost_s"]):
                self.stop(f"超過 {sp['lost_s']} 秒找不到線")
        # ---- 保護機制（213809 在坡道上誤轉摔下去）
        sf = self.safe
        if self.state in (RUN, LOST) and sf:
            if pitch is not None and abs(pitch) >= sf["fall_pitch"]:
                # 1. 摔落／翻車：車身傾斜超過坡道會有的角度（坡道最多約 ±22°，摔下去時 −30 ~ −70°）
                self.stop(f"車身俯仰 {pitch:+.0f}°：摔落保護，停車")
            elif self.state == LOST and self.elevated and t - self.t_lost > sf["elevated_lost_s"]:
                # 2. 在高處（坡道、平台）找不到線：不要慢慢找，馬上停（掉下去比停下來嚴重）
                self.stop(f"高處超過 {sf['elevated_lost_s']} 秒找不到線：停車")
            elif self.mode == "turn" and pitch is not None and pitch <= -sf["turn_drop_pitch"] and not self.elevated_down_ok(t):
                # 3. 左轉時車身往下掉（轉出平台邊緣）：馬上停
                self.stop(f"左轉時車身往下掉（俯仰 {pitch:+.0f}°）：停車")
        running = self.state in (RUN, LOST)
        if running and self.t_go is None:
            self.t_go = t

        # ---- Y 字岔路口：繼續往前走 = 從 V 分隔線尖端「離車子遠的那一邊」過去
        #  尖端在車子右邊 → 走左邊那條；尖端在左邊 → 走右邊那條（車子本來在哪一邊就留在哪一邊，不切過去）。
        #  舊程式成功的圈：140849 15 s 尖端偏右走左、32 s 尖端偏左走右；200621 兩次都硬靠右 → 跑反了
        #  第一次認到時決定，看不到 V 之後再維持 fork_hold_s 秒
        self.n_fork = self.n_fork + 1 if s.fork else 0
        if (mp and self.map_lap and self.n_fork
                and not any(self.map_near(f["s"], self.mc.get("fork_ignore_m", 4.0), self.mc.get("fork_ignore_m", 4.0))
                            for f in self.tmap.forks)):
            # 這一圈已經用上坡對過地圖、附近 4 m 內沒有岔路：看到的 V 是誤判（212647 平台上左轉後看到 V）
            self.n_fork = 0
        tip_now = s.fork_tip_y if s.fork else None
        if self.n_fork == 0:
            self.fork_anchored = False
        if self.loc is not None and running and self.n_fork >= self.jc["fork_n"] and not self.fork_anchored:
            self.fork_anchored = True
            self._landmark("v")
        if self.loc is not None and running:
            if s.dark and not self.was_dark:
                self._landmark("dark")
            self.was_dark = s.dark
        mp = self.map_ok()
        if self.n_fork >= self.jc["fork_n"]:
            if t >= self.fork_until or self.fork_by_map:
                self.fork_by_map = False
                tip = s.fork_tip_y if s.fork_tip_y is not None else 0.0
                # 差不多正對尖端時「另一邊」不可靠：155206 同一個岔路口第一圈尖端偏左 11 cm 走右（對），
                # 第二圈偏右 10 cm 走左（開進柱子底下）。尖端偏不到 fork_tip_min 就走 fork_default
                fm = self.map_fork() if mp else None
                if fm is not None:
                    # 地圖定位確定、在地圖岔路附近：一律照地圖（9 圈正確紀錄建的）。影像尖端在車子偏一點時會判反
                    #  （200223 第 4 圈：30.4 m 尖端判成在左 17 cm 走右、39.8 m 判成在右 20 cm 走左，兩個都走反）
                    want = "left" if tip < 0 else "right"
                    if want != fm["side"]:
                        print(f"[driver] 岔路：影像尖端說走{'左' if want == 'left' else '右'}，地圖說走{'左' if fm['side'] == 'left' else '右'} → 照地圖")
                        if abs(tip) >= self.jc.get("fork_cross_m", 0.08):
                            # 車子已經在 V 的另一側、要橫切過尖端：先降速慢慢切（11-22 195822 第 3 圈尖端在左 15 cm、
                            #  照地圖走左，用 55% 硬切，前輪 −44°、後輪 12/98，甩出去）
                            self.t_fork_cross = t
                            print(f"[driver] 岔路：要橫切過 V 尖端（{abs(tip) * 100:.0f} cm），降到 {self.jc.get('v_cross', 35)}%")
                    self.fork_side = fm["side"]
                elif self._fork_by_landmarks() is not None:
                    # 地圖還不能用（出發後還沒對過地圖）：用「V 前面剛看過什麼」認是哪一個岔路
                    #  （11-22 144642：在兩個岔路中間出發，第一個 V 尖端只偏右 4 cm → 走左，實際是 39.8 m 該走右的岔路；
                    #   最新 3 趟影像尖端規則 13 次錯 3 次：4 cm / 14 cm / 32 cm 都有判反）
                    side, why_lm = self._fork_by_landmarks()
                    want = "left" if tip < 0 else "right"
                    print(f"[driver] 岔路：地圖還不能用，{why_lm} → 走{'左' if side == 'left' else '右'}"
                          + ("" if want == side else f"（影像尖端說走{'左' if want == 'left' else '右'}，不採用）"))
                    self.fork_side = side
                elif abs(tip) < self.jc["fork_tip_min"]:
                    self.fork_side = self.jc["fork_default"]
                else:
                    self.fork_side = "left" if tip < 0 else "right"
                where = "右" if tip < 0 else "左"
                go = "左" if self.fork_side == "left" else "右"
                print(f"[driver] 岔路口：V 尖端在車子{where}邊 {abs(tip) * 100:.0f} cm → 走{go}邊")
            self.fork_until = t + self.jc["fork_hold_s"]
        elif mp and t >= self.fork_until:
            # 影像沒看到 V（或還沒確認）：地圖上的岔路附近照地圖走
            f = self.map_fork()
            if f is not None:
                if self.fork_side != f["side"] or getattr(self, "t_mapfork", -1e9) < t - 1.0:
                    print(f"[driver] 地圖岔路（{f['s']:.1f} m）：影像沒看到 V，照地圖走{'左' if f['side'] == 'left' else '右'}邊")
                self.t_mapfork = t
                self.fork_side = f["side"]
                self.fork_by_map = True
                self.fork_until = t + 0.1
        self.eye.fork_side = self.fork_side if t < self.fork_until else None
        self.eye.post_turn = self.mode == "turn" or t - self.t_turn_done < self.jc["post_turn_s"]
        # 虛線左轉完到下坡前：虛線當左線（使用者：一律沿虛線右側走）
        self.eye.dash_left = (self.mode == "follow" and self.t_turn_done > self.t_down_start
                              and 0 <= t - self.t_turn_done <= self.jc.get("after_s", 8) and not down)
        self.eye.slope = slope or t - self.t_up_end < sl["top_s"] or (pitch is not None and abs(pitch) >= sl["foot_pitch"])
        self.eye.slope_down = down or (pitch is not None and pitch <= -sl["foot_pitch"])

        # ---- 坡頂：上坡一結束，卡住的推力歸零、先慢一下（152948 用 80% 衝上平台）
        if self.was_up and not up and (pitch is None or pitch < sl["up_pitch"]):
            self.t_up_end = t
            self.boost = 0.0
            self.elevated = True
        self.was_up = up
        # ---- 坡底：下坡一結束，推力歸零、先慢一下（160642 坡底被當成卡住推到 70%，接右彎甩出去）
        if down or (self.safe and t - self.t_up_end > self.safe.get("elevated_s", 15)):
            self.elevated = False          # 開始下坡（或上完坡很久了）= 回到地面
        if down and not self.was_down:
            self.t_down_start = t
            self.down_cmd = 0.0
        if self.was_down and not down:
            self.t_down_end = t
            self.boost = 0.0
        self.was_down = down

        # ---- 陀螺儀檢查：車道方向一張就跳很多、車子自己卻沒轉 = 這張看錯（只剩一條線左右認反、雜點），
        #  維持上一個前輪角。152948 第二圈坡上單線認反，方向 −15° → +23°，90% 往左衝向坡道木邊；
        #  各圈也常有單張 −178° 的跳動讓前輪猛打。連續 jump_n 張都這樣就相信它（真的變了）。
        # ---- 坡上 / 坡頂：單線認錯邊修正。持續追蹤偏移，只看到一條線、偏移一張就跳約一個車道寬（陀螺儀沒轉）= 左右認反
        #  （191239 坡頂車偏左 −18 cm，下一張單線說 +27 cm，前輪修不回來壓上平台左邊緣）
        zone = up or (self.elevated and t - self.t_up_end < sl.get("crest_zone_s", 3.0))
        self.off_fixed = False
        if zone:
            w = self.g["lane_width"]
            # 預測：車頭和坡道方向的夾角 × 車速 = 橫向移動（車往左 → 中線相對往右 → 偏移變小）
            if self.off_trk is not None and self.ramp_dir_last is not None and yaw is not None:
                vms = max(0.0, (self.cmd[1] + self.cmd[2]) / 2) / 100 * 1.57
                self.off_trk -= vms * dt * math.sin(math.radians(yaw - self.ramp_dir_last))
            if s.conf == 2:
                if self.off_trk is None or abs(s.offset - self.off_trk) <= sl.get("trk_gate2", 0.12):
                    self.off_trk = s.offset if self.off_trk is None else 0.5 * self.off_trk + 0.5 * s.offset
            elif s.conf == 1 and self.off_trk is not None:
                best = min((s.offset, s.offset - w, s.offset + w), key=lambda c: abs(c - self.off_trk))
                if abs(best - self.off_trk) <= sl.get("trk_gate1", 0.08):
                    self.off_fixed = best != s.offset
                    self.off_trk = 0.75 * self.off_trk + 0.25 * best
                if self.off_trk is not None:
                    s.offset = self.off_trk          # 單線：坡上 / 坡頂用追蹤的偏移（不信單張）
            if self.ramp_dir is not None:
                self.ramp_dir_last = self.ramp_dir
        else:
            self.off_trk, self.ramp_dir_last = None, None
        glitch = False
        if s.conf > 0 and yaw is not None:
            if self.h_ref is not None and t - self.h_ref[2] <= 0.4:
                pred = self.h_ref[0] - (yaw - self.h_ref[1])
                n_max = sc["jump_n_slope"] if slope else sc["jump_n"]
                jump_off = (s.conf == 1 and len(self.h_ref) > 3
                            and abs(s.offset - self.h_ref[3]) > sc["jump_off"])   # 單線左右認反：偏移一張跳很多
                if (abs(s.heading - pred) > sc["jump_deg"] or jump_off) and self.n_glitch < n_max:
                    glitch = True
                    # 平地：連續 jump_consist_n 張「看錯」的方向彼此一致（扣掉車頭轉動，差 ≤ jump_consist_deg）= 彎道真的收緊，
                    #  提早接受（11-23 120220 右緩彎出口 16.7 m：單線方向 −24 → −35 ~ −40° 連 6 張一致，全被當看錯，
                    #  前輪卡在 21.4° 往外漂，之後看不到線、衝出左邊緣掉下去）。單線左右認反（jump_off）、高處平台 / 虛線左轉後照舊擋
                    gp = getattr(self, "g_prev", None)
                    #  起步 start_s 秒內（一定從大直線出發，車道是直的）跳太大（和陀螺儀推算差 > jump_consist_max）的不算：
                    #  11-23 193042 起步單線方向 −54 ~ −64°（影像兩條線清楚、車在中間）連 3 張一致被接受，前輪往右打到底 +44° 0.4 秒，
                    #  起步後蛇行 4 秒（車頭 −11 → +7 → −9 → +8°）；121713 1.07 s 同樣（前輪 +30.7°）。
                    #  不全程套用：回放 131445 坡前急彎真的轉進來時方向也一次跳 > 35°，擋了會晚打（44 → 16.5°）
                    starting = self.t_go is not None and t - self.t_go < sp["start_s"]
                    if (not slope and not self.elevated and not self.after_dash(t) and not jump_off and gp is not None
                            and not (starting and abs(s.heading - pred) > sc.get("jump_consist_max", 1e9))
                            and abs(s.heading - (gp[0] - (yaw - gp[1]))) <= sc.get("jump_consist_deg", 0)):
                        self.n_consist = getattr(self, "n_consist", 0) + 1
                    else:
                        self.n_consist = 0
                    self.g_prev = (s.heading, yaw)
                    if sc.get("jump_consist_n", 0) and self.n_consist >= sc["jump_consist_n"] - 1:
                        glitch = False
            if glitch:
                self.n_glitch += 1
            else:
                self.n_glitch = 0
                self.g_prev, self.n_consist = None, 0
                self.h_ref = (s.heading, yaw, t, s.offset)
        s.glitch = glitch
        # 居中程度（0–1）：偏移越小、車頭和車道越平行越接近 1；看不到線當作很不居中
        if s.conf > 0 and not glitch:
            self.center_c = (max(0.0, 1 - abs(s.offset) / sl["center_off"]) *
                             max(0.0, 1 - abs(s.heading) / sl["center_deg"]))
        elif s.conf == 0:
            self.center_c = 0.0

        # ---- 方向
        if not glitch:
            fwa_pp, self.target = self._pursuit(s, self.v)
            if s.conf > 0:
                a = sc["smooth"]
                self.fwa = a * self.fwa + (1 - a) * fwa_pp
            # 自動修正置中角：平地直線上車道一直說「要往同一邊修」= 車子本身在拉（前輪歪、左右輪出力不同），
            #  慢慢把置中角往那邊調。175457：前輪置中、車頭每秒往右 5°，2 秒撞右側。每次出發重來，不記賽道
            if (s.conf == 2 and self.mode == "follow" and self.t_dash_seen is None and not slope
                    and (pitch is None or abs(pitch) < sl["flat_pitch"]) and self.sev <= sp["sev_low"]
                    and s.far >= sp["straight_far"] and abs(s.heading) < 8 and self.state == RUN and self.v > 0):
                # 只在「看得遠又直」的直線上學（彎道進出的誤差會帶偏：這條賽道右彎多）
                v_ms = 0.3 + 0.02 * max(0.0, self.v)
                e = s.heading + math.degrees(math.atan2(sc["k_off"] * s.offset, v_ms))
                e = max(-5.0, min(5.0, e))
                self.trim_i = max(-sc["trim_max"], min(sc["trim_max"], self.trim_i - sc["trim_ki"] * e * dt))
                # 後輪差速補償一起學：車一直要往左修（e > 0）→ 右輪多出一點（速度快時前輪轉向作用小，
                #  11-19 速度越快越往右拉：58% 以上前輪左打還偏右 11 cm）
                self.diff_trim = max(-sc["diff_trim_max"], min(sc["diff_trim_max"], self.diff_trim + sc["diff_trim_ki"] * e * dt))
        # （看不到線 / 看錯：維持上一個角度）
        if up and self.ramp_dir is not None and yaw is not None:
            # 坡道方向量到之後用陀螺儀判斷有沒有對正：坡上影像亂跳（185357 車頭方向 ±150° 亂跳、陀螺儀很穩），
            #  照影像會一直判「沒對正」把速度壓在 50%，卡住 1.7 秒
            self.up_mis = abs(yaw - self.ramp_dir) > sl["up_align_deg"]
        else:
            self.up_mis = up and (s.conf == 0 or abs(s.heading) > sl["up_align_deg"])
        self.ramp_d = None
        if yaw is not None and self.yaw_prev is not None and dt > 0:
            self.yaw_rate = 0.7 * self.yaw_rate + 0.3 * (yaw - self.yaw_prev) / dt
        self.yaw_prev = yaw
        # ---- 坡道是直的：剛上坡 ramp_learn_s 秒內用「車頭角 + 車道方向」估坡道方向，之後用陀螺儀維持車頭，
        #  視覺只做小幅左右修正（最多 ramp_off_max 度）。154336 坡上視覺跟著車頭一起轉，車頭往右轉了 20°，
        #  偏到右邊木板上，坡頂把白柱子當成車道線
        #  坡道方向只在「兩條線的消失點回到地平線附近」時量（車子和前面的坡道在同一個平面）：
        #  160004 坡底量的方向偏了 ~12°（−461，實際約 −448），車頭被帶往右、坡上轉不回來，壓線上木板
        if up and yaw is not None:
            if self.t_up0 is None:
                self.t_up0, self.ramp_pts, self.ramp_dir, self.ramp_i = t, [], None, 0.0
                self.ramp_early = []
            # 量坡道方向：真的在坡上（俯仰 ≥ ramp_learn_pitch 持續 ramp_learn_s 秒）、兩條線、路寬合理。
            # （消失點判斷在 180923 整段都不過，坡道方向一直沒量到；坡底俯仰還在變的時候方向會偏，所以等俯仰穩定）
            if pitch is not None and pitch >= sl["ramp_learn_pitch"]:
                self.t_steep = t if self.t_steep is None else self.t_steep
            else:
                self.t_steep = None
            steady = self.t_steep is not None and t - self.t_steep >= sl["ramp_learn_s"]
            lw = s.lane_w
            # 11-23 173958：坡道上段影像車道方向 −14 ~ −18°（兩條線、車其實已對正），正式坡道方向被拉偏 ~15°（−635 → −652，
            #  覆蓋掉正確的備用值且沒印出來），陀螺儀「維持」錯的方向，前輪一直右打 +19.6°，坡頂從右邊衝出去。
            #  只收車道方向小的畫面；已經有坡道方向時，差太多的不收
            pt = yaw + s.heading
            if (s.conf == 2 and not glitch and steady and lw is not None
                    and 0.75 * self.g["lane_width"] <= lw <= 1.3 * self.g["lane_width"]
                    and abs(s.heading) <= sl.get("ramp_learn_hd", 999)
                    and (self.ramp_dir is None or abs(pt - self.ramp_dir) <= sl.get("ramp_learn_tol", 999))):
                self.ramp_pts = (self.ramp_pts + [pt])[-sl["ramp_n"]:]
                if len(self.ramp_pts) >= 3:
                    new_dir = float(statistics.median(self.ramp_pts))
                    if self.ramp_dir is None:
                        print(f"[driver] 坡道方向 {new_dir:.0f}°（車頭 {yaw:.0f}°）：坡上用陀螺儀維持")
                    elif abs(new_dir - self.ramp_dir) >= 3 and abs(new_dir - getattr(self, "ramp_dir_said", self.ramp_dir)) >= 3:
                        print(f"[driver] 坡道方向更新 {self.ramp_dir:.0f}° → {new_dir:.0f}°（車頭 {yaw:.0f}°）")
                        self.ramp_dir_said = new_dir
                    self.ramp_dir = new_dir
            # 備用坡道方向：坡底剛對正（俯仰 ≥ up_pitch、還沒穩定）時兩條線、路寬合理、車道方向小的畫面先存起來；
            #  穩定後如果正式的量不到（坡上只看到一條線），用備用的。
            #  11-23 124429：坡前彎出得晚，坡上幾乎只有單線，正式坡道方向沒量到，只靠單線偏移修（認反、一直說偏右 +8 cm），
            #  車頭往左轉 14°，坡頂從左邊衝出去。9 次正常上坡備用和正式只差 ±1.6°
            if (sl.get("ramp_early", False) and s.conf == 2 and not glitch and not steady and lw is not None
                    and pitch is not None and pitch >= sl["up_pitch"]
                    and 0.75 * self.g["lane_width"] <= lw <= 1.3 * self.g["lane_width"]
                    and abs(s.heading) <= sl.get("ramp_early_hd", 6)):
                self.ramp_early = (getattr(self, "ramp_early", []) + [yaw + s.heading])[-sl["ramp_n"]:]
            if (sl.get("ramp_early", False) and self.ramp_dir is None and steady
                    and len(getattr(self, "ramp_early", [])) >= 3):
                self.ramp_dir = float(statistics.median(self.ramp_early))
                print(f"[driver] 坡道方向（備用：坡底對正時量）{self.ramp_dir:.0f}°（車頭 {yaw:.0f}°）：坡上用陀螺儀維持")
        elif (not up and t - self.t_up_end >= sl["ramp_top_s"]) or self.mode == "turn":
            self.t_up0, self.ramp_dir = None, None              # 虛線左轉一開始就不再維持坡道方向
        if up and self.ramp_dir is None and self.mode == "follow" and s.conf > 0 and not glitch and not self.up_mis:
            # 坡底（還沒量到坡道方向）：車道方向不可信，只照左右偏移修正
            v_ms = 0.3 + 0.02 * max(0.0, self.v)
            lim = sl["ramp_vis_max"]
            d = max(-lim, min(lim, math.degrees(math.atan2(sl["ramp_k_off"] * s.offset, v_ms))))
            self.fwa = sc["fwa_trim"] + self.trim_i - sl["ramp_gain"] * d   # 坡上前輪修得比平地用力（差速幾乎轉不動車）
            self.ramp_d = d
        if self.ramp_dir is not None and yaw is not None and self.mode == "follow" and not self.up_mis:
            # 陀螺儀（坡道方向）和視覺（車道方向 + 偏移）各一半；只看到一條線也用視覺。
            # 154852：只有兩條線才修正，單線說「偏右 12–20 cm」被丟掉，車壓過右線開上木板
            d = self.ramp_dir - yaw
            # 坡頂到平台入口（上坡剛結束）：視野從坡道換到平台，只看到一條線時常認錯邊（190254 坡頂把左線當右線，
            #  偏移 −5 → +35 cm，前輪往左打，車壓上平台左邊緣），這段只相信兩條線，一條線就純陀螺儀維持坡道方向
            crest_phase = not up
            if (s.conf > (1 if crest_phase else 0) or (crest_phase and s.conf == 1 and self.off_trk is not None)) and not glitch:
                v_ms = 0.3 + 0.02 * max(0.0, self.v)
                lim = sl["ramp_vis_max"]
                hd = s.heading
                if abs(hd - d) > sl.get("ramp_vis_hd", 999):
                    # 坡道是直的：影像說的車道方向和陀螺儀（坡道方向 − 車頭）差太多 = 影像看錯（坡頂線斷掉、單線認錯），改用陀螺儀的
                    #  11-23 173958：陀螺儀說已對正（差 0~7°），影像車道方向 −14 ~ −43°，前輪被拉到右邊 +19.6°，坡頂從右邊衝出去
                    hd = d
                d_vis = hd + math.degrees(math.atan2(sl["ramp_k_off"] * s.offset, v_ms))
                w = sl["ramp_vis_w"]
                d = (1 - w) * d + w * max(-lim, min(lim, d_vis))
            self.fwa = sc["fwa_trim"] + self.trim_i - sl["ramp_gain"] * d   # 坡上前輪修得比平地用力（差速幾乎轉不動車）
            self.ramp_d = d
        # ---- 下坡：前輪照循線打，另外算「想往哪邊修」給後輪差速用（反打時前輪幾乎轉不動）
        #  183932：下坡口車偏右 18 cm、前輪左打 26–31°，後輪反打煞車，車還是往右壓線
        self.down_d = None
        if down and s.conf > 0 and not glitch and self.mode == "follow":
            off = math.degrees(math.atan2(sl["ramp_k_off"] * s.offset, sl["down_v_ms"]))
            lim = sl["ramp_vis_max"]
            self.down_d = max(-lim, min(lim, 0.5 * s.heading + off))
        # 坡底還沒對正（車頭和車道差很多、或看不到線）：前輪不限制、速度先壓住，對正了才全力爬
        #  203013：坡底右急彎轉到一半（前輪 +38°、看不到線），一認到上坡前輪被限到 +16°、推到 90%，車頭還偏左 37° 就衝上去撞左邊
        if up and self.ramp_dir is None and self.mode == "follow" and sl.get("ramp_foot_kd", 0) > 0:
            # 坡底（還沒有坡道方向）：車頭轉速阻尼。這段原本只看左右偏移，不管車頭在轉
            #  11-23 135702：坡前右急彎還沒轉完就上坡（前輪 +40°、車頭 −45°/s），0.5 s 又往右轉 30°，
            #  從坡道底段右邊衝出去。車頭往右轉（轉速負）就往左壓：前輪 += ramp_foot_kd × 轉速
            self.fwa += sl["ramp_foot_kd"] * self.yaw_rate
        if up and self.up_mis and self.boost >= sl["up_align_boost"]:
            # 對正中又卡住：先爬上去要緊，前輪收一點降低阻力（205751 坡底前輪 +42°、壓 50% 完全不動）
            lim = sc["up_fwa"]
            self.fwa = max(sc["fwa_trim"] - lim, min(sc["fwa_trim"] + lim, self.fwa))
        if up and not self.up_mis:
            # 坡上前輪不要打太大：打大了阻力大會卡住（152948 第二圈坡底打 +26° 推 90% 卡 2 秒，回正就動了）
            lim = sc["up_fwa_stuck"] if self.boost > 0 else sc["up_fwa"]
            self.fwa = max(sc["fwa_trim"] - lim, min(sc["fwa_trim"] + lim, self.fwa))

        # ---- 4. 虛線左轉
        # 剛上完坡 dash_after_top 秒內不認虛線：213809 坡道上段變平、被當成坡頂，坡道上的反光被認成虛線，
        #  在坡道上左轉摔下去（平台上真的虛線在坡頂後 1.3–2 s 才會出現）
        if (self.jc["enable"] and self.state == RUN and self.mode == "follow"
                and (pitch is None or abs(pitch) < self.jc["max_pitch"])
                and t - self.t_turn_done > self.jc["cooldown_s"]
                and not up and t - self.t_up_end >= self.jc["dash_after_top"]
                # 真的虛線只在平台上：IMU 有在讀就要「上過坡、還在高處」才認
                #  11-22 211032 / 211121：出隧道（41.3–41.9 m）後在平地誤判虛線左轉（前 6 s 俯仰最多 2.1 / 2.6°、不在高處）；
                #  真的那次 205458 前 6 s 俯仰到 14.7°、在高處。211032 地圖定位錯（以為在 5.7 m），地圖擋不住，211121 只有地標定位也擋不住
                and (not self.jc.get("dash_need_elevated", True) or pitch is None or self.elevated)
                and (not (mp and self.map_lap) or self.map_near(self.tmap.m["dash"], self.mc.get("dash_before", 2.0), self.mc.get("dash_after", 1.2)))):
            self._dash_track(s, t, yaw)
            # 照地圖左轉只在「這一圈真的上過坡、現在在高處」時（172340：定位錯、沒上坡就在平地照地圖左轉）
            # 預設關閉（11-21 191914 / 192035：坡頂後慢速時定位跑在前面 1.5–2 m，還沒到虛線就照地圖左轉，每次同一個地方）
            if (self.mc.get("dash_fallback", False) and mp and self.map_lap and self.elevated and self.mode == "follow" and not self.map_dash_lap
                    and self.map_near(self.tmap.m["dash"], -self.mc.get("dash_late", 0.6), self.mc.get("dash_after", 1.2))):
                # 地圖：虛線位置已經過了 dash_late m、影像還沒觸發 → 照地圖左轉（漏判保護）
                self.mode, self.t_turn, self.yaw_turn0 = "turn", t, yaw
                self.t_dash_seen, self.dtrk = None, None
                print(f"[driver] 地圖：虛線位置到了（{self.loc.s:.1f} m），影像沒看到，照地圖左轉")
        else:
            self.dtrk = None
            self.t_dash_seen = None if self.mode == "follow" else self.t_dash_seen
        if self.t_dash_seen is not None and self.mode == "follow":
            # 看到虛線到開始左轉之間：前輪回正直走、慢下來。虛線會被當成車道線，
            # 201718 這段循線往右打到 +20°，車頭偏右才開始左轉，轉不過去
            self.fwa = sc["fwa_trim"] + self.trim_i
        if self.mode == "turn":
            turned = None if (yaw is None or self.yaw_turn0 is None) else yaw - self.yaw_turn0
            el = t - self.t_turn
            done = (el >= self.jc["time_s"]) if turned is None else (turned >= self.jc["yaw"] or el >= self.jc["max_s"])
            if done or not running:
                print(f"[driver] 左轉結束（{el:.2f} 秒" + ("" if turned is None else f"，車頭 {turned:+.0f}°") + "）")
                self.mode, self.t_turn_done = "follow", t
                # 轉完前輪先回正、陀螺儀的車道方向參考重來（不要沿用轉彎中看錯的方向）
                self.fwa, self.h_ref, self.n_glitch = sc["fwa_trim"] + self.trim_i, None, 0
            else:
                self.fwa = self.jc["fwa"]

        if self.mode == "turn" and mode_prev == "follow":
            # 單圈計時：每圈只有一次虛線左轉（要在高處才認，不會誤判），兩次左轉開始的間隔 = 一圈
            if self.t_lap is not None:
                self.laps.append(t - self.t_lap)
                print(f"[driver] 第 {len(self.laps)} 圈 {self.laps[-1]:.2f} 秒（平均 {sum(self.laps) / len(self.laps):.2f} 秒）")
            self.t_lap = t
        if self.loc is not None and self.mode == "turn" and mode_prev == "follow":
            self.map_dash_lap = True
            if self.loc.anchor(self.tmap.m["dash"], 0.4):
                pass

        # ---- 平台出口（虛線左轉完 → 下坡前）：地圖上這段是直的（車頭角 +73 ~ +83），影像會看到平台下面下層賽道的線亂跳
        #  （194823 出口偏移 −28 ~ +31 亂跳、前輪 −16 → +44，下坡前 1.5 s 右轉 36°，騎在線上衝下坡；正常圈 15–20°）
        #  → 陀螺儀維持車道方向；車道方向只用兩條線、沒看錯的畫面慢慢更新；左右只用兩條線的偏移修
        ex = self.sl.get("exit_hold_s")
        # 虛線左轉完、還沒開始下坡（不用 elevated：195351 從平台出發沒上過坡，出口照影像左右打滿 ±44° 才下坡）
        in_exit = (ex is not None and self.mode == "follow" and not up and not down and yaw is not None
                   and self.t_turn_done > self.t_down_start and ex <= t - self.t_turn_done <= self.jc.get("after_s", 8))
        # 出口方向用地圖：左轉開始時的車頭角 + exit_dpsi（地圖 psi：虛線 6.7 m −0.7° → 出口 8.5–10.2 m +73 ~ +75°）
        #  11-22 191603：左轉本身 +59°（OK），轉完後影像「看錯」又左打 −20 ~ −44° 0.6 秒，1.0 秒後才開始維持方向，
        #  當時車頭已經 +100°，照這個方向下坡 → 下坡時 +110°（地圖 +74°），坡底偏 −35 cm 衝出去
        dpsi = sl.get("exit_dpsi")
        exit_map = dpsi is not None and self.yaw_turn0 is not None
        self.in_exit = in_exit
        if in_exit:
            if exit_map:
                tgt = self.yaw_turn0 + dpsi
                if self.exit_ref is None:
                    self.exit_ref = tgt
                elif s.conf == 2 and not glitch and abs(s.heading) < sl.get("exit_ref_hd", 10):
                    # 11-22 215150 第 2 圈：出口兩條線一直說車道在右 −5 ~ −11°（畫面 2890–2930），0.15 修太慢，
                    #  車頭偏左 ~8° 一路漂過左線、下坡從左邊摔下去（俯仰 −27°）→ 比例 exit_ref_k
                    k = sl.get("exit_ref_k", 0.15)
                    self.exit_ref = (1 - k) * self.exit_ref + k * (yaw + s.heading)
                tol = sl.get("exit_ref_tol", 10)                      # 影像只能在地圖方向 ±tol 內微調
                self.exit_ref = max(tgt - tol, min(tgt + tol, self.exit_ref))
            elif self.exit_ref is None:
                self.exit_ref = yaw + (s.heading if (s.conf == 2 and not glitch and abs(s.heading) < 15) else 0.0)
            elif s.conf == 2 and not glitch and abs(s.heading) < 10:
                self.exit_ref = 0.85 * self.exit_ref + 0.15 * (yaw + s.heading)
            d = self.exit_ref - yaw                                  # 正 = 要往左
            dash_ok = any(l is not None and getattr(l, "is_dash", False) for l in (s.left, s.right))   # 虛線當邊線：單線也可靠
            # 單線但連續幾張偏移很穩（變化 < exit_stable_tol）：也拿來修左右（上限一樣 exit_off_max）
            #  11-22 214145：轉完只看得到左邊實線，47.66–48.56 s 單線偏移穩定 −5.4 ~ −12 cm、方向說車道在右 −5 ~ −15°，
            #  但單線不修左右，車頭照地圖方向一路漂到左邊條，保險桿撞上（畫面 1412–1418）
            hist = getattr(self, "exit_off_hist", [])
            hist = (hist + [s.offset if (s.conf >= 1 and not glitch) else None])[-self.sl.get("exit_stable_n", 5):]
            self.exit_off_hist = hist
            stable = (len(hist) >= self.sl.get("exit_stable_n", 5) and None not in hist
                      and max(hist) - min(hist) < self.sl.get("exit_stable_tol", 0.05))
            if (s.conf == 2 or dash_ok or stable) and not glitch:
                # 左右修正最多 exit_off_max 度：單線的偏移常跳 ±15–25 cm，不限制會把車頭帶離出口方向
                #  （11-22 201101 第 2 圈：單線偏移 +16 cm → 修 +17.7°，蓋過方向目標，車頭 +88 → +105.7°，下坡時偏左 38 cm 跑出去）
                om = sl.get("exit_off_max", 90)
                # 跑到虛線右邊（看得到虛線、中線在左邊 ≥ exit_wrong_off 連續幾張）：往左修可以放大到 exit_wrong_max
                #  11-22 204034：轉完車子在虛線上 / 右邊，偏移 +46 ~ +62 cm 0.7 s，但修正被限 8°，前輪只有 −10 ~ +3°，
                #  一路沿地圖方向開到平台右邊緣掉下去。（201101 的 +16 cm 單線誤判不會觸發：門檻 25 cm、要看到虛線）
                self.n_exit_wrong = self.n_exit_wrong + 1 if (dash_ok and s.offset >= sl.get("exit_wrong_off", 0.25)) else 0
                corr = math.degrees(math.atan2(sl["exit_k_off"] * s.offset, 0.5))
                if self.n_exit_wrong >= sl.get("exit_wrong_n", 3):
                    # 錯車道：往左修優先，不讓「維持地圖方向」把車頭拉回右邊（回放 204034：車頭已比地圖方向左 13°，
                    #  方向誤差 −13° 抵消了左修，前輪只剩 −3°）。車頭最多到出口方向 + exit_wrong_tol，回到虛線左邊就恢復
                    want = min(yaw + min(sl.get("exit_wrong_max", 15), corr), self.exit_ref + sl.get("exit_wrong_tol", 20))
                    d = max(d + max(-om, min(om, corr)), want - yaw)
                else:
                    d += max(-om, min(om, corr))
            else:
                self.n_exit_wrong = 0
            lim = sl.get("exit_fwa", 20)
            self.fwa = sc["fwa_trim"] + self.trim_i + max(-lim, min(lim, -sl.get("exit_gain", 1.2) * d))
            self.exit_last = (self.exit_ref, t, self.loc.s if (self.loc is not None and self.map_ok()) else None)
        else:
            self.exit_ref = None
            self.n_exit_wrong = 0
            self.exit_off_hist = []

        # ---- 暗處 / 隧道（map.hold_zones，地圖上是直的）：陀螺儀維持「進去前量到的車道方向」，影像只小幅修左右
        #  11-22 202211：暗處 20.9 m 影像方向 −67 / +71° 亂跳，第 3 張就被相信，前輪 −27 ~ −31° 0.4 s，撞到東西（俯仰 −12.6°）
        #  11-22 202516：隧道 27.5 m 單線偏移 −9 → +33 ~ +38 cm，前輪 −18 ~ −29° 0.6 s 再亂修到 +44°，掉出邊緣（俯仰 −14.8°）
        # 岔路區（map.fork_hold_zones，地圖上走的那邊是直的）也維持方向：分岔處兩條線會配到 V 的外側線，
        #  11-23 113124 第 1 圈 30.4 m 岔路偏移 +21 → −6 → +26 → −13 cm、前輪 −20 ~ +18° 來回打，39.8 m 偏移 −27 cm；
        #  地圖 29–33 m、37.5–41 m 車頭方向只有 ±4° 雜訊（走左 / 走右那邊都是直的）
        nh_ = len(self.mc.get("hold_zones") or [])
        hz = (list(self.mc.get("hold_zones") or []) + list(self.mc.get("fork_hold_zones") or [])) if self.tmap is not None else None
        in_hold = False
        self.hold_fork = False
        if not self.map_ok():
            self.t_map_bad = t
        if hz and yaw is not None and self.mode == "follow" and not slope and self.map_ok():
            for k_, (a, b) in enumerate(hz):
                if self.tmap.near(self.loc.s, (a + b) / 2, (b - a) / 2, (b - a) / 2):
                    # 岔路區：地圖要連續定位 fork_hold_after_s 秒以上（進岔路前就定位好）才用
                    #  11-23 115540：在 30.4 m 的 V 才定位（s 30.39），維持方向馬上啟動，基準方向被分岔處配錯的線
                    #  （路寬 0.61–0.64 m、方向 −10°）帶右 10° 以上，車走進右邊分岔
                    if k_ >= nh_ and t - getattr(self, "t_map_bad", -1e9) < self.mc.get("fork_hold_after_s", 5.0):
                        continue
                    in_hold = True
                    self.hold_fork = k_ >= nh_
                    break
        # 兩條線連續看出是彎道（|方向| > hold_vis_curve_hd）→ 之後 hold_vis_curve_s 秒不用影像觸發的維持方向
        #  （維持方向假設前面是直的隧道；11-23 105609 / 105632 第一個右彎被誤判成隧道口，
        #  兩條線已經說車道往右 −14 ~ −25°（第 110–116 張），前輪還被拉回 −2 ~ −7°，沒轉）
        if s.conf == 2 and not glitch:
            if abs(s.heading) > self.mc.get("hold_vis_curve_hd", 15):
                self.n_curve2 = getattr(self, "n_curve2", 0) + 1
                if self.n_curve2 >= self.mc.get("hold_vis_curve_n", 2):
                    self.t_curve2 = t
            else:
                self.n_curve2 = 0
        # 地圖還不能用（出發後第一圈）：影像看到隧道口 / 暗處就同樣維持方向（岔路口不用：要照 V 選邊）
        #  11-22 212023 第一圈：出髮夾彎後單線偏移說車偏右 +16 ~ +33 cm（其實車在往左），前輪 −13 ~ −32° 往左，
        #  地圖沒定位所以沒維持方向，10.3 s 撞隧道左邊（俯仰 −17.8°）
        if (not in_hold and self.mc.get("hold_vis", False) and yaw is not None and self.tmap is not None
                and self.mode == "follow" and not slope and not self.map_ok()
                and t - getattr(self, "t_tunnel_vis", -1e9) < self.mc.get("hold_vis_s", 0.8)
                and t - getattr(self, "t_curve2", -1e9) >= self.mc.get("hold_vis_curve_s", 1.5)
                and self.n_fork == 0 and t >= self.fork_until):
            in_hold = True
        if (not in_hold and yaw is not None and self.tmap is not None and self.mode == "follow" and not slope
                and s.conf == 2 and not glitch and abs(s.heading) < 12):
            self.hold_hist = (getattr(self, "hold_hist", []) + [(t, yaw + s.heading, self.loc.s, s.lane_w)])[-30:]
        if in_hold:
            if getattr(self, "hold_ref", None) is None:
                rec = [h for h in getattr(self, "hold_hist", []) if t - h[0] <= self.mc.get("hold_hist_s", 0.8)]
                if self.hold_fork:
                    # 岔路區：基準方向只用路寬像一個車道的兩條線（115540 用了配到 V 外側線的 0.61–0.64 m，方向偏右 10° 以上）
                    lw_lo, lw_hi = self.mc.get("fork_hold_lw", [0.42, 0.58])
                    rec = [h for h in rec if len(h) > 3 and h[3] is not None and lw_lo <= h[3] <= lw_hi]
                if len(rec) >= 3:
                    self.hold_ref = (float(np.median([h[1] for h in rec])), rec[-1][2])
            if self.hold_ref is not None:
                ds_ = self.tmap.ds
                i1, i0 = int(self.loc.s / ds_) % self.tmap.n, int(self.hold_ref[1] / ds_) % self.tmap.n
                # 兩條線的偏移和上一張（任何線數）差太多 = 這張兩條線配錯（隧道口柱子、旁邊的線），方向和偏移都不用
                #  11-22 211945：69.42 s 偏移 −13.9 → +16.1 cm（兩條線），被拿來修維持方向（−1352.6 → −1348.3）和偏移，
                #  前輪往左 −6 ~ −13° 0.5 s，撞隧道左邊（舊版同一段 −7 ~ +5°）
                prev = getattr(self, "hold_off_prev", None)
                jump = prev is not None and abs(s.offset - prev) > self.mc.get("hold_jump", 0.12)
                if s.conf > 0:
                    self.hold_off_prev = s.offset
                ok2 = s.conf == 2 and not glitch and not jump
                if ok2 and self.hold_fork:
                    # 岔路區：路寬不像一個車道（配到 V 外側線 0.6–0.7 m、或沒量到）就不拿來修左右
                    lw_lo, lw_hi = self.mc.get("fork_hold_lw", [0.42, 0.58])
                    ok2 = s.lane_w is not None and lw_lo <= s.lane_w <= lw_hi
                if ok2 and abs(s.heading) < self.mc.get("hold_hd_max", 15):
                    # 兩條線的方向慢慢修正維持方向（最多離進去時 ±hold_ref_tol 度）；單線還是不信
                    #  11-22 205458：髮夾彎剛出來量到的方向偏左 ~7°，隧道裡兩條線一直說車道在右（−8.8 / −13.6°）沒被採用，
                    #  偏移 −6 → −22 → −32 cm 撞到隧道左邊
                    if not hasattr(self, "hold_ref0") or self.hold_ref0 is None:
                        self.hold_ref0 = self.hold_ref[0]
                    k = self.mc.get("hold_ref_k", 0.3)
                    s_at = self.hold_ref[1]
                    meas = yaw + s.heading - float(self.tmap.psi[i1] - self.tmap.psi[i0])
                    r = (1 - k) * self.hold_ref[0] + k * meas
                    tol = self.mc.get("hold_ref_tol", 10)
                    self.hold_ref = (max(self.hold_ref0 - tol, min(self.hold_ref0 + tol, r)), s_at)
                # 暗處 / 隧道在地圖上是直的，地圖方向在這段只有雜訊（psi_sd 5–11°）：預設不加地圖方向變化
                #  11-22 213131 第 3 圈：地圖 20.3 → 21.0 m 方向 −131.8 → −127.9°（往左 3.9°），但三圈兩條線量到的車道方向
                #  18–22 m 都在 ±2° 內（直的）→ 維持方向被帶左 ~4°，偏移 −2 → −22 cm，21.97 m 撞起點柱子（俯仰 −16.4°）
                dpsi = float(self.tmap.psi[i1] - self.tmap.psi[i0]) if self.mc.get("hold_map_psi", True) else 0.0
                tgt = self.hold_ref[0] + dpsi
                d = tgt - yaw                                         # 正 = 要往左
                if ok2:
                    om = self.mc.get("hold_off_max", 6)
                    d += max(-om, min(om, math.degrees(math.atan2(sc["k_off"] * s.offset, 0.5))))
                lim = self.mc.get("hold_fwa", 15)
                # 轉速阻尼：車頭往左轉（yaw_rate 正）就往右壓（211945 進隧道前輪 −15° 0.4 s，車頭 0.6 s 左轉 15° 衝過頭）
                u = -self.mc.get("hold_gain", 1.2) * d + self.mc.get("hold_kd", 0.0) * self.yaw_rate
                if self.hold_fork and self.mc.get("fork_hold_ki", 0) > 0:
                    # 岔路區維持方向加積分：只有比例時，前輪置中修正（trim_i）和真的直走角度不同就會停在偏一個角度直走
                    #  （11-23 122950 第 2 次 30.4 m：基準 −1038.1°，車頭停在 −1034.5（偏左 3.6°）、前輪 0 ~ +1.4°，
                    #  3 m 偏到 −50 cm）。偏差持續就慢慢多修，最多 ±fork_hold_i_max 度
                    #  只在車頭穩定（|轉速| < fork_hold_i_rate）時累積：剛進岔路車頭還在轉時不累積，免得過衝
                    im = self.mc.get("fork_hold_i_max", 6.0)
                    hi_ = getattr(self, "hold_i", 0.0)
                    if abs(self.yaw_rate) < self.mc.get("fork_hold_i_rate", 3.0):
                        hi_ = hi_ - self.mc["fork_hold_ki"] * d * dt
                    self.hold_i = max(-im, min(im, hi_))
                    u += self.hold_i
                self.fwa = sc["fwa_trim"] + self.trim_i + max(-lim, min(lim, u))
        else:
            self.hold_ref = None
            self.hold_ref0 = None
            self.hold_off_prev = None
            self.hold_i = 0.0
        self.in_hold =in_hold and getattr(self, "hold_ref", None) is not None

        # ---- 地圖方向保護：地圖上一定是某個方向的彎（guard 區段），前輪不准往反方向打
        #  （11-21 173105：下坡底應該右彎，影像在坡底選到左邊那條線，車頭往左轉 84° 衝出賽道；
        #  正常的 6 圈坡底都是往右、前輪 +20 ~ +30°）
        if self.map_ok() and self.mode == "follow" and self.mc.get("guard"):
            g = self.map_guard()
            if g is not None:
                trim = sc["fwa_trim"] + self.trim_i
                rel = self.fwa - trim
                ff = self.mc.get("guard_ff", 0.35) * g                 # 地圖彎度 → 至少要打這麼多（正 = 右）
                # 影像要往反方向打（被擋）時不要只回正：改打地圖彎度需要的角度（軸距 × 曲率）
                #  11-22 211827：坡底影像把左邊別的車道的弧線和右線配成兩條線（偏移 +16 cm、要左打 −3 ~ −6°），
                #  被擋成置中 −1.7° 整整 1 秒，右彎沒轉、衝出右線（畫面 1474–1486）。地圖這段要右打 7–10°
                fb = (math.degrees(math.atan(self.g["wheelbase"] * math.radians(abs(g)))) * self.mc.get("guard_block_k", 0.0)
                      * (1 if g > 0 else -1))
                if g > 0:
                    rel = max(rel, ff) if rel >= 0 else max(ff, fb)    # 右彎：不准往左打
                elif g < 0:
                    rel = min(rel, ff) if rel <= 0 else min(ff, fb)    # 左彎：不准往右打
                self.fwa = trim + rel

        # ---- 下坡方向穩定（11-21 164610 下坡左右甩 +84 / −41°/s）：
        #  1. 前輪從置中算最多 ±down_fwa 度（坡上視覺方向 0.5 秒內 +24° → −38° 亂跳，前輪跟著打滿 ±44°）
        #  2. 陀螺儀阻尼：車頭往哪邊轉就往反方向壓 down_kd × 轉速（左轉 = 正）
        #  3. 看不到線：不維持打滿的前輪（164610 看不到線 0.6 秒前輪停在 +44°，一直往右轉），只留阻尼 = 維持車頭方向直走
        if down and not self.was_down_hold:
            # 下坡一開始：沿用平台出口維持的車道方向（陀螺儀），下坡也照這個方向直直下去
            self.down_ref = self.exit_last[0] if (self.exit_last and t - self.exit_last[1] < 1.5) else None
            self.down_ref_s = self.exit_last[2] if self.down_ref is not None else None
        self.was_down_hold = down
        down_tgt = self.down_ref
        if (down_tgt is not None and getattr(self, "down_ref_s", None) is not None and self.mc.get("down_follow_map", True)
                and self.map_ok()):
            # 下坡方向跟著地圖彎：出口方向 + 地圖從出口到現在的車頭變化（坡道 −6 ~ −9°、坡底開始右彎）
            #  11-22 205458：下坡照固定的出口方向，車頭 −578°（影像說和車道平行）被拉回 −568° 往左 5°，
            #  坡底右彎時車頭比車道左 26°、偏左 25 cm，保險桿撞到起點柱子（畫面 1492–1515 卡住不動）
            ds_ = self.tmap.ds
            i1, i0 = int(self.loc.s / ds_) % self.tmap.n, int(self.down_ref_s / ds_) % self.tmap.n
            down_tgt = self.down_ref + float(self.tmap.psi[i1] - self.tmap.psi[i0])
        if down and self.mode == "follow" and sl.get("down_fwa"):
            trim = sc["fwa_trim"] + self.trim_i
            base = self.fwa if s.conf > 0 else trim
            if down_tgt is not None and yaw is not None:
                # 影像（坡上很亂）和出口方向各一半（202028 下坡口車頭歪 30°，照影像修不回來）
                hold = trim - sl.get("exit_gain", 1.2) * (down_tgt - yaw)
                # 影像比重 down_img_w：11-22 220412 第 4 圈下坡，影像把左邊坡道線和坡下平地賽道的線配成兩條線
                #  （畫面 5950–5965，偏移 +15 ~ +18 cm），前輪往左 −9 ~ −16°，車頭 0.4 s 左轉 19°，從左邊摔下（俯仰 −31°）
                wi = sl.get("down_img_w", 0.5)
                base = (wi * base + (1 - wi) * hold) if s.conf > 0 else hold
            base = trim + max(-sl["down_fwa"], min(sl["down_fwa"], base - trim))
            self.fwa = base + sl["down_kd"] * self.yaw_rate
            self.fwa = trim + max(-sl["down_fwa"], min(sl["down_fwa"], self.fwa - trim))

        # ---- 速度
        self._obstacle(frame, s, t, pitch, slope)
        v, why = self._speed(s, t, dt, up, down, running)
        if getattr(self, "in_hold", False):
            why += "/暗處維持方向"
        self.why = why
        fwa = max(-sc["fwa_limit"], min(sc["fwa_limit"], self.steer_out(self.fwa)))
        if not running:
            self.cmd = (fwa, 0.0, 0.0)
            self.v = 0.0
            return self.cmd
        self.v = v
        if v > 0 and not slope and self.boost == 0 and (pitch is None or abs(pitch) < sl["flat_pitch"]):
            bwl, bwr = self._wheels(v, fwa)
            # 平台上（高處、虛線左轉完到下坡前）不用平地學到的差速補償：11-22 201101 出口後輪一直 37.4/42.6（右輪多 5%），
            #  第 2 圈前輪已經右打 +9 ~ +20° 車頭還一路往左（+94 → +105.7°），下坡偏左 38 cm 跑出去
            u = 0.0 if (self.elevated or getattr(self, "in_exit", False)) else self.diff_trim * v
            bwl, bwr = self._diff((bwl + bwr) / 2, (bwr - bwl) / 2 + u)
        elif up and v > 0 and self.ramp_d is not None and not self.cresting:
            # 坡上用後輪差速修方向（帶積分）：爬坡負載大時左右輪出力不同，車一直往右偏，
            # 前輪在坡上又抓不住（160004 / 154852 用 66% 爬，前輪左打 9–13° 車頭還是往右；152948 推 90% 沒偏）
            # 坡道是直的：車頭在轉就用差速擋回去（不靠視覺，坡底視覺方向不準）；量到坡道方向後才累加積分
            if self.ramp_dir is not None:
                self.ramp_i = max(-sl["ramp_i_max"], min(sl["ramp_i_max"], self.ramp_i + sl["ramp_ki"] * self.ramp_d * dt))
            u = sl["ramp_kp"] * self.ramp_d + self.ramp_i - sl["ramp_k_rate"] * self.yaw_rate
            u = max(-sl["ramp_diff_max"], min(sl["ramp_diff_max"], u))
            bwl, bwr = self._diff(v, u)
        elif down and sl.get("down_coast") and abs(v) < 1:
            bwl = bwr = 0.0
        elif down and self.down_d is not None:
            # 下坡差速：想往左 → 左輪反打多、右輪少（推的時候右輪多推）；兩輪平均不變
            u = max(-sl["down_diff_max"], min(sl["down_diff_max"], sl["down_kp"] * self.down_d))
            bwl, bwr = self._diff(v, u)
        else:
            bwl = bwr = v
        wm = self.safe.get("elevated_wheel_max") if self.safe else None
        if wm and not up and not down and self.mode != "turn" and self.boost == 0 and (self.elevated or self.after_dash(t)):
            # 平台上（高處、虛線左轉完到下坡前）急彎時外側輪不要衝太快（161638 差速讓外側輪到 67%）
            m = max(abs(bwl), abs(bwr))
            if m > wm:
                bwl, bwr = bwl * wm / m, bwr * wm / m
        if self.mode == "turn" and self.safe:
            # 4. 虛線左轉時外側輪不要太快（213809 外側輪 64%，車子轉得太猛、畫面變化衝到 50）
            m = max(abs(bwl), abs(bwr))
            if m > self.safe["turn_wheel_max"]:
                k = self.safe["turn_wheel_max"] / m
                bwl, bwr = bwl * k, bwr * k
        # 每個後輪的指令也漸進（每秒最多變 wheel_slew %）。平均速度本來就有 accel / decel 限制，
        #  但差速（前輪過 diff_from 才開、boost 歸零那一張才切回差速）和卡住推力是直接加在輪子上的：
        #  11-21 204006 22.37 s 兩輪 43/43 → 64/24 一張就跳；最新 3 趟平地單輪 > 100 %/s 的跳變 200 張，平均速度幾乎沒變的佔 121 張
        #  平地循線才限；反打（緊急煞車）、坡道、虛線左轉、剛出發（上一張停著）不限
        ws = sp.get("wheel_slew")
        pl, pr = self.cmd[1], self.cmd[2]
        if (ws and self.mode == "follow" and not slope and t >= self.brake_until
                and min(bwl, bwr) >= 0 and min(pl, pr) >= 0 and max(pl, pr) >= 1):
            lim = ws * dt
            bwl = pl + max(-lim, min(lim, bwl - pl))
            bwr = pr + max(-lim, min(lim, bwr - pr))
        self.cmd = (fwa, bwl, bwr)
        return self.cmd

    def _dash_track(self, s, t, yaw):
        """前方橫越虛線的距離追蹤（11-21 155804 改）：
        每張「看到」（≥ dash_min 塊排成橫線）的距離，要和預測（上次距離 − 車速 × 經過時間）差 dash_trk_tol 以內
        才算同一條虛線；一致的畫面累計 dash_hits 張（其中 dash_hits3 張 ≥ 3 塊）才確認。
        地上的虛線會隨車速穩定靠近；雜物、反光、標籤的距離會亂跳，累積不到。
        確認後用預測的距離決定什麼時候轉（塊數 2/3 跳動、某幾張沒看到都不影響），到 go_x 就左轉。"""
        jc, k = self.jc, self.dtrk
        v_ms = max(0.0, (self.cmd[1] + self.cmd[2]) / 2) / 100 * self.full_mps
        xp = None
        if k is not None:
            xp = k["x"] - k["r"] * (t - k["t"])
            k["x"], k["t"] = xp, t
        seen = s.dash >= jc["dash_min"] and s.dash_x is not None and 0.2 <= s.dash_x <= jc["dash_x_max"]
        if seen:
            if k is not None and abs(s.dash_x - xp) <= jc["dash_trk_tol"] + jc["dash_trk_rel"] * xp:
                # α-β：位置往量測拉一半；靠近速度也依誤差修（斜著進彎時虛線靠近得比車速快：155804 約 1.3 vs 0.9 m/s）
                res, dt_u = s.dash_x - xp, max(0.03, t - k["t_seen"])
                k["x"] = xp + 0.5 * res
                k["r"] = max(max(0.2, 0.6 * v_ms), min(max(1.0, 2.0 * v_ms), k["r"] - 0.1 * res / dt_u))
                k["hits"] += 1
                k["hits3"] += int(s.dash >= 3)
                k["t_seen"], k["x_seen"] = t, k["x"]
            elif (k is None or t - k["t_seen"] > jc["dash_lost_s"] or (not k["ok"] and k["hits"] <= 1)
                  or (not k["ok"] and k["hits3"] == 0 and s.dash >= 3)):
                # 還沒確認、只有 2 塊證據的追蹤，遇到 3 塊的新虛線就換過去（195325 先鎖到 0.6 m 的雜點，
                #  真的虛線 0.89 m 一路被丟掉，到車頭下才轉）
                k = self.dtrk = dict(x=s.dash_x, r=max(0.2, v_ms), t=t, t_seen=t, x_seen=s.dash_x, hits=1,
                                     hits3=int(s.dash >= 3), ok=False)
        if os.environ.get("DASH_DEBUG") and (k is not None or seen):
            print(f"  [dash] t={t:.2f} n={s.dash} 量={s.dash_x if s.dash_x is None else round(s.dash_x, 2)} "
                  f"預測={None if xp is None else round(xp, 2)} 追={None if k is None else (round(k['x'], 2), round(k['r'], 2), k['hits'], k['hits3'], k['ok'])} v={v_ms:.2f}")
        if k is None:
            return
        lost = t - k["t_seen"] > jc["dash_lost_s"]
        if not k["ok"]:
            if lost:
                self.dtrk = None
            elif k["hits"] >= jc["dash_hits"] and k["hits3"] >= jc["dash_hits3"]:
                k["ok"], self.t_dash_seen = True, t
                print(f"[driver] 確認前方橫越的虛線（{k['x']:.2f} m，{k['hits']} 張距離一致）：等靠近再左轉")
            return
        near = k["x_seen"] <= jc["go_x"] + jc["dash_near_m"]
        if k["x"] <= jc["go_x"] or lost:
            if not lost or near:
                # 到了（或虛線很近時跑到鏡頭下面看不到了）：左轉
                self.mode, self.t_turn, self.yaw_turn0 = "turn", t, yaw
                print(f"[driver] 虛線到了（{k['x']:.2f} m）：左轉")
            else:
                print(f"[driver] 虛線還很遠就不見了（最後 {k['x_seen']:.2f} m）：取消左轉")
            self.t_dash_seen, self.dtrk = None, None

    # ------------------------------------------------------------------ 地圖
    def _map_step(self, t, pitch, yaw, up_start, down_start):
        if self.loc is None:
            return
        if self.loc.p is None:
            if self.mc.get("start_anywhere"):
                self.loc.reset(0.0, self.tmap.L)
            else:
                self.loc.reset()
        running = self.state in (RUN, LOST)
        dt = 0.0 if getattr(self, "t_map", None) is None else max(0.0, min(0.2, t - self.t_map))
        self.t_map = t
        v = (self.cmd[1] + self.cmd[2]) / 2 if running else 0.0
        m = self.s.motion if self.s is not None else 9.0
        if m < self.mc.get("frozen_m", 0.7):
            self.t_frozen = t if self.t_frozen is None else self.t_frozen
        else:
            self.t_frozen = None
        if self.t_frozen is not None and t - self.t_frozen > 0.4:
            v = 0.0                   # 畫面幾乎完全不動（真的卡住，0.4–0.6）：位置不要再往前算
            # （平地花紋均勻時正常跑畫面變化也常 < 1.5，不能用一般的卡住判斷：會讓定位一圈落後 20 m）
        vms = trackmap.odo_speed(v, pitch, self.tmap.m.get("full_mps", 1.57))
        self.odo += vms * dt
        self.loc.step(dt, vms, yaw)
        if up_start and running and (self.map_lap or self.map_fork_trust) and os.environ.get("MAP_DEBUG"):
            print(f"[driver] 上坡起點：定位原本在 {self.loc.s:.1f} m（誤差 {(self.loc.s + self.tmap.L / 2) % self.tmap.L - self.tmap.L / 2:+.1f} m）")
        # 上坡起點整圈唯一：但要「真的上坡」才定位（俯仰在 up_anchor_m 內到過 up_anchor_pitch）。
        #  11-22 203956：IMU 校正偏 +4°，起步加速俯仰讀到 +9.2° 被當成上坡起點，定位跳到 0 m（實際約 18 m），
        #  地圖以為不在岔路附近 → 30.4 m 的 V 被忽略、沒走左邊。真的坡道俯仰 ≥ 15.6°（202516），起步最多 5.2°。
        if up_start and running:
            self.up_pend = self.odo
        pend = getattr(self, "up_pend", None)
        if pend is not None:
            d_up = self.odo - pend
            if d_up > self.mc.get("up_anchor_m", 1.5) or not running:
                print(f"[driver] 上坡起點不採用：{d_up:.1f} m 內俯仰沒到 {self.mc.get('up_anchor_pitch', 12)}°（起步 / 顛簸）")
                self.up_pend = None
            elif pitch is not None and pitch >= self.mc.get("up_anchor_pitch", 12):
                self.up_pend = None
                if self.loc.anchor(d_up, 0.3, gate=1e9):          # 確認時已經走了 d_up m：定位設在 d_up
                    self.map_dash_lap = False
                    self.map_lap = True       # 這一圈的上坡有對到地圖
        if down_start and self.tmap.m.get("down") is not None and running:
            # 下坡起點整圈也只有一個：直接定位（任何地方出發時也能確定位置）
            trusted = self.map_lap or self.map_fork_trust
            if self.loc.anchor(self.tmap.m["down"], 0.3, gate=None if trusted else 1e9) and not trusted:
                self.map_fork_trust = True
                print(f"[driver] 地圖：下坡起點 = {self.tmap.m['down']:.1f} m，定位確定，開始用地圖")

    # ---- 地標序列定位（任何地方出發）：V、隧道入口；上坡 / 下坡起點另外直接定位
    def _landmarks(self):
        if getattr(self, "lm", None) is None:
            m = self.tmap.m
            fa = next((f["s"] for f in self.tmap.forks if f["side"] == "right"), None)
            self.lm = {"v": [f["s"] for f in self.tmap.forks] + ([fa + self.mc.get("fork_second_v_s", 2.2)] if fa is not None else []),
                       "dark": [d[0] for d in m.get("dark", [])]}
        return self.lm

    def _landmark(self, typ):
        """看到一個地標：記下行駛距離；最近幾個地標彼此的距離只有一種地圖位置解釋得通 → 定位確定。"""
        lm, L = self._landmarks(), self.tmap.L
        if self.lm_ev and self.lm_ev[-1][0] == typ and self.odo - self.lm_ev[-1][1] < self.mc.get("lm_merge_m", 1.0):
            return                                            # 同一個地標閃爍
        self.lm_ev = (self.lm_ev + [(typ, self.odo)])[-4:]
        if self.map_lap or self.map_fork_trust or len(self.lm_ev) < 2:
            return
        o = self.odo
        circ = lambda a, b: abs((a - b + L / 2) % L - L / 2)
        cands = []
        for c in lm.get(typ, []):
            ok, matched = True, 0
            for pt, po in self.lm_ev[:-1]:
                d = o - po
                if d > self.mc.get("lm_max_m", 45.0):
                    continue
                ps = (c - d) % L
                tol = self.mc.get("lm_tol0", 1.5) + self.mc.get("lm_tol_rel", 0.15) * d
                if any(circ(x, ps) <= tol for x in lm.get(pt, [])):
                    matched += 1
                else:
                    ok = False
                    break
            if ok and matched >= 1:
                cands.append(c)
        desc = "、".join(f"{'V' if pt == 'v' else '隧道'}@{o - po:.1f}m前" for pt, po in self.lm_ev[:-1])
        if cands and max(circ(c, cands[0]) for c in cands) <= 1.0:
            self.loc.anchor(cands[0], 0.3, gate=1e9)
            self.map_fork_trust = True
            print(f"[driver] 地圖：地標序列（{desc} → 現在{'V' if typ == 'v' else '隧道'}）只對得上 {cands[0]:.1f} m，定位確定，開始用地圖")
        else:
            print(f"[driver] 地圖：地標序列（{desc} → 現在{'V' if typ == 'v' else '隧道'}）對得上 {len(cands)} 個位置，還不確定")

    def map_ok(self):
        """可以用地圖做決定：定位確定，而且（任何地方出發時）出發後已經過一次上坡起點對過地圖。
        還沒過上坡前完全照影像跑（從哪裡出發都一樣安全）。"""
        if self.loc is None or not self.loc.ok or not self.mc.get("use", True):
            return False
        return self.map_lap or self.map_fork_trust or not self.mc.get("start_anywhere")

    def map_near(self, target, before, after):
        return self.tmap.near(self.loc.s, target, before, after)

    def map_guard(self):
        """現在在 guard 區段內：回傳前方 1 m 的平均彎度換成的右打量（°/m，正 = 往右彎），否則 None。"""
        s = self.loc.s
        for a, b in self.mc.get("guard", []):
            if (s - a) % self.tmap.L <= (b - a) % self.tmap.L:
                i0 = int(s / self.tmap.ds)
                idx = (i0 + np.arange(int(1.0 / self.tmap.ds))) % self.tmap.n
                return -float(np.mean(self.tmap.kappa[idx]))
        return None

    def _fork_by_landmarks(self):
        """地圖還不能用時，用最近看過的地標認岔路（地圖上每個岔路前面看到的東西不一樣）：
          ・fork_lm_v_m 內剛過一個 V → 同一個岔路口的第二個 V（39.8 m 後 2.2 m 還有一個），照剛才那邊
          ・fork_lm_dark_m 內剛過隧道 → 前面有隧道的岔路（30.4 m，隧道 27.4–27.9 m）
          ・都沒有 → 前面沒有隧道的岔路（39.8 m）
        回傳 (side, 說明) 或 None（沒有地圖 / 認不出來 → 照影像尖端）。"""
        mc = self.mc
        if self.tmap is None or not mc.get("fork_lm", True) or not self.tmap.forks:
            return None
        dark_m, v_m = mc.get("fork_lm_dark_m", 6.0), mc.get("fork_lm_v_m", 4.0)
        merge = mc.get("lm_merge_m", 1.0)
        seen_v = seen_dark = None
        for typ, o in reversed(self.lm_ev):
            d = self.odo - o
            if typ == "v" and d <= merge:
                continue                                  # 就是現在這個 V
            if typ == "v" and seen_v is None:
                seen_v = d
            if typ == "dark" and seen_dark is None:
                seen_dark = d
        if seen_v is not None and seen_v <= v_m and self.fork_side:
            return self.fork_side, f"{seen_v:.1f} m 前剛過 V（同一個岔路口）"
        L = self.tmap.L
        dark_ends = [d[1] for d in self.tmap.m.get("dark", [])]
        has_dark = lambda f: any((f["s"] - e) % L <= dark_m for e in dark_ends)
        near = seen_dark is not None and seen_dark <= dark_m
        cands = [f for f in self.tmap.forks if has_dark(f) == near]
        if len({f["side"] for f in cands}) != 1:
            return None
        f = cands[0]
        why = (f"{seen_dark:.1f} m 前剛過隧道" if near else f"前 {dark_m:.0f} m 內沒有隧道") + f" = {f['s']:.1f} m 的岔路"
        return f["side"], why

    def _in_map_guard(self):
        """現在在 map.guard（地圖方向保護區段）裡。"""
        if self.loc is None or not self.map_ok():
            return False
        return any(self.tmap.near(self.loc.s, (a + b) / 2, (b - a) / 2, (b - a) / 2) for a, b in self.mc.get("guard", []))

    def _map_curve_v(self):
        """現在在 map.curve_v 指定的彎裡：回傳那個彎的速度，否則 None。"""
        if self.loc is None or not self.map_ok():
            return None
        for a, b, zv in self.mc.get("curve_v", []):
            if self.tmap.near(self.loc.s, (a + b) / 2, (b - a) / 2, (b - a) / 2):
                return zv
        return None

    def map_fork(self):
        for f in self.tmap.forks:
            if self.map_near(f["s"], self.mc.get("fork_before", 1.5), self.mc.get("fork_after", 1.5)):
                return f
        return None

    def _obstacle(self, frame, s, t, pitch, slope):
        """前車偵測 + 連續確認。坡道區不偵測（坡上鏡頭和前面路面不在同一平面，沒車影片的誤判幾乎都在那裡）。"""
        self.obs_raw = None
        if self.obs_det is None:
            self.obs_d = None
            return
        oc = self.oc
        skip = slope or self.elevated or (pitch is not None and abs(pitch) >= oc.get("max_pitch", 5))
        if not skip and self.map_ok():
            for a, b in oc.get("skip_zones", []):
                if self.tmap.ahead(a, self.loc.s) <= (b - a) % self.tmap.L:
                    skip = True
                    break
        if not skip:
            small = cv2.resize(frame, (self.eye.W, self.eye.H), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            self.obs_raw, _ = self.obs_det.process(gray, s.center, s.far if s.center is not None else None)
        self.obs_hist.append(self.obs_raw)
        ds = [x for x in self.obs_hist if x is not None]
        ok = len(ds) >= oc.get("need", 4) and max(ds) - min(ds) <= oc.get("tol", 0.15)
        self.obs_d = float(np.median(ds)) if ok else None
        if ok and not self.obs_on:
            print(f"[driver] 前方有車 {self.obs_d:.2f} m" + ("" if oc.get("act", False) else "（只記錄，不減速）"))
        self.obs_on = ok

    def after_dash(self, t):
        """虛線左轉完、還沒開始下坡（最多 after_s 秒）。"""
        return (0 <= t - self.t_turn_done < self.jc.get("after_s", 0)
                and self.t_down_start < self.t_turn_done)

    def _far_dropping(self, s, t):
        """遠處亮度一直往下掉 = 前面要進隧道（161638：1.3 秒內 58.8 → 41.8，原本 < 44 才認，進去前 0.1 秒才降速）。
        過去 1 秒的最大值減現在 ≥ tunnel_drop，而且現在 < tunnel_far_max。"""
        sp = self.sp
        self.far_hist.append((t, s.far_bright))
        while self.far_hist and t - self.far_hist[0][0] > 1.0:
            self.far_hist.popleft()
        if "tunnel_drop" not in sp:
            return False
        hi = max(b for _, b in self.far_hist)
        return (s.far_bright < sp["tunnel_far_max"] * s.dark_k
                and hi - s.far_bright >= sp["tunnel_drop"] * s.dark_k)

    def steer_out(self, fwa):
        """舵機左右不對稱補償：從置中算的偏轉量，往右乘 steer_k_right、往左乘 steer_k_left。
        11-21 維修後地上實測（前輪 ±25°、35%）：右轉 49°/s、左轉 20°/s（11-19 是 25 / 19）。"""
        sc = self.sc
        d = fwa - sc["fwa_trim"]
        # 置中附近 ±steer_k_from 度內不放大（直線上的小修正左右一樣），超過的部分才乘 k（彎道才補）
        #  11-21 163926：整段乘 1.6，直線右修被放大 → 左右來回蛇行（前輪 sd 12.6°、車頭轉速 sd 20°/s；
        #  11-19 k = 1 時 4–6° / 7–11°/s）
        d0 = sc.get("steer_k_from", 0.0)
        k = sc.get("steer_k_right", 1.0) if d > 0 else sc.get("steer_k_left", 1.0)
        a = abs(d)
        out = a if a <= d0 else d0 + (a - d0) * k
        return sc["fwa_trim"] + math.copysign(out, d)

    def elevated_down_ok(self, t):
        """左轉時往下掉是不是正常的（平台左轉後馬上接下坡）：目前一律視為不正常。"""
        return False

    # ------------------------------------------------------------------ 方向
    def _pursuit(self, s, v):
        """回傳 (前輪角度, 目標點)。目標點在中線上、離後輪軸 ld 的地方。"""
        sc = self.sc
        if s.center is None:
            return self.fwa, None
        ld = (sc["ld_dark"] if s.dark else sc["ld"]) + sc["ld_per_v"] * max(0.0, v) / 100
        if self.sev <= self.sp["sev_low"]:
            # 直線：看遠一點，打方向比較平順（191431 直線蛇行 ±15 cm 撞起點柱子；模擬加了延遲後前視短就會擺）
            ld += sc["ld_straight"]
        pts = s.center.copy()
        pts[:, 0] += self.g["cam_to_axle"]
        tgt = self._point_at(pts, ld)
        d = self._steer_deg(tgt)
        # 車道方向前饋（Stanley）：車道轉了多少前輪就先打多少，加上偏移修正。和 Pure Pursuit 各半。
        # 194747 髮夾彎：車道已經轉 −29°，Pure Pursuit 只打 11°，車子一路壓著外側線
        if sc["k_head"] > 0:
            v_ms = 0.3 + 0.02 * max(0.0, v)
            d_st = s.heading + math.degrees(math.atan2(sc["k_off"] * s.offset, v_ms))
            d_pp = d
            d = (1 - sc["k_head"]) * d + sc["k_head"] * d_st
            # 直線 / 緩彎（上一張彎的程度 < sev_high）轉向用 k_off_straight：11-23 193042 直線蛇行週期約 2.7 s、偏移 ±10–13 cm；
            #  指令到車頭轉動延遲約 10 張（0.33 s），閉迴路模擬 k_off 1.4 → 1.0 過衝約減半。
            #  只改轉向，速度判斷（彎的程度）照舊用 d（回放：彎的程度一起變小會讓岔路區速度多 10%，等於沒經同意提速）；
            #  地圖指定的彎（curve_v）、地圖方向保護區（guard）照舊（回放：坡底右緩彎 12.7 m 入彎前輪 +9° → 0°）
            d_steer = d
            if (sc.get("k_off_straight") is not None and self.sev < self.sp["sev_high"]
                    and self._map_curve_v() is None and not self._in_map_guard()):
                d_st2 = s.heading + math.degrees(math.atan2(sc["k_off_straight"] * s.offset, v_ms))
                d_steer = (1 - sc["k_head"]) * d_pp + sc["k_head"] * d_st2
        else:
            d_steer = d
        # 彎的程度也要算「車頭和車道差幾度」：202536 撞柱前車道已經差 −20~−27°，彎只算 7–13 → 用 50% 衝進彎
        # 看前方多遠的彎：速度越快看越遠，才來得及漸進減速
        pv = self.sp["preview"] + self.sp.get("preview_per_v", 0.0) * max(0.0, v) / 100
        self.sev = max(abs(d), abs(self._steer_deg(self._point_at(pts, pv))),
                       self.sp.get("sev_head", 0.0) * abs(s.heading))
        tgt_cam = (tgt[0] - self.g["cam_to_axle"], tgt[1])
        return sc["fwa_trim"] + self.trim_i - sc["gain"] * d_steer, tgt_cam

    @staticmethod
    def _point_at(pts, dist):
        """中線上離原點 dist 的點；中線比 dist 短就沿最後一段延長。"""
        r = (pts[:, 0] ** 2 + pts[:, 1] ** 2) ** 0.5
        if r[-1] >= dist:
            i = max(1, int((r >= dist).argmax()))
            if r[i - 1] >= dist:
                return pts[0]
            k = (dist - r[i - 1]) / max(1e-6, r[i] - r[i - 1])
            return pts[i - 1] + k * (pts[i] - pts[i - 1])
        if len(pts) >= 2:
            d = pts[-1] - pts[-2]
            n = (d[0] ** 2 + d[1] ** 2) ** 0.5
            if n > 1e-6:
                return pts[-1] + d / n * (dist - r[-1])
        return pts[-1]

    def _steer_deg(self, p):
        """Pure Pursuit：往左為正的前輪角（度）。"""
        x, y = float(p[0]), float(p[1])
        l2 = x * x + y * y
        if l2 < 1e-4:
            return 0.0
        return math.degrees(math.atan(2 * self.g["wheelbase"] * y / l2))

    # ------------------------------------------------------------------ 速度
    def _speed(self, s, t, dt, up, down, running):
        sp, sl = self.sp, self.sl
        why = []
        # 1. / 2. 彎道 / 直線
        sev = self.sev
        # 地圖上指定的彎（map.curve_v）用自己的彎道速度：只放在「已經證明轉得過去」的彎
        #  （11-22：最新 3 趟 50–60% 過彎側向加速度 90 百分位 0.90–0.94 m/s²、偏移 90 百分位 11–16 cm；
        #   緩彎 R ≈ 1.4 m 用 65% 只要 0.75 m/s²，在已證明的範圍內；髮夾彎 R 0.6–0.7 m 用 65% 要 1.5–1.7 m/s²，沒證據，維持原速）
        vc_img, vt_img = sp["v_curve"], sp["v_sat"]
        zone_v = self._map_curve_v()
        if zone_v is not None:
            vc_img = vt_img = zone_v
        straight_v = sp["v_straight"] if (s.conf == 2 and s.far >= sp["straight_far"]) else sp["v_mid"]
        if sev >= sp["sev_sat"]:
            v, w = vt_img, "急彎"
        elif sev >= sp["sev_high"]:
            v, w = vc_img, "彎道"
        elif sev <= sp["sev_low"]:
            v, w = straight_v, "大直線" if straight_v == sp["v_straight"] else "直線"
        else:
            k = (sev - sp["sev_low"]) / (sp["sev_high"] - sp["sev_low"])
            v, w = straight_v + k * (vc_img - straight_v), "進彎"
        if self.map_ok():
            # 影像為主、地圖輔助（11-21 使用者要求）：
            #  1. 地圖看到前方有彎 → 提早減速（取小的）
            #  2. 地圖說前方直 / 彎得緩 → 只有影像也確認（兩條線、靠中間、車頭正）才拉高到地圖的速度
            #  （181722 第 3 圈：地圖直線推 100%，影像只有一條線、偏 13 cm，衝出去撞邊）
            mc = self.mc
            look = mc.get("preview", 1.0) + mc.get("preview_per_v", 1.2) * max(0.0, self.v) / 100
            kmax = self.tmap.max_kappa_ahead(self.loc.s - mc.get("behind", 0.3), look + mc.get("behind", 0.3))
            vs, vc, vt = mc.get("v_straight", sp["v_straight"]), mc.get("v_curve", sp["v_curve"]), mc.get("v_sat", sp["v_sat"])
            if zone_v is not None:
                vc = vt = zone_v
            if kmax <= mc["k_straight"]:
                vm = vs
            elif kmax >= mc["k_sat"]:
                vm = vt
            elif kmax >= mc["k_curve"]:
                vm = vc + (kmax - mc["k_curve"]) / (mc["k_sat"] - mc["k_curve"]) * (vt - vc)
            else:
                vm = vs + (kmax - mc["k_straight"]) / (mc["k_curve"] - mc["k_straight"]) * (vc - vs)
            # 地圖沒記到的暗處 / 隧道口（map.slow_zones）：前方 look m 內有就先降到該速度
            #  （11-22 194748：s ≈ 20.3 兩圈都出現隧道口；第一圈 44% 進去沒事，第二圈大直線剛跑到 100%、59.6% 進去，
            #   暗處看錯偏移 +64 cm，前輪 −37 → +44°、後輪 89/14，衝出賽道）
            #  離區段越近限越低（每 m 多 slow_k %），一路漸進降，不會一次降很多觸發反打
            for a, b, zv in mc.get("slow_zones", []):
                if self.tmap.near(self.loc.s, (a + b) / 2, (b - a) / 2 + look, (b - a) / 2):
                    d_in = max(0.0, (a - self.loc.s + self.tmap.L / 2) % self.tmap.L - self.tmap.L / 2)
                    zcap = zv + mc.get("slow_k", 25.0) * d_in
                    if zcap < vm:
                        vm = zcap
                        w += "/地圖暗處"
            good = (s.conf == 2 and abs(s.offset) <= mc.get("boost_off", 0.08)
                    and abs(s.heading) <= mc.get("boost_head", 8.0) and not s.dark)
            # 只看到一條線但很正：也可以加速，但最多 boost_one_v
            good1 = (s.conf == 1 and abs(s.offset) <= mc.get("boost_one_off", 0.06)
                     and abs(s.heading) <= mc.get("boost_one_head", 6.0) and not s.dark)
            if getattr(self, "in_hold", False) and getattr(self, "hold_fork", False):
                # 岔路區維持方向中：影像看到的是 V 的線（方向亂跳 → 被當成進彎 55%），不拿來決定速度，照地圖加速
                #  （上限 fork_hold_v）。11-23 121713 上限 65 時岔路區實際只有 55–64%，「岔路 / 進彎」一直壓著
                good = True
            near_ramp = self.loc.s >= self.tmap.L - mc.get("ramp_approach", 1.5) or self.loc.s < 0.3
            if vm < v and not near_ramp:      # 上坡前 ramp_approach m 不減速：要衝力上坡（185357 坡底被壓到 40% 卡住）
                v, w = vm, w + "/地圖減速"
            elif vm > v and good:
                v, w = vm, w + "/地圖加速"
            elif min(vm, mc.get("boost_one_v", 80)) > v and good1:
                v, w = min(vm, mc.get("boost_one_v", 80)), w + "/地圖加速"
        if self.map_ok() and self.mc.get("pre_ramp"):
            # 上坡前的右急彎：慢慢過、出彎在車道中間再上坡（192638：40–45% 外側輪 73%，出彎被甩到左邊 16 cm 才上坡，
            #  坡上一路偏左；歷次進坡前都偏左 10–20 cm）
            a0, b0 = self.mc["pre_ramp"]
            at_foot = self.pitch is not None and self.pitch >= sl["foot_pitch"]   # 已經到坡腳：放開，要衝力（194359 坡底卡 1.7 s）
            if (not at_foot and (self.loc.s - a0) % self.tmap.L <= (b0 - a0) % self.tmap.L
                    and v > self.mc.get("pre_ramp_v", 32)):
                v, w = self.mc.get("pre_ramp_v", 32), w + "/坡前彎慢行"
        why.append(w)
        sf = self.safe
        if sf and s.conf > 0 and not (up or down):
            # 5. 靠邊漸進減速：離中線 edge_off0 開始降，到 edge_off 降到 edge_v（高處 edge_v_high），修回中間再加速
            #  （11-21 161139：右彎常跑寬 15–25 cm 壓線；原本 20 cm 才一次降到 40，來不及）
            lo = sf["edge_v_high"] if self.elevated else sf["edge_v"]
            e0, e1 = sf.get("edge_off0", sf["edge_off"]), sf["edge_off"]
            a_off = abs(s.offset)
            if a_off >= e0:
                k = 1.0 if a_off >= e1 or e1 <= e0 else (a_off - e0) / (e1 - e0)
                cap = sp["v_curve"] + k * (lo - sp["v_curve"])
                if cap < v:
                    v = cap
                    why.append("靠邊")
        if self.elevated and not up and sf.get("elevated_v"):
            # 6. 在高處（上完坡、還沒下坡：平台）整段限速：平台出口就是下坡，先慢下來再下坡
            #  （11-21 161139 用 53% 衝進下坡，只能猛反打）
            if v > sf["elevated_v"]:
                v = sf["elevated_v"]
                why.append("高處")
        if self.after_dash(t) and not up:
            # 7. 虛線左轉完到下坡前：再慢一點（11-21 161638 左轉後急彎外側輪衝到 67%，進下坡偏 44 cm）
            if v > self.jc["v_after"]:
                v = self.jc["v_after"]
                why.append("轉完慢行")
        if s.dark:
            v = min(v, sp["v_dark"])
            why.append("暗處")
            self.t_tunnel_vis = t
        elif s.far_bright < self.eye.v["far_dark"] * s.dark_k or self._far_dropping(s, t):
            # 隧道口：遠處先變暗就先降速（190231 用 52–58% 衝進隧道，進去才降；遠處 < 44 約在進隧道前 0.5 s，
            #  一般路段只有 3% 的時間會這樣）
            v = min(v, sp["v_dark"])
            why.append("隧道口")
            self.t_tunnel_vis = t
        if self.n_fork >= 1 or t < self.fork_until:
            # 岔路口：看到 V 分隔線就把目標降下來（配合下面的漸進減速，慢慢降、過了再慢慢加）
            vf = self.jc["v_fork"]
            if getattr(self, "in_hold", False) and getattr(self, "hold_fork", False):
                # 岔路區維持方向中（地圖上是直線、陀螺儀維持方向）：用 fork_hold_v（11-23 使用者要求岔路直線加速）
                vf = self.mc.get("fork_hold_v", vf)
            v = min(v, vf)
            why.append("岔路")
        if t - getattr(self, "t_fork_cross", -1e9) < self.jc.get("cross_s", 1.5):
            v = min(v, self.jc.get("v_cross", 35))
            why.append("岔路橫切")
        if self.state == LOST:
            v = sp["v_lost"]
            why.append("找線")
        if self.mode == "turn":
            v = self.jc["v"]
            why = ["虛線左轉"]
        elif self.t_dash_seen is not None:
            v = min(v, self.jc["v"])
            why = ["虛線直走"]
        if self.t_go is not None and t - self.t_go < sp["start_s"]:
            v = min(v, sp["start_v"])
            why.append("起步")
        if t - self.t_up_end < sl["top_s"] and not up:
            v = min(v, sl["top_v"])
            why.append("坡頂")
        if t - self.t_down_end < sl["bottom_s"] and not down:
            v = min(v, sl["bottom_v"])
            why.append("坡底")
        if getattr(s, "glitch", False):
            why.append("看錯")

        # 加速限制、減速反打
        prev = self.v
        if v > prev:
            v = min(v, prev + sp["accel"] * dt) if prev > 0 else min(v, sp["start_v"])
        elif v < prev:
            if (prev - v >= sp["brake_dv"] and not (up or down) and self.boost == 0 and v <= sp["v_curve"] + 5
                    and t - self.t_up_end >= self.sl["top_s"]):  # 坡頂不反打（還在坡邊，反打會往後滑）
                self.brake_until = t + sp["brake_s"]               # 一次要降很多（緊急）才反打
            elif (self.mode == "follow" and self.state == RUN and not (up or down) and prev > 0
                  and t - self.t_down_end >= sl["bottom_s"]):
                # 進彎、岔路口漸進減速：每秒最多降 decel %
                v = max(v, prev - sp["decel"] * dt)
        braking = t < self.brake_until and not (up or down)

        # 卡住（畫面幾乎不動）：遲滯判斷，慢慢加推力
        # 動起來要連續 moving_n 張都夠大才算（200621 坡上卡住，輪子打滑畫面晃一下就把推力歸零，又停住）
        self.n_moving = self.n_moving + 1 if s.motion >= sl["moving_m"] else 0
        # 坡上畫面本來就變化小（坡道近、花紋均勻）：152948 第一圈 62% 爬 1.4 秒畫面變化 1.6–2.9，沒算卡住、幾乎停住
        starting = self.t_go is not None and t - self.t_go < sp["start_s"] + 1.0 and not up
        # 起步：要真的沒在動（畫面變化 < start_stuck_m）才算卡住（184602 / 175457 起步誤判卡住多推 6%，往右拉撞右側）
        # 下坡滑行時本來就慢（畫面變化 1.1–2.7）：要 < down_stuck_m 才算卡住（173105 滑行被當卡住，坡上推到 55%）
        if s.motion < (sl["stuck_m_up"] if up else sl["down_stuck_m"] if (down and sl.get("down_coast"))
                       else sl["start_stuck_m"] if starting else sl["stuck_m"]):
            self.t_still = t if self.t_still is None else self.t_still
        elif self.n_moving >= sl["moving_n"]:
            self.t_still = None
        # 只在坡上 / 坡底、剛出發時才加推力：平地直線上地墊花紋很均勻，畫面變化也會很小（不是卡住）
        # 急彎卡住：前輪打很大、車頭卻完全沒在轉 = 真的卡住（平地花紋均勻造成的「畫面不動」不會這樣）
        #  160927 51 s 前輪 −39°、內側輪只剩 6%，停在原地，平地不能加推力就一直卡著
        turn_stuck = (abs(self.fwa - self.sc["fwa_trim"]) >= sl["turn_stuck_fwa"]
                      and abs(self.yaw_rate) < sl["turn_stuck_rate"])
        # 坡頂邊緣：剛上完坡、或在高處俯仰還沒回平（後輪還在坡道轉折上）也可以加推力
        #  （11-21 162248 停在坡頂俯仰 3.6°、38–42% 推不上去卡 6 秒：不算上坡也不算坡底，加推力被關掉）
        crest = self.elevated and (t - self.t_up_end < sl["crest_s"]
                                   or (self.pitch is not None and self.pitch >= sl["crest_pitch"]))
        # 坡頂用比較嚴的「真的不動」：畫面變化 < crest_still_m 連續 crest_still_s 秒
        #  （真的卡住 0.4–0.6；平台上正常跑 1.6–3.8，偶爾掉到 1.5 以下，用一般門檻會誤推到 70%：212647 / 203451 重播）
        if s.motion < sl["crest_still_m"]:
            self.t_crest_still = t if self.t_crest_still is None else self.t_crest_still
        else:
            self.t_crest_still = None
        crest = crest and self.t_crest_still is not None and t - self.t_crest_still >= sl["crest_still_s"]
        can_boost = up or down or self.n_up > 0 or (self.t_go is not None and t - self.t_go < sp["start_s"] + 1.0)             or (self.pitch is not None and self.pitch >= sl["foot_pitch"]) or turn_stuck or crest
        stuck = running and can_boost and self.t_still is not None and t - self.t_still >= sl["stuck_s"]
        if stuck and not braking:
            # 坡上卡住加推力快一點（185357 每秒 40% 要 1 秒以上才推得動）
            rate = sl.get("up_boost_per_s", sl["boost_per_s"]) if up else sl["boost_per_s"]
            self.boost = min(sl["boost_max"], self.boost + rate * dt)
        else:
            # 坡上推力慢慢退（剛推動就退光會再停住）
            k = 0.5 if (up or self.n_up > 0) else 2.0
            self.boost = max(0.0, self.boost - k * sl["boost_per_s"] * dt)

        # 3. 上坡
        if up:
            v = max(v, sl["up_v"])
            why = ["上坡"]
            if self.up_mis:
                v = min(v, sl["up_align_v"])
                why = ["上坡/對正中"]
        # 5. 下坡
        if down and sl.get("down_coast"):
            # 下坡滑行 + 適時輕反打（11-21 使用者方案）：慢的時候滑行（馬達 0），畫面變化超過 down_coast_m（開始變快）
            #  才反打，越快打越多（down_brake_min → down_brake），用低通讓反打量連續變化、不要開關式切換
            #  （152406：反打 / 滑行來回切，坡上速度忽快忽慢、車左右甩出線；163209 / 163405 全滑行：畫面變化中位 11、
            #  車頭甩 56–98°，一路反打的 203451 / 210913 只有 4–7、22–38°）
            m = s.motion
            if m < sl["down_coast_m"]:
                tgt = 0.0
                # 坡下半段（俯仰比 −down_foot_pitch 緩）而且已經很慢：輕推，不要滑到停在坡底
                #  （11-23 113124 兩次坡底：俯仰 −13 ~ −8° 還在滑行（馬達 0），畫面變化 0.65–1.9，停頓 0.4–0.7 s）
                if (sl.get("down_foot_v", 0) > 0 and self.pitch is not None
                        and self.pitch > -sl.get("down_foot_pitch", 14) and m < sl.get("down_foot_m", 2.0)):
                    tgt = sl["down_foot_v"]
            else:
                k = min(1.0, (m - sl["down_coast_m"]) / max(1e-3, sl["down_fast_m"] - sl["down_coast_m"]))
                tgt = sl["down_brake_min"] + k * (sl["down_brake"] - sl["down_brake_min"])
            a = sl.get("down_smooth", 0.6)
            self.down_cmd = a * self.down_cmd + (1 - a) * tgt
            v = self.down_cmd
            why = [("下坡輕推" if v > 1 else "下坡滑行") if v > -1 else f"下坡輕反打{v:.0f}"]
            if stuck and self.pitch is not None and self.pitch <= -sl["down_stuck_pitch"]:
                v = max(v, min(sl["down_stuck_v"], sl["down_v"] + 2 * self.boost))
                why = ["下坡卡住"]
        elif down:
            # 下坡一路反打後輪：畫面變化越大（越快）反打越多，太慢才推。
            # 152406 第二圈：反打 −10 / 滑行來回切，坡上速度 10–30（平地 3–6），車左右甩出線
            m = s.motion
            if m < sl["down_push_m"]:
                v, w = sl["down_v"], "下坡推"
            else:
                k = min(1.0, (m - sl["down_push_m"]) / max(1e-3, sl["down_fast_m"] - sl["down_push_m"]))
                v = sl["down_brake_min"] + k * (sl["down_brake"] - sl["down_brake_min"])
                w = f"下坡反打{v:.0f}"
            why = [w]
            # 真的還在陡坡上才算卡住（坡底坡度變緩、反打後畫面變化小，不是卡住）
            if stuck and self.pitch is not None and self.pitch <= -sl["down_stuck_pitch"]:
                v = max(v, min(sl["down_stuck_v"], sl["down_v"] + 2 * self.boost))
                why = ["下坡卡住"]
        elif self.boost > 0:
            if up and self.up_mis:
                cap = sl["up_align_stuck_v"] if self.boost >= sl["up_align_boost"] else sl["up_align_v"]
            else:
                cap = sl["up_cap"] if up else 80.0
            if crest:
                cap = min(cap, sl["crest_stuck_cap"])         # 坡頂邊緣卡住：最多到這個
            elif not up and self.t_go is not None and t - self.t_go < sp["start_s"] + 1.0:
                cap = sp["start_v"] + sp["start_boost"]       # 起步：只多加一點（不然平地起步會衝到 80%）
            elif not up and not down and not (self.pitch is not None and self.pitch >= sl["foot_pitch"]):
                cap = min(cap, sl["turn_stuck_cap"])          # 平地急彎卡住：最多到這個
            v = min(cap, v + self.boost)
            why.append(f"卡住+{self.boost:.0f}")
        # 上坡速度漸進提升（每秒最多 up_accel %），給車頭回正的時間；往下降不限制
        #  210415：一認到上坡 46% → 70% 瞬間跳、對正中 / 全力 50% ↔ 86% 來回跳，車頭還沒回正就衝出去
        if up or self.n_up > 0:
            # 上坡漸進加速，和居中程度連動：越居中加得越快、能加到越高；越偏加得越慢、上限越低
            #  （212647 第 3 圈坡上偏右 25–44 cm 還推到 100%，衝出去）。卡住時至少給 up_stuck_min 推得動
            c = self.center_c
            allowed = sl["up_min_v"] + (sl["up_cap"] - sl["up_min_v"]) * c
            if self.boost >= sl["up_align_boost"]:
                allowed = max(allowed, sl["up_stuck_min"])
            v = min(v, allowed)
            if v > prev:
                v = min(v, max(prev, sl["up_v0"]) + sl["up_accel"] * (sl["up_accel_min"] + (1 - sl["up_accel_min"]) * c) * dt)
        if braking and v >= 0:
            v = sp["brake_v"]
            why.append("反打減速")
        if self.elevated and not up and self.mode == "follow" and self.t_up_end > -1e8 and not self.after_dash(t):
            # 坡頂後緩慢提速（使用者建議）：先降到 post_crest_v0，只在居中（兩條線、偏移小）時每秒加 post_crest_accel，
            #  不居中就不加，給前輪修正的時間（191239 坡頂後 35% 直走偏左修不回來）
            if self.pc_v is None:
                self.pc_v = sl.get("post_crest_v0", 25)
            centered = s.conf == 2 and abs(s.offset) <= sl.get("post_crest_off", 0.08)
            if centered:
                self.pc_v += sl.get("post_crest_accel", 8) * dt
            if v > self.pc_v:
                v = self.pc_v
                why.append("坡頂慢行")
        else:
            self.pc_v = None
        if getattr(self, "cresting", False) and v > sl["top_v"]:
            # 過坡頂：降到坡頂速度、兩輪等速（正常的 6 次都是 40–45% 過坡頂；190254 60/40、190823 90/70 都偏出平台）
            v = sl["top_v"]
            why.append("過坡頂")
        if self.obs_d is not None and self.oc.get("act", False):
            # 跟車：stop_d 以內停，slow_d 以內照距離線性限速（最多 follow_v）
            oc = self.oc
            k = (self.obs_d - oc.get("stop_d", 0.35)) / max(1e-3, oc.get("slow_d", 1.2) - oc.get("stop_d", 0.35))
            cap = 0.0 if k <= 0 else min(oc.get("follow_v", 60), oc.get("follow_v", 60) * k)
            if v > cap:
                v = cap
                why.append("前車%.2fm" % self.obs_d)
        return v, "/".join(why)

    @staticmethod
    def _diff(v, u):
        """左右輪 v − u / v + u；超過 ±100 時兩輪一起平移，保留差距（184602 坡上推到 90%，右輪頂到 100 差速就沒了）。"""
        l, r = v - u, v + u
        hi, lo = max(l, r), min(l, r)
        if hi > 100.0:
            l, r = l - (hi - 100.0), r - (hi - 100.0)
        elif lo < -100.0:
            l, r = l + (-100.0 - lo), r + (-100.0 - lo)
        return max(-100.0, min(100.0, l)), max(-100.0, min(100.0, r))

    def _wheels(self, v, fwa):
        """後輪差速：前輪超過 diff_from 度才用，外側快、內側慢。fwa 正 = 右轉（右輪內側）。"""
        sc = self.sc
        d0, lim = sc["diff_from"], sc["fwa_limit"]
        r = max(0.0, min(1.0, (abs(fwa - sc["fwa_trim"]) - d0) / max(1e-3, lim - d0)))
        r = r if fwa > sc["fwa_trim"] else -r
        clip = lambda u: max(-100.0, min(100.0, u))
        return clip(v * (1 + sc["diff_gain"] * r)), clip(v * (1 - sc["diff_gain"] * r))
