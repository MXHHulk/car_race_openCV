"""爬坡與坡頂平台（檔名不用 platform.py：會蓋掉 Python 內建的 platform 模組）。

正式賽道（2026-10-28 run_20261028_183945）：彎道長坡 26–50 s 卡住 → 加速 → 衝一下 → 又卡住，共 6 次；
50 s 到坡頂平台，平台右邊有白色虛線、藍色膠帶、白色 U 形框，循線時被當成車道線，
車子先左後右亂轉（53–61 s）差點開出平台。而且卡住加速還有保底沒退，平台上速度 43–69%。

只靠鏡頭分不出「坡道上又動起來」和「到坡頂了」（兩者畫面變化、路寬、坡度都差不多），所以不去判斷坡頂：
  1. 卡住加速到 arm_boost 以上 = 開始爬坡 → 進入爬坡模式
  2. 爬坡模式中：vision 只用左線（和它保持半個路寬，右邊的東西都不看）；
     車子跑快了（畫面變化中位數超過 fast_motion）就限制總速度（含卡住加速）最多 v_max，
     降回 stall.release 以下才解除。坡道上慢慢爬（畫面變化 1–2.3）不限速，卡住加速照常作用
     沒在限速時速度至少 v_min（run_20261029_181936：坡道上降到 24–29% 就停住，40–50% 都還爬得動）
  3. 卡住加速完全退掉後 time_s 秒（坡頂平台走完）回到一般循線

run_20261028_190458（第一版）：
  - 限速用「畫面變化超過 release」判斷，坡道上慢慢爬時變化在 1.0–2.3 跳來跳去，速度 32 ↔ 50 一直切換（一衝一衝）
    → 改成 fast_motion（坡頂平台實測 3–5，坡道上慢爬 1–2.3）加遲滯
  - 逾時從「最後一次卡住」算，37 s 卡住後加速維持在 20% 爬了 8 秒，45 s 在坡道中間就結束了
    → 改成從卡住加速完全退掉開始算
run_20261029_185111：
  - 起步時的卡住加速就超過門檻（10%），平地上開了 12 秒爬坡模式 → 出發後 start_ignore_s 秒內不算
  - 坡道上 45% 穩穩爬到 70 s（偏移 ±8 cm 內），快到坡頂畫面變化一大就限速到 32%，
    車子在坡頂轉折處失去衝力卡住，之後加速到 71% 也推不動（11 秒，車底可能卡在轉折上）→ 上限改 45
"""
import math

import cv2
import numpy as np

DASH_TOP, DASH_BOTTOM, DASH_THR = 160, 400, 150   # 取樣範圍（640×480 的列）、白色門檻
RIGHT_ROW, RIGHT_UP, RIGHT_LEAN = 330, 290, 130   # 右線位置：看第幾列、上方對照列、上下最多差幾欄


def dash_blobs(frame):
    """畫面中段的白色虛線方塊（扁的、實心、不碰到取樣範圍邊緣）：左右車道線又長又斜，不會被算進來。
    門檻 150：地墊接縫不會過；120 在坡道上會誤判（run_20261029_192728 73.1 s）。
    從第 160 列開始看：要在虛線前就開始轉。左轉開始時車速低、車頭起步慢，第 200 列才看到會太晚，
    轉完走到虛線右邊（run_20261030_211236）。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[DASH_TOP:DASH_BOTTOM]
    _, _, st, _ = cv2.connectedComponentsWithStats((g > DASH_THR).astype(np.uint8), 8)
    return [b for b in st[1:] if 30 <= b[2] <= 240 and 8 <= b[3] <= 90 and b[2] >= 1.3 * b[3]
            and b[4] >= 0.45 * b[2] * b[3] and b[1] > 2 and b[1] + b[3] < DASH_BOTTOM - DASH_TOP - 2]


def _white_runs(row):
    """一列灰階裡的白色段 [(起, 迄)]，太窄的不算。"""
    d = np.diff(np.r_[0, (row > DASH_THR).astype(np.int8), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= 15]


def right_line_x(frame):
    """第 330 列（640×480）右線左緣的像素位置，看不到回傳 640。直接看畫面、不換算公尺：
    坡道上換算的偏移幾乎不動，這個數字卻分得很清楚（關 inner_bias 後成功最低 505，壓上右側時 1 秒內掉到 250）。
    右線往上一定往左斜、左線往右斜，用 40 列上方的位置分辨，車子偏多少都找得到
    （run_20261030_215546 145 s 右線跑到第 250 欄左邊，舊寫法回傳 640，保護失效）。"""
    g = cv2.cvtColor(frame[RIGHT_UP - 3:RIGHT_ROW + 4], cv2.COLOR_BGR2GRAY).astype(np.float32)
    low, up = _white_runs(g[-7:].mean(0)), _white_runs(g[:7].mean(0))
    for a, b in low:
        if any(0 < (a + b - c - d) / 2 < RIGHT_LEAN for c, d in up):
            return int(a)
    return 640


class HilltopMode:
    def __init__(self, cfg):
        c = cfg["control"].get("hilltop") or {}
        self.ramp_stuck_add = c.get("ramp_stuck_add", 0)   # 坡上卡住時速度上限可以再加多少 %
        self.ramp_stuck_window = c.get("ramp_stuck_window", 2.5)   # 進坡幾秒內才允許（進坡口卡護條）
        self.t_climb_start = None
        self.arm_n = c.get("arm_n", 1)              # 俯仰角連續幾張超過 arm_pitch 才算上坡
        self.n_up = 0
        self.on = bool(c.get("enable", False))
        self.arm_boost = c.get("arm_boost", 10)     # 卡住加速超過這個 % 才算在爬坡（出發時起步約 3–5；v_max 40 時坡道入口只加到 17）
        self.time_s = c.get("time_s", 8.0)          # 最後一次卡住後維持幾秒
        self.v_max = c.get("v_max", 45)             # 跑快時的總速度上限 %（太低會在坡頂失去衝力卡住，見下方說明）
        self.ignore_s = c.get("start_ignore_s", 3.0)  # 出發後幾秒內的卡住不算爬坡（從靜止起步一定會先卡一下）
        self.fast = c.get("fast_motion", 4.0)       # 畫面變化超過這個 = 跑快了（坡頂衝出去時 5–15、坡道 40–50% 在爬 2–3.4）
        self.v_min = c.get("v_min", 40)             # 爬坡模式最低速度 %：坡道上 24–29% 就停住，40–50% 都還在爬
        self.rearm_s = c.get("rearm_s", 20.0)      # 爬坡模式結束後幾秒內不再啟動（下坡底部也會卡一下）
        # IMU 俯仰角（車頭往上，度）要超過這個才算真的在上坡，0 = 不看 IMU。
        # 進坡提早加速（stall.early）在平地急彎也會觸發：run_20261103_173116 25 s 平地右彎前輪 +40°，
        # 加速 15% 就進了爬坡模式，固定 50%、關差速，轉不過彎開出賽道。畫面變化分不出來（真的進坡時 1.4–3.6、這次 2.5）
        # run_20261103_184744：平地最多 +5.8°、上坡 +13.5 ~ +16°、下坡是負的（下坡底卡住也不會誤進）
        self.arm_pitch = c.get("arm_pitch", 0)
        # 俯仰角超過 arm_pitch（在坡上）時的速度 %，0 = 照 v_min / v_max。
        # 50% 坡上畫面變化只有 2–4（平地 4–7），看起來像停頓；到坡頂俯仰角掉下來就回到 v_min / v_max，不會衝出坡頂
        self.ramp_v = c.get("ramp_v", 0)
        # 過了坡頂（俯仰角先超過 arm_pitch、再掉到一半以下）到爬坡模式結束的速度 %，0 = 照 v_min / v_max。
        # 平台上 50% 衝向路口，左轉一慢就來不及：run_20261103_194718 左轉第 1 秒只轉 33°（平常 55–63°），
        # 越過虛線開出平台。坡頂轉折處還是用 v_min，不然會失去衝力卡住
        self.top_v = c.get("top_v", 0)
        # 有 IMU 時爬坡模式改成坡頂後 crest_hold_s 秒結束（不看卡住加速）：進坡只加速一次，加速很早就退掉，
        # 電量低、爬得慢時 7 s 在左轉前就到了，路口變成 30%（run_20261103_203923、204141）
        self.crest_hold_s = c.get("crest_hold_s", 6.0)
        self.crest = self.ramp_seen = False
        self.t_crest = None
        self.reset()
        self.t_drive = None                         # 這次開始行駛的時間（停車才清掉）
        self.t_end = None                           # 上一次爬坡模式結束的時間

    def reset(self):
        self.mode, self.t_boost = "idle", None      # idle / climb；t_boost：最後一次還有卡住加速的時間
        self.crest = self.ramp_seen = False
        self.t_crest = None
        self.capping = False

    @property
    def active(self):
        return self.mode == "climb"

    def update(self, stall, t, driving, pitch=None):
        """每張畫面呼叫（在 stall.update 之後）。pitch：IMU 俯仰角（度，上坡正），None = 沒有 IMU。
        回傳 (最低速度 %, 總速度上限 %)，None = 不限。"""
        if not self.on or not driving:
            self.reset()
            self.t_drive = self.t_end = None
            return None, None
        self.t_drive = t if self.t_drive is None else self.t_drive
        # 起步那次卡住的加速會拖好幾秒才退（保底 floor_s），只看「現在幾秒」擋不住：
        # 要求「最後一次卡住」發生在出發 ignore_s 秒之後（run_20261029_190202 出發 3 秒後還是被起步的加速觸發）
        late = stall.t_stall is not None and stall.t_stall - self.t_drive >= self.ignore_s
        # 剛爬完坡不再啟動：run_20261029_201418 下坡到底（41 s）卡了一下又觸發爬坡模式，
        # 平地上以最低 45% 只沿左線走，岔路附近大晃、開到賽道外
        cool = self.t_end is None or t - self.t_end >= self.rearm_s
        # 沒有 IMU（或讀不到）時照舊只看卡住加速
        # 俯仰角要連續 arm_n 張都超過 arm_pitch 才算上坡：113139 25.5 s 平地上撞到邊條，車身一跳俯仰角瞬間超過 9°，
        # 剛好前輪打大有卡住加速，就進了爬坡模式，接著在平地上強制左轉，整趟毀掉
        self.n_up = self.n_up + 1 if (pitch is not None and self.arm_pitch and pitch >= self.arm_pitch) else 0
        uphill = not self.arm_pitch or pitch is None or self.n_up >= self.arm_n
        if stall.extra >= self.arm_boost and self.mode == "idle" and late and cool and uphill:
            self.mode = "climb"
            self.t_climb_start = t
            print(f"[hilltop] 卡住加速到 {stall.extra:.0f}%" + ("" if pitch is None else f"、俯仰 {pitch:+.1f}°")
                  + "：爬坡模式，只沿左線走")
        if self.mode != "climb":
            return None, None
        if stall.extra > 0 or self.t_boost is None:
            self.t_boost = t
        if pitch is not None and self.arm_pitch and pitch >= self.arm_pitch:
            self.ramp_seen = True
        if self.ramp_seen and not self.crest and pitch is not None and pitch < self.arm_pitch / 2:
            self.crest, self.t_crest = True, t
        if self.ramp_seen:
            over = (self.crest and t - self.t_crest >= self.crest_hold_s) or t - self.t_boost >= self.time_s + 8.0
        else:
            over = t - self.t_boost >= self.time_s
        if over:
            print(f"[hilltop] 過坡頂 {self.crest_hold_s:.0f} 秒了，回到一般循線" if self.crest
                  else f"[hilltop] 卡住加速退掉 {self.time_s:.0f} 秒了，回到一般循線")
            self.reset()
            self.t_end = t
            return None, None
        # 一跑快就馬上限速（中位數本身已經濾掉單張雜訊）：run_20261029_180918 坡頂衝出時，
        # 等 0.3 秒的那段速度是 69–74%
        if stall.med > self.fast:
            self.capping = True
        elif stall.med < stall.release:
            self.capping = False
        if self.ramp_v and pitch is not None and pitch >= self.arm_pitch:
            # 坡上真的卡住（畫面不動）時讓卡住加速能加上去：上限原本固定 ramp_v，加速再多也是 62%，
            # 133246 進坡卡在左側護條上 5 秒（加速到 45 也沒用）
            # 只在進坡口那幾秒：143427 坡頂前的隆起卡一下，上限放到 82% 一衝過坡頂就往右滑出平台
            # （原本 62% 在隆起卡住也推得過去）
            early = self.t_climb_start is not None and t - self.t_climb_start <= self.ramp_stuck_window
            if stall.med < stall.thresh and early:
                return self.ramp_v, self.ramp_v + self.ramp_stuck_add
            return self.ramp_v, self.ramp_v
        if self.top_v and self.crest:
            return self.top_v, self.top_v
        return self.v_min, (max(self.v_max, self.v_min) if self.capping else None)   # 限速時也維持最低速度，坡頂轉折要有衝力


class Junction:
    """坡頂 T 字路口：左轉後直線開到下坡（run_20261029_191149：69.7 s 前輪打左到底轉進去，
    70.6 s 起畫面上的轉角線、藍膠帶、U 形框讓循線往右拉，72 s 前輪 +38° 衝向平台右邊緣）。

    畫面上分不出路口（坡道地墊的接縫和虛線長得太像），改用「車子自己的動作」判斷：
    爬坡模式中（或結束後 window_s 秒內）、而且已經連續 free_s 秒沒有卡住加速（車子順順在跑，不是在坡道上推），
    前輪連續 detect_s 秒打左超過 detect_fwa → 是路口左轉。
    （坡道中間偏右時修正也會打左到 −30° 以上，但那時通常還有卡住加速：181936 39 s、184312 53 s）
    之後固定動作：前輪 turn_fwa 打 turn_s 秒 → 前輪回正直走 straight_s 秒 → 回到一般循線。
    時間要在賽道上試：轉不夠就加長 turn_s，還沒到下坡就切回循線就加長 straight_s。
    """

    def __init__(self, cfg):
        c = (cfg["control"].get("hilltop") or {}).get("junction") or {}
        self.on = bool(c.get("enable", False))
        self.detect_fwa = c.get("detect_fwa", -30)   # 前輪打左超過這個（負 = 左）
        self.detect_s = c.get("detect_s", 0.2)
        self.window_s = c.get("window_s", 3.0)       # 爬坡模式結束後還接受幾秒
        self.free_s = c.get("free_s", 2.0)           # 卡住加速要已經退掉這麼久
        self.t_free = None
        self.turn_fwa = c.get("turn_fwa", -40)
        self.turn_s = c.get("turn_s", 0.6)
        self.straight_s = c.get("straight_s", 2.0)
        self.v = c.get("speed", 35)
        self.lap_s = c.get("lap_s", 60.0)            # 做完後至少過這麼久、又開始爬坡，才算下一圈（一圈約 110 s）
        # 前輪不一定會往左：run_20261030_205113 到路口時循線往右打 +26°，沒觸發就開過去。
        # 所以 free_s 之後也看虛線：連續 2 張看到 2 塊以上就左轉。force_s 是連虛線都沒看到時的保底，0 = 不用
        # （只靠時間不夠：run_20261030_210239 第二圈 8.25 s 比路口早約 0.8 s，左轉後越過左邊界）
        self.n_dash = 0
        # 看到虛線後先不轉，等最近一塊虛線到畫面第 dash_near 列（車頭快壓到）再轉，0 = 看到就轉。
        # 太早轉會切到平台左邊界的木條角（run_20261031_152350）；最多等 dash_wait 秒，車子歪了虛線可能出畫面
        self.dash_near = c.get("dash_near", 0)
        self.dash_wait = c.get("dash_wait", 0.6)
        self.t_dash = None
        self.force_s = c.get("force_s", 0)
        # 有 IMU 時改從坡頂（俯仰角掉回 top_pitch 以下）開始算：坡頂到左轉每趟都是 3.5–3.7 s，
        # 從加速退掉算是 5.3–6.4 s，而且爬坡變快就會提早（run_20261103 四趟）。0 = 用加速退掉的時間
        self.top_pitch = c.get("top_pitch", 0)       # 上坡時要先超過這個才算爬過坡
        self.top_free_s = c.get("top_free_s", 2.0)   # 坡頂後幾秒開始找虛線（虛線最早在坡頂後 2.9 s 看到）
        self.top_force_s = c.get("top_force_s", 5.0) # 坡頂後幾秒還沒看到虛線就直接左轉
        self.on_ramp = False
        self.t_top = None
        self.ramp_fwa = c.get("ramp_fwa", 0)         # 坡道直線段前輪上限（度），0 = 不限
        self.ramp_k = c.get("ramp_k", 0.0)           # 坡道直線段：車頭每偏 1° 前輪多修幾度
        self.post_fwa = c.get("post_fwa", 0)         # 直走結束後前輪上限（度），0 = 不限
        self.post_s = c.get("post_s", 2.5)           # 這個上限維持幾秒
        self.post_lmax = c.get("post_lmax", self.post_fwa)  # 這段往左最多打幾度
        self.ramp_xr = c.get("ramp_xr", 0)           # 右線位置小於這個（像素）= 太靠右，0 = 不用
        self.ramp_px = c.get("ramp_px", 0.25)        # 每靠右 1 像素前輪往左修幾度
        self.ramp_guard = c.get("ramp_guard", 30)    # 靠太右時前輪最多往左打幾度（比 ramp_fwa 大：−18° 救不回來）
        # 靠右保護從進爬坡模式幾秒後開始（None = 舊做法：限制開始後 2 s，約進坡 3.8 s）。
        # 170204 第二圈進坡 2.8 s 右線已到 496、一路掉到 332，前輪還往右 +8 ~ +28°（坡上只剩右線，被當成左線），
        # 進坡 3.7 s 壓上右護條掉下去；保護 3.8 s 才開始。第一圈右線最低 513，不會誤觸發
        self.ramp_xr_after = c.get("ramp_xr_after", None)
        self.ramp_ref, self.ramp_hist = None, []
        self.ramp_early_fwa = c.get("ramp_early_fwa", self.ramp_fwa)
        self.ramp_hold_k = c.get("ramp_hold_k", 0.0)        # 坡上每偏 1° 前輪多修幾度（0 = 不用）
        self.ramp_hold_after = c.get("ramp_hold_after", 0.5)
        self.ramp_hold_max = c.get("ramp_hold_max", 20.0)
        self.crest_yaw_add = c.get("crest_yaw_add", 0.0)    # 過坡頂後車頭目標往左多幾度
        self.after_crest = False
        self.t_c0 = self.yaw_c0 = self.hold_ref = None
        self.ramp_start_s = c.get("ramp_start_s", None)   # 卡住加速沒退時，進入爬坡幾秒後開始限制前輪（None = 只等加速退掉）
        self.t_c0 = None
        # 以量到的車頭角度決定左轉多久（見 yaw.py），0 = 只用 turn_s 固定時間
        self.turn_yaw = c.get("turn_yaw", 0)         # 車頭比坡道轉了這麼多度就回正（回正後還會滑約 20°）
        self.turn_max_s = c.get("turn_max_s", 1.5)   # 畫面看不出轉角時最多左轉幾秒
        self.hold_yaw = c.get("hold_yaw", 0)         # 直走時維持的車頭角度（相對坡道）
        self.hold_after_s = c.get("hold_after_s", 0.5)  # 直走開始後幾秒才修（回正後還會滑）
        self.hold_k = c.get("hold_k", 1.0)           # 每差 1° 前輪修幾度
        self.hold_fwa = c.get("hold_fwa", 12)        # 直走時前輪最多修幾度
        self.align_fwa = c.get("align_fwa", self.hold_fwa)   # 對準下坡時前輪最多修幾度（221701 第二圈 12° 修不回來，往左開出平台）
        # 往左最多修幾度：第二圈車頭基準不準，算出來轉不夠、前輪 −12°，其實已經快壓左線
        # （run_20261031_141947 206 s、144024 169.8 s 都這樣開出平台左邊）
        self.hold_left = c.get("hold_left", 3)
        # 直走時看虛線修正橫向位置：虛線保持在車子 dash_y 公尺的地方（負 = 右邊），0 = 不用。
        # 左轉完車子都在虛線另一邊（虛線在左邊 +25 cm）。185435 一路靠過去壓著虛線（+3 cm），
        # 對準下坡時往右衝出去；184744 越過虛線、最後虛線在右邊 15 cm，順利下坡（run_20261103）。
        # 路寬 54 cm、車寬 30 cm：−0.20 = 車道中間偏一點，車輪離虛線約 5 cm
        self.dash_y = c.get("dash_y", 0.0)
        # 每差 1 m，車頭目標比 hold_yaw 多偏幾度（最多 ±dash_yaw_max）。原本直接加到前輪，車頭一偏左
        # hold_yaw 那項就把前輪拉回右邊：run_20261103_191003 虛線只到右邊 8 cm、190800 停在虛線上就結束直走
        self.dash_k = c.get("dash_k", 50.0)
        self.dash_yaw_max = c.get("dash_yaw_max", 20)
        # 直走、對準時車頭轉速的阻尼（每 °/s 前輪反向修幾度）：左轉結束時車頭還在轉，
        # run_20261103_192609 車頭一路轉到 108°，越過虛線到右邊 38 cm（目標 20 cm），對準時位置偏左
        self.hold_d = c.get("hold_d", 0.0)
        # 已經下坡後的車頭目標：下坡本身略往右偏，成功的 184744 下坡時從 80° 轉到 67°
        self.down_yaw = c.get("down_yaw", None)
        self.down_fwa = c.get("down_fwa", 8)          # 下坡上往右最多修幾度（184744 當時 +4 ~ +8°）
        self.down_lmax = c.get("down_lmax", 0)        # 下坡上車頭比 down_yaw 還右時，往左最多修幾度
        self.down_brake = c.get("down_brake", None)   # 坡上（俯仰 ≤ −top_pitch）的速度 %，負 = 反打煞車；None = 照 down_sv
        self.down_brake_m = c.get("down_brake_m", 2.0)  # 畫面變化超過這個（太快）才反打
        self.down_stuck_v = c.get("down_stuck_v", 0)      # 對準 / 下坡卡住時最多推到幾 %（0 = 不用）
        self.down_stuck_m = c.get("down_stuck_m", 1.0)    # 畫面變化低於這個算沒在動
        self.down_stuck_s = c.get("down_stuck_s", 0.5)    # 沒在動多久才開始加
        self.down_push_m = c.get("down_push_m", 0.0)    # 畫面變化低於這個（太慢）照 down_sv 推；兩者之間滑行（0）
        self.down_off_k = c.get("down_off_k", 0.0)      # 坡上車子每偏右 1 m，車頭目標往左幾度（0 = 不看鏡頭）
        self.down_off_max = c.get("down_off_max", 12.0)
        self.down_shift = 0.0
        self.dash_left = c.get("dash_left", 12)      # 看虛線修正時往左最多打幾度（取代 hold_left）
        self.gmap = None                             # vision.GroundMap，main.py 設定
        self.dash_seen = None                        # 最近一次量到的虛線橫向位置（m），寫紀錄用
        self.yaw0 = None
        # 下坡到底的右彎：車子落地顛一下（畫面變化 > bend_m）之後要馬上右轉約 60°。
        # 成功的圈前輪都打到 +33° 以上；失敗的圈畫面上線很亂、前輪沒打夠，直直開出賽道（run_20261031_155923、160423 第二圈）。
        # 落地後依車頭角度右轉到 bend_yaw（相對坡道），bend_fwa = 0 不用
        self.bend_fwa = c.get("bend_fwa", 0)
        self.bend_yaw = c.get("bend_yaw", 30)
        self.bend_end = c.get("bend_end", 8)         # 離目標這麼多度內就交回循線
        self.bend_s = c.get("bend_s", 3.0)           # 最多幾秒
        self.bend_m = c.get("bend_m", 20)            # 畫面變化超過這個算落地
        self.bend_from = c.get("bend_from", 3.0)     # 直走結束後這段時間內才看（s）
        self.bend_to = c.get("bend_to", 8.0)         # 到這個時間還沒落地就直接轉
        # 直走結束到落地這段不循線，維持車頭方向對準下坡。畫面上線很雜，循線時車頭 47–126° 都有，
        # 太右或太左都會開出賽道（run_20261101_162335、171628）；車頭一路在 95–115° 的圈（165533）最穩
        self.align = c.get("align", False)
        self.align_v = c.get("align_v", 30)
        # 已經下坡後的速度：對準時要快一點進坡，慢慢進底盤會卡在坡口打滑（203923 30% 卡了 6 s），坡上再放慢
        self.down_sv = c.get("down_sv", self.align_v)
        # 直走時間到了但虛線還沒到右邊（dash_y + straight_tol 以內）就多走，最多 straight_more_s 秒。
        # 電量低變慢時 3 s 不夠越過虛線，對準時位置不對（203923、204141 結束時虛線還在 −1 ~ −3 cm）
        self.straight_tol = c.get("straight_tol", 0.08)
        self.straight_more_s = c.get("straight_more_s", 1.5)
        self.dash_last = None                        # (時間, 最近一次量到的虛線橫向位置)
        # 直走段改用鏡頭：沿虛線左側半個路寬走（Pure Pursuit），可以開快一點。cam = false 就是原本的 IMU 鎖車頭。
        # 車頭離 hold_yaw 超過 cam_guard 度、或看不到虛線 → 這一張退回 IMU 鎖車頭。
        # 直走結束改看「走了多遠」（速度 % × 秒，換算成原本 speed × straight_s），開快了才不會衝過頭
        self.cam = bool(c.get("cam", False))
        self.cam_v = c.get("cam_v", self.v)          # 鏡頭直走的速度 %
        self.cam_slow = c.get("cam_slow", 0.7)       # 走到預計距離的這個比例後降回 speed，準備對準下坡
        self.cam_k_lat = c.get("cam_k_lat", 60.0)    # 離目標每 1 m，車頭多偏幾度
        self.cam_cut = c.get("cam_cut", 18.0)        # 車頭最多比車道多偏幾度（斜切過去）
        self.cam_half = c.get("cam_half", -self.dash_y if self.dash_y else 0.27)   # 車道中線在虛線左邊幾 m
        self.cam_fwa = c.get("cam_fwa", 20)          # 往右最多打幾度
        self.cam_left = c.get("cam_left", 15)        # 往左最多打幾度
        self.cam_guard = c.get("cam_guard", 15)      # 車頭離 hold_yaw 超過這個就不信鏡頭
        self.wheelbase = cfg["geometry"]["wheelbase"]
        self.cam_used = False                        # 這一張有沒有用鏡頭（寫紀錄用）
        self.dist, self.t_prev, self.v_prev = 0.0, None, 0.0
        # 對準下坡時看左右位置（見 _align_shift）：每偏 1 m 車頭目標多轉幾度、最多幾度、取幾張中位數
        self.align_off_k = c.get("align_off_k", 0.0)
        self.align_off_max = c.get("align_off_max", 10.0)
        self.align_off_n = c.get("align_off_n", 9)
        self.align_off_pitch = c.get("align_off_pitch", 3.0)
        self.off_hist, self.align_shift = [], 0.0
        # 平台跑得比較快時，虛線出現得早、轉彎前多滑一段：找虛線、轉彎的時機依速度提早。
        # 原本的 top_free_s / top_force_s / dash_wait / dash_near 都是在 dash_ref_v（top_v 42%）時調的，
        # 速度 v 時時間乘上 dash_ref_v / v（最少 0.4 倍），dash_near 每快 1% 提早 dash_lead_px 列（畫面上越上面 = 越遠）。
        self.dash_ref_v = c.get("dash_ref_v", 0)     # 0 = 不依速度調整
        self.dash_lead_px = c.get("dash_lead_px", 0.0)
        # 看到虛線後先把速度降到 speed（左轉的速度），轉彎從同樣的速度開始，轉的幅度才穩定
        self.dash_brake = bool(c.get("dash_brake", False))
        self.v_cap = None                            # main.py 用：不為 None 時速度不超過這個
        self.k_speed = 1.0                           # 最近一次的時間縮放（寫紀錄 / 除錯用）
        # 鏡頭算的前輪和上一張平均：方塊一張多一塊少一塊，算出來會差 10° 以上（221701 重播 −4° ↔ −20°）
        self.cam_smooth = c.get("cam_smooth", 0.5)
        self.cam_f = None
        # 下坡是略往右偏的：車頭維持 96° 會開到左側木條（run_20261101_172605、173140）；
        # 車頭 66–86° 的圈都順利下去（141924 第二圈、170631 第二圈）
        self.align_yaw = c.get("align_yaw", 75)
        self.bend_done = True
        self.down_seen = False
        self.mode, self.t0, self.t_left, self.t_climb, self.done = None, None, None, None, False
        self.t_done = None

    @property
    def active(self):
        return self.mode is not None

    def after_turn(self, t, s):
        """路口動作結束後 s 秒內。"""
        return self.t_done is not None and t - self.t_done < s

    def post_limit(self, fwa, t):
        """直走剛結束時畫面上還有虛線，循線會把它當車道線：run_20261030_215031 35.3 s 車子在虛線左邊，
        卻算出偏 −34 cm、前輪 +44° 往右衝，開出平台右邊。成功那趟（201418）這時前輪只有 +12 ~ +28°。
        這段看到的線常是下坡的左線，車子在它左邊，循線當成右線往左修，車身一半在賽道外（run_20261101_141924），
        所以往左限制得比往右小。不強制當左線：143948 那樣改，算出偏 −50 cm，一路往右開到下坡右側摔下去。"""
        if self.post_fwa and self.t_done is not None and t - self.t_done < self.post_s:
            return max(-self.post_lmax, min(self.post_fwa, fwa))
        return fwa

    def ramp_steer(self, fwa, climbing, t, yaw, frame=None):
        """坡道直線段：限制前輪、維持車頭方向。卡住加速退掉後一直到爬坡模式結束。
        坡道中段有一處隆起（卡住加速退掉後 3–4 s），地面不是平的，畫面換算成公尺會突然偏 30–40 cm。
        run_20261030_202526 26.9 s 算出偏 −42 cm、前輪 +44°，差速 81/9，從坡道右側掉下去（194955 25 s 同樣打到 +41°）。
        正常爬這段最多 15.5°，所以限制前輪。
        畫面上偏移也不可靠：run_20261030_211948 第二圈 142–144.5 s 車頭往右偏 25°、開上右側，偏移只算出 3–14 cm。
        所以取退掉後 0.5–2 s 的平均方向當坡道方向，之後依車頭偏差多修一點（yaw.py）。"""
        # 有 IMU 後爬坡的卡住加速常一路維持到坡頂（112404 加速 10.5–15 沒退），t_free 一直是 None，
        # 這個限制從沒生效：坡上前輪打到 +43°，撞上右側防護板。沒退的話從進入爬坡 ramp_start_s 秒後開始限制
        if not climbing:
            self.t_c0 = self.yaw_c0 = None
            self.hold_ref = None
        elif self.t_c0 is None:
            self.t_c0 = t
            self.yaw_c0 = yaw.value if yaw is not None else None
        # 坡上鎖車頭方向：坡道方向 = 進爬坡模式那一刻的車頭（31 趟：之後坡頂路口的車頭和它差 +1.5°，順利的圈 ±5°）。
        # 只看鏡頭時坡上會左右甩：142649 進坡 1 s 後車頭往右偏 25°、接著往左 46° 開上左護條翻起來（俯仰 74°），
        # 142107 往右偏 18° 壓右線掉下去。進坡 ramp_hold_after 秒後，前輪多修 ramp_hold_k × 車頭偏差
        # 只在坡上鎖（過坡頂就放開）：坡頂平台上鎖著車頭（+12°）和循線互搶，145048 坡頂後被拉來拉去、往右開出平台掉下去
        if (climbing and self.ramp_hold_k and self.yaw_c0 is not None and yaw is not None
                and t - self.t_c0 >= self.ramp_hold_after and (self.crest_yaw_add or not self.after_crest)):
            # 過了坡頂（after_crest，main.py 設定）目標改成往左 crest_yaw_add 度：37 次過坡頂，順利的圈坡頂後 1 s 車頭
            # 都往左 +6 ~ +20°；沒過的圈是 0 或往右，直直開出平台右緣（143930 第二圈俯仰掉到 −11° 卡住）
            target = self.yaw_c0 + (self.crest_yaw_add if self.after_crest else 0.0)
            err = yaw.value - target                      # 車頭太左（正）→ 往右修（前輪正）
            fwa += max(-self.ramp_hold_max, min(self.ramp_hold_max, self.ramp_hold_k * err))
        base = self.t_free if self.t_free is not None else (
            self.t_c0 + self.ramp_start_s if (self.t_c0 is not None and self.ramp_start_s is not None) else None)
        if not self.ramp_fwa or not climbing or base is None or t < base:
            self.ramp_ref, self.ramp_hist = None, []
            return self._right_guard(fwa, climbing, t, None, frame)
        el = t - base
        if el < 2.0:
            if el >= 0.5 and yaw is not None:
                self.ramp_hist.append(yaw.value)
        elif self.ramp_ref is None and self.ramp_hist:
            self.ramp_ref = float(np.mean(self.ramp_hist))
        if self.ramp_ref is not None and self.ramp_k:
            fwa += self.ramp_k * (yaw.value - self.ramp_ref)       # 車頭太右（yaw 變小）→ 往左修（fwa 負）
        # 卡住加速還沒退（坡道前段、彎坡）時放寬到 ramp_early_fwa：順利過坡的圈坡上前輪 −29 ~ +35° 都有，
        # ±18° 太緊（113949 坡上要往左修卻卡在 −18°，偏右 50 cm 掉出內側）；只擋打到底（112404 +43° 撞防護板）
        lim = self.ramp_fwa if self.t_free is not None else self.ramp_early_fwa
        fwa = max(-lim, min(lim, fwa))
        return self._right_guard(fwa, climbing, t, el, frame)

    def _right_guard(self, fwa, climbing, t, el, frame):
        """run_20261030_212836 81.5 s、213126 第二圈 147.5 s：到路口前頂上右側。靠太右就不准再往右打。"""
        if not self.ramp_xr or frame is None or not climbing:
            return fwa
        if self.ramp_xr_after is None:
            on = el is not None and el >= 2.0
        else:
            on = self.t_c0 is not None and t - self.t_c0 >= self.ramp_xr_after
        if not on:
            return fwa
        e = self.ramp_xr - right_line_x(frame)
        if e > 0:
            fwa = max(-self.ramp_guard, min(fwa, 0.0) - self.ramp_px * e)
        return fwa

    def update(self, fwa, climbing, boosting, t, driving, running=True, yaw=None, frame=None, motion=0.0, pitch=None, v=None,
               offset=None):
        """回傳 (強制前輪角度, 強制速度)，None = 不介入。fwa 是循線算出來的前輪角度。
        running：這一趟還在跑（RUN / LOST）；停車後才允許下一趟再做一次。
        yaw：YawEstimator，有給而且設定了 turn_yaw 才用量到的角度。"""
        self.cam_used = False
        self.v_cap = None
        # 每圈做一次：下坡口卡住也會啟動爬坡模式（run_20261029_194232 89 s），所以要隔夠久才算下一圈
        if not running or (climbing and self.done and t - self.t_done >= self.lap_s):
            self.done = False
        if not self.on or not driving:
            self.mode, self.t_left = None, None
            return None, None
        if climbing:
            self.t_climb = t
        if self.top_pitch and pitch is not None:
            if pitch >= self.top_pitch:
                self.on_ramp, self.t_top = True, None
            elif self.on_ramp and pitch < self.top_pitch / 2:
                self.on_ramp, self.t_top = False, t
        self.t_free = None if boosting else (self.t_free or t)
        if self.mode in (None, "align") and self.done and not self.bend_done and self.bend_fwa and yaw is not None and self.yaw0 is not None:
            el = t - self.t_done
            if self.top_pitch and pitch is not None:
                # 有 IMU：下坡（俯仰 ≤ −top_pitch）之後俯仰回到一半以上才算落地。畫面變化在坡上顛一下就會誤判：
                # run_20261103_195613、192609 還在坡上（−16°）就右轉，切到彎道內側壓線，之後循線修過頭開出去。
                # 成功的圈落地比原本右轉早 0.2–0.7 s。沒偵測到落地時 bend_to 再多等 2 s 保底
                if self.top_pitch and pitch <= -self.top_pitch:
                    self.down_seen = True
                # 保底時間到了也要不在坡上才右轉（180259 還在 −16° 的坡上就右轉，開出去）
                go = (self.down_seen and pitch > -self.top_pitch / 2) or (el >= self.bend_to + 2.0 and pitch > -self.top_pitch)
            else:
                go = el >= self.bend_to or (el >= self.bend_from and motion >= self.bend_m)
            if go:
                self.bend_done = True
                print(f"[junction] 下坡到底：右轉到車頭 {self.bend_yaw}°（現在 {yaw.value - self.yaw0:.0f}°）")
                self.mode, self.t0 = "bend", t
        if self.mode == "align":
            if self.top_pitch and pitch is not None and pitch <= -self.top_pitch:
                self.down_seen = True
            target = self.down_yaw if self.down_seen and self.down_yaw is not None else self.align_yaw
            if not self.down_seen:
                target -= self._align_shift(t, offset, pitch)
            elif self.down_off_k and offset is not None:
                # 坡上也看鏡頭偏移：181705 下坡滑行 5 秒，車頭目標 down_yaw 一直往右帶，偏移從 +7 一路到 +72 cm，
                # 落地卡在右側。偏右（offset 正）→ 目標往左（加大）
                self.down_shift = max(-self.down_off_max, min(self.down_off_max, self.down_off_k * offset))
                target += self.down_shift
            e = self.hold_k * (yaw.value - self.yaw0 - target) + self.hold_d * yaw.rate(t)
            if self.down_seen:
                # 已經下坡了（到右轉前都算）：下坡本身往右偏，車頭自然往右轉（184744 從 80° 到 67°，當時前輪 +4 ~ +8° 順利下去），
                # 這時不要往左拉回 align_yaw
                # 但比 down_yaw 還右太多時要能往左修一點：175302 車頭下坡一路往右到 down_yaw 右邊 13°，前輪卡在 0，
                # 落地偏右 70–80 cm，接右彎開出去（down_lmax，0 = 舊做法）
                e = max(-self.down_lmax, min(self.down_fwa, e))
            v = self.down_sv if self.down_seen else self.align_v
            # 下坡反打煞車（down_brake，負 = 馬達反轉）：滿電電池下坡衝得快，同樣前輪車頭轉得多，175538 下坡往右轉 23°、
            # 落地偏右 75 cm。只在坡上（俯仰 ≤ −top_pitch）反打；畫面幾乎不動（快停了）就改成 0 滑行，免得倒退
            if self.down_seen and self.down_brake is not None and pitch is not None and self.top_pitch and pitch <= -self.top_pitch:
                # 三段：太快反打、太慢照 down_sv 推、中間滑行。180259 一直反打 / 滑行，坡上爬 4 秒（畫面變化 1–1.6），
                # 還在坡上（俯仰 −16°）就到時間右轉，開出去
                if motion >= self.down_brake_m:
                    v = self.down_brake
                elif motion < self.down_push_m:
                    v = self.down_sv
                else:
                    v = 0.0
            # 對準 / 下坡時卡住（畫面幾乎不動超過 down_stuck_s）：慢慢加到 down_stuck_v 推出去。
            # 095220 坡口（俯仰 −12°）以 30% 停住 8 秒，卡住加速在路口動作裡不會加上去
            if motion < self.down_stuck_m:
                self.t_still = t if getattr(self, "t_still", None) is None else self.t_still
            else:
                self.t_still = None
            if self.down_stuck_v and self.t_still is not None and t - self.t_still >= self.down_stuck_s:
                k = min(1.0, (t - self.t_still - self.down_stuck_s) / 1.0)
                v = max(v, self.v + (self.down_stuck_v - self.v) * k)
            return max(-self.align_fwa, min(self.align_fwa, e)), v
        if self.mode == "bend":
            e = yaw.value - self.yaw0 - self.bend_yaw
            if e <= self.bend_end or t - self.t0 >= self.bend_s:
                print(f"[junction] 右轉結束（{t - self.t0:.2f} 秒，還差 {e:.0f}°），回到循線")
                self.mode = None
                return None, None
            return min(self.bend_fwa, e), self.v
        if self.mode is None:
            k = self.k_speed = (max(0.4, min(1.0, self.dash_ref_v / v)) if self.dash_ref_v and v and v > 0 else 1.0)
            near = self.t_climb is not None and t - self.t_climb <= self.window_s
            if self.t_top is not None:                     # 有 IMU：從坡頂算
                near = near and t - self.t_top >= self.top_free_s * k
                forced = near and t - self.t_top >= self.top_force_s * k
            else:
                near = near and self.t_free is not None and t - self.t_free >= self.free_s
                forced = near and bool(self.force_s) and t - self.t_free >= self.force_s
            look = near and not self.done and frame is not None
            blobs = dash_blobs(frame) if look else []
            self.n_dash = self.n_dash + 1 if len(blobs) >= 2 else 0
            if self.n_dash >= 2 and self.t_dash is None:
                self.t_dash = t
            if self.t_dash is not None:
                # 不算畫面左右邊緣的方塊：左線彎過來的尾端常被算進去
                near_row = self.dash_near
                if self.dash_near and self.dash_ref_v and v:
                    near_row = max(DASH_TOP + 20, self.dash_near - self.dash_lead_px * max(0.0, v - self.dash_ref_v))
                close = any(b[0] >= 40 and b[0] + b[2] <= 600 and b[1] + b[3] + DASH_TOP >= near_row for b in blobs)
                forced = forced or close or t - self.t_dash >= self.dash_wait * k
                if self.dash_brake and not self.done:
                    self.v_cap = self.v
            if near and not self.done and (forced or fwa <= self.detect_fwa):
                self.t_left = t if self.t_left is None else self.t_left
                if forced or t - self.t_left >= self.detect_s:
                    self.mode, self.t0, self.t_dash, self.dash_last = "turn", t, None, None
                    self.yaw0 = yaw.ref(t) if yaw is not None else None
                    print(f"[junction] 坡頂路口：左轉 {self.turn_s:.1f} 秒、直走 {self.straight_s:.1f} 秒")
            else:
                self.t_left = None
            return None, None
        el = t - self.t0
        self.dash_seen = None
        turned = None if self.yaw0 is None or not self.turn_yaw else yaw.value - self.yaw0
        if self.mode == "turn" and (el >= self.turn_s if turned is None else (turned >= self.turn_yaw or el >= self.turn_max_s)):
            print(f"[junction] 左轉 {el:.2f} 秒結束" + ("" if turned is None else f"，車頭轉了 {turned:.0f}°"))
            self.mode, self.t0, el = "straight", t, 0.0
            self.dist, self.t_prev, self.v_prev = 0.0, t, self.v
            self.cam_f = None
        self.cam_used = False
        prog = el                                    # 直走「相當於用 speed 走了幾秒」
        if self.mode == "straight" and self.cam:
            self.dist += self.v_prev * (t - self.t_prev)
            self.t_prev = t
            prog = self.dist / self.v
        crossed = (not self.dash_y or self.dash_last is None or t - self.dash_last[0] > 0.6
                   or self.dash_last[1] <= self.dash_y + self.straight_tol)
        if self.mode == "straight" and prog >= self.straight_s and (crossed or prog >= self.straight_s + self.straight_more_s):
            print("[junction] 直走結束，回到一般循線（下坡）")
            self.mode, self.done, self.t_done, self.bend_done = None, True, t, False   # 這一圈做過了
            if self.align and self.bend_fwa and self.yaw0 is not None:
                self.mode, self.down_seen, self.t_still = "align", False, None
            return None, None
        if self.mode == "turn":
            return self.turn_fwa, self.v
        if turned is None or not self.hold_yaw or el < self.hold_after_s:   # 回正後車頭還在滑，等滑完再修
            return 0.0, self.v
        rate = self.hold_d * yaw.rate(t)
        e = self.hold_k * (turned - self.hold_yaw) + rate   # 車頭太左 → 往右修（fwa 正）
        pts = self._dash_points(frame) if (self.dash_y or self.cam) else []
        yd = next((y for x, y in pts if x <= 1.0), None) if self.dash_y else None
        self.dash_seen = yd
        if yd is not None:
            self.dash_last = (t, yd)
        v = self.v
        if self.cam:
            v = self.cam_v if prog < self.cam_slow * self.straight_s else self.v
            f = self._cam_steer(pts, turned)
            if f is not None:
                if self.cam_f is not None:
                    f = self.cam_smooth * self.cam_f + (1 - self.cam_smooth) * f
                self.cam_f = f
                self.cam_used = True
                self.v_prev = v
                return max(-self.cam_left, min(self.cam_fwa, f + rate)), v
            self.v_prev = self.v                     # 這張退回 IMU：用原本的速度
        if yd is not None:
            # 虛線比目標偏左（車子太右）→ 車頭目標往左多偏一點，斜斜切過去；到目標位置車頭目標就回到 hold_yaw
            off = max(-self.dash_yaw_max, min(self.dash_yaw_max, self.dash_k * (yd - self.dash_y)))
            e = self.hold_k * (turned - self.hold_yaw - off) + rate
            return max(-self.dash_left, min(self.hold_fwa, e)), self.v
        return max(-self.hold_left, min(self.hold_fwa, e)), self.v

    def _align_shift(self, t, offset, pitch=None):
        """對準下坡時依車子左右位置調整車頭目標（度，正 = 往右多轉）。
        只看車頭方向的話，直走時偏掉的位置修不回來：104046、104748 開始下坡時車子都偏左 25–43 cm，卡在左側邊條。
        鏡頭算出的偏移（正 = 車子偏右）取最近 align_off_n 張的中位數（平台上偶爾一張判錯），
        車子偏左 → 目標往右多轉 align_off_k × 偏移，最多 align_off_max 度；這段時間看不到線就不修。"""
        if not self.align_off_k:
            return 0.0
        # 到坡口車身開始前傾，畫面上的線換算會錯（104748 80.3 s 前傾 −5.7° 時一下算出偏右 +0.23 m），
        # 前傾超過 align_off_pitch 度就不再更新，沿用平地上算的
        if pitch is not None and pitch < -self.align_off_pitch:
            return self.align_shift
        if offset is not None:
            self.off_hist.append((t, offset))
        self.off_hist = [(tt, o) for tt, o in self.off_hist[-self.align_off_n:] if t - tt <= 0.5]
        if len(self.off_hist) < 3:
            self.align_shift = 0.0
            return 0.0
        med = float(np.median([o for _, o in self.off_hist]))
        self.align_shift = max(-self.align_off_max, min(self.align_off_max, -self.align_off_k * med))
        return self.align_shift

    def _dash_points(self, frame, x_max=1.5):
        """虛線方塊底邊中點換算成車子座標 [(x 前方 m, y 左正 m)]，由近到遠。
        只用車前 0.25 m 以外的方塊：太近會被車身擋住一半。"""
        if self.gmap is None or frame is None:
            return []
        sx, sy = 320.0 / frame.shape[1], 240.0 / frame.shape[0]   # GroundMap 用 320×240
        pts = []
        for b in dash_blobs(frame):
            if b[0] < 40 or b[0] + b[2] > 600:             # 畫面左右邊緣的常常是左線彎過來的尾端
                continue
            x, y = self.gmap.to_vehicle((b[0] + b[2] / 2) * sx, (b[1] + b[3] + DASH_TOP) * sy)
            x, y = float(x), float(y)
            if np.isfinite(x) and np.isfinite(y) and 0.25 <= x <= x_max:
                pts.append((x, y))
        return sorted(pts)

    def _dash_lateral(self, frame):
        """最近一塊虛線（車前 0.25–1.0 m）離車子中心的橫向距離（m，左正），看不到回傳 None。"""
        return next((y for x, y in self._dash_points(frame) if x <= 1.0), None)

    def _cam_steer(self, pts, turned):
        """直走段沿虛線左側 cam_half 走：依虛線方向和橫向距離算目標車頭，回傳前輪角度（負左正右）；
        不可靠就回傳 None（退回 IMU 鎖車頭）。
        虛線方向：兩塊以上、前後差 0.25 m 以上就自己擬合（221701 實測車道在 IMU 的 91–93°，不是 hold_yaw 96°，
        只用 IMU 方向時一塊和兩塊方塊算出來差 15°，前輪一張 −5° 一張 −21° 跳來跳去）；擬合結果和 IMU 算的方向差太多（被別的白色東西騙）或方塊太少，
        改用 IMU：車道方向 = hold_yaw − 現在車頭角度。"""
        if not pts or turned is None or abs(turned - self.hold_yaw) > self.cam_guard:
            return None
        th = math.radians(self.hold_yaw - turned)          # 車道相對車頭的方向（左正）
        p = np.array(pts)
        if len(p) >= 2 and np.ptp(p[:, 0]) >= 0.25:
            m, c = np.polyfit(p[:, 0], p[:, 1], 1)
            if abs(math.degrees(math.atan(m) - th)) <= self.cam_guard:
                th = math.atan(m)
                x0 = float(p[0, 0])
                p0 = (x0, float(m * x0 + c))
            else:
                p0 = (float(p[0, 0]), float(p[0, 1]))
        else:
            p0 = (float(p[0, 0]), float(p[0, 1]))
        # 不追前方一點（Pure Pursuit）：轉完彎車子在虛線右邊、離目標 40–50 cm，追點會一路打到底，
        # 車頭甩到 115–120°（191956、192130，目標車道約 92°），橫向到位時車頭還歪 20°，對準下坡修不回來。
        # 改成「目標車頭 = 車道方向 + 依橫向距離多偏（最多 cam_cut 度）」：模擬車頭最多 111–113°、到位時已經轉正
        tx = math.tan(th)
        c0 = p0[1] - tx * p0[0]                            # 虛線在車子正橫方向的位置（m，左正）
        lat = c0 * math.cos(th) + self.cam_half             # 車子在目標右邊多少（正 = 要往左）
        want = math.degrees(th) + max(-self.cam_cut, min(self.cam_cut, self.cam_k_lat * lat))
        return -self.hold_k * want
