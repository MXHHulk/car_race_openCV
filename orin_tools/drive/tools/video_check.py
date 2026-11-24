#!/usr/bin/env python3
"""專門讀錄影（run_*.mp4 + 同名 CSV）的檢查程式：找 CSV 數字本身看不出來、或主程式不會報的問題。

  python tools/video_check.py ../logs/run_20261121_203112.mp4            # 一趟
  python tools/video_check.py --latest 3                                 # logs/ 裡最新 3 趟（含日期資料夾）
  python tools/video_check.py x.mp4 --no-replay                          # 不重跑辨識（快，只查檔案 / 畫面 / 時間）
  python tools/video_check.py x.mp4 --start 20 --end 40                  # 只看一段（秒，從錄影開頭算）

檢查項目（每項都印出時間點當證據；同一種問題最多列 --show 個）：
  A 檔案   影片打不開、讀不出來的張數、影片張數和 CSV 列數對不上、CSV 尾巴斷電（\\x00）
  B 時間   CSV 時間戳跳格（主程式來不及處理 → 鏡頭畫面被丟掉）、處理時間 > 1 張的時間、fps 掉
  C 畫面   畫面凍結（連續幾張一模一樣 = 鏡頭卡住，但主程式還在照舊畫面開車）、太暗 / 過曝、模糊（晃動）
  D 鏡頭   開頭幾張用前板比對，看鏡頭和校正當時差幾度（同 tools/cam_align.py）
  E 重跑   用「現在的」程式和設定把每張畫面重跑一次（餵 CSV 的俯仰 / 車頭角），和當時紀錄的看到幾條線、
           偏移、前輪角比；對不上 = 程式 / 設定 / 校正已經和當時不同（回放結論不能直接套到當時），或程式有不確定性
  F 辨識   一條線 ↔ 兩條線單張閃爍、偏移單張大跳、量到的路寬離設定太遠、看不到線的連續時間
  G 控制   前輪打到底的連續時間、IMU 沒資料 / 車頭角卡住不動（但前輪有在打）
"""
import argparse
import collections
import csv
import io
import math
import os
import re
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
LOGS = os.path.normpath(os.path.join(HERE, "..", "logs"))


def fnum(v):
    try:
        return None if v in (None, "", "None") else float(v)
    except ValueError:
        return None


def load_csv(path):
    raw = open(path, "rb").read()
    nul = b"\x00" in raw
    text = raw.split(b"\x00")[0].decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, nul


def latest(n):
    found = []
    for d, _, names in os.walk(LOGS):
        for x in names:
            if re.fullmatch(r"run_\d{8}_\d{6}\.mp4", x):
                found.append(os.path.join(d, x))
    return sorted(found, key=os.path.basename)[-n:][::-1]


class Report:
    def __init__(self, show):
        self.show = show
        self.items = collections.OrderedDict()   # (類別, 名稱) -> [證據...]
        self.notes = []

    def add(self, cat, name, ev):
        self.items.setdefault((cat, name), []).append(ev)

    def segs(self, cat, name, ts, mask, min_len, fmt):
        """mask 連續為真、長度 ≥ min_len 秒的區段。"""
        i, n = 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                if ts[j] - ts[i] >= min_len:
                    self.add(cat, name, fmt(ts[i], ts[j]))
                i = j + 1
            else:
                i += 1

    def print(self):
        if not self.items:
            print("  沒有找到問題。")
        for (cat, name), evs in self.items.items():
            more = f"（共 {len(evs)} 處，只列前 {self.show}）" if len(evs) > self.show else ""
            print(f"  [{cat}] {name}{more}")
            for e in evs[: self.show]:
                print(f"        {e}")
        for nt in self.notes:
            print(f"  · {nt}")


def check(video, args, cfg):
    csv_path = video[:-4] + ".csv"
    print(f"\n=== {os.path.relpath(video)} ===")
    R = Report(args.show)
    if not os.path.exists(csv_path):
        print("  沒有同名 CSV，跳過")
        return
    rows, nul = load_csv(csv_path)
    if nul:
        R.add("A 檔案", "CSV 尾巴有 \\x00（寫到一半斷電 / 被拔掉），之後的列遺失", csv_path)
    rows = [r for r in rows if fnum(r.get("t")) is not None]
    if not rows:
        print("  CSV 沒有資料")
        return
    ts = np.array([fnum(r["t"]) for r in rows])
    t0 = ts[0]
    rel = ts - t0
    col = lambda k: [fnum(r.get(k)) for r in rows]
    states = [r.get("state", "") for r in rows]
    print(f"  CSV {len(rows)} 列、{rel[-1]:.1f} s，結束狀態 {states[-1]}"
          + (f"（{rows[-1].get('reason')}）" if rows[-1].get("reason") else ""))

    # ---------------- B 時間（只看 CSV）
    dts = np.diff(ts)
    if len(dts):
        nominal = float(np.median(dts))
        gaps = np.where(dts > 1.8 * nominal)[0]
        lost_frames = int(np.sum(np.round(dts[gaps] / nominal) - 1)) if len(gaps) else 0
        for g in gaps:
            R.add("B 時間", f"畫面時間跳格（正常每張 {nominal * 1000:.0f} ms）：主程式來不及處理，中間的鏡頭畫面被丟掉",
                  f"{rel[g]:6.1f}s 跳 {dts[g] * 1000:.0f} ms（約丟 {round(dts[g] / nominal) - 1} 張）")
        if len(gaps):
            R.notes.append(f"跳格 {len(gaps)} 次，估計共丟 {lost_frames} 張（{100 * lost_frames / (len(rows) + lost_frames):.1f}%）")
    pm = np.array([v if v is not None else 0 for v in col("proc_ms")])
    if len(dts):
        slow = pm > 1000 * nominal
        R.segs("B 時間", f"辨識 + 控制 > {1000 * nominal:.0f} ms（比一張畫面還久）", rel, slow, 0.0,
               lambda a, b: f"{a:6.1f}–{b:6.1f}s 最大 {pm[(rel >= a) & (rel <= b)].max():.0f} ms")

    # ---------------- G 控制（只看 CSV）
    lim = cfg["steer"].get("fwa_limit", 44)
    fwa = np.array([v if v is not None else 0 for v in col("fwa")])
    run = np.array([s in ("RUN", "LOST") for s in states])
    R.segs("G 控制", f"前輪打到底（|fwa| ≥ {lim - 0.5}°）超過 0.5 s：轉向已經沒有餘量", rel,
           run & (np.abs(fwa) >= lim - 0.5), 0.5, lambda a, b: f"{a:6.1f}–{b:6.1f}s（{b - a:.1f} s）")
    pitch, yaw = col("pitch"), col("yaw")
    none_p = np.array([p is None for p in pitch])
    if none_p.all():
        R.add("G 控制", "整趟沒有 IMU 資料：上坡 / 下坡 / 虛線左轉 / 地圖定位全部失效", "pitch 欄全空")
    else:
        R.segs("G 控制", "IMU 沒資料", rel, none_p & run, 0.3, lambda a, b: f"{a:6.1f}–{b:6.1f}s")
        y = np.array([v if v is not None else np.nan for v in yaw])
        stuck = np.zeros(len(y), bool)
        for i in range(1, len(y)):
            stuck[i] = y[i] == y[i - 1]
        R.segs("G 控制", "車頭角連續不變但前輪有在打（IMU 卡住？）", rel,
               stuck & run & (np.abs(fwa) > 15), 1.0, lambda a, b: f"{a:6.1f}–{b:6.1f}s 車頭角固定在 {y[np.searchsorted(rel, a)]:.1f}°")

    # ---------------- F 辨識（CSV）
    conf = np.array([int(v) if v is not None else -1 for v in col("conf")])
    off = np.array([v if v is not None else np.nan for v in col("offset")])
    R.segs("F 辨識", "看不到線（conf 0）超過 0.3 s", rel, run & (conf == 0), 0.3,
           lambda a, b: f"{a:6.1f}–{b:6.1f}s（{b - a:.1f} s）")
    for i in range(1, len(conf) - 1):
        if run[i] and conf[i - 1] == conf[i + 1] == 2 and conf[i] < 2:
            R.add("F 辨識", "兩條線中間單張掉成一條 / 沒有（閃爍）", f"{rel[i]:6.1f}s conf 2→{conf[i]}→2")
        if run[i] and conf[i - 1] == conf[i] == 2 and abs(off[i] - off[i - 1]) > 0.12:
            R.add("F 辨識", "兩條線都有、偏移單張跳 > 12 cm（認錯線？）",
                  f"{rel[i]:6.1f}s 偏移 {off[i - 1] * 100:+.0f} → {off[i] * 100:+.0f} cm")
    lw0 = cfg["geometry"]["lane_width"]
    lw = np.array([v if v is not None else np.nan for v in col("lane_w")])
    with np.errstate(invalid="ignore"):
        R.segs("F 辨識", f"量到的路寬偏離設定 {lw0} m 超過 20% 達 0.3 s", rel, run & (np.abs(lw - lw0) > 0.2 * lw0), 0.3,
               lambda a, b: f"{a:6.1f}–{b:6.1f}s 路寬 {np.nanmedian(lw[(rel >= a) & (rel <= b)]):.2f} m")

    # ---------------- A / C / D / E：讀影片
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        R.add("A 檔案", "影片打不開", video)
        R.print()
        return
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    if abs(n_meta - len(rows)) > 1:
        R.add("A 檔案", "影片張數和 CSV 列數對不上（每張畫面應該剛好一列）",
              f"影片標頭 {n_meta} 張 / CSV {len(rows)} 列；影片 {n_meta / (vfps or 30):.1f} s vs CSV {rel[-1]:.1f} s")
    if len(dts) and vfps:
        R.notes.append(f"影片標示 {vfps:.0f} fps，實際 {1 / nominal:.1f} fps：用影片時間軸看會{'變快' if 1 / nominal < vfps - 1 else '一致'}"
                       "（main.py 一律寫 30 fps）")

    drv = None
    if not args.no_replay:
        from driver import Driver
        drv = Driver(cfg, HERE)
    plate = None
    try:
        from tools.cam_align import REF, measure, MIN_SCORE, GOOD_SCORE
        if os.path.exists(REF):
            plate = (np.load(REF)["tpl"].astype(np.float32), measure, MIN_SCORE, GOOD_SCORE)
    except Exception as e:  # noqa: BLE001
        R.notes.append(f"鏡頭比對略過（{e}）")

    prev, bad, bad_ts, plate_vals = None, 0, [], []
    frozen, means, blur = [], [], []
    mism = collections.Counter()
    first_mis = {}
    n_cmp = 0
    was_run = False
    for i, r in enumerate(rows):
        ok, frame = cap.read()
        if not ok:      # 車上錄的 mp4 連續讀有時會壞，跳到這一張重讀
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
        if not ok:
            bad += 1
            bad_ts.append(rel[i])
            frozen.append(False), means.append(np.nan), blur.append(np.nan)
            continue
        g = cv2.cvtColor(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        frozen.append(prev is not None and float(cv2.absdiff(g, prev).mean()) < 0.25)
        prev = g
        means.append(float(g.mean()))
        blur.append(float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
        if plate and len(plate_vals) < 5 and frame.shape[:2] == (480, 640):
            f = 320 / math.tan(math.radians(cfg["geometry"]["hfov"] / 2))
            plate_vals.append(plate[1](cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), plate[0], f))
        if rel[i] < args.start or rel[i] > args.end:
            if drv is not None and rel[i] < args.start:
                drv.eye.process(frame)
            continue
        if drv is not None:
            st = states[i]
            now_run = st in ("RUN", "LOST")
            if now_run and (not was_run or drv.state not in ("RUN", "LOST")):
                drv.state = "IDLE"
                drv.start()
            elif not now_run and drv.state in ("RUN", "LOST"):
                drv.stop("紀錄上停了")
            was_run = now_run
            p, y = fnum(r.get("pitch")), fnum(r.get("yaw"))
            fw, _, _ = drv.step(frame, ts[i], p, y, 0.0)
            if not now_run:
                continue
            n_cmp += 1
            s = drv.s
            checks = {"看到幾條線": s.conf != conf[i],
                      "偏移差 > 2 cm": not np.isnan(off[i]) and abs(s.offset - off[i]) > 0.02,
                      "前輪角差 > 2°": abs(fw - fwa[i]) > 2.0,
                      "速度原因不同": drv.why.split("/")[0] != (r.get("why") or "").split("/")[0]}
            for k, v in checks.items():
                if v:
                    mism[k] += 1
                    first_mis.setdefault(k, f"{rel[i]:6.1f}s 當時 conf{conf[i]} 偏{off[i] * 100:+.0f}cm "
                                            f"fwa{fwa[i]:+.0f} {r.get('why')} ／ 現在 conf{s.conf} "
                                            f"偏{s.offset * 100:+.0f}cm fwa{fw:+.0f} {drv.why}")

    # A
    if bad:
        R.add("A 檔案", f"影片有 {bad} 張讀不出來（重試後仍失敗）", f"第一張在 {bad_ts[0]:.1f}s、最後在 {bad_ts[-1]:.1f}s")
    # C
    m = np.array(means)
    fz = np.array(frozen)
    R.segs("C 畫面", "畫面凍結（連續幾張幾乎一模一樣）且車在跑：鏡頭卡住，主程式照舊畫面開車", rel[: len(fz)],
           fz & run[: len(fz)], 0.15, lambda a, b: f"{a:6.1f}–{b:6.1f}s")
    with np.errstate(invalid="ignore"):
        R.segs("C 畫面", "整張畫面太暗（平均 < 25）", rel[: len(m)], m < 25, 0.3, lambda a, b: f"{a:6.1f}–{b:6.1f}s")
        R.segs("C 畫面", "整張畫面過曝（平均 > 220）", rel[: len(m)], m > 220, 0.3, lambda a, b: f"{a:6.1f}–{b:6.1f}s")
        b = np.array(blur)
        med = np.nanmedian(b) if np.isfinite(b).any() else np.nan
        if np.isfinite(med):
            R.segs("C 畫面", f"畫面模糊（清晰度 < 整趟中位數 {med:.0f} 的 30%，車身晃動 / 撞到）", rel[: len(b)],
                   (b < 0.3 * med) & run[: len(b)], 0.2, lambda a, b_: f"{a:6.1f}–{b_:6.1f}s")
        dm = np.abs(np.diff(m))
        for i in np.where(dm > 40)[0]:
            R.add("C 畫面", "亮度單張跳 > 40（曝光突變 / 進出隧道太快）", f"{rel[i + 1]:6.1f}s {m[i]:.0f} → {m[i + 1]:.0f}")
    # D
    if plate_vals:
        yaw_d, pit_d, roll_d, sc = [float(np.median([v[k] for v in plate_vals])) for k in range(4)]
        if sc < plate[2]:
            R.notes.append(f"鏡頭比對：開頭畫面對不到前板（相似度 {sc:.2f}），不能判斷鏡頭有沒有動")
        else:
            txt = (f"左右 {yaw_d:+.1f}°、上下 {pit_d:+.1f}°、歪斜 {roll_d:+.1f}°（相似度 {sc:.2f}"
                   + ("，偏低：車子可能已經在動或前板反光，數字參考用" if sc < plate[3] else "") + "）")
            if max(abs(yaw_d), abs(pit_d), abs(roll_d)) > 0.5:
                R.add("D 鏡頭", "鏡頭和校正當時差超過 0.5°（地面座標會偏，偏移 / 路寬 / 前視點都受影響）", txt)
            else:
                R.notes.append("鏡頭位置 OK：" + txt)
    # E
    if drv is not None and n_cmp:
        for k, c in mism.items():
            if c / n_cmp > 0.02:
                R.add("E 重跑", f"現在的程式重跑和當時紀錄不同：{k} {100 * c / n_cmp:.1f}% 的張數", "第一處 " + first_mis[k])
        if not any(c / n_cmp > 0.02 for c in mism.values()):
            R.notes.append(f"重跑 {n_cmp} 張和當時紀錄一致（差異都 < 2%）：回放結果可以代表當時")
    R.print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--latest", type=int, default=0, help="檢查 logs/ 裡最新 N 趟")
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--no-replay", action="store_true", help="不重跑辨識（E 項略過）")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--show", type=int, default=5, help="每種問題最多列幾處")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    vids = list(args.videos) + (latest(args.latest) if args.latest else [])
    if not vids:
        ap.error("給影片路徑或 --latest N")
    import contextlib
    for v in vids:
        if args.no_replay:
            check(v, args, cfg)
        else:
            with contextlib.redirect_stdout(io.StringIO()) as buf:   # Driver 的「出發 / 停止」訊息不印
                check(v, args, cfg)
            out = buf.getvalue()
            print("\n".join(ln for ln in out.splitlines() if not ln.startswith("[driver]")))
    print("\n這支查不到的（要靠其他方法）：" + BLIND_SPOTS)


BLIND_SPOTS = """
1. 車子實際做了什麼：影片是鏡頭看到的，CSV 是程式「下的指令」，舵機有沒有真的打到那個角度（連桿鬆、11-06 那次）、
   後輪有沒有打滑、BMduino 有沒有收到，都看不到 → 要舵機回授 / 輪速計，或從畫面移動量反推（目前沒做）。
2. 重跑是開迴路：車子走的路線是錄影當時的，改了參數後「車會不會走不同的路」重跑不出來 → 只能實車驗證。
3. 影片沒錄到的時間：被主程式丟掉的鏡頭畫面（B 項只能算張數，看不到內容）、錄影開始前 / 結束後。
4. 鏡頭比對只在開頭幾張：中途撞到把鏡頭撞歪看不出來（會表現成路寬 / 偏移異常，F 項可能抓到）。
5. IMU 數值對不對：只能查有沒有資料、有沒有卡住，漂移、裝歪、震動雜訊查不出來。
6. 通訊延遲 / STA：CSV 沒記 sta_age，序列埠斷斷續續看不到（主程式超時才會停車）。
7. 電池電壓、馬達出力下降：同樣指令跑得比較慢，影片看得到變慢但程式不會自動判斷。
"""


if __name__ == "__main__":
    main()
