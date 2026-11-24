#!/usr/bin/env python3
"""用跑完整圈的紀錄（CSV）建賽道地圖 track_map.json（2026-11-21：使用者改成背地圖跑）。

一圈從「上坡開始」（俯仰 ≥ 9° 連續 6 張）算起 s = 0，到下一次上坡開始。
每圈的行駛距離用指令速度估（平均輪速 % × full_mps），各圈都縮放到平均圈長 L 再平均。
地圖記：每 0.1 m 的車頭角（相對上坡開始）、坡頂、虛線左轉、下坡、暗處（隧道）、岔路（位置 + 走哪邊）、出發區間。

  python tools/build_map.py                  # 用預設的多圈紀錄建
  python tools/build_map.py --laps ../logs/run_20261119_203451.csv ...
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, HERE)
from trackmap import odo_speed  # noqa: E402
LOGS = os.path.join(HERE, "..", "logs")
DEFAULT = ["run_20261119_203451", "run_20261119_210913", "run_20261119_212647", "run_20261121_163926"]
DS = 0.1


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))
    f = lambda k, d=0.0: np.array([float(r[k]) if r.get(k) not in (None, "", "None") else d for r in rows])
    t = f("t")
    t = t - t[0]
    v = np.clip((f("bwl") + f("bwr")) / 2, 0, None)
    return dict(t=t, yaw=f("yaw"), p=f("pitch"), v=v, dark=f("dark"),
                mode=[r["mode"] for r in rows], why=[r["why"] for r in rows], state=[r["state"] for r in rows])


def runs_of(mask, n):
    """mask 連續 n 張為真的起點索引。"""
    c = np.convolve(mask.astype(int), np.ones(n, int), "valid")
    out, last = [], -10 ** 9
    for i in np.where(c >= n)[0]:
        if i - last > 150:
            out.append(int(i))
        last = i
    return out


def episodes(mask, min_len=2, gap=10):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and (mask[j] or (j + gap < n and mask[j:j + gap].any())):
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", nargs="*", help="CSV 檔（預設用 11-19 / 11-21 的多圈紀錄）")
    ap.add_argument("--full-mps", type=float, default=1.57)
    ap.add_argument("--out", default=os.path.join(HERE, "track_map.json"))
    args = ap.parse_args()
    def find(r):   # tools/log_tidy.py --archive 會把舊的趟搬進 logs/YYYY-MM-DD/
        p = os.path.join(LOGS, r + ".csv")
        q = os.path.join(LOGS, f"{r[4:8]}-{r[8:10]}-{r[10:12]}", r + ".csv")
        return p if os.path.exists(p) or not os.path.exists(q) else q
    paths = args.laps or [find(r) for r in DEFAULT]

    laps = []
    for path in paths:
        d = load(path)
        dt = np.diff(d["t"], prepend=d["t"][0])
        vms = np.array([odo_speed(v, p, args.full_mps) for v, p in zip(d["v"], d["p"])])
        dist = np.cumsum(vms * dt)
        ups = runs_of(d["p"] >= 9, 6)
        for a, b in zip(ups[:-1], ups[1:]):
            laps.append(dict(name=os.path.basename(path), d=d, a=a, b=b, dist=dist[a:b] - dist[a]))
    if not laps:
        raise SystemExit("沒有完整的圈（要兩次上坡）")
    L = float(np.mean([lp["dist"][-1] for lp in laps]))
    grid = np.arange(0, L, DS)
    psis, ev = [], {"crest": [], "dash": [], "down": [], "down_end": [], "dark": [], "fork": []}
    for lp in laps:
        d, a, b = lp["d"], lp["a"], lp["b"]
        s = lp["dist"] * L / lp["dist"][-1]
        y = d["yaw"][a:b] - d["yaw"][a]
        psis.append(np.interp(grid, s, y))
        p = d["p"][a:b]
        i_cr = next((i for i in range(len(p) - 5) if (p[i:i + 5] < 6).all()), None)
        if i_cr is not None:
            ev["crest"].append(s[i_cr])
        i_dash = next((i for i in range(1, len(p)) if d["mode"][a + i] == "turn" and d["mode"][a + i - 1] != "turn"), None)
        if i_dash is not None:
            ev["dash"].append(s[i_dash])
            i_dn = next((i for i in range(i_dash, len(p) - 4) if (p[i:i + 4] <= -7).all()), None)
            if i_dn is not None:
                ev["down"].append(s[i_dn])
                i_de = next((i for i in range(i_dn, len(p) - 5) if (p[i:i + 5] > -4).all()), None)
                if i_de is not None:
                    ev["down_end"].append(s[i_de])
        for i, j in episodes(d["dark"][a:b] > 0, 5):
            ev["dark"].append((s[i], s[j - 1]))
        fk = np.array(["岔路" in w for w in d["why"][a:b]])
        for i, j in episodes(fk, 3):
            ev["fork"].append(s[i])
    psi = np.mean(psis, axis=0)
    spread = np.std(psis, axis=0)

    def cluster(vals, tol=1.5, min_n=None):
        """位置相近（tol m 內）的歸成一群，至少出現在一半的圈才算數（誤判的位置每圈不一樣）。"""
        min_n = min_n or max(2, len(laps) // 2)
        vals, out = sorted(vals), []
        for v in vals:
            if out and v - out[-1][-1] <= tol:
                out[-1].append(v)
            else:
                out.append([v])
        return [float(np.median(c)) for c in out if len(c) >= min_n]

    forks = cluster(ev["fork"], tol=1.5, min_n=3)
    # 使用者規則：V 尖端在左走右、在右走左。這條賽道一圈兩個岔路（重播 203451 / 210913 每圈都一樣）：
    #  虛線左轉後的第一個岔路尖端在右 → 走左；第二個尖端在左 → 走右
    dash_s = float(np.median(ev["dash"])) if ev["dash"] else None
    after = sorted(forks, key=lambda f: (f - (dash_s or 0)) % L)
    sides = {}
    if len(after) >= 2:
        sides = {after[0]: "left", after[1]: "right"}
    dark = cluster([x for x, _ in ev["dark"]], tol=2.0)
    dark_rng = []
    for d0 in dark:
        ends = [y for x, y in ev["dark"] if abs(x - d0) <= 2.0]
        dark_rng.append([d0, float(np.median(ends))])
    out = dict(
        L=round(L, 3), ds=DS, laps=len(laps), full_mps=args.full_mps,
        psi=[round(float(x), 2) for x in psi],
        psi_sd=[round(float(x), 1) for x in spread],
        crest=float(np.median(ev["crest"])) if ev["crest"] else None,
        dash=dash_s,
        down=float(np.median(ev["down"])) if ev["down"] else None,
        down_end=float(np.median(ev["down_end"])) if ev["down_end"] else None,
        dark=dark_rng,
        forks=[dict(s=round(f, 2), side=sides.get(f, "right")) for f in sorted(forks)],
        start=[17.0, 44.0],          # 出發區間：11-19 / 11-21 各趟出發在 19.5、29–32、35–41 m（172340 從 ~20 m 出發，原本 28–42 太窄定位錯 13 m）
        sources=sorted(set(lp["name"] for lp in laps)),
    )
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"地圖：{args.out}")
    print(f"  {len(laps)} 圈，圈長 {L:.2f} m（各圈 {min(lp['dist'][-1] for lp in laps):.1f}–{max(lp['dist'][-1] for lp in laps):.1f}），"
          f"總轉 {psi[-1]:+.0f}°，車頭角各圈差 中位 {np.median(spread):.1f}° / 最大 {spread.max():.1f}°")
    print(f"  坡頂 {out['crest']:.1f}  虛線 {out['dash']:.1f}  下坡 {out['down']:.1f}–{out['down_end']:.1f}  暗處 {dark_rng}")
    print(f"  岔路 {out['forks']}（原始位置 {sorted(round(x, 1) for x in ev['fork'])}）")
    for k in ("crest", "dash", "down", "down_end"):
        if ev[k]:
            print(f"    {k}: 各圈 {np.round(sorted(ev[k]), 1).tolist()}")


if __name__ == "__main__":
    main()
