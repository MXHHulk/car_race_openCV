#!/usr/bin/env python3
"""整理紀錄檔（車上 ~/orin_tools/logs，也可以用在電腦的 orin_tools/logs）。

main.py / webctl.py 每跑一趟就留 run_*.csv + run_*.mp4（約 8 MB / 1000 張）、console_*.txt，從來不刪，會越積越多。
這支只動檔名是 run_YYYYmmdd_HHMMSS.* / console_YYYYmmdd_HHMMSS.txt 的檔案，其他檔案（README、設定備份）不碰。

  python3 tools/log_tidy.py                    # 只看：總量、磁碟剩多少、最新 3 趟的摘要（什麼都不改）
  python3 tools/log_tidy.py --archive          # 試跑：列出要搬進日期資料夾（logs/2026-11-21/）的檔案
  python3 tools/log_tidy.py --archive --yes    # 真的搬（最新 --keep 趟留在 logs/ 底下，方便看最新 1–3 次）
  python3 tools/log_tidy.py --prune --yes      # 刪：--keep-days 天前的 mp4、不到 --min-sec 秒的短趟；csv / console 保留
  python3 tools/log_tidy.py --prune --max-gb 5 --yes   # 另外從最舊的 mp4 開始刪，直到全部紀錄 ≤ 5 GB

沒加 --yes 都只是列出來（試跑）。最新 --keep 趟（預設 3）不管哪個條件都不會被搬或刪。
刪掉的影片拿不回來：要留的趟先抓回電腦，例如
  scp "jetson@<ORIN_IP>:~/orin_tools/logs/run_20261121_*" orin_tools/logs/2026-11-21/
"""
import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.normpath(os.path.join(HERE, "..", "logs"))
PAT = re.compile(r"^(run|console)_(\d{8})_(\d{6})\.(csv|mp4|txt)$")


def scan(root):
    """回傳 {趟的時間戳: {"run": [路徑...], "console": [路徑...], "dir": 所在資料夾}}；包含已搬進日期資料夾的。"""
    out = {}
    for d in [root] + [os.path.join(root, x) for x in sorted(os.listdir(root))
                       if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x) and os.path.isdir(os.path.join(root, x))]:
        for name in os.listdir(d):
            m = PAT.match(name)
            if not m:
                continue
            kind, stamp = m.group(1), m.group(2) + "_" + m.group(3)
            e = out.setdefault((kind, stamp), [])
            e.append(os.path.join(d, name))
    return out


def size(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def gb(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def csv_summary(path):
    """(張數, 秒數, 最後狀態, 最後原因)。CSV 斷電時尾巴會有 \\x00，切掉再讀。"""
    import csv
    import io
    try:
        raw = open(path, "rb").read().split(b"\x00")[0].decode("utf-8", "replace")
        rows = list(csv.DictReader(io.StringIO(raw)))
    except OSError:
        return 0, 0.0, "?", ""
    if not rows:
        return 0, 0.0, "空的", ""
    try:
        dur = float(rows[-1]["t"]) - float(rows[0]["t"])
    except (KeyError, ValueError):
        dur = len(rows) / 30.0
    last = rows[-1]
    reason = next((r.get("reason") for r in reversed(rows) if r.get("reason")), "")
    return len(rows), dur, last.get("state", "?"), reason


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"紀錄資料夾（預設 {DEFAULT_DIR}）")
    ap.add_argument("--keep", type=int, default=3, help="最新幾趟一律不動（預設 3）")
    ap.add_argument("--archive", action="store_true", help="把舊的趟搬進 YYYY-MM-DD 資料夾")
    ap.add_argument("--prune", action="store_true", help="刪舊影片和短趟")
    ap.add_argument("--keep-days", type=float, default=7, help="--prune：幾天前的 mp4 刪掉（預設 7）")
    ap.add_argument("--min-sec", type=float, default=3, help="--prune：不到幾秒的短趟整趟刪掉（預設 3，0 = 不刪）")
    ap.add_argument("--max-gb", type=float, default=None, help="--prune：全部紀錄超過這個大小就從最舊的 mp4 開始刪")
    ap.add_argument("--yes", action="store_true", help="真的執行（沒加只是列出來）")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        sys.exit(f"找不到紀錄資料夾 {root}")
    files = scan(root)
    runs = sorted(st for k, st in files if k == "run")
    keep = set(runs[-args.keep:]) if args.keep > 0 else set()
    total = size(p for v in files.values() for p in v)
    du = shutil.disk_usage(root)

    print(f"紀錄資料夾：{root}")
    print(f"  {len(runs)} 趟 run、{sum(1 for k, _ in files if k == 'console')} 個 console，共 {gb(total)}；"
          f"磁碟剩 {gb(du.free)} / {gb(du.total)}")
    print(f"最新 {min(args.keep, len(runs))} 趟（不會被搬或刪）：")
    for st in runs[-args.keep:][::-1]:
        paths = files[("run", st)]
        csvp = next((p for p in paths if p.endswith(".csv")), None)
        n, dur, state, reason = csv_summary(csvp) if csvp else (0, 0.0, "沒有 csv", "")
        has_mp4 = any(p.endswith(".mp4") for p in paths)
        print(f"  run_{st}  {n:5d} 張 {dur:6.1f} s  {gb(size(paths)):>9}  {'有' if has_mp4 else '沒有'}影片  "
              f"結束 {state}" + (f"（{reason}）" if reason else ""))

    if not (args.archive or args.prune):
        print("\n只有查看。要整理加 --archive，要刪舊影片加 --prune；確認列表後再加 --yes。")
        return

    acts = []   # (動作, 路徑, 目標或說明)
    if args.prune:
        now = time.time()
        stamp_t = lambda st: datetime.strptime(st, "%Y%m%d_%H%M%S").timestamp()
        for st in runs:
            if st in keep:
                continue
            paths = files[("run", st)]
            csvp = next((p for p in paths if p.endswith(".csv")), None)
            n, dur, _, _ = csv_summary(csvp) if csvp else (0, 0.0, "", "")
            if args.min_sec > 0 and dur < args.min_sec:
                acts += [("刪", p, f"短趟 {dur:.1f} s / {n} 張") for p in paths]
                continue
            age = (now - stamp_t(st)) / 86400
            if age > args.keep_days:
                acts += [("刪", p, f"{age:.0f} 天前的影片") for p in paths if p.endswith(".mp4")]
        if args.max_gb is not None:
            gone = {p for a, p, _ in acts}
            left = total - size(gone)
            for st in runs:
                if left <= args.max_gb * 1e9:
                    break
                if st in keep:
                    continue
                for p in files[("run", st)]:
                    if p.endswith(".mp4") and p not in gone:
                        acts.append(("刪", p, f"超過 {args.max_gb} GB"))
                        left -= os.path.getsize(p)
    if args.archive:
        gone = {p for a, p, _ in acts}
        # console 是 webctl 開主程式時開的，排在它之後的那幾趟前面：最新幾趟用到的 console 也留在外面
        cons = sorted(st for k, st in files if k == "console")
        before = [c for c in cons if keep and c <= min(keep)]
        keep_con = set(c for c in cons if keep and c >= (before[-1] if before else min(keep)))
        for (kind, st), paths in files.items():
            if (kind == "run" and st in keep) or (kind == "console" and st in keep_con):
                continue
            day = f"{st[:4]}-{st[4:6]}-{st[6:8]}"
            for p in paths:
                if p in gone or os.path.basename(os.path.dirname(p)) == day:
                    continue
                acts.append(("搬", p, os.path.join(root, day, os.path.basename(p))))

    if not acts:
        print("\n沒有要整理的檔案。")
        return
    freed = size(p for a, p, _ in acts if a == "刪")
    print(f"\n{'執行' if args.yes else '試跑（加 --yes 才會真的做）'}：{sum(a == '搬' for a, _, _ in acts)} 個要搬、"
          f"{sum(a == '刪' for a, _, _ in acts)} 個要刪（{gb(freed)}）")
    for a, p, x in sorted(acts, key=lambda v: v[1]):
        rel = os.path.relpath(p, root)
        print(f"  {a} {rel}  → {os.path.relpath(x, root)}" if a == "搬" else f"  {a} {rel}（{x}）")
        if args.yes:
            if a == "搬":
                os.makedirs(os.path.dirname(x), exist_ok=True)
                if os.path.exists(x):
                    print(f"    ⚠ 目標已存在，跳過")
                    continue
                shutil.move(p, x)
            else:
                os.remove(p)
    if args.yes:
        print(f"完成。磁碟剩 {gb(shutil.disk_usage(root).free)}")


if __name__ == "__main__":
    main()
