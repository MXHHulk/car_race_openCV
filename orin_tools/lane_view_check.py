#!/usr/bin/env python3
"""鏡頭視角檢查：確認目前鏡頭的位置與角度，抓到的畫面能不能做「黑底白線」辨識，
並判斷前方路段是 直線 / 左彎 / 右彎，以及 上坡 / 下坡。

不需要鳥瞰校正就能跑：
  - 直線 vs 彎道：平面上的直線投影到畫面仍是直線，所以看左右白線「彎了多少」(bow)。
  - 上坡 vs 下坡：把白線分成近段、遠段，各自延長求交點（消失點）。同一個平面上的
    平行線，消失點都落在同一條地平線上；前方路面往上折，遠段的地平線就會比近段高，
    往下折就比較低。兩者的角度差 = 前方坡度相對於車身的變化（slope_deg）。
    注意這是「相對變化」：上坡途中快到坡頂時，前方平地會被判成「下坡」，這是正確的，
    代表前方變緩，控制上要準備收油門。

用法（在 Orin 上）：
  python3 lane_view_check.py                         # /dev/video0，即時檢查，瀏覽器開 http://<orin-ip>:8080
  python3 lane_view_check.py --frames 90 --label straight --out runs/straight
                                                      # 拍 90 張後輸出檢查報告（逐張 CSV + 標註圖）
  python3 lane_view_check.py --frames 60 --save-baseline baseline.json
                                                      # 車停在平坦直線正中央，記錄基準（修正鏡頭變形）
  python3 lane_view_check.py --baseline baseline.json --label uphill --frames 60 --out runs/uphill
  python3 lane_view_check.py --source runs/video.mp4  # 也可以吃影片、單張圖或整個資料夾
  python3 lane_view_check.py --selftest               # 用合成賽道畫面驗證演算法本身

--label 可用：straight、left、right、curve、uphill、downhill（用來算判斷正確率）。
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

W, H = 320, 240  # 辨識用解析度，與系統架構設計一致

SHAPE_ZH = {"straight": "直線", "left": "左彎", "right": "右彎", "none": "看不到線"}
SLOPE_ZH = {"flat": "平路", "uphill": "上坡", "downhill": "下坡", "unknown": "坡度無法判斷"}
LEVEL = {"OK": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------- 找線

def _peaks(hist, thr, sep, k=3):
    hist = hist.copy()
    out = []
    for _ in range(k):
        i = int(np.argmax(hist))
        if hist[i] < thr:
            break
        out.append(i)
        hist[max(0, i - sep):i + sep] = 0
    return out


def _track(binary, claimed, x0, y_start, y0, cfg, roi_h):
    """從 (x0, y_start) 往上用滑動視窗追一條線。視窗會依前幾格的走向預測下一格位置。"""
    win_h = max(6, roi_h // cfg.nwin)
    margin = cfg.margin
    x, dx = float(x0), 0.0
    xs, ys, wins, thick = [], [], [], []
    miss, exit_reason = 0, "top"
    y_hi = y_start
    while y_hi > y0:
        y_lo = max(y0, y_hi - win_h)
        xc = x + dx
        xl, xr = int(round(xc - margin)), int(round(xc + margin))
        if xr <= 2 or xl >= W - 2:
            exit_reason = "side"
            break
        xl, xr = max(0, xl), min(W, xr)
        sub = binary[y_lo:y_hi, xl:xr] > 0
        sub &= ~claimed[y_lo:y_hi, xl:xr]
        yy, xx = np.nonzero(sub)
        if len(xx) >= cfg.minpix:
            nx = float(xx.mean()) + xl
            if wins:
                dx = 0.6 * (nx - x) + 0.4 * dx
            x = nx
            xs.append(xx + xl)
            ys.append(yy + y_lo)
            wins.append((xl, y_lo, xr, y_hi))
            thick.append(len(xx) / max(1, y_hi - y_lo))
            miss = 0
        else:
            x = xc
            miss += 1
            if miss >= 2:
                exit_reason = "end"
                break
        y_hi = y_lo
    if not wins:
        return None
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return {"xs": xs, "ys": ys, "wins": wins, "thick": thick, "exit": exit_reason,
            "top": int(ys.min()), "bot": int(ys.max()), "n": int(len(xs))}


def find_lines(binary, y0, cfg, y1=H):
    """在 ROI（第 y0 列到第 y1 列）裡找左右兩條線。"""
    roi_h = y1 - y0
    claimed = np.zeros_like(binary, dtype=bool)
    lines = []
    band = max(8, roi_h // 6)
    # 從最下面一條帶開始找線的起點，找不到兩條就往上一條帶再找，一直到 ROI 頂端。
    # （原本只找下面 4 條帶 = ROI 的下 2/3：只出現在 ROI 上方的線永遠找不到，ROI 上緣一動結果就跳來跳去）
    for k in range(6):
        b1 = y1 - k * band
        b0 = b1 - band
        if b0 < y0:
            break
        hist = ((binary[b0:b1] > 0) & ~claimed[b0:b1]).sum(0).astype(float)
        hist = np.convolve(hist, np.ones(5) / 5, "same")
        for p in _peaks(hist, thr=max(3.0, 0.25 * band), sep=int(W * 0.08)):
            ln = _track(binary, claimed, p, b1, y0, cfg, roi_h)
            if ln is None or ln["bot"] - ln["top"] < max(6, roi_h * 0.08):
                continue
            for (xl, ya, xr, yb) in ln["wins"]:
                claimed[ya:yb, xl:xr] = True
            lines.append(ln)
        if len(lines) >= 2:
            break
    lines.sort(key=lambda l: -l["n"])
    lines = lines[:2]
    if len(lines) == 2:
        # 粗線有時被拆成兩段追蹤：兩條「線」在共同的列上幾乎重疊，就只留像素多的那條
        a, b = lines
        ra, xa = _row_centers(a)
        rb, xb = _row_centers(b)
        common, ia, ib = np.intersect1d(ra, rb, return_indices=True)
        if len(common) >= 3 and np.median(np.abs(xa[ia] - xb[ib])) < max(12.0, 2.5 * np.median(a["thick"])):
            lines = [a]
        elif len(common) == 0:
            # 上下接續、沒有重疊列：看接點附近是不是同一條線
            lo, hi = (a, b) if a["top"] > b["top"] else (b, a)   # lo 在下方
            if abs(lo["top"] - hi["bot"]) < 3 * max(6, roi_h // cfg.nwin):
                xlo = _row_centers(lo)[1][0]
                xhi = _row_centers(hi)[1][-1]
                if abs(xlo - xhi) < max(12.0, 2.5 * np.median(lo["thick"])):
                    lines = [a]
    for ln in lines:
        # 被畫面左右邊緣切掉的那幾列，白線中心會被算偏，不拿來擬合
        edge_rows = np.unique(ln["ys"][(ln["xs"] <= 1) | (ln["xs"] >= W - 2)])
        keep = ~np.isin(ln["ys"], edge_rows)
        if keep.sum() >= 20:
            ln["xs"], ln["ys"] = ln["xs"][keep], ln["ys"][keep]
            ln["top"], ln["bot"] = int(ln["ys"].min()), int(ln["ys"].max())
        span = ln["bot"] - ln["top"]
        deg = 2 if span >= roi_h * 0.2 else 1
        ln["fit"] = np.polyfit(ln["ys"], ln["xs"], deg)
        if deg == 2:
            # bow：曲線中點偏離「頭尾連線」多少像素；正值 = 往左彎
            ln["bow"] = float(-ln["fit"][0] * (span / 2.0) ** 2)
        else:
            ln["bow"] = None
        ln["x_bot"] = float(np.polyval(ln["fit"], ln["bot"]))
        far = ln["thick"][-3:] if len(ln["thick"]) >= 3 else ln["thick"]
        ln["thick_far"] = float(np.median(far))
    # 分左右
    left = right = None
    if len(lines) == 2:
        a, b = lines
        yc = min(a["bot"], b["bot"])
        if np.polyval(a["fit"], yc) <= np.polyval(b["fit"], yc):
            left, right = a, b
        else:
            left, right = b, a
    elif len(lines) == 1:
        ln = lines[0]
        if ln["x_bot"] < W / 2:
            left = ln
        else:
            right = ln
    return left, right, claimed


def _segfit(ln, ya, yb):
    m = (ln["ys"] >= ya) & (ln["ys"] <= yb)
    if m.sum() < 10 or np.ptp(ln["ys"][m]) < 4:
        return None
    return np.polyfit(ln["ys"][m], ln["xs"][m], 1)


def _row_centers(ln):
    """每一列白線像素的平均 x（不用擬合曲線，坡道轉折處才不會被抹平）。"""
    cnt = np.bincount(ln["ys"], minlength=H)
    sx = np.bincount(ln["ys"], weights=ln["xs"], minlength=H)
    rows = np.nonzero(cnt)[0]
    return rows, sx[rows] / cnt[rows]


def ground_grade(left, right, v_h, hfov):
    """假設近處是平地，把兩條線投影回地面，用「路寬應該固定」反推前方路面高度。

    在相機高度 = 1 的單位下：投到平地的點若實際高 z，看起來的路寬會放大 1/(1-z) 倍。
    所以 r = 路寬 / 近處路寬 → z = 1 - 1/r，實際距離 = 投影距離 / r。
    彎道不影響（量的是垂直於線的路寬），回傳遠段的坡度（度，正 = 上坡）與高度剖面。
    """
    f = (W / 2) / math.tan(math.radians(hfov / 2))
    p = math.atan((H / 2 - v_h) / f)
    rl, xl = _row_centers(left)
    rr, xr = _row_centers(right)
    lo, hi = max(rl.min(), rr.min(), v_h + 4), min(rl.max(), rr.max())
    if hi - lo < 20:
        return None
    yy = np.arange(math.ceil(lo), math.floor(hi) + 1, dtype=float)
    k = np.ones(5) / 5
    xl = np.convolve(np.interp(yy, rl, xl), k, "same")[2:-2]
    xr = np.convolve(np.interp(yy, rr, xr), k, "same")[2:-2]
    yy = yy[2:-2]

    def to_ground(x, y):
        zc = f / ((y - v_h) * math.cos(p))
        return (x - W / 2) * zc / f, (zc - math.sin(p)) / math.cos(p)

    XL, DL = to_ground(xl, yy)
    XR, DR = to_ground(xr, yy)
    d2 = (XL[:, None] - XR[None, :]) ** 2 + (DL[:, None] - DR[None, :]) ** 2
    j = d2.argmin(1)
    ok = (j > 0) & (j < len(yy) - 1)  # 最近點落在另一條線端點上 → 不可靠
    if ok.sum() < 10:
        return None
    w = np.sqrt(d2[np.arange(len(yy)), j])[ok]
    D = ((DL + DR[j]) / 2)[ok]
    order = np.argsort(D)
    w, D = w[order], D[order]
    n = len(D)
    w0 = np.median(w[: max(3, n // 4)])
    if not w0 > 1e-6:
        return None
    r = w / w0
    keep = np.isfinite(r) & (r > 0.3) & (r < 3)  # 路寬差太多倍的點是誤配對
    r, D = r[keep], D[keep]
    n = len(r)
    z = 1 - 1 / r
    Dt = D / r
    far = slice(int(n * 0.6), n)
    if n - int(n * 0.6) < 5 or np.ptp(Dt[far]) < 1e-3:
        return None
    a = np.polyfit(Dt[far], z[far], 1)[0]
    return math.degrees(math.atan(a)), (Dt, z)


def _width_at(left, right, y):
    """第 y 列的路寬佔畫面寬的比例。"""
    return float(np.polyval(right["fit"], y) - np.polyval(left["fit"], y)) / W


def _vp(fl, fr):
    """兩條直線 x = m·y + c 的交點；幾乎平行時回傳 None。"""
    dm = fl[0] - fr[0]
    if abs(dm) < 1e-3:
        return None
    y = (fr[1] - fl[1]) / dm
    return float(np.polyval(fl, y)), float(y)


# ---------------------------------------------------------------- 單張分析

def analyse(bgr, cfg, base=None, mask=None):
    """mask：外部給的白線遮罩（W×H，0 / 255，例如 YOLO 分割模型的輸出），有給就取代 Otsu 二值化。"""
    img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    y0 = int(round(cfg.roi_top * H))
    y1 = max(y0 + 10, int(round(getattr(cfg, "roi_bottom", 1.0) * H)))  # 下緣以下是車身，不算
    roi_h = y1 - y0
    g_roi = blur[y0:y1]

    otsu, _ = cv2.threshold(g_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thr = cfg.thresh if cfg.thresh else max(otsu, cfg.min_thresh)
    _, b_roi = cv2.threshold(g_roi, thr, 255, cv2.THRESH_BINARY)
    b_roi = cv2.morphologyEx(b_roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if mask is not None:
        b_roi = np.ascontiguousarray(mask[y0:y1]).astype(np.uint8)
    binary = np.zeros((H, W), np.uint8)
    binary[y0:y1] = b_roi

    r = {"roi_top": y0, "roi_bot": y1, "thresh": float(thr)}
    # --- 畫質
    white = g_roi[b_roi > 0]
    black = g_roi[b_roi == 0]
    r["white_mean"] = float(white.mean()) if white.size else 0.0
    r["black_mean"] = float(black.mean()) if black.size else 0.0
    r["contrast"] = r["white_mean"] - r["black_mean"] if white.size else 0.0
    r["overexp"] = float((gray[y0:y1] >= 250).mean())
    r["white_frac"] = float((b_roi > 0).mean())
    r["sharp"] = float(cv2.Laplacian(gray[y0:y1], cv2.CV_64F).var())

    if r["white_frac"] > 0.5:  # 一半以上都是白的：二值化失效（過曝或門檻錯），不找線
        binary[:] = 0
    left, right, claimed = find_lines(binary, y0, cfg, y1)
    r["n_lines"] = (left is not None) + (right is not None)
    r["clutter"] = float(((binary > 0) & ~claimed)[y0:y1].mean())
    r["confidence"] = r["n_lines"] / 2.0

    lines = [l for l in (left, right) if l is not None]
    r["lookahead"] = max([(y1 - l["top"]) / roi_h for l in lines], default=0.0)
    r["top_row"] = min([l["top"] for l in lines], default=H)
    r["thick_far"] = min([l["thick_far"] for l in lines], default=0.0)
    r["exits"] = "/".join(l["exit"] for l in lines)
    bows = [l["bow"] for l in lines if l["bow"] is not None]
    r["bow_l"] = left["bow"] if left is not None else None
    r["bow_r"] = right["bow"] if right is not None else None

    # --- 近處路寬、置中（用擬合曲線延伸到 ROI 最下面那一列）
    r["lane_w_bot"] = r["center_bot"] = None
    if left is not None and right is not None:
        xl = float(np.polyval(left["fit"], y1 - 1))
        xr = float(np.polyval(right["fit"], y1 - 1))
        r["lane_w_bot"] = (xr - xl) / W
        r["center_bot"] = ((xl + xr) / 2 - W / 2) / W

    # --- 近段 / 遠段消失點 → 地平線 → 坡度變化
    r["h_near"] = r["h_far"] = r["slope_deg"] = None
    r["vp_near"] = r["vp_far"] = None
    f_px = (W / 2) / math.tan(math.radians(cfg.hfov / 2))
    if left is not None and right is not None:
        ct, cb = max(left["top"], right["top"]), min(left["bot"], right["bot"])
        span = cb - ct
        r["common_span"] = span / roi_h
        if span >= roi_h * cfg.min_span:
            seg = 0.4 * span
            nl, nr = _segfit(left, cb - seg, cb), _segfit(right, cb - seg, cb)
            fl, fr = _segfit(left, ct, ct + seg), _segfit(right, ct, ct + seg)
            if all(v is not None for v in (nl, nr, fl, fr)):
                vn, vf = _vp(nl, nr), _vp(fl, fr)
                r["seg"] = (nl, nr, fl, fr, ct, cb, seg)
                if vn and vf:
                    r["vp_near"], r["vp_far"] = vn, vf
                    r["h_near"], r["h_far"] = vn[1], vf[1]
                    a_n = math.degrees(math.atan((H / 2 - vn[1]) / f_px))
                    a_f = math.degrees(math.atan((H / 2 - vf[1]) / f_px))
                    r["slope_deg"] = a_f - a_n
        # 有基準時用基準的地平線（車身所在平面），沒有就用這一張近段的消失點
        v_h = base["h_near"] if base and base.get("h_near") is not None else r["h_near"]
        if v_h is not None and v_h < y0:
            g = ground_grade(left, right, v_h, cfg.hfov)
            if g is not None:
                r["slope_deg"] = g[0]
    else:
        r["common_span"] = 0.0

    if base:
        if r["slope_deg"] is not None:
            r["slope_deg"] -= base.get("slope_deg", 0.0)
        if r["bow_l"] is not None:
            r["bow_l"] -= base.get("bow_l", 0.0)
        if r["bow_r"] is not None:
            r["bow_r"] -= base.get("bow_r", 0.0)
        bows = [b for b in (r["bow_l"], r["bow_r"]) if b is not None]

    # --- 路段判斷
    r["bow"] = float(np.mean(bows)) if bows else None
    if not lines:
        r["shape"] = "none"
    elif r["bow"] is None:
        r["shape"] = "straight" if r["n_lines"] == 2 else "none"
    elif r["bow"] > cfg.curve_px:
        r["shape"] = "left"
    elif r["bow"] < -cfg.curve_px:
        r["shape"] = "right"
    else:
        r["shape"] = "straight"
    # 兩條線在差不多同一列同時「斷掉」、斷掉時線還很粗（不是遠到看不清）：
    # 前方路面被擋住，通常是坡頂、下坡開始
    r["cutoff"] = bool(left is not None and right is not None
                       and left["exit"] == "end" and right["exit"] == "end"
                       and abs(left["top"] - right["top"]) <= 0.08 * roi_h
                       and r["thick_far"] >= 3.0
                       and _width_at(left, right, max(left["top"], right["top"])) >= cfg.cutoff_w)
    if r["slope_deg"] is not None and r["slope_deg"] > cfg.slope_deg:
        r["slope"] = "uphill"
    elif r["cutoff"] or (r["slope_deg"] is not None and r["slope_deg"] < -cfg.slope_deg):
        r["slope"] = "downhill"
    elif r["slope_deg"] is None:
        r["slope"] = "unknown"
    else:
        r["slope"] = "flat"

    r["checks"] = frame_checks(r, cfg)
    r["_img"], r["_bin"], r["_left"], r["_right"] = img, binary, left, right
    r["_claimed"] = claimed
    return r


# ---------------------------------------------------------------- 判定標準

def _grade(v, ok, warn, higher_better=True):
    if v is None:
        return "FAIL"
    if higher_better:
        return "OK" if v >= ok else "WARN" if v >= warn else "FAIL"
    return "OK" if v <= ok else "WARN" if v <= warn else "FAIL"


def frame_checks(r, cfg):
    c = {}
    c["exposure"] = _grade(r["overexp"], 0.03, 0.10, higher_better=False)
    c["contrast"] = _grade(r["contrast"], 80, 50)
    c["sharpness"] = _grade(r["sharp"], 60, 25)
    c["clutter"] = _grade(r["clutter"], 0.01, 0.03, higher_better=False)
    c["lines"] = {2: "OK", 1: "WARN", 0: "FAIL"}[r["n_lines"]]
    c["far_width"] = _grade(r["thick_far"], 3.0, 2.0) if r["n_lines"] else "FAIL"
    c["lookahead"] = _grade(r["lookahead"], 0.6, 0.35)
    if r["lane_w_bot"] is None:
        c["lane_width"] = "WARN"
    else:
        w = r["lane_w_bot"]
        c["lane_width"] = "OK" if 0.35 <= w <= 1.0 else "WARN" if 0.2 <= w <= 1.6 else "FAIL"
    if r["h_near"] is None:
        c["horizon"] = "WARN"
    else:
        h = r["h_near"]
        c["horizon"] = "FAIL" if h > r["roi_top"] else "WARN" if h < -3 * H else "OK"
    return c


CHECK_ZH = {
    "exposure": ("曝光", "過曝像素比例", "{:.1%}"),
    "contrast": ("對比", "白線與黑底的灰階差", "{:.0f}"),
    "sharpness": ("清晰度", "Laplacian 變異數", "{:.0f}"),
    "clutter": ("雜訊", "不屬於兩條線的白點比例", "{:.2%}"),
    "lines": ("抓到線數", "左右兩條都抓到", "{}"),
    "far_width": ("遠端線寬", "最遠處白線的像素寬度", "{:.1f} px"),
    "lookahead": ("前視距離", "白線往上延伸到 ROI 的比例", "{:.0%}"),
    "lane_width": ("近處路寬", "ROI 最下面一列的路寬 / 畫面寬", "{:.0%}"),
    "horizon": ("地平線位置", "近段消失點的列（負值在畫面上方）", "{:.0f}"),
}
CHECK_KEY = {"exposure": "overexp", "contrast": "contrast", "sharpness": "sharp", "clutter": "clutter",
             "lines": "n_lines", "far_width": "thick_far", "lookahead": "lookahead",
             "lane_width": "lane_w_bot", "horizon": "h_near"}
ADVICE = {
    "exposure": "畫面有過曝（反光或曝光太高）：用 v4l2-ctl 關掉自動曝光、調低曝光值，或改變鏡頭角度避開燈光反射。",
    "contrast": "白線和黑底的對比不足：調高曝光或補光；若是黑底反光，降低曝光並避開直射燈。",
    "sharpness": "畫面模糊：檢查對焦、鏡頭髒污；行駛中模糊代表曝光時間太長，要調短曝光並補光。",
    "clutter": "ROI 裡有很多不是白線的白點：多半是 ROI 包到地平線以上的背景、或黑底反光。調低 --roi-top 範圍或調整角度。",
    "lines": "有時抓不到兩條線：直線路段一定要兩條都看到，否則鏡頭太低、太偏或視角太窄。",
    "far_width": "遠處的白線太細（< 2~3 px）：遠段會時有時無。可以讓鏡頭往下壓一點、或提高辨識解析度。",
    "lookahead": "看得不夠遠：白線只佔 ROI 下方一小段，來不及看到彎道/坡道。鏡頭往上抬、或往高處裝。",
    "lane_width": "近處路寬比例不理想：>100% 代表 ROI 下緣處兩條線已在畫面外（鏡頭太低或視角太窄），車子一偏就只剩一條線；"
                  "<35% 代表看太遠、路面解析度不足。",
    "horizon": "地平線位置不理想：落在 ROI 內代表背景會被當成路面（調小 ROI 或鏡頭往下）；在畫面上方很遠代表鏡頭太朝下，坡道判斷會變弱。",
}


# ---------------------------------------------------------------- 疊圖

def draw(r):
    img = r["_img"].copy()
    bin_c = cv2.cvtColor(r["_bin"], cv2.COLOR_GRAY2BGR) // 3
    y0 = r["roi_top"]
    cv2.line(img, (0, y0), (W, y0), (0, 200, 255), 1)
    if r["roi_bot"] < H:
        cv2.line(img, (0, r["roi_bot"]), (W, r["roi_bot"]), (0, 200, 255), 1)
        cv2.line(bin_c, (0, r["roi_bot"]), (W, r["roi_bot"]), (0, 200, 255), 1)
    for ln, col in ((r["_left"], (80, 255, 80)), (r["_right"], (255, 120, 255))):
        if ln is None:
            continue
        bin_c[ln["ys"], ln["xs"]] = col
        for (xl, ya, xr, yb) in ln["wins"]:
            cv2.rectangle(bin_c, (xl, ya), (xr - 1, yb - 1), (0, 200, 255), 1)
        yy = np.arange(ln["top"], ln["bot"] + 1)
        pts = np.stack([np.polyval(ln["fit"], yy), yy], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 2)
    if "seg" in r:
        nl, nr, fl, fr, ct, cb, seg = r["seg"]
        for f, (ya, yb), col in ((nl, (cb - seg, cb), (0, 255, 255)), (nr, (cb - seg, cb), (0, 255, 255)),
                                 (fl, (ct, ct + seg), (255, 200, 0)), (fr, (ct, ct + seg), (255, 200, 0))):
            ytop = max(-H, min(ya, (r["h_near"] if f is nl or f is nr else r["h_far"]) or ya))
            p0 = (int(np.polyval(f, yb)), int(yb))
            p1 = (int(np.polyval(f, ytop)), int(ytop))
            cv2.line(img, p0, p1, col, 1, cv2.LINE_AA)
        for h, col in ((r["h_near"], (0, 255, 255)), (r["h_far"], (255, 200, 0))):
            if h is not None and 0 <= h < H:
                cv2.line(img, (0, int(h)), (W, int(h)), col, 1)
    top = np.hstack([img, bin_c])
    panel = np.zeros((78, 2 * W, 3), np.uint8)
    worst = max(r["checks"].values(), key=lambda s: LEVEL[s])
    vcol = {"OK": (80, 220, 80), "WARN": (0, 200, 255), "FAIL": (60, 60, 255)}[worst]
    slope = f"{r['slope_deg']:+.1f}deg" if r["slope_deg"] is not None else "n/a"
    bow = f"{r['bow']:+.1f}px" if r["bow"] is not None else "n/a"
    txt = [
        f"VIEW {worst}   shape={r['shape']}  slope={r['slope']} ({slope})  bow={bow}"
        + ("  CUTOFF" if r["cutoff"] else ""),
        f"lines={r['n_lines']}  look={r['lookahead']:.0%}  farW={r['thick_far']:.1f}px  "
        f"contrast={r['contrast']:.0f}  overexp={r['overexp']:.1%}  clutter={r['clutter']:.1%}",
        "  ".join(f"{k}:{v}" for k, v in r["checks"].items() if v != "OK") or "all checks OK",
    ]
    for i, t in enumerate(txt):
        cv2.putText(panel, t, (6, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    vcol if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([top, panel])


# ---------------------------------------------------------------- 彙總報告

def _label_ok(label, r):
    if label in ("straight", "left", "right"):
        return r["shape"] == label
    if label == "curve":
        return r["shape"] in ("left", "right")
    if label in ("uphill", "downhill", "flat"):
        return r["slope"] == label
    return None


def summarize(results, label=None):
    n = len(results)
    s = {"frames": n}
    if not n:
        return s
    s["detect_rate"] = sum(r["n_lines"] > 0 for r in results) / n
    s["both_rate"] = sum(r["n_lines"] == 2 for r in results) / n
    s["checks"] = {}
    for k in results[0]["checks"]:
        cnt = {lv: sum(r["checks"][k] == lv for r in results) / n for lv in LEVEL}
        vals = [r[CHECK_KEY[k]] for r in results if r[CHECK_KEY[k]] is not None]
        med = float(np.median(vals)) if vals else None
        verdict = "FAIL" if cnt["FAIL"] > 0.1 else "WARN" if cnt["FAIL"] + cnt["WARN"] > 0.2 else "OK"
        s["checks"][k] = {"verdict": verdict, "median": med, **cnt}
    s["shape"] = {k: sum(r["shape"] == k for r in results) / n for k in SHAPE_ZH}
    s["slope"] = {k: sum(r["slope"] == k for r in results) / n for k in SLOPE_ZH}
    s["cutoff_rate"] = sum(r["cutoff"] for r in results) / n
    for key in ("slope_deg", "bow", "bow_l", "bow_r", "h_near", "center_bot", "lane_w_bot"):
        v = [r[key] for r in results if r[key] is not None]
        s[key] = {"median": float(np.median(v)), "std": float(np.std(v))} if v else None
    if label:
        oks = [_label_ok(label, r) for r in results]
        s["label"] = label
        s["label_acc"] = sum(bool(o) for o in oks) / n
    s["verdict"] = max((c["verdict"] for c in s["checks"].values()), key=lambda v: LEVEL[v])
    if s["detect_rate"] < 0.98:
        s["verdict"] = "FAIL"
    return s


def print_report(s, cfg):
    if not s.get("frames"):
        print("沒有任何畫面")
        return
    p = print
    p("\n" + "=" * 64)
    p(f" 鏡頭視角檢查報告   共 {s['frames']} 張   總判定：{s['verdict']}")
    p("=" * 64)
    p(f" 抓到線的比例   {s['detect_rate']:.1%}（標準 ≥ 98%）   兩條都抓到 {s['both_rate']:.1%}")
    p("\n 項目          判定   中位數        OK / WARN / FAIL 比例")
    for k, c in s["checks"].items():
        name, _, fmt = CHECK_ZH[k]
        med = fmt.format(c["median"]) if c["median"] is not None else "-"
        p(f"  {name:<10}{c['verdict']:<6} {med:<12}  {c['OK']:.0%} / {c['WARN']:.0%} / {c['FAIL']:.0%}")
    sh = "  ".join(f"{SHAPE_ZH[k]} {v:.0%}" for k, v in s["shape"].items() if v)
    sl = "  ".join(f"{SLOPE_ZH[k]} {v:.0%}" for k, v in s["slope"].items() if v)
    p(f"\n 路段判斷（形狀）  {sh}")
    p(f" 路段判斷（坡度）  {sl}")
    if s["slope_deg"]:
        p(f"   坡度變化 {s['slope_deg']['median']:+.1f}° ± {s['slope_deg']['std']:.1f}"
          f"（> +{cfg.slope_deg}° 判上坡，< -{cfg.slope_deg}° 判下坡）")
    if s["bow"]:
        p(f"   彎曲量 bow {s['bow']['median']:+.1f} px ± {s['bow']['std']:.1f}"
          f"（|bow| > {cfg.curve_px} px 判彎道，正 = 左彎）")
    if s["cutoff_rate"] > 0.2:
        p(f"   {s['cutoff_rate']:.0%} 的畫面兩條線同時中斷：前方路面被擋住（坡頂）或賽道盡頭")
    if s["center_bot"]:
        p(f"   車道中心偏離畫面中心 {s['center_bot']['median']:+.1%}（抖動 ± {s['center_bot']['std']:.1%}）")
    if "label" in s:
        p(f"\n 標記為「{s['label']}」，判斷正確率 {s['label_acc']:.0%}（建議 ≥ 90%）")
    bad = [k for k, c in s["checks"].items() if c["verdict"] != "OK"]
    if bad:
        p("\n 建議：")
        for k in bad:
            p(f"  - {ADVICE[k]}")
    p("=" * 64)


# ---------------------------------------------------------------- 影像來源

def open_source(src):
    """回傳一個 generator，逐張產生 (name, bgr)。"""
    if src.isdigit():
        dev = int(src)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(dev)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"無法開啟 /dev/video{dev}，先用 `v4l2-ctl --list-devices` 確認裝置編號")
        for _ in range(10):  # 讓曝光穩定
            cap.read()

        def gen():
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                yield f"cam_{i:05d}", f
                i += 1
        return gen(), True
    if os.path.isdir(src):
        files = sorted(sum((glob.glob(os.path.join(src, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")), []))
        return ((os.path.basename(f), cv2.imread(f)) for f in files), False
    ext = os.path.splitext(src)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        return iter([(os.path.basename(src), cv2.imread(src))]), False
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 {src}")

    def vgen():
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f"frame_{i:05d}", f
            i += 1
    return vgen(), False


# ---------------------------------------------------------------- MJPEG 串流

class Stream:
    def __init__(self, port):
        self.jpg = None
        self.lock = threading.Lock()
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with stream.lock:
                            jpg = stream.jpg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    def push(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpg = jpg.tobytes()


# ---------------------------------------------------------------- 主程式

CSV_KEYS = ["name", "shape", "slope", "slope_deg", "bow", "bow_l", "bow_r", "cutoff", "n_lines", "confidence",
            "lookahead", "top_row", "thick_far", "lane_w_bot", "center_bot", "h_near", "h_far", "contrast",
            "overexp", "clutter", "sharp", "thresh", "exits"]


def run(cfg):
    base = None
    if cfg.baseline:
        with open(cfg.baseline, encoding="utf-8") as f:
            base = json.load(f)
        if cfg.roi_top is None and base.get("roi_top") is not None:
            cfg.roi_top = base["roi_top"]
        print(f"已載入基準 {cfg.baseline}")
    if cfg.roi_top is None:
        cfg.roi_top = 0.35

    frames, live = open_source(cfg.source)
    stream = Stream(cfg.port) if live and not cfg.no_stream else None
    if stream:
        print(f"即時畫面：瀏覽器開 http://<orin-ip>:{cfg.port}/   (Ctrl+C 結束並輸出報告)")
    if cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        fcsv = open(os.path.join(cfg.out, "frames.csv"), "w", newline="", encoding="utf-8")
        wcsv = csv.DictWriter(fcsv, CSV_KEYS, extrasaction="ignore")
        wcsv.writeheader()

    results = []
    t_last = time.time()
    t_start = time.time()
    try:
        for name, bgr in frames:
            if bgr is None:
                continue
            r = analyse(bgr, cfg, base)
            r["name"] = name
            vis = draw(r) if (stream or cfg.out or cfg.show) else None
            if stream:
                stream.push(vis)
            if cfg.show:
                cv2.imshow("lane_view_check", vis)
                if cv2.waitKey(0 if not live else 1) & 0xFF == ord("q"):
                    break
            if cfg.out:
                wcsv.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in CSV_KEYS})
                if not live or len(results) % cfg.save_every == 0 or LEVEL[max(r["checks"].values(), key=LEVEL.get)] == 2:
                    cv2.imwrite(os.path.join(cfg.out, os.path.splitext(name)[0] + ".jpg"), vis)
            for k in [k for k in r if k.startswith("_")]:
                del r[k]
            r.pop("seg", None)
            results.append(r)
            if not live:
                bad = [k for k, v in r["checks"].items() if v != "OK"]
                sd = f"{r['slope_deg']:+5.1f}°" if r["slope_deg"] is not None else "   - "
                bw = f"{r['bow']:+5.1f}" if r["bow"] is not None else "   - "
                print(f"{name:<28} {SHAPE_ZH[r['shape']]:<4} {SLOPE_ZH[r['slope']]:<6} 坡度{sd} bow{bw}"
                      f"  線{r['n_lines']}  前視{r['lookahead']:.0%}  {' '.join(bad)}")
            elif time.time() - t_last >= 2:
                s = summarize(results[-30:])
                fps = len(results) / (time.time() - t_start)
                print(f"[{len(results):5d}] {fps:4.1f} fps  最近：{SHAPE_ZH[r['shape']]} {SLOPE_ZH[r['slope']]}  "
                      f"抓線 {s['detect_rate']:.0%}  前視 {r['lookahead']:.0%}  對比 {r['contrast']:.0f}  "
                      f"判定 {s['verdict']}")
                t_last = time.time()
            if cfg.frames and len(results) >= cfg.frames:
                break
    except KeyboardInterrupt:
        pass

    s = summarize(results, cfg.label)
    print_report(s, cfg)
    if cfg.out:
        fcsv.close()
        with open(os.path.join(cfg.out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        print(f"逐張結果與標註圖存在 {cfg.out}/")
    if cfg.save_baseline and results:
        med = lambda k: float(np.median([r[k] for r in results if r[k] is not None] or [0.0]))
        b = {"slope_deg": med("slope_deg"), "bow_l": med("bow_l"), "bow_r": med("bow_r"),
             "h_near": med("h_near"), "lane_w_bot": med("lane_w_bot"), "roi_top": cfg.roi_top,
             "roi_bottom": cfg.roi_bottom,
             "hfov": cfg.hfov, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        if s["both_rate"] < 0.9:
            print("警告：建立基準時有超過 10% 的畫面沒抓到兩條線，基準可能不準。請停在平坦直線正中央重拍。")
        with open(cfg.save_baseline, "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=2)
        print(f"已存基準 {cfg.save_baseline}：{b}")
    return s


def selftest(cfg):
    """用合成賽道畫面確認演算法能分出直線、左右彎、上下坡。"""
    import lane_synth
    if cfg.roi_top is None:
        cfg.roi_top = 0.1  # 合成畫面的地平線在畫面上方，ROI 可以開大
    out = cfg.out or "selftest_out"
    os.makedirs(out, exist_ok=True)
    # 跟實車一樣：先在平坦直線上建立基準
    straight = lane_synth.render()
    r0 = analyse(straight, cfg)
    base = {"h_near": r0["h_near"], "slope_deg": 0.0, "bow_l": 0.0, "bow_r": 0.0}
    r0 = analyse(straight, cfg, base)
    base.update(slope_deg=r0["slope_deg"] or 0.0, bow_l=r0["bow_l"] or 0.0, bow_r=r0["bow_r"] or 0.0)
    print(f"基準：地平線 y={base['h_near']:.1f}  坡度偏差 {base['slope_deg']:+.2f}°\n")
    total = good = 0
    print(f"{'場景':<22}{'預期':<10}{'判斷':<14}{'坡度°':>7}{'bow':>7}  前視  檢查")
    for name, expect, scene in lane_synth.SCENES:
        img = lane_synth.render(**scene)
        r = analyse(img, cfg, base)
        cv2.imwrite(os.path.join(out, name + ".jpg"), draw(r))
        cv2.imwrite(os.path.join(out, name + "_raw.png"), img)
        got = (r["shape"], r["slope"])
        ok = all(e is None or e == g for e, g in zip(expect, got))
        total += 1
        good += ok
        sd = f"{r['slope_deg']:+.1f}" if r["slope_deg"] is not None else "-"
        bw = f"{r['bow']:+.1f}" if r["bow"] is not None else "-"
        exp = "/".join(e or "*" for e in expect)
        bad = " ".join(k for k, v in r["checks"].items() if v != "OK")
        print(f"{name:<22}{exp:<10}{got[0] + '/' + got[1]:<14}{sd:>7}{bw:>7}  {r['lookahead']:.0%}  "
              f"{'PASS' if ok else 'MISS'} {bad}")
    print(f"\n自我測試 {good}/{total} 通過，標註圖在 {out}/")
    return good == total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="鏡頭編號、影片、圖片或資料夾（預設 0）")
    ap.add_argument("--frames", type=int, default=0, help="處理幾張就結束並出報告（0 = 直到 Ctrl+C / 檔案結束）")
    ap.add_argument("--label", choices=["straight", "left", "right", "curve", "uphill", "downhill", "flat"],
                    help="這段測試的實際路段，用來算判斷正確率")
    ap.add_argument("--out", help="輸出資料夾：frames.csv、report.json、標註圖")
    ap.add_argument("--save-every", type=int, default=15, help="鏡頭模式下每幾張存一張標註圖")
    ap.add_argument("--baseline", help="讀入基準檔（平坦直線上用 --save-baseline 建立）")
    ap.add_argument("--save-baseline", help="把這次的中位數存成基準檔")
    ap.add_argument("--roi-top", type=float, default=None, help="ROI 從畫面高度的多少比例開始（預設 0.35）")
    ap.add_argument("--roi-bottom", type=float, default=1.0,
                    help="ROI 到畫面高度的多少比例為止（畫面下緣拍到車身時調小，例如 0.85）")
    ap.add_argument("--thresh", type=int, default=0, help="固定二值化門檻（0 = Otsu）")
    ap.add_argument("--min-thresh", type=int, default=60, help="Otsu 門檻的下限，避免全黑畫面被雜訊切出白點")
    ap.add_argument("--hfov", type=float, default=70.0, help="鏡頭水平視角（度），只影響坡度角度的換算")
    ap.add_argument("--curve-px", type=float, default=2.5, help="|bow| 超過幾像素算彎道")
    ap.add_argument("--slope-deg", type=float, default=3.0, help="坡度變化超過幾度算上/下坡")
    ap.add_argument("--min-span", type=float, default=0.3, help="兩條線共同高度至少佔 ROI 多少才判斷坡度")
    ap.add_argument("--cutoff-w", type=float, default=0.2,
                    help="兩條線同時斷掉時，斷點的路寬還 ≥ 畫面寬的這個比例，視為前方路面被擋住（坡頂）")
    ap.add_argument("--nwin", type=int, default=14)
    ap.add_argument("--margin", type=int, default=18)
    ap.add_argument("--minpix", type=int, default=6)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-stream", action="store_true", help="鏡頭模式不開瀏覽器串流")
    ap.add_argument("--show", action="store_true", help="用 cv2.imshow 顯示（需要螢幕）")
    ap.add_argument("--selftest", action="store_true", help="用合成畫面測試演算法")
    cfg = ap.parse_args()
    if cfg.selftest:
        sys.exit(0 if selftest(cfg) else 1)
    s = run(cfg)
    sys.exit(0 if s.get("verdict") in ("OK", "WARN") else 1)


if __name__ == "__main__":
    main()
