#!/usr/bin/env python3
"""產生作品集用的完整賽道圖 docs/track_map.png。

路線不是手畫的：用 orin_tools/drive/track_map.json（車子 9 圈平均的車頭角，每 0.1 m 一點）積分出行駛路線，
頭尾誤差（推算里程，約 4 m）照距離比例平均分攤讓它閉合。再標上各路段、隧道、岔路，旁邊放賽道 / 車子照片。

  python docs/tools/make_track_map.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib import font_manager                               # noqa: E402
from matplotlib.patches import FancyArrowPatch                    # noqa: E402
import numpy as np                                                # noqa: E402
from PIL import Image                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP = os.path.join(ROOT, "orin_tools", "drive", "track_map.json")
OUT = os.path.join(ROOT, "docs", "track_map.png")
PHOTOS = os.path.join(ROOT, "docs", "賽道照片")
CAR = os.path.join(ROOT, "docs", "車子照片.jpg")

for f in (r"C:\Windows\Fonts\NotoSansTC-VF.ttf", r"C:\Windows\Fonts\msjh.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BG, ROAD, LINE, UP = "#f4f1ea", "#2b2b2b", "#ffffff", "#c8873a"
ACC = "#d1495b"


def path():
    m = json.load(open(MAP, encoding="utf-8"))
    psi = np.radians(np.array(m["psi"]))
    ds = m["ds"]
    s = np.arange(len(psi) + 1) * ds
    x = np.concatenate([[0], np.cumsum(np.cos(psi)) * ds])
    y = np.concatenate([[0], np.cumsum(np.sin(psi)) * ds])
    x -= x[-1] * s / s[-1]
    y -= y[-1] * s / s[-1]
    # 輕微平滑（推算里程的鋸齒）
    k = np.ones(9) / 9
    xs = np.convolve(np.concatenate([x[-5:-1], x, x[1:5]]), k, "same")[4:-4]
    ys = np.convolve(np.concatenate([y[-5:-1], y, y[1:5]]), k, "same")[4:-4]
    return m, s, xs, ys


def at(s, x, y, v):
    i = int(np.argmin(np.abs(s - v)))
    return x[i], y[i], i


def draw_road(ax, x, y, color, lw, z):
    ax.plot(x, y, color=LINE, lw=lw + 3, solid_capstyle="round", zorder=z)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=z + 0.1)


def main():
    m, s, x, y = path()
    L = m["L"]
    fig = plt.figure(figsize=(18, 10.5), facecolor=BG)
    ax = fig.add_axes([0.02, 0.06, 0.62, 0.84])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    up_end = m["down_end"]
    low = s >= up_end
    hi = s <= up_end
    draw_road(ax, x[low], y[low], ROAD, 14, 1)                 # 地面層
    draw_road(ax, x[hi], y[hi], UP, 14, 3)                     # 高架：上坡 → 平台 → 下坡（畫在上層）
    ax.plot(x, y, color="#e9e9e9", lw=1, ls=(0, (4, 4)), zorder=4)

    # 方向箭頭
    for v in (2.5, 15.0, 21.0, 32.0, 38.5, 45.0):
        _, _, i = at(s, x, y, v)
        ax.add_patch(FancyArrowPatch((x[i], y[i]), (x[i + 4], y[i + 4]), arrowstyle="-|>", mutation_scale=22,
                                     color="#ffd166", lw=0, zorder=5))

    def tag(v, text, dx, dy, color=ACC, num=None):
        px, py, _ = at(s, x, y, v)
        ax.plot(px, py, "o", ms=9, mfc=color, mec="white", mew=2, zorder=7)
        label = (f"{num}  " if num else "") + text
        ax.annotate(label, (px, py), (px + dx, py + dy), fontsize=12.5, color="#222", zorder=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.5),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.4))

    tag(0.0, "上坡起點  s=0", -1.6, -0.9, UP)
    tag(m["crest"], "坡頂", 0.2, -1.5, UP)
    tag(m["dash"], "坡頂平台虛線左轉", 1.7, -1.6, UP)
    tag(m["down"], "下坡", 2.0, -0.3, UP)
    tag(18.5, "起點 / 大直線", 1.9, 0.9, "#118ab2", "①")
    tag(25.0, "髮夾彎 1", 1.8, -0.7, ACC)
    tag(m["dark"][0][0], "隧道 1（高架下方）", -2.4, -1.1, "#555", "③")
    tag(m["forks"][0]["s"], "岔路 30.4 m：走左", -2.6, 0.7, "#06a77d", "②")
    tag(35.2, "髮夾彎 2", 1.6, 0.9, ACC)
    tag(m["forks"][1]["s"], "岔路 39.8 m：走右", 2.6, 0.9, "#06a77d")
    tag(m["dark"][1][0], "隧道 2（高架下方）", -2.6, 0.9, "#555")
    tag(45.2, "坡前右急彎", -1.4, -1.4, ACC)

    ax.set_xlim(x.min() - 3.6, x.max() + 3.4)
    ax.set_ylim(y.min() - 2.2, y.max() + 1.8)
    # 比例尺
    x0, y0 = x.min() - 3.2, y.min() - 1.7
    ax.plot([x0, x0 + 2], [y0, y0], color="#222", lw=3)
    ax.text(x0 + 1, y0 + 0.25, "2 m", ha="center", fontsize=11)

    fig.text(0.02, 0.95, "car_race_openCV — 賽道全圖", fontsize=26, weight="bold", color="#222")
    fig.text(0.02, 0.915, f"一圈 {L:.1f} m，兩層立體交叉（上坡 → 坡頂平台 → 下坡），兩個隧道、兩個 Y 字岔路。"
             "路線由車子 9 圈的陀螺儀 + 里程紀錄積分而成（頭尾誤差已平均分攤），非手繪。",
             fontsize=12.5, color="#444")
    fig.text(0.02, 0.025, "■ 橘色＝高架段（上坡、平台、下坡）  ■ 黑色＝地面層  ● 編號對應右側照片", fontsize=12, color="#444")

    # 右側照片
    shots = [("S__67567620_0.jpg", "① 起點（大直線）與坡頂平台"), ("S__67567621_0.jpg", "② Y 字岔路與上坡道"),
             ("S__67567622_0.jpg", "③ 隧道：地面層從高架下方穿過"), ("S__67567623_0.jpg", "④ 全景")]
    W, H = 0.165, 0.27
    for k, (fn, cap) in enumerate(shots):
        cx, cy = 0.655 + (k % 2) * (W + 0.01), 0.60 - (k // 2) * (H + 0.06)
        a = fig.add_axes([cx, cy, W, H])
        a.imshow(Image.open(os.path.join(PHOTOS, fn)))
        a.axis("off")
        a.set_title(cap, fontsize=12, color="#222", pad=4)
    a = fig.add_axes([0.655, 0.03, 0.335, 0.21])
    im = Image.open(CAR)
    a.imshow(im)
    a.axis("off")
    a.set_title("車子：Jetson Orin Nano + BMduino，單鏡頭循線", fontsize=12, color="#222", pad=4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, facecolor=BG)
    print("已存", OUT)


if __name__ == "__main__":
    main()
