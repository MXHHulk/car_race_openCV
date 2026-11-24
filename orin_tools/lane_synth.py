#!/usr/bin/env python3
"""合成黑底白線賽道畫面（針孔相機投影），給 lane_view_check.py --selftest 用。

賽道用「沿中線的里程 s」描述：曲率 kappa(s) 決定彎道，坡度 grade(s)（度）決定上下坡。
由遠到近依序畫路面（黑）和白線，近處的路面自然會擋住坡頂後面的部分。

  python3 lane_synth.py out_dir   # 把所有場景存成 png
"""
import math
import os
import sys

import cv2
import numpy as np


def render(kappa=0.0, curve_from=0.3, grade=0.0, grade_from=0.6, ramp_len=0.15,
           cam_h=0.22, pitch=28.0, hfov=70.0, cam_x=0.0, yaw=0.0,
           lane_w=0.45, line_w=0.05, length=4.0, size=(640, 480), noise=4.0, seed=0,
           kappa_fn=None, grade_fn=None):
    """回傳 BGR 影像。單位：公尺、度。kappa > 0 為左彎；grade > 0 為上坡。

    kappa_fn(s) / grade_fn(s)：可直接給「鏡頭前方 s 公尺處」的曲率 / 坡度（度），取代上面的單一彎道、坡道。
    """
    w, h = size
    f = (w / 2) / math.tan(math.radians(hfov / 2))
    cx, cy = w / 2, h / 2
    ds = 0.01
    s = np.arange(-0.3, length, ds)
    k = np.where(s > curve_from, kappa, 0.0) if kappa_fn is None else kappa_fn(s)
    psi = np.cumsum(k) * ds                         # 行進方向（0 = 正前方，正值往左）
    g = np.clip((s - grade_from) / ramp_len, 0, 1) * math.radians(grade)
    if grade_fn is not None:
        g = np.radians(grade_fn(s))
    X = np.cumsum(-np.sin(psi) * np.cos(g)) * ds    # 右為正
    Y = np.cumsum(np.cos(psi) * np.cos(g)) * ds     # 前為正
    Z = np.cumsum(np.sin(g)) * ds
    i0 = np.argmin(np.abs(s))
    X, Y, Z = X - X[i0], Y - Y[i0], Z - Z[i0]
    nx, ny = -np.cos(psi), -np.sin(psi)             # 左法向量

    p, yw = math.radians(pitch), math.radians(yaw)
    fwd = np.array([-math.sin(yw) * math.cos(p), math.cos(yw) * math.cos(p), -math.sin(p)])
    right = np.array([math.cos(yw), math.sin(yw), 0.0])
    down = np.cross(fwd, right)
    cam = np.array([cam_x, 0.0, cam_h])

    def proj(px, py, pz):
        rel = np.stack([px, py, pz], -1) - cam
        zc = rel @ fwd
        u = cx + f * (rel @ right) / np.maximum(zc, 1e-6)
        v = cy + f * (rel @ down) / np.maximum(zc, 1e-6)
        return np.stack([u, v], -1), zc

    img = np.full((h, w), 18, np.uint8)
    # 黑色路面：只比車道寬一點。太寬（例如 ±1.5 m）時，急彎附近的路面會蓋掉出彎直線上的白線
    floor = lane_w / 2 + 0.12
    strips = [(-floor, floor, 18)]
    for side in (1, -1):                            # 左線、右線
        c = side * lane_w / 2
        strips.append((c - line_w / 2, c + line_w / 2, 225))
    pts = {}
    for a, b, col in strips:
        pa, za = proj(X + nx * a, Y + ny * a, Z)
        pb, zb = proj(X + nx * b, Y + ny * b, Z)
        pts[(a, b)] = (pa, pb, np.minimum(za, zb), col)
    for i in range(len(s) - 2, -1, -1):             # 由遠到近
        for (a, b), (pa, pb, zc, col) in pts.items():
            if zc[i] < 0.03 or zc[i + 1] < 0.03:
                continue
            quad = np.array([pa[i], pb[i], pb[i + 1], pa[i + 1]]) * 8
            cv2.fillConvexPoly(img, quad.astype(np.int32), int(col), lineType=cv2.LINE_AA, shift=3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    rng = np.random.default_rng(seed)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# (名稱, (預期形狀, 預期坡度), 參數)；None 代表不檢查那一項
SCENES = [
    ("straight",            ("straight", "flat"),   {}),
    ("straight_offset_yaw", ("straight", "flat"),   {"cam_x": 0.06, "yaw": 6}),
    ("left_curve_r1.0",     ("left", "flat"),       {"kappa": 1 / 1.0}),
    ("right_curve_r1.0",    ("right", "flat"),      {"kappa": -1 / 1.0}),
    ("left_curve_r0.6",     ("left", None),         {"kappa": 1 / 0.6, "curve_from": 0.2}),
    ("right_curve_r0.6",    ("right", None),        {"kappa": -1 / 0.6, "curve_from": 0.2}),
    ("uphill_15deg",        ("straight", "uphill"), {"grade": 15}),
    ("uphill_10deg_near",   ("straight", "uphill"), {"grade": 10, "grade_from": 0.4}),
    ("downhill_15deg",      ("straight", "downhill"), {"grade": -15, "grade_from": 0.5}),
    ("downhill_10deg_near", ("straight", "downhill"), {"grade": -10, "grade_from": 0.4}),
    ("uphill_15deg_far",    ("straight", "uphill"), {"grade": 15, "grade_from": 1.0}),
    ("left_curve_uphill",   ("left", "uphill"),     {"kappa": 1 / 1.2, "grade": 15}),
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_out"
    os.makedirs(out, exist_ok=True)
    for name, _, sc in SCENES:
        cv2.imwrite(os.path.join(out, name + ".png"), render(**sc))
    print(f"已輸出 {len(SCENES)} 張到 {out}/")
