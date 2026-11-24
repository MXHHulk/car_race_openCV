#!/usr/bin/env python3
"""鏡頭調校工具：在瀏覽器用拉條調整鏡頭設定（曝光、對焦、亮度…）與辨識參數（ROI、門檻），
即時看到辨識結果與畫質指標，調好後存進 lane/config.yaml 或存成場地設定檔。

每換一個場地（燈光不同）就開這支程式重新調一次。

用法（在 ~/orin_tools 底下）：
  python3 cam_tuner.py                 # /dev/video0，電腦或手機瀏覽器開 http://<orin-ip>:8080
  python3 cam_tuner.py -d 1 --port 8081
  python3 cam_tuner.py --source runs/track.mp4   # 沒有鏡頭時用影片，鏡頭設定改為「模擬」

注意：鏡頭同一時間只能給一支程式用，執行前先關掉 main.py、lane_view_check.py、cam_stream.py。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lane"))
import lane_view_check as lvc  # noqa: E402

CONFIG = os.path.join(HERE, "lane", "config.yaml")
BASELINE = os.path.join(HERE, "baseline.json")
PROFILE_DIR = os.path.join(HERE, "cam_profiles")

CTRL_ZH = {
    "brightness": ("亮度", "整體加亮 / 變暗。優先調曝光時間，這個當微調"),
    "contrast": ("對比", "拉大亮暗差，太大會讓白線過曝"),
    "saturation": ("飽和度", "辨識只用灰階，不影響"),
    "hue": ("色相", "辨識只用灰階，不影響"),
    "white_balance_automatic": ("自動白平衡", "灰階辨識影響小，關掉可避免畫面亮度跳動"),
    "gamma": ("Gamma", "調高會讓暗部變亮（黑底會變灰）"),
    "power_line_frequency": ("防閃爍", "台灣用 50 Hz 以外的燈可能閃，室內一般選 60 Hz"),
    "white_balance_temperature": ("色溫", "自動白平衡關掉後才有效"),
    "sharpness": ("銳利度", "太高會在線邊緣產生白邊雜點"),
    "backlight_compensation": ("背光補償", "建議關（0），會把畫面拉亮造成過曝"),
    "auto_exposure": ("曝光模式", "選 Manual 才能手動調曝光時間"),
    "exposure_time_absolute": ("曝光時間", "單位 0.1 ms。越短越暗、跑動時越不糊。最重要的一項。"
                                         "超過約 330（33 ms）鏡頭就會掉幀，車子會來不及轉彎"),
    "exposure_dynamic_framerate": ("動態幀率", "建議關，否則暗處會自動降 fps"),
    "focus_absolute": ("對焦位置", "連續自動對焦關掉後才有效"),
    "focus_automatic_continuous": ("連續自動對焦", "建議關，行駛中會對焦跑掉"),
    "gain": ("增益", "提高亮度但雜訊變多，優先調曝光"),
}
AUTO_FIRST = ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous", "exposure_dynamic_framerate")


# ---------------------------------------------------------------- 鏡頭設定（v4l2-ctl）

class V4L2:
    fake = False

    def __init__(self, dev):
        self.dev = f"/dev/video{dev}"

    def _run(self, *args):
        r = subprocess.run(["v4l2-ctl", "-d", self.dev, *args], capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr.strip()

    def list(self):
        code, out, err = self._run("--list-ctrls-menus")
        if code:
            raise RuntimeError(err)
        ctrls, group, cur = [], "", None
        for line in out.splitlines():
            m = re.match(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$", line)
            if m:
                name, typ, rest = m.groups()
                kv = dict(re.findall(r"(\w+)=(-?\d+|[\w,\-]+)", rest))
                cur = {"name": name, "type": typ, "group": group,
                       "min": int(kv.get("min", 0)), "max": int(kv.get("max", 1)),
                       "step": int(kv.get("step", 1)), "default": int(kv.get("default", 0)),
                       "value": int(kv.get("value", 0)), "inactive": "inactive" in kv.get("flags", ""),
                       "readonly": "read-only" in kv.get("flags", ""), "menu": {}}
                ctrls.append(cur)
                continue
            m = re.match(r"^\s+(\d+):\s*(.+)$", line)
            if m and cur is not None and cur["type"] == "menu":
                cur["menu"][m.group(1)] = m.group(2).strip()
                continue
            if line.strip() and "0x" not in line:
                group, cur = line.strip(), None
        return ctrls

    def set(self, name, value):
        code, _, err = self._run("-c", f"{name}={int(value)}")
        return None if code == 0 else (err or "設定失敗")

    def transform(self, frame):
        return frame


class FakeCtrls:
    """沒有 v4l2-ctl（例如在電腦上用影片測試）時，用軟體模擬曝光 / 亮度 / 對比。"""
    fake = True

    def __init__(self):
        self.c = {
            "brightness": dict(type="int", min=-64, max=64, step=1, default=0, value=0, group="User Controls"),
            "contrast": dict(type="int", min=0, max=95, step=1, default=32, value=32, group="User Controls"),
            "backlight_compensation": dict(type="int", min=0, max=1, step=1, default=1, value=1, group="User Controls"),
            "auto_exposure": dict(type="menu", min=1, max=3, step=1, default=3, value=3, group="Camera Controls",
                                  menu={"1": "Manual Mode", "3": "Aperture Priority Mode"}),
            "exposure_time_absolute": dict(type="int", min=3, max=2047, step=1, default=166, value=166,
                                           group="Camera Controls"),
            "focus_automatic_continuous": dict(type="bool", min=0, max=1, step=1, default=0, value=1,
                                               group="Camera Controls"),
        }

    def list(self):
        out = []
        for k, v in self.c.items():
            inactive = (k == "exposure_time_absolute" and self.c["auto_exposure"]["value"] != 1)
            out.append({"name": k, "readonly": False, "inactive": inactive, "menu": v.get("menu", {}),
                        **{kk: vv for kk, vv in v.items() if kk != "menu"}})
        return out

    def set(self, name, value):
        if name not in self.c:
            return "沒有這個設定"
        c = self.c[name]
        c["value"] = int(min(c["max"], max(c["min"], int(value))))
        return None

    def transform(self, frame):
        c = {k: v["value"] for k, v in self.c.items()}
        f = frame.astype(np.float32)
        if c["auto_exposure"] == 1:
            gain = c["exposure_time_absolute"] / 100.0
        else:
            gain = 120.0 / max(1.0, f.mean())          # 自動曝光：把平均亮度拉到 120（白線容易過曝）
        gain *= 1.25 if c["backlight_compensation"] else 1.0
        f = (f * gain - 128) * (0.6 + c["contrast"] / 80.0) + 128 + c["brightness"] * 1.5
        return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 影像來源

class FileSource:
    """圖片或影片，循環播放，介面和 lane/camera.Camera 相同。"""
    live = False

    def __init__(self, path):
        self.img = cv2.imread(path) if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp") else None
        self.cap = None if self.img is not None else cv2.VideoCapture(path)
        if self.img is None and not self.cap.isOpened():
            raise SystemExit(f"無法開啟 {path}")
        self.fps = 30.0 if self.img is not None else (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._id = 0

    def read(self, last_id=0, timeout=1.0):
        time.sleep(1 / self.fps)
        if self.img is not None:
            f = self.img.copy()
        else:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
                if not ok:
                    return None
        self._id += 1
        return self._id, time.time(), f


# ---------------------------------------------------------------- 調校主體

class Tuner:
    def __init__(self, src, ctrls, device):
        self.src, self.ctrls, self.device = src, ctrls, device
        self.lock = threading.Lock()
        self.ctrl_lock = threading.Lock()
        self.vis_cfg = SimpleNamespace(roi_top=0.35, roi_bottom=1.0, thresh=0, min_thresh=60, hfov=70.0,
                                       curve_px=2.5, slope_deg=3.0, min_span=0.3, cutoff_w=0.2,
                                       nwin=14, margin=18, minpix=6)
        self._load_vision_from_config()
        self.base = self._load_baseline()
        self.jpg, self.metrics, self.frame_no = None, {}, 0
        self.fps, self._fps_t, self._fps_n = 0.0, time.time(), 0
        self.collect = None                 # 建立基準時收集的結果
        self.autoexp = {"running": False}
        self.message = ""
        threading.Thread(target=self._loop, daemon=True).start()

    # --- 設定檔
    def _load_vision_from_config(self):
        try:
            import yaml
            with open(CONFIG, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[tuner] 讀不到 {CONFIG}：{e}")
            return
        v = cfg.get("vision", {})
        for k in ("roi_top", "roi_bottom", "thresh", "min_thresh"):
            if v.get(k) is not None:
                setattr(self.vis_cfg, k, v[k])
        self.vis_cfg.hfov = cfg.get("geometry", {}).get("hfov", 70.0)
        self.config_ctrls = cfg.get("camera", {}).get("v4l2_ctrls") or {}

    def _load_baseline(self):
        if os.path.exists(BASELINE):
            with open(BASELINE, encoding="utf-8") as f:
                return json.load(f)
        return None

    # --- 主迴圈：取像 → 模擬設定 → 辨識 → 疊圖
    def _loop(self):
        last = 0
        while True:
            got = self.src.read(last)
            if got is None:
                continue
            last, _, frame = got
            frame = self.ctrls.transform(frame)
            try:
                r = lvc.analyse(frame, self.vis_cfg, self.base)
            except Exception as e:  # noqa: BLE001  辨識出錯不要讓整個工具掛掉
                self.message = f"辨識錯誤：{e}"
                continue
            vis = lvc.draw(r)
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            m = self._metrics(r, frame)
            with self.lock:
                if ok:
                    self.jpg = jpg.tobytes()
                self.metrics = m
                self.frame_no += 1
            self._fps_n += 1
            if time.time() - self._fps_t >= 1.0:
                self.fps = self._fps_n / (time.time() - self._fps_t)
                self._fps_t, self._fps_n = time.time(), 0
            if self.collect is not None:
                self._collect(r)

    def _metrics(self, r, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (lvc.W, lvc.H)), cv2.COLOR_BGR2GRAY)
        y0, y1 = r["roi_top"], r["roi_bot"]
        hist = np.histogram(gray[y0:y1], bins=32, range=(0, 256))[0]
        checks = []
        for k, lv in r["checks"].items():
            name, desc, fmt = lvc.CHECK_ZH[k]
            v = r[lvc.CHECK_KEY[k]]
            checks.append({"key": k, "name": name, "desc": desc, "level": lv,
                           "value": fmt.format(v) if v is not None else "-", "advice": lvc.ADVICE[k]})
        verdict = max(r["checks"].values(), key=lambda s: lvc.LEVEL[s])
        return {"verdict": verdict, "checks": checks, "hist": (hist / max(1, hist.max())).round(3).tolist(),
                "thresh": r["thresh"], "white_mean": round(r["white_mean"]), "black_mean": round(r["black_mean"]),
                "shape": lvc.SHAPE_ZH[r["shape"]], "slope": lvc.SLOPE_ZH[r["slope"]],
                "slope_deg": None if r["slope_deg"] is None else round(r["slope_deg"], 1),
                "n_lines": r["n_lines"], "contrast": r["contrast"], "overexp": r["overexp"],
                "clutter": r["clutter"], "lookahead": r["lookahead"], "sharp": r["sharp"],
                "center_bot": r["center_bot"], "h_near": r["h_near"]}

    # --- 建立基準
    def start_baseline(self, n=60):
        self.collect = {"n": n, "res": []}

    def _collect(self, r):
        c = self.collect
        c["res"].append({k: r[k] for k in ("slope_deg", "bow_l", "bow_r", "h_near", "lane_w_bot", "n_lines")})
        if len(c["res"]) < c["n"]:
            return
        res = c["res"]
        self.collect = None
        med = lambda k: float(np.median([x[k] for x in res if x[k] is not None] or [0.0]))
        both = sum(x["n_lines"] == 2 for x in res) / len(res)
        if both < 0.9 or all(x["h_near"] is None for x in res):
            self.message = (f"建立基準失敗：只有 {both:.0%} 的畫面看到兩條線。"
                            "請把車放在平坦直線正中央、確認兩條線都有標色後再試。")
            return
        # 基準要在「沒有套用舊基準」的狀態下量；把舊基準的修正加回去
        old = self.base or {}
        b = {"slope_deg": med("slope_deg") + old.get("slope_deg", 0.0),
             "bow_l": med("bow_l") + old.get("bow_l", 0.0), "bow_r": med("bow_r") + old.get("bow_r", 0.0),
             "h_near": med("h_near"), "lane_w_bot": med("lane_w_bot"),
             "roi_top": self.vis_cfg.roi_top, "roi_bottom": self.vis_cfg.roi_bottom,
             "hfov": self.vis_cfg.hfov, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        if os.path.exists(BASELINE):
            shutil.copy(BASELINE, BASELINE + ".bak")
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=2)
        self.base = b
        self.message = f"已建立基準 baseline.json（地平線 y={b['h_near']:.1f}，兩條線 {both:.0%}）"

    # --- 鏡頭設定
    def ctrl_list(self):
        with self.ctrl_lock:
            try:
                return self.ctrls.list()
            except Exception as e:  # noqa: BLE001
                self.message = f"讀取鏡頭設定失敗：{e}"
                return []

    def ctrl_set(self, name, value):
        with self.ctrl_lock:
            return self.ctrls.set(name, value)

    def apply(self, ctrls):
        """依序套用一組設定：自動模式類的先設，其餘的後設。"""
        errs = []
        names = sorted(ctrls, key=lambda k: (k not in AUTO_FIRST, 0))
        for k in names:
            e = self.ctrl_set(k, ctrls[k])
            if e:
                errs.append(f"{k}: {e}")
        return errs

    def current_ctrls(self):
        """目前所有可設定的值（自動模式類排前面）。被自動模式蓋掉的項目設了也無效，不存。"""
        out = {}
        for c in self.ctrl_list():
            if c["readonly"] or c["inactive"]:
                continue
            out[c["name"]] = c["value"]
        return dict(sorted(out.items(), key=lambda kv: (kv[0] not in AUTO_FIRST, 0)))

    # --- 自動找曝光
    def start_autoexp(self):
        if self.autoexp.get("running"):
            return "已經在跑了"
        names = {c["name"]: c for c in self.ctrl_list()}
        if "exposure_time_absolute" not in names:
            return "這顆鏡頭沒有 exposure_time_absolute"
        threading.Thread(target=self._autoexp, args=(names,), daemon=True).start()
        return None

    def _wait_frames(self, n):
        with self.lock:
            target = self.frame_no + n
        t0 = time.time()
        while time.time() - t0 < 5:
            with self.lock:
                if self.frame_no >= target:
                    return dict(self.metrics)
            time.sleep(0.02)
        with self.lock:
            return dict(self.metrics)

    def _autoexp(self, names):
        st = self.autoexp = {"running": True, "progress": 0, "results": [], "best": None}
        ae = names.get("auto_exposure")
        if ae:
            manual = next((int(k) for k, v in ae["menu"].items() if "manual" in v.lower()), 1)
            self.ctrl_set("auto_exposure", manual)
        ex = names["exposure_time_absolute"]
        # 曝光時間超過 1/30 秒（330 × 0.1 ms）鏡頭就無法維持 30 fps，車子會來不及反應，所以不找更長的
        lo, hi = max(ex["min"], 3), min(ex["max"], 330)

        def measure(e):
            self.ctrl_set("exposure_time_absolute", e)
            self._wait_frames(6)                      # 等曝光生效
            ms = [self._wait_frames(1) for _ in range(4)]
            avg = lambda k: float(np.mean([m[k] for m in ms]))
            contrast, over, clutter, lines = avg("contrast"), avg("overexp"), avg("clutter"), avg("n_lines")
            score = (contrast - 3000 * max(0.0, over - 0.02) - 1500 * max(0.0, clutter - 0.01)
                     - 40 * (2 - lines))
            r = {"exposure": int(e), "score": round(score, 1), "contrast": round(contrast),
                 "overexp": round(over * 100, 1), "clutter": round(clutter * 100, 2), "lines": round(lines, 1)}
            st["results"].append(r)
            return r

        def pick(rs):
            # 所有檢查都過（不過曝、沒雜訊、兩條線、對比夠）的裡面挑對比最高的；都不過才用分數
            ok = [r for r in rs if r["overexp"] <= 3 and r["clutter"] <= 1 and r["lines"] >= 2 and r["contrast"] >= 80]
            return max(ok, key=lambda r: r["contrast"]) if ok else max(rs, key=lambda r: r["score"])

        coarse = sorted(set(np.geomspace(lo, hi, 10).astype(int).tolist()))
        for i, e in enumerate(coarse):                # 粗掃：對數間隔
            measure(e)
            st["progress"] = (i + 1) / (len(coarse) + 5)
        b = pick(st["results"])
        j = coarse.index(b["exposure"])
        fine = np.linspace(coarse[max(0, j - 1)], coarse[min(len(coarse) - 1, j + 1)], 7)[1:-1]
        for i, e in enumerate(sorted(set(int(round(x)) for x in fine) - set(coarse))):  # 細掃：最佳值兩側
            measure(e)
            st["progress"] = (len(coarse) + i + 1) / (len(coarse) + 5)
        st["results"].sort(key=lambda r: r["exposure"])
        best = pick(st["results"])
        self.ctrl_set("exposure_time_absolute", best["exposure"])
        st["best"], st["running"] = best, False
        self.message = (f"自動找曝光完成：曝光 {best['exposure']}（對比 {best['contrast']}、過曝 {best['overexp']}%）。"
                        "這是起點，可再用拉條微調。")

    # --- 存檔
    def save_config(self):
        # 存全部設定：只存「和預設不同」的話，等於預設的項目（例如曝光 166）不會寫進去，
        # 下次啟動鏡頭會沿用上一次調整時留下的值
        ctrls = self.current_ctrls()
        with open(CONFIG, encoding="utf-8", newline="") as f:  # 保留原本的換行字元
            s = f.read()
        nl = "\r\n" if "\r\n" in s else "\n"
        s = s.replace("\r\n", "\n")
        block = ("  v4l2_ctrls:\n" + "".join(f"    {k}: {v}\n" for k, v in ctrls.items())) if ctrls \
            else "  v4l2_ctrls: {}\n"
        s, n = re.subn(r"(?m)^  v4l2_ctrls:.*\n(?:(?:    |  #   ).*\n)*", lambda m: block, s)
        if n != 1:
            return "config.yaml 裡找不到 camera.v4l2_ctrls，沒有存檔"
        vc = self.vis_cfg
        for k, v in (("roi_top", round(vc.roi_top, 3)), ("roi_bottom", round(vc.roi_bottom, 3)),
                     ("thresh", int(vc.thresh)), ("min_thresh", int(vc.min_thresh))):
            s, n = re.subn(rf"(?m)^(  {k}:[ \t]*)([^\s#]+)", lambda m, v=v: m.group(1) + str(v), s)
            if n != 1:
                return f"config.yaml 裡找不到 vision.{k}，沒有存檔"
        shutil.copy(CONFIG, CONFIG + ".bak")
        with open(CONFIG, "w", encoding="utf-8", newline="") as f:
            f.write(s.replace("\n", nl))
        return None

    def save_profile(self, name):
        name = re.sub(r"[^\w\-]", "_", name.strip()) or time.strftime("profile_%m%d_%H%M")
        os.makedirs(PROFILE_DIR, exist_ok=True)
        vc = self.vis_cfg
        data = {"ctrls": self.current_ctrls(), "vision": {"roi_top": vc.roi_top, "roi_bottom": vc.roi_bottom,
                                                          "thresh": vc.thresh, "min_thresh": vc.min_thresh},
                "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(PROFILE_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return name

    def load_profile(self, name):
        with open(os.path.join(PROFILE_DIR, os.path.basename(name) + ".json"), encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.get("vision", {}).items():
            setattr(self.vis_cfg, k, v)
        return self.apply(data.get("ctrls", {}))

    def profiles(self):
        if not os.path.isdir(PROFILE_DIR):
            return []
        return sorted(os.path.splitext(f)[0] for f in os.listdir(PROFILE_DIR) if f.endswith(".json"))

    def state(self):
        b = self.base
        return {"fake": self.ctrls.fake, "device": self.device, "ctrls": self.ctrl_list(),
                "labels": CTRL_ZH, "vision": vars(self.vis_cfg), "profiles": self.profiles(),
                "baseline": None if not b else {"h_near": round(b["h_near"], 1), "created": b.get("created", "")}}


# ---------------------------------------------------------------- HTTP

def make_handler(t):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with t.lock:
                            jpg = t.jpg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(1 / 15)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            elif self.path == "/api/state":
                self._json(t.state())
            elif self.path == "/api/metrics":
                with t.lock:
                    m = dict(t.metrics)
                m.update(fps=round(t.fps, 1), message=t.message, autoexp=t.autoexp,
                         collecting=None if t.collect is None else len(t.collect["res"]) / t.collect["n"])
                self._json(m)
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "JSON 格式錯誤"}, 400)
            p = self.path
            if p == "/api/ctrl":
                err = t.ctrl_set(req["name"], req["value"])
                return self._json({"error": err, "ctrls": t.ctrl_list()})
            if p == "/api/vision":
                for k in ("roi_top", "roi_bottom", "thresh", "min_thresh"):
                    if k in req:
                        setattr(t.vis_cfg, k, type(getattr(t.vis_cfg, k))(req[k]) if k in ("thresh", "min_thresh")
                                else float(req[k]))
                if t.vis_cfg.roi_bottom <= t.vis_cfg.roi_top + 0.1:
                    t.vis_cfg.roi_bottom = min(1.0, t.vis_cfg.roi_top + 0.1)
                return self._json({"vision": vars(t.vis_cfg)})
            if p == "/api/reset":
                errs = t.apply({c["name"]: c["default"] for c in t.ctrl_list() if not c["readonly"]})
                return self._json({"error": "；".join(errs) or None, "ctrls": t.ctrl_list()})
            if p == "/api/autoexp":
                return self._json({"error": t.start_autoexp()})
            if p == "/api/baseline":
                t.start_baseline(int(req.get("n", 60)))
                return self._json({"error": None})
            if p == "/api/save_config":
                err = t.save_config()
                if not err:
                    t.message = "已存到 lane/config.yaml（舊檔備份成 config.yaml.bak）"
                return self._json({"error": err})
            if p == "/api/save_profile":
                name = t.save_profile(req.get("name", ""))
                t.message = f"已存成設定檔 cam_profiles/{name}.json"
                return self._json({"error": None, "profiles": t.profiles()})
            if p == "/api/load_profile":
                try:
                    errs = t.load_profile(req["name"])
                except FileNotFoundError:
                    return self._json({"error": "找不到這個設定檔"})
                t.message = f"已載入設定檔 {req['name']}" + (f"（部分失敗：{'；'.join(errs)}）" if errs else "")
                return self._json({"error": None, "state": t.state()})
            self.send_error(404)

    return H


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>鏡頭調校</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#1b1f24;--muted:#5d6672;--line:#dde1e6;--accent:#1d6fa5;
--ok:#1e7a4f;--okbg:#e3f3ea;--warn:#8a5b00;--warnbg:#fbf0d9;--bad:#b0322a;--badbg:#fbe7e5;
--sans:"Noto Sans TC","Microsoft JhengHei","PingFang TC",system-ui,sans-serif;--mono:Consolas,"Cascadia Mono",monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1a1d21;--fg:#e7eaee;--muted:#9aa3ad;--line:#2c3137;
--accent:#6cb4e4;--ok:#6fd0a0;--okbg:#14301f;--warn:#f0c46a;--warnbg:#33280f;--bad:#f2938a;--badbg:#3a1d1a;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 var(--sans)}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5}
header h1{font-size:17px;margin:0}
.pill{font-size:12px;padding:2px 8px;border-radius:99px;background:var(--line);color:var(--muted)}
.pill.fake{background:var(--warnbg);color:var(--warn)}
.pill.low{background:var(--badbg);color:var(--bad);font-weight:700}
#verdict{font-weight:700;padding:2px 10px;border-radius:6px}
.lv-OK{background:var(--okbg);color:var(--ok)}.lv-WARN{background:var(--warnbg);color:var(--warn)}.lv-FAIL{background:var(--badbg);color:var(--bad)}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,1fr);gap:16px;padding:16px;max-width:1500px;margin:0 auto}
@media (max-width:900px){main{grid-template-columns:1fr;padding:12px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.card h2{font-size:15px;margin:0 0 10px;display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card h2 small{font-weight:400;color:var(--muted);font-size:12px}
#view{width:100%;border-radius:6px;background:#000;display:block;aspect-ratio:640/318}
.stats{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:8px;font-size:13px;color:var(--muted)}
.stats b{color:var(--fg);font-family:var(--mono);font-weight:500}
.checks{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.chk{border-radius:6px;padding:6px 8px;font-size:13px;cursor:help}
.chk .v{font-family:var(--mono);float:right}
#advice{font-size:13px;margin:10px 0 0;padding-left:18px;color:var(--muted)}
#advice:empty{display:none}
canvas{width:100%;height:90px;display:block}
.row{display:grid;grid-template-columns:118px minmax(0,1fr) 64px;gap:8px;align-items:center;padding:6px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row label{font-size:14px}
.row .hint{grid-column:1/-1;font-size:12px;color:var(--muted);margin-top:-4px}
.row.inactive{opacity:.45}
.row input[type=range]{width:100%;accent-color:var(--accent)}
.row output{font-family:var(--mono);font-size:13px;text-align:right}
.row select{width:100%;padding:4px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.row .key{font-family:var(--mono);font-size:11px;color:var(--muted);display:block}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;font-size:14px;padding:7px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.5;cursor:default}
input[type=text]{font:inherit;padding:6px 8px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);min-width:0;flex:1}
#msg{font-size:13px;color:var(--accent);min-height:1.5em;margin:8px 0 0}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;background:var(--accent);width:0}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px;font-family:var(--mono)}
td,th{border-bottom:1px solid var(--line);padding:3px 4px;text-align:right}th{color:var(--muted);font-weight:500}
tr.best td{color:var(--ok);font-weight:700}
.tip{font-size:13px;color:var(--muted);margin:0 0 8px}
</style></head><body>
<header><h1>鏡頭調校</h1><span id="verdict" class="lv-WARN">…</span>
<span class="pill" id="dev"></span><span class="pill" id="fps"></span><span class="pill" id="base"></span></header>
<main>
<section>
  <div class="card">
    <img id="view" src="/stream" alt="辨識畫面：左邊是鏡頭畫面與找到的線，右邊是二值化結果">
    <div class="stats">
      <span>路段 <b id="shape">-</b></span><span>坡度 <b id="slope">-</b></span>
      <span>白線灰階 <b id="wm">-</b></span><span>黑底灰階 <b id="bm">-</b></span><span>門檻 <b id="th">-</b></span>
    </div>
  </div>
  <div class="card"><h2>畫質與辨識檢查 <small>游標停在項目上看說明</small></h2>
    <div class="checks" id="checks"></div><ul id="advice"></ul></div>
  <div class="card"><h2>ROI 灰階分布 <small>左峰 = 黑底、右峰 = 白線，兩峰分得越開越好；紅線 = 二值化門檻</small></h2>
    <canvas id="hist" width="600" height="90"></canvas></div>
</section>
<section>
  <div class="card"><h2>建議流程</h2>
    <p class="tip">① 按「自動找曝光」取得起點 → ② 用拉條微調到檢查全綠（至少不要紅）→ ③ 調 ROI 讓橘線以上沒有背景、下緣不含車身 →
    ④ 車放直線正中央按「建立基準」→ ⑤「存到 config」。換場地時可以存成設定檔，下次直接載入。</p>
    <div class="btns">
      <button class="primary" id="b-auto">自動找曝光</button>
      <button id="b-base">建立基準（直線正中央）</button>
      <button id="b-save">存到 config</button>
      <button id="b-reset">鏡頭設定恢復預設</button>
    </div>
    <div class="bar" id="bar" hidden><i></i></div>
    <p id="msg"></p>
    <div id="autoexp"></div>
  </div>
  <div class="card"><h2>辨識參數</h2><div id="vision"></div></div>
  <div class="card"><h2>鏡頭設定 <small id="ctrl-note"></small></h2><div id="ctrls"></div></div>
  <div class="card"><h2>場地設定檔</h2>
    <div class="btns"><input type="text" id="pname" placeholder="名稱，例如 lab_daylight"><button id="b-psave">存成設定檔</button></div>
    <div class="btns" style="margin-top:8px"><select id="plist" style="flex:1;padding:6px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg)"></select><button id="b-pload">載入</button></div>
  </div>
</section>
</main>
<script>
const $ = s => document.querySelector(s);
const api = (p, body) => fetch(p, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}).then(r => r.json());
let labels = {}, busy = {};
const VISION = [
  ['roi_top','ROI 上緣',0,0.9,0.01,'畫面高度的比例。往下移到背景（賽道外）不會出現在橘線以下'],
  ['roi_bottom','ROI 下緣',0.3,1,0.01,'畫面下方拍到車身時往上移'],
  ['thresh','固定門檻',0,255,1,'0 = 自動（Otsu）。反光或光線不均時改用固定值'],
  ['min_thresh','門檻下限',0,200,1,'自動門檻的最低值，避免全黑時雜訊被切成白點'],
];
function throttle(key, fn, ms=120){ // 拖拉條時不要每一格都送
  if (busy[key]) { busy[key].pending = fn; return; }
  busy[key] = {}; fn().finally(() => { setTimeout(() => { const p = busy[key].pending; delete busy[key]; if (p) throttle(key, p, ms); }, ms); });
}
function rowRange(id, label, key, min, max, step, val, hint, onset, inactive){
  const d = document.createElement('div'); d.className = 'row' + (inactive ? ' inactive' : '');
  d.innerHTML = `<label for="${id}">${label}<span class="key">${key}</span></label>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}">
    <output for="${id}">${val}</output>${hint ? `<div class="hint">${hint}</div>` : ''}`;
  const inp = d.querySelector('input'), out = d.querySelector('output');
  inp.addEventListener('input', () => { out.textContent = inp.value; throttle(id, () => onset(Number(inp.value))); });
  return d;
}
function renderVision(v){
  const box = $('#vision'); box.innerHTML = '';
  for (const [k, lab, mn, mx, st, hint] of VISION)
    box.appendChild(rowRange('v-'+k, lab, k, mn, mx, st, v[k], hint, val => api('/api/vision', {[k]: val})));
}
function renderCtrls(ctrls){
  const box = $('#ctrls'); box.innerHTML = '';
  for (const c of ctrls) {
    if (c.readonly) continue;
    const [zh, hint] = labels[c.name] || [c.name, ''];
    const extra = c.inactive ? '（目前是自動模式，這項無效）' : '';
    if (c.type === 'menu' || c.type === 'bool') {
      const d = document.createElement('div'); d.className = 'row' + (c.inactive ? ' inactive' : '');
      const opts = c.type === 'bool' ? {0:'關', 1:'開'} : c.menu;
      d.innerHTML = `<label>${zh}<span class="key">${c.name}</span></label><select>${Object.entries(opts).map(([k,v]) =>
        `<option value="${k}" ${Number(k)===c.value?'selected':''}>${k}: ${v}</option>`).join('')}</select><output>${c.default}</output>
        <div class="hint">${hint}${extra}（預設 ${c.default}）</div>`;
      d.querySelector('select').addEventListener('change', e => setCtrl(c.name, Number(e.target.value), true));
      box.appendChild(d);
    } else {
      box.appendChild(rowRange('c-'+c.name, zh, c.name, c.min, c.max, c.step, c.value,
        `${hint}${extra}（預設 ${c.default}）`, val => setCtrl(c.name, val, false), c.inactive));
    }
  }
}
function setCtrl(name, value, rerender){
  return api('/api/ctrl', {name, value}).then(r => {
    if (r.error) msg('設定 ' + name + ' 失敗：' + r.error);
    if (rerender) renderCtrls(r.ctrls);  // 切換自動 / 手動會改變其他項目能不能調
  });
}
function msg(t){ $('#msg').textContent = t; }
function loadState(){
  return api('/api/state').then(s => {
    labels = s.labels;
    $('#dev').textContent = s.fake ? '模擬鏡頭設定（沒有 v4l2-ctl）' : s.device;
    $('#dev').classList.toggle('fake', s.fake);
    $('#base').textContent = s.baseline ? `基準：地平線 ${s.baseline.h_near}（${s.baseline.created}）` : '還沒有基準';
    renderVision(s.vision); renderCtrls(s.ctrls);
    $('#plist').innerHTML = s.profiles.map(p => `<option>${p}</option>`).join('') || '<option value="">（還沒有設定檔）</option>';
  });
}
function drawHist(h, th){
  const cv = $('#hist'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
  const css = getComputedStyle(document.documentElement);
  g.clearRect(0,0,W,H); const bw = W / h.length;
  g.fillStyle = css.getPropertyValue('--accent');
  h.forEach((v,i) => g.fillRect(i*bw+1, H - v*(H-4), bw-2, v*(H-4)));
  g.fillStyle = css.getPropertyValue('--bad'); g.fillRect(th/256*W - 1, 0, 2, H);
}
let lastBaseMsg = '';
function poll(){
  api('/api/metrics').then(m => {
    if (!m.checks) return;
    $('#verdict').textContent = '視角 ' + m.verdict; $('#verdict').className = 'lv-' + m.verdict;
    $('#fps').textContent = m.fps + ' fps' + (m.fps < 20 ? '：太低！曝光時間調短（≤ 330）' : '');
    $('#fps').classList.toggle('low', m.fps < 20);
    $('#shape').textContent = m.shape; $('#slope').textContent = m.slope + (m.slope_deg === null ? '' : ` ${m.slope_deg}°`);
    $('#wm').textContent = m.white_mean; $('#bm').textContent = m.black_mean; $('#th').textContent = Math.round(m.thresh);
    $('#checks').innerHTML = m.checks.map(c => `<div class="chk lv-${c.level}" title="${c.desc}">${c.name}<span class="v">${c.value}</span></div>`).join('');
    $('#advice').innerHTML = m.checks.filter(c => c.level !== 'OK').map(c => `<li>${c.advice}</li>`).join('');
    drawHist(m.hist, m.thresh);
    const ae = m.autoexp || {}, bar = $('#bar');
    const prog = ae.running ? ae.progress : (m.collecting !== null ? m.collecting : null);
    bar.hidden = prog === null; if (prog !== null) bar.firstElementChild.style.width = (prog*100) + '%';
    $('#b-auto').disabled = !!ae.running; $('#b-base').disabled = m.collecting !== null;
    if (ae.results && ae.results.length) {
      $('#autoexp').innerHTML = `<table><tr><th>曝光</th><th>分數</th><th>對比</th><th>過曝%</th><th>雜訊%</th><th>線數</th></tr>` +
        ae.results.map(r => `<tr class="${ae.best && ae.best.exposure===r.exposure?'best':''}"><td>${r.exposure}</td><td>${r.score}</td><td>${r.contrast}</td><td>${r.overexp}</td><td>${r.clutter}</td><td>${r.lines}</td></tr>`).join('') + '</table>';
    }
    if (m.message && m.message !== lastBaseMsg) {
      lastBaseMsg = m.message; msg(m.message);
      if (/基準|自動找曝光完成/.test(m.message)) loadState();
    }
  }).catch(() => { $('#verdict').textContent = '連線中斷'; $('#verdict').className = 'lv-FAIL'; })
    .finally(() => setTimeout(poll, 300));
}
$('#b-auto').onclick = () => api('/api/autoexp', {}).then(r => r.error ? msg(r.error) : msg('自動找曝光中，約 10 秒，請不要移動車子…'));
$('#b-base').onclick = () => { if (confirm('車子已經放在平坦直線正中央、車頭對正了嗎？')) api('/api/baseline', {n:60}).then(() => msg('建立基準中（收集 60 張）…')); };
$('#b-save').onclick = () => api('/api/save_config', {}).then(r => { if (r.error) msg(r.error); });
$('#b-reset').onclick = () => { if (confirm('把所有鏡頭設定恢復成出廠預設？')) api('/api/reset', {}).then(r => { renderCtrls(r.ctrls); msg(r.error ? '部分失敗：' + r.error : '已恢復預設'); }); };
$('#b-psave').onclick = () => api('/api/save_profile', {name: $('#pname').value}).then(loadState);
$('#b-pload').onclick = () => { const n = $('#plist').value; if (n) api('/api/load_profile', {name: n}).then(r => r.error ? msg(r.error) : loadState()); };
loadState(); poll();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--source", help="改用影片 / 圖片（鏡頭設定變成軟體模擬）")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-apply", action="store_true", help="啟動時不要先套用 config.yaml 裡的鏡頭設定")
    args = ap.parse_args()

    if args.source:
        src, ctrls, dev = FileSource(args.source), FakeCtrls(), args.source
    else:
        from camera import Camera
        src = Camera(args.device)
        dev = f"/dev/video{args.device}"
        ctrls = V4L2(args.device) if shutil.which("v4l2-ctl") else FakeCtrls()
        if ctrls.fake:
            print("找不到 v4l2-ctl（sudo apt install v4l-utils），鏡頭設定改用軟體模擬")
    t = Tuner(src, ctrls, dev)
    if not args.no_apply and getattr(t, "config_ctrls", None):
        errs = t.apply(t.config_ctrls)
        print("已套用 config.yaml 的鏡頭設定" + (f"（失敗：{errs}）" if errs else ""))
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(t))
    print(f"鏡頭調校：電腦或手機瀏覽器開 http://<orin-ip>:{args.port}/   (Ctrl+C 結束)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n結束。記得按「存到 config」才會保留設定。")


if __name__ == "__main__":
    main()
