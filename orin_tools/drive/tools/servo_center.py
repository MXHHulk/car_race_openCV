#!/usr/bin/env python3
"""手動調舵機置中：開網頁用按鈕一度一度調前輪，調到打直後按「儲存為置中」。
儲存會：1) 把角度寫進 config.yaml 的 steer.fwa_trim（行駛程式、hw_check、speed_test 都用這個）
        2) 舊的 config.yaml 先備份成 config.yaml.bak_YYYYmmdd_HHMMSS
        3) 在 logs/servo_center_log.csv 加一筆紀錄（時間、舊值、新值、備註），當作基準
主程式要先關（序列埠才不會被佔用）。後輪不會轉。

  python3 tools/servo_center.py               # 手機開 http://<車子 IP>:8002/
  python3 tools/servo_center.py --port 8002

韌體只收整數度（每度約 12 µs），所以一次最少調 1°。
離開時舵機停在最後的角度（韌體失聯時舵機維持原位），不會自己回 0°。
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
CFG = os.path.join(HERE, "config.yaml")
LOG = os.path.join(HERE, "..", "logs", "servo_center_log.csv")
LIMIT = 45          # 調整範圍 ±45°（避免舵機頂住機構）

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>舵機置中</title>
<style>
:root{--bg:#f3f4f1;--panel:#fff;--ink:#1d2328;--muted:#5d6870;--rule:#d5d9d6;--go:#2f7d4f;--accent:#d9661f}
@media (prefers-color-scheme:dark){:root{--bg:#171b1e;--panel:#20262a;--ink:#e4e7e3;--muted:#9aa5ab;--rule:#343c41;--go:#3f9a63;--accent:#f08a45}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;padding:16px}
.wrap{max-width:520px;margin:0 auto;display:grid;gap:12px}
h1{font-size:20px;margin:0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px}
.big{font-size:56px;font-weight:700;text-align:center;font-variant-numeric:tabular-nums}
.sub{text-align:center;color:var(--muted);font-size:14px}
.row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
button{font:inherit;font-size:20px;font-weight:700;padding:16px 4px;border:0;border-radius:10px;background:var(--muted);color:#fff;cursor:pointer;touch-action:manipulation}
#save{background:var(--go);width:100%}
#back,#sweep{background:var(--accent);font-size:16px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
input{font:inherit;font-size:16px;width:100%;padding:10px;border:1px solid var(--rule);border-radius:8px;background:var(--bg);color:var(--ink)}
.msg{color:var(--muted);font-size:14px;min-height:1.3em;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:4px;border-bottom:1px solid var(--rule)}
</style></head><body><div class="wrap">
<h1>舵機置中調整</h1>
<div class="card"><div class="big" id="ang">—</div>
<div class="sub">目前指令（負 = 往左、正 = 往右）｜config 裡的置中：<b id="trim">—</b>°</div></div>
<div class="row"><button data-d="-5">◀◀ 5</button><button data-d="-1">◀ 1</button><button data-d="1">1 ▶</button><button data-d="5">5 ▶▶</button></div>
<div class="row2"><button id="back">回到 config 置中</button><button id="sweep">左右掃一次再回來</button></div>
<div class="card"><input id="note" placeholder="備註（例如：11-21 維修後重調）"><div style="height:8px"></div>
<button id="save">儲存為置中（寫入 fwa_trim）</button></div>
<div class="msg" id="msg"></div>
<div class="card"><b>儲存紀錄</b><table id="hist"></table></div>
</div><script>
const $=id=>document.getElementById(id);
async function api(p,b){const r=await fetch(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});
 const j=await r.json(); show(j); return j;}
function show(j){$("ang").textContent=(j.angle>0?"+":"")+j.angle+"°";$("trim").textContent=j.trim;
 if(j.msg)$("msg").textContent=j.msg;
 $("hist").innerHTML="<tr><th>時間</th><th>舊</th><th>新</th><th>備註</th></tr>"+(j.hist||[]).map(h=>"<tr><td>"+h.time+"</td><td>"+h.old+"</td><td>"+h.new+"</td><td>"+(h.note||"")+"</td></tr>").join("");}
document.querySelectorAll("[data-d]").forEach(b=>b.onclick=()=>api("/api/nudge",{d:+b.dataset.d}));
$("back").onclick=()=>api("/api/set",{angle:null});
$("sweep").onclick=()=>api("/api/sweep");
$("save").onclick=()=>{if(confirm("把 "+$("ang").textContent+" 存成新的置中（fwa_trim）？"))api("/api/save",{note:$("note").value});};
fetch("/api/state").then(r=>r.json()).then(show);
</script></body></html>
"""


def main_running():
    out = subprocess.run(["pgrep", "-af", "drive/main.py|python3 main.py"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "pgrep" not in l and "bash -c" not in l]


def read_trim():
    return yaml.safe_load(open(CFG, encoding="utf-8"))["steer"]["fwa_trim"]


def write_trim(new):
    """只改 fwa_trim 那一行的數字，保留註解和原本的換行（CRLF）。"""
    raw = open(CFG, "rb").read().decode("utf-8")
    pat = re.compile(r"^(  fwa_trim:\s*)(-?[0-9.]+)", re.M)
    if len(pat.findall(raw)) != 1:
        raise RuntimeError("config.yaml 找不到唯一的 fwa_trim 行")
    bak = CFG + time.strftime(".bak_%Y%m%d_%H%M%S")
    shutil.copy2(CFG, bak)
    raw = pat.sub(lambda m: m.group(1) + f"{new:g}", raw)
    open(CFG, "wb").write(raw.encode("utf-8"))
    return bak


def read_hist():
    if not os.path.exists(LOG):
        return []
    with open(LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))[-10:][::-1]


def add_hist(old, new, note):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    first = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["time", "old", "new", "note"])
        if first:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "old": old, "new": new, "note": note})


class Servo:
    """背景一直送目前角度（後輪 0），按鈕只改角度。"""

    def __init__(self, link, angle):
        self.link, self.angle, self.lock = link, int(round(angle)), threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            with self.lock:
                a = self.angle
            self.link.send(a, 0, 0)
            self.link.poll()
            time.sleep(0.05)

    def set(self, a):
        with self.lock:
            self.angle = max(-LIMIT, min(LIMIT, int(round(a))))
            return self.angle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    if main_running():
        print("主程式正在跑（會佔用序列埠）：先從網頁按「關閉主程式」再執行")
        return
    from motor_cmd import BMduinoLink
    lk = yaml.safe_load(open(CFG, encoding="utf-8"))["link"]
    link = BMduinoLink(lk["port"], lk["baud"])
    servo = Servo(link, read_trim())

    def state(msg=""):
        return {"angle": servo.angle, "trim": read_trim(), "msg": msg, "hist": read_hist()}

    def sweep():
        a = servo.angle
        for b in (a - 25, a + 25, a):
            servo.set(b)
            time.sleep(1.5)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._send(200, json.dumps(state()))
            else:
                self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            p, msg = self.path.split("?")[0], ""
            if p == "/api/nudge":
                a = servo.set(servo.angle + float(body.get("d", 0)))
                msg = f"前輪指令 {a:+d}°" + ("（已到調整上限）" if abs(a) == LIMIT else "")
            elif p == "/api/set":
                a = body.get("angle")
                a = servo.set(read_trim() if a is None else float(a))
                msg = f"前輪指令 {a:+d}°"
            elif p == "/api/sweep":
                threading.Thread(target=sweep, daemon=True).start()
                msg = "左右各 25° 掃一次（約 4.5 秒）後回到目前角度：看左右打的幅度是否差不多"
            elif p == "/api/save":
                old, new, note = read_trim(), servo.angle, str(body.get("note", ""))[:80]
                try:
                    bak = write_trim(new)
                    add_hist(old, new, note)
                    msg = f"已儲存：fwa_trim {old} → {new}（舊 config 備份成 {os.path.basename(bak)}）"
                    print(msg, flush=True)
                except Exception as e:  # noqa: BLE001
                    msg = f"儲存失敗：{e}"
            else:
                self._send(404, "{}")
                return
            self._send(200, json.dumps(state(msg)))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    print(f"舵機置中調整：http://{ip[0] if ip else '<車子 IP>'}:{args.port}/   Ctrl+C 結束", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"結束：舵機停在 {servo.angle:+d}°", flush=True)


if __name__ == "__main__":
    main()
