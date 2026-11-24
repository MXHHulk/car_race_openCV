#!/usr/bin/env python3
"""錄其他車的樣子（避撞用資料）：手機開 http://<orin-ip>:8010/

用法（車上）：主程式要開著（IDLE 就好，鏡頭在主程式裡），然後
  python3 tools/car_rec.py            # 預設存到 ~/orin_tools/cars/
把別台車放在我們車子前方，網頁填「車名 / 距離 / 方向 / 位置」，按「拍 1 張」或「連拍」。
照片存成 cars/<車名>/<車名>_<距離>m_<方向>_<位置>_<時間>.jpg，每張一列寫進 cars/index.csv。
另外建議拍「沒有車」的空賽道（車名填「無車」），偵測時拿來比較。
畫面從主程式的 http://127.0.0.1:8001/raw.jpg 抓（原解析度、沒有畫線）。
"""
import argparse
import csv
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_OUT = os.path.join(HERE, "..", "cars")

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>錄其他車</title>
<style>
:root{--bg:#f6f7f9;--fg:#1b1f24;--card:#fff;--line:#d7dbe0;--acc:#2563eb;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e8eaed;--card:#1d2024;--line:#33373d;--acc:#60a5fa;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 8px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
img{width:100%;max-width:640px;border-radius:8px;background:#000;display:block}
label{display:block;margin:6px 0 2px;font-size:13px;opacity:.8}
input,select{width:100%;padding:8px;font-size:16px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{width:100%;padding:12px;font-size:16px;border:0;border-radius:8px;background:var(--acc);color:#fff;margin-top:8px}
button.sec{background:transparent;color:var(--fg);border:1px solid var(--line)}
#msg{margin-top:8px;min-height:1.4em}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
</style></head><body>
<h1>錄其他車（避撞資料）</h1>
<div class="card"><img id="pv" alt="鏡頭畫面"><div id="pvs" style="font-size:12px;opacity:.7"></div></div>
<div class="card">
<label>車名（例：紅車A、白車B；空賽道填「無車」）</label><input id="car" value="">
<div class="row">
<div><label>距離（m，車頭到對方車尾）</label><input id="dist" type="number" step="0.1" min="0" value="1.0"></div>
<div><label>方向</label><select id="dir"><option>同向</option><option>停著</option><option>逆向</option><option>橫向</option></select></div>
</div>
<div class="row">
<div><label>位置</label><select id="pos"><option>正前方</option><option>偏左</option><option>偏右</option><option>彎道中</option></select></div>
<div><label>地點 / 備註</label><input id="note" placeholder="例：大直線、隧道、坡上"></div>
</div>
<button onclick="shot(1)">拍 1 張</button>
<button class="sec" onclick="shot(10)">連拍 10 張（3 秒，可以慢慢推車改距離）</button>
<button class="sec" onclick="undo()">刪除最後一次拍的</button>
<div id="msg"></div>
</div>
<div class="card"><b>已錄</b><table id="tb"></table></div>
<script>
const $=id=>document.getElementById(id);
for(const k of ["car","dist","dir","pos","note"]){try{const v=localStorage.getItem("cr_"+k);if(v!==null)$(k).value=v}catch(e){}
  $(k).addEventListener("change",()=>{try{localStorage.setItem("cr_"+k,$(k).value)}catch(e){}})}
async function pv(){const t=Date.now();try{const r=await fetch("/preview.jpg?"+t,{cache:"no-store"});
  if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);const o=$("pv").src;$("pv").src=u;if(o.startsWith("blob:"))URL.revokeObjectURL(o);$("pvs").textContent=""}
  else $("pvs").textContent="沒有畫面：主程式要開著（IDLE 即可）"}catch(e){$("pvs").textContent="連線中斷"}
  setTimeout(pv,300)}
async function stats(){try{const r=await fetch("/api/stats");const d=await r.json();
  let h="<tr><th>車名</th><th>同向</th><th>停著</th><th>逆向</th><th>橫向</th><th>合計</th></tr>";
  for(const [c,v] of Object.entries(d.cars)){h+=`<tr><td>${c}</td><td>${v["同向"]||0}</td><td>${v["停著"]||0}</td><td>${v["逆向"]||0}</td><td>${v["橫向"]||0}</td><td>${v.total}</td></tr>`}
  $("tb").innerHTML=h}catch(e){}}
async function shot(n){const car=$("car").value.trim();if(!car){msg("先填車名",1);return}
  msg(n>1?"連拍中…":"拍照中…");
  const r=await fetch("/api/shot",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({car,dist:$("dist").value,dir:$("dir").value,pos:$("pos").value,note:$("note").value,n})});
  const d=await r.json();msg(d.msg,!d.ok);stats()}
async function undo(){const r=await fetch("/api/undo",{method:"POST"});const d=await r.json();msg(d.msg,!d.ok);stats()}
function msg(t,bad){$("msg").textContent=t;$("msg").className=bad?"bad":"ok"}
pv();stats();
</script></body></html>"""


def safe(s):
    s = re.sub(r"[^\w一-鿿.-]+", "_", str(s).strip())
    return s[:40] or "未命名"


class Rec:
    def __init__(self, out, src):
        self.out, self.src = os.path.abspath(out), src
        os.makedirs(self.out, exist_ok=True)
        self.index = os.path.join(self.out, "index.csv")
        self.lock = threading.Lock()
        self.last = []                  # 最後一次拍的檔案（undo 用）

    def grab(self):
        with urllib.request.urlopen(self.src, timeout=2.0) as r:
            return r.read()

    def preview(self):
        try:
            return self.grab()
        except Exception:
            return None

    def shot(self, d):
        car, n = safe(d.get("car", "")), max(1, min(30, int(d.get("n", 1))))
        try:
            dist = float(d.get("dist", 0))
        except ValueError:
            dist = 0.0
        dirn, pos, note = safe(d.get("dir", "")), safe(d.get("pos", "")), str(d.get("note", ""))[:80]
        folder = os.path.join(self.out, car)
        os.makedirs(folder, exist_ok=True)
        saved, prev = [], None
        self.grab()                       # 第一次要求會讓主程式開始編碼原始畫面
        time.sleep(0.15)
        for k in range(n):
            jpg = None
            for _ in range(10):
                try:
                    jpg = self.grab()
                except Exception:
                    jpg = None
                if jpg and jpg != prev:
                    break
                time.sleep(0.05)
            if not jpg:
                break
            prev = jpg
            ts = time.strftime("%Y%m%d_%H%M%S") + "_%03d" % int((time.time() % 1) * 1000)
            name = f"{car}_{dist:.1f}m_{dirn}_{pos}_{ts}.jpg"
            path = os.path.join(folder, name)
            with open(path, "wb") as f:
                f.write(jpg)
            saved.append(path)
            with self.lock:
                new = not os.path.exists(self.index)
                with open(self.index, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "file", "car", "dist_m", "direction", "position", "note"])
                    w.writerow([ts, os.path.relpath(path, self.out), car, dist, dirn, pos, note])
            if k < n - 1:
                time.sleep(0.3)
        self.last = saved
        if not saved:
            return {"ok": False, "msg": "抓不到畫面：主程式有開嗎？"}
        return {"ok": True, "msg": f"已存 {len(saved)} 張到 cars/{car}/"}

    def undo(self):
        if not self.last:
            return {"ok": False, "msg": "沒有可以刪的"}
        names = {os.path.relpath(p, self.out) for p in self.last}
        for p in self.last:
            try:
                os.remove(p)
            except OSError:
                pass
        with self.lock:
            if os.path.exists(self.index):
                rows = list(csv.reader(open(self.index, encoding="utf-8")))
                keep = [r for r in rows if len(r) < 2 or r[1] not in names]
                with open(self.index, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(keep)
        n = len(self.last)
        self.last = []
        return {"ok": True, "msg": f"刪除 {n} 張"}

    def stats(self):
        cars = {}
        if os.path.exists(self.index):
            for r in csv.DictReader(open(self.index, encoding="utf-8")):
                c = cars.setdefault(r["car"], {"total": 0})
                c[r["direction"]] = c.get(r["direction"], 0) + 1
                c["total"] += 1
        return {"cars": cars}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--src", default="http://127.0.0.1:8001/raw.jpg")
    a = ap.parse_args()
    rec = Rec(a.out, a.src)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif p == "/preview.jpg":
                j = rec.preview()
                self._send(200, j, "image/jpeg") if j else self._send(503, '{"msg":"no frame"}')
            elif p == "/api/stats":
                self._send(200, json.dumps(rec.stats(), ensure_ascii=False))
            else:
                self._send(404, '{"msg":"not found"}')

        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if p == "/api/shot":
                self._send(200, json.dumps(rec.shot(body), ensure_ascii=False))
            elif p == "/api/undo":
                self._send(200, json.dumps(rec.undo(), ensure_ascii=False))
            else:
                self._send(404, '{"msg":"not found"}')

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"錄其他車：http://<orin-ip>:{a.port}/  存到 {os.path.abspath(a.out)}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
