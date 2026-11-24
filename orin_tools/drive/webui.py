"""網頁控制台：手機 / 電腦瀏覽器開 http://<orin-ip>:<port>/

- 即時畫面（循線疊圖，約 10 張 / 秒）
- 開始 / 停止
- 勾選「紀錄」（這一趟寫 CSV + 錄原始畫面）、「起跑模式」（按開始後等中控台綠燈才出發）
  行駛中（含等綠燈）兩個勾選都不能改，伺服器端也會擋
只用標準函式庫 + OpenCV。控制動作在 HTTP 執行緒裡呼叫 pilot.start / stop，和終端機鍵盤一樣。
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>循跡車控制台</title>
<style>
:root{--bg:#f3f4f1;--panel:#fff;--ink:#1d2328;--muted:#5d6870;--rule:#d5d9d6;--go:#2f7d4f;--stop:#b2401f;--accent:#d9661f}
@media (prefers-color-scheme:dark){:root{--bg:#171b1e;--panel:#20262a;--ink:#e4e7e3;--muted:#9aa5ab;--rule:#343c41;--go:#3f9a63;--stop:#d05a3a;--accent:#f08a45}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;
 padding:max(12px,env(safe-area-inset-top)) max(12px,env(safe-area-inset-right)) 24px max(12px,env(safe-area-inset-left))}
.wrap{max-width:900px;margin:0 auto;display:grid;gap:12px}
h1{font-size:20px;margin:0}
.cam{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:4/3;display:grid;place-items:center}
.cam img{width:100%;height:100%;object-fit:contain;display:block}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.status div{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:8px 10px;min-width:0}
.status b{display:block;font-size:12px;color:var(--muted);font-weight:500}
.status span{font-size:17px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{font:inherit;font-size:20px;font-weight:700;padding:18px 10px;border:0;border-radius:10px;color:#fff;cursor:pointer;touch-action:manipulation}
button:disabled{opacity:.4;cursor:not-allowed}
#start{background:var(--go)}#stop{background:var(--stop)}
.opts{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:6px 12px;display:grid}
.opts label{display:flex;align-items:center;gap:12px;padding:10px 0;font-size:17px}
.opts label+label{border-top:1px solid var(--rule)}
.opts input{width:24px;height:24px;flex:none}
.opts small{display:block;color:var(--muted);font-size:13px}
.lock{color:var(--accent);font-size:13px;min-height:1.2em}
.msg{color:var(--muted);font-size:13px;min-height:1.2em}
.mrow{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;justify-content:space-between;padding:8px 0;font-size:16px}
.mbtn{display:flex;gap:8px}
.mbtn button{font-size:15px;padding:10px 12px;background:var(--muted)}
#mainon{background:var(--go)}
.exit{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--panel);border:1px solid var(--stop);border-radius:8px;padding:8px 10px;font-size:12px;margin:0}
.status .wide{grid-column:1/-1}
.status small{display:block;color:var(--muted);font-size:13px}
.bar{display:block;height:8px;background:var(--rule);border-radius:4px;margin:6px 0 4px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--go);transition:width .2s}
.alert{background:var(--stop);color:#fff;border-radius:8px;padding:10px 12px;font-size:16px;font-weight:700}
.hud{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.6);color:#fff;border-radius:8px;padding:6px 10px;text-align:right;font-variant-numeric:tabular-nums;line-height:1.15}
.hud b{display:block;font-size:30px}
.hud small{font-size:13px;opacity:.85}
#camoff{color:#ccc;font-size:15px;padding:16px;text-align:center}
</style></head><body><div class="wrap">
<h1>循跡車控制台</h1>
<div class="opts" id="mainbox" hidden>
 <div class="mrow"><span>主程式：<b id="mainst">—</b></span>
  <span class="mbtn"><button id="mainon">啟動主程式</button><button id="mainoff">關閉主程式</button></span></div>
</div>
<div class="cam"><img id="cam" alt="車上鏡頭即時畫面"><div class="hud" id="hud" hidden><b id="hudv">—</b><small id="hudw">—</small></div><span id="camoff" hidden>主程式沒有在跑，沒有畫面</span></div>
<div class="btns"><button id="start">開始</button><button id="stop">停止</button></div>
<div class="opts">
 <label><input type="checkbox" id="record"><span>紀錄<small>這一趟寫 CSV 並錄下鏡頭畫面</small></span></label>
 <label><input type="checkbox" id="race"><span>起跑模式<small>按開始後等中控台綠燈才出發</small></span></label>
</div>
<div class="lock" id="lock"></div>
<div class="alert" id="alert" hidden></div>
<div class="status">
 <div><b>狀態</b><span id="st">—</span></div>
 <div class="wide"><b>速度</b><span id="spd">—</span><i class="bar"><i id="spdbar"></i></i><small id="spdw">—</small></div>
 <div class="wide"><b>單圈平均</b><span id="lapavg">—</span><small id="lapw">虛線左轉到下一次虛線左轉算一圈</small></div>
 <div><b>前輪</b><span id="fwa">—</span></div>
 <div><b>畫面</b><span id="fps">—</span></div>
 <div><b>起跑旗標</b><span id="rs">—</span></div>
 <div><b>紀錄檔</b><span id="log">—</span></div>
</div>
<div class="msg" id="msg"></div>
<pre class="exit" id="exit" hidden></pre>
</div>
<script>
const $=id=>document.getElementById(id);
const names={IDLE:"待命",RUN:"行駛中",LOST:"找線中",STOP:"已停止"};
let busy=false;
async function post(path,body){
  busy=true;
  try{const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
      const j=await r.json(); $("msg").textContent=j.msg||""; show(j.state);}
  catch(e){$("msg").textContent="連不到車子";}
  busy=false;
}
function show(s){
  if(!s)return;
  if("main" in s){
    $("mainbox").hidden=false;
    $("mainst").textContent=s.main?(s.main_child?"執行中（網頁啟動）":"執行中（終端機啟動）"):"未啟動";
    $("mainon").disabled=s.main||s.main_starting;
    $("mainoff").disabled=!s.main||!s.main_child;
    $("cam").hidden=!s.main; $("camoff").hidden=!!s.main;
    $("exit").hidden=!s.main_exit; $("exit").textContent=s.main_exit||"";
    if(!s.main){
      $("hud").hidden=true;
      $("st").textContent=s.main_starting?"主程式啟動中…":"主程式未啟動";
      $("start").disabled=$("stop").disabled=true;
      $("record").disabled=$("race").disabled=false;
      if(!busy){$("record").checked=s.record;$("race").checked=s.race;}
      $("lock").textContent="";
      return;
    }
  }
  $("st").textContent=(names[s.state]||s.state)+(s.waiting?"（等綠燈）":"")+(s.reason?"："+s.reason:"");
  const avg=(s.bwl+s.bwr)/2, mps=Math.max(0,avg)/100*(s.full_mps||1.57);
  $("spd").textContent=(avg<0?"倒轉煞車 ":"")+mps.toFixed(2)+" m/s（"+(mps*3.6).toFixed(1)+" km/h）";
  $("hud").hidden=false; $("hudv").textContent=mps.toFixed(2)+" m/s";
  $("hudw").textContent=(avg<0?"倒轉煞車":avg.toFixed(0)+"%")+(s.why?"・"+s.why:"");
  $("spdbar").style.width=Math.min(100,Math.abs(avg)).toFixed(0)+"%";
  $("spdbar").style.background=avg<0?"var(--stop)":"var(--go)";
  $("spdw").textContent="目標 "+(s.v||0).toFixed(0)+"%｜左 "+s.bwl.toFixed(0)+"% / 右 "+s.bwr.toFixed(0)+"%"+(s.why?"｜"+s.why:"");
  const bad=s.state==="STOP"&&s.reason&&s.reason!=="手動停止";
  $("alert").hidden=!bad; $("alert").textContent=bad?"⚠ 自動停車："+s.reason+(s.log?"（已記在 "+s.log.replace("（已存）","")+"）":""):"";
  if(s.laps){
    $("lapavg").textContent=s.lap_avg.toFixed(2)+" 秒（"+s.laps+" 圈）";
    $("lapw").textContent="上一圈 "+s.lap_last.toFixed(2)+" 秒｜最快 "+s.lap_best.toFixed(2)+" 秒";
  }else{
    $("lapavg").textContent="—";
    $("lapw").textContent="還沒跑完一圈（虛線左轉到下一次虛線左轉算一圈）";
  }
  $("fwa").textContent=(s.fwa>0?"+":"")+s.fwa.toFixed(0)+"°";
  $("fps").textContent=s.fps.toFixed(1)+" fps";
  $("rs").textContent=s.race_status||"—";
  $("log").textContent=s.log||"不紀錄";
  if(!busy){$("record").checked=s.record;$("race").checked=s.race;}
  $("record").disabled=$("race").disabled=s.running;
  $("start").disabled=s.running;
  $("stop").disabled=!s.running && s.state!=="STOP";
  $("lock").textContent=s.running?"行駛中（含等綠燈）不能改勾選，先按停止":"";
}
$("start").onclick=()=>post("/api/start");
$("mainon").onclick=()=>post("/api/main/start");
$("mainoff").onclick=()=>{if(confirm("關閉主程式？（車子會先停下）"))post("/api/main/stop");};
$("stop").onclick=()=>post("/api/stop");
$("record").onchange=$("race").onchange=()=>post("/api/opts",{record:$("record").checked,race:$("race").checked});
// 主程式重新啟動後畫面要重連（併進同一個輪詢，不再另外每 1.5 s 多打一次 /api/state）
let lastMain=null;
async function poll(){
  try{const r=await fetch("/api/state",{cache:"no-store"}); const s=await r.json(); show(s);
    lastMain=s.main;}
  catch(e){$("msg").textContent="連不到車子";}
  setTimeout(poll,250);
}
poll();
// 畫面：一次抓一張最新的（/snapshot.jpg），抓完才抓下一張 → 不會排隊延遲；
// 2 秒沒回應就放棄重抓（MJPEG 卡住時瀏覽器不會報錯，畫面就一直停著）
let camUrl=null;
async function cam(){
  const t0=Date.now();
  try{
    const ctl=new AbortController(); const to=setTimeout(()=>ctl.abort(),2000);
    const r=await fetch("/snapshot.jpg?"+t0,{cache:"no-store",signal:ctl.signal});
    clearTimeout(to);
    if(r.ok){
      const u=URL.createObjectURL(await r.blob());
      $("cam").src=u; if(camUrl)URL.revokeObjectURL(camUrl); camUrl=u;
    }
  }catch(e){}
  setTimeout(cam,Math.max(0,100-(Date.now()-t0)));   // 最多每秒 10 張
}
cam();
</script></body></html>
"""


class WebUI:
    def __init__(self, port, pilot, race=None, record=False, race_mode=False, run_states=("RUN", "LOST"),
                 host="127.0.0.1"):
        self.pilot, self.race = pilot, race
        self.record, self.race_mode = bool(record), bool(race_mode and race is not None)
        self.run_states = run_states
        self.lock = threading.Lock()
        self.jpg = None
        self.raw_jpg = None              # 原始鏡頭畫面（tools/car_rec.py 錄其他車用），有人要才編碼
        self.raw_req = 0.0
        self.info = {"bwl": 0.0, "bwr": 0.0, "fwa": 0.0, "fps": 0.0, "log": ""}
        self.session = False            # 按了開始之後到停止：紀錄用
        self.full_mps = 1.57            # 100% 的實測車速（m/s），main 會用 config 覆蓋
        ui = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # 關 Nagle（見 webctl.py）

            def log_message(self, *a):
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
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, PAGE, "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._send(200, json.dumps(ui.state()))
                elif path == "/snapshot.jpg":
                    # 只給最新一張：網頁抓完一張才抓下一張，Wi-Fi 慢時不會像 MJPEG 一樣把舊畫面排隊、越來越卡
                    with ui.lock:
                        jpg = ui.jpg
                    if jpg is None:
                        self._send(503, '{"msg":"no frame"}')
                    else:
                        self._send(200, jpg, "image/jpeg")
                elif path == "/raw.jpg":
                    # 原始鏡頭畫面（沒有畫線、原解析度）：tools/car_rec.py 錄其他車用；最近 2 秒有人要，主迴圈才編碼
                    with ui.lock:
                        ui.raw_req = time.time()
                        jpg = ui.raw_jpg
                    if jpg is None:
                        self._send(503, '{"msg":"no raw frame yet"}')
                    else:
                        self._send(200, jpg, "image/jpeg")
                elif path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with ui.lock:
                                jpg = ui.jpg
                            if jpg is not None and jpg is not last:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                                last = jpg
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self._send(404, '{"msg":"not found"}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                path = self.path.split("?")[0]
                if path == "/api/start":
                    msg = ui.start()
                elif path == "/api/stop":
                    msg = ui.stop()
                elif path == "/api/opts":
                    msg = ui.set_opts(body.get("record"), body.get("race"))
                else:
                    self._send(404, '{"msg":"not found"}')
                    return
                self._send(200, json.dumps({"msg": msg, "state": ui.state()}))

        self.srv = ThreadingHTTPServer((host, port), Handler)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    # ------------------------------------------------------------ 狀態 / 控制
    def running(self):
        """行駛中或在等綠燈：這段時間勾選不能改。"""
        return self.pilot.state in self.run_states or (self.race is not None and self.race.armed)

    def state(self):
        with self.lock:
            info = dict(self.info)
        return {"state": self.pilot.state, "reason": self.pilot.reason or "", "running": self.running(),
                "waiting": self.race is not None and self.race.armed,
                "record": self.record, "race": self.race_mode,
                "race_status": self.race.status() if self.race is not None else "",
                **(self.pilot.lap_info() if hasattr(self.pilot, "lap_info") else {}),
                **info}

    def start(self):
        with self.lock:
            if self.running():
                return "已經在跑了"
            self.session = True
            if self.race_mode and self.race is not None:
                if self.pilot.state == "STOP":
                    self.pilot.toggle()              # STOP → 待命，綠燈時才能出發
                self.race.arm()
                return "起跑模式：等中控台綠燈"
            self.pilot.start()
            return "出發"

    def stop(self):
        with self.lock:
            if self.race is not None and self.race.armed:
                self.race.disarm("關閉（網頁停止）")
            self.pilot.stop("網頁停止")
            return "已停止"

    def set_opts(self, record, race):
        with self.lock:
            if self.running():
                return "行駛中不能改，先按停止"
            if record is not None:
                self.record = bool(record)
            if race is not None:
                if race and self.race is None:
                    return "沒有起跑旗標功能（不是用鏡頭在跑）"
                self.race_mode = bool(race)
            return "已更新"

    # ------------------------------------------------------------ 主迴圈呼叫
    def push(self, frame, fwa, bwl, bwr, fps, log_name, v=0.0, why=""):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self.lock:
            if ok:
                self.jpg = jpg.tobytes()
            self.info = {"bwl": float(bwl), "bwr": float(bwr), "fwa": float(fwa), "fps": float(fps),
                         "log": log_name or "", "v": float(v), "why": why or "",
                         "full_mps": self.full_mps}

    def want_raw(self):
        return time.time() - self.raw_req < 2.0

    def push_raw(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            with self.lock:
                self.raw_jpg = jpg.tobytes()

    def close(self):
        self.srv.shutdown()
