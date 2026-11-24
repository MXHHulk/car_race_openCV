"""網頁服務（和主程式分開啟動）：python3 webctl.py

手機 / 電腦開 http://<orin-ip>:<web.public_port>/（預設 8000）：
- 啟動 / 關閉主程式 main.py（網頁啟動的才能從網頁關）
- 主程式在跑時：即時畫面、開始 / 停止、勾選紀錄與起跑模式（轉接主程式的本機控制埠 web.port，預設 8001）
- 主程式沒在跑時，勾選先記在這裡，主程式一啟動就帶過去
主程式也可以照舊從終端機啟動，網頁一樣看得到、控制得到（只是不能從網頁關掉）。
只用標準函式庫。
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webui import PAGE  # noqa: E402


class MainProc:
    """管理網頁啟動的主程式。"""

    def __init__(self, cfg, cfg_path, extra=()):
        wc = cfg.get("web") or {}
        self.base = f"http://127.0.0.1:{int(wc.get('port', 8001))}"
        self.cfg_path = cfg_path
        self.extra = list(extra)            # 帶給主程式的額外參數（例如測試用 --dry-run）
        self.log_dir = os.path.join(HERE, (cfg.get("debug") or {}).get("log_dir", "../logs"))
        self.proc = None
        self.t_start = None
        self.record = bool(wc.get("record", True))
        self.race = False
        self.lock = threading.Lock()

    def call(self, path, body=None, timeout=0.6):
        """呼叫主程式的控制埠；連不到回 None。"""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def child_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def state(self):
        s = self.call("/api/state")
        starting = self.child_alive() and s is None and time.time() - (self.t_start or 0) < 30
        if s is None:
            return {"main": False, "main_child": self.child_alive(), "main_starting": starting,
                    "state": "", "reason": "", "running": False, "waiting": False,
                    "record": self.record, "race": self.race, "race_status": "",
                    "bwl": 0.0, "bwr": 0.0, "fwa": 0.0, "fps": 0.0, "log": ""}
        self.record, self.race = s.get("record", self.record), s.get("race", self.race)
        s.update(main=True, main_child=self.child_alive(), main_starting=False)
        return s

    def start_main(self):
        with self.lock:
            if self.call("/api/state") is not None:
                return "主程式已經在跑了"
            if self.child_alive():
                return "主程式啟動中，等一下"
            os.makedirs(self.log_dir, exist_ok=True)
            out = open(os.path.join(self.log_dir, time.strftime("console_%Y%m%d_%H%M%S.txt")), "w")
            self.proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "main.py"), "-c", self.cfg_path] + self.extra,
                                         cwd=HERE, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self.t_start = time.time()
        # 等控制埠起來後把勾選帶過去（背景做，不卡住網頁）
        threading.Thread(target=self._push_opts, daemon=True).start()
        return "主程式啟動中（鏡頭、IMU 準備好約 5–10 秒）"

    def _push_opts(self):
        for _ in range(150):
            if not self.child_alive():
                return
            if self.call("/api/opts", {"record": self.record, "race": self.race}) is not None:
                return
            time.sleep(0.2)

    def stop_main(self):
        with self.lock:
            if not self.child_alive():
                if self.call("/api/state") is not None:
                    return "主程式是從終端機啟動的，請在終端機輸入 q 關閉"
                return "主程式沒有在跑"
            self.call("/api/stop", {})                         # 先停車
            time.sleep(0.3)
            self.proc.send_signal(signal.SIGINT)               # 讓主程式正常收尾（關紀錄檔、送停車）
            try:
                self.proc.wait(8)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            return "主程式已關閉"

    def set_opts(self, record, race):
        r = self.call("/api/opts", {"record": record, "race": race})
        if r is not None:
            return r.get("msg", "")
        if record is not None:
            self.record = bool(record)
        if race is not None:
            self.race = bool(race)
        return "已記住，主程式啟動時套用"


def lan_ip():
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.connect(("192.168.0.1", 80))
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except OSError:
        return "<orin-ip>"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--main-arg", action="append", default=[],
                    help="啟動主程式時多帶的參數，可重複（例如 --main-arg=--dry-run 測試時不接馬達）")
    args = ap.parse_args()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    port = args.port or int((cfg.get("web") or {}).get("public_port", 8000))
    mp = MainProc(cfg, cfg_path, args.main_arg)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

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
                self._send(200, json.dumps(mp.state()))
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, '{"msg":"not found"}')

        def _stream(self):
            """轉接主程式的即時畫面；主程式沒在跑就回 503（網頁會每秒重連）。"""
            try:
                up = urllib.request.urlopen(mp.base + "/stream.mjpg", timeout=2)
            except (urllib.error.URLError, OSError):
                self._send(503, '{"msg":"no stream"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    chunk = up.read1(65536) if hasattr(up, "read1") else up.read(16384)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout):
                pass
            finally:
                up.close()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            path = self.path.split("?")[0]
            if path == "/api/main/start":
                msg = mp.start_main()
            elif path == "/api/main/stop":
                msg = mp.stop_main()
            elif path in ("/api/start", "/api/stop"):
                r = mp.call(path, {})
                msg = "主程式沒有在跑" if r is None else r.get("msg", "")
            elif path == "/api/opts":
                msg = mp.set_opts(body.get("record"), body.get("race"))
            else:
                self._send(404, '{"msg":"not found"}')
                return
            self._send(200, json.dumps({"msg": msg, "state": mp.state()}))

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"網頁服務：http://{lan_ip()}:{port}/（手機連同一個 Wi-Fi）。Ctrl+C 結束（網頁啟動的主程式會一起關）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if mp.child_alive():
            print("關閉網頁啟動的主程式…")
            mp.stop_main()
        srv.server_close()


if __name__ == "__main__":
    main()
