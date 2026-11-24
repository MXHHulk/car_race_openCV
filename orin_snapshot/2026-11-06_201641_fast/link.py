"""通訊：獨立執行緒每 50 ms 送一次「目前最新」的指令，並讀 STA。

dry_run=True 時不開序列埠（電腦上離線測試用）。
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Sender:
    def __init__(self, port, baud, hz=20, dry_run=False):
        self.dry = dry_run
        self.period = 1.0 / hz
        self.cmd = (0, 0, 0)
        self.lock = threading.Lock()
        self.sent = 0
        self.link = None
        if not dry_run:
            from motor_cmd import BMduinoLink  # 延後 import：電腦上沒有 pyserial / curses 也能 dry-run
            self.link = BMduinoLink(port, baud)
        self.t_start = time.time()
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def set(self, fwa, bwl, bwr):
        with self.lock:
            self.cmd = (int(round(fwa)), int(round(bwl)), int(round(bwr)))

    def sta_age(self):
        """距離上次收到有效 STA 幾秒；dry-run 永遠回 0。剛啟動還沒收到時從啟動時間算。"""
        if self.dry:
            return 0.0
        return time.time() - (self.link.sta_time or self.t_start)

    def _loop(self):
        nxt = time.time()
        while self._running:
            with self.lock:
                cmd = self.cmd
            if self.link:
                self.link.send(*cmd)
                self.link.poll()
            self.sent += 1
            nxt += self.period
            time.sleep(max(0.0, nxt - time.time()))

    def close(self):
        """停止執行緒，送 3 次停車再關序列埠。"""
        self._running = False
        self._th.join(timeout=0.5)
        if self.link:
            self.link.close()
