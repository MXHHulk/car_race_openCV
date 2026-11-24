"""圈數賽起跑旗標：等中控台 /go 由 0 變 1（見 CAR-INTEGRATION.md）。

只用標準函式庫。RaceStart 在背景執行緒輪詢，不會卡住控制迴圈：
  - 開啟（arm）後每 50 ms 讀一次 /go（timeout 0.2 s）
  - 旗標要先讀到 0、之後變 1 才起跑（開啟時已經是 1 = 上一場沒重置，不起跑）
  - 起跑後自動關閉，不再輪詢；下一場要再手動開
  - stop_on_zero：起跑後改成每 0.2 s 讀一次，讀到 0（裁判按停止）就停車；讀不到不算停止
  - 沒開啟時完全不碰網路；讀不到旗標只顯示離線，不影響其他功能
"""
import threading
import time
from urllib.request import urlopen

PI_GO_URL = "http://192.168.0.149:8080/go"


def read_go(url=PI_GO_URL, timeout=0.2):
    """回傳 1 或 0；讀不到（斷線、逾時）回 None。"""
    try:
        with urlopen(url, timeout=timeout) as r:
            return 1 if r.read(8).strip() == b"1" else 0
    except (OSError, ValueError):
        return None


class RaceStart:
    def __init__(self, on_go, on_stop=None, url=PI_GO_URL, interval=0.05, stop_on_zero=False):
        """on_go()：綠燈時呼叫（起跑）。on_stop(reason)：stop_on_zero 時旗標回 0 呼叫。"""
        self.on_go, self.on_stop = on_go, on_stop
        self.url, self.interval, self.stop_on_zero = url, interval, stop_on_zero
        self.value = None          # 最近一次讀到的旗標（1 / 0 / None = 離線）
        self.saw_zero = False
        self.t_go = None           # 起跑時間（time.time()）
        self._armed = threading.Event()
        self._watch = threading.Event()   # 起跑後看停止旗標
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def armed(self):
        return self._armed.is_set()

    def arm(self):
        with self._lock:
            self.saw_zero, self.value = False, -1   # -1 = 還沒讀過（第一次讀到什麼都會顯示，含離線）
            self._watch.clear()
            self._armed.set()
        print(f"[race] 起跑模式開啟：等待綠燈（{self.url}）")

    def disarm(self, why="關閉"):
        if self._armed.is_set():
            self._armed.clear()
            print(f"[race] 起跑模式{why}")
        self._watch.clear()

    def status(self):
        """給畫面 / 紀錄用的一行字。"""
        if self._armed.is_set():
            v = {None: "離線", -1: "?"}.get(self.value, str(self.value))
            return f"等綠燈 旗標={v}" + ("" if self.saw_zero else "（等歸零）")
        if self._watch.is_set():
            return "比賽中"
        return ""

    def _show(self, value):
        if value != self.value:
            print("[race] 旗標：" + ("中控台連不到（離線）" if value is None else str(value)))
        self.value = value

    def _run(self):
        while True:
            if self._armed.is_set():
                value = read_go(self.url)
                with self._lock:
                    if not self._armed.is_set():      # 讀的時候被關掉了
                        continue
                    self._show(value)
                    if value == 0:
                        self.saw_zero = True
                    elif value == 1 and self.saw_zero:
                        self._armed.clear()           # 一次性：起跑後自動關閉
                        self.t_go = time.time()
                        print("[race] 綠燈！起跑（起跑模式已自動關閉）")
                        if self.stop_on_zero:
                            self._watch.set()
                        self.on_go()
                        continue
                time.sleep(self.interval)
            elif self._watch.is_set():
                time.sleep(0.2)
                if read_go(self.url) == 0 and self._watch.is_set():
                    self._watch.clear()
                    print("[race] 旗標回到 0（裁判停止）")
                    if self.on_stop:
                        self.on_stop("裁判停止")
            else:
                time.sleep(0.05)
