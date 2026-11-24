"""取像：獨立執行緒一直讀鏡頭，只保留最新一張。也可以讀影片檔（離線測試用，不丟張）。"""
import subprocess
import sys
import threading
import time

import cv2


class Camera:
    def __init__(self, device=0, width=640, height=480, fps=30, v4l2_ctrls=None):
        if v4l2_ctrls and sys.platform.startswith("linux"):
            for k, v in v4l2_ctrls.items():
                r = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}", "-c", f"{k}={v}"],
                                   capture_output=True, text=True)
                if r.returncode:
                    print(f"[camera] 設定 {k}={v} 失敗：{r.stderr.strip()}")
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, api)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # 不要把 CAP_PROP_BUFFERSIZE 設成 1：V4L2 只有一個緩衝時，程式取出畫面的那段時間鏡頭沒地方放下一張，
        # 會每隔一張掉一張（30 → 15 fps）。這裡有執行緒一直把畫面讀走，用預設緩衝也不會累積延遲。
        if not self.cap.isOpened():
            raise SystemExit(f"無法開啟 /dev/video{device}，先用 `v4l2-ctl --list-devices` 確認裝置編號")
        self.live = True
        self._frame, self._t, self._id = None, 0.0, 0
        self._cond = threading.Condition()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._t, self._id = f, time.time(), self._id + 1
                self._cond.notify_all()

    def read(self, last_id=0, timeout=1.0):
        """等到比 last_id 新的畫面，回傳 (id, 取得時間, 影像)；逾時回傳 None。"""
        with self._cond:
            if not self._cond.wait_for(lambda: self._id > last_id, timeout):
                return None
            return self._id, self._t, self._frame

    def close(self):
        self._running = False
        self.cap.release()


class VideoFile:
    """影片或圖片資料夾，逐張回傳；時間戳依影片 fps 推算，讓控制和實車一樣用秒計算。"""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"無法開啟 {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.live = False
        self._id = 0

    def read(self, last_id=0, timeout=1.0):
        ok, f = self.cap.read()
        if not ok:
            return None
        self._id += 1
        return self._id, self._id / self.fps, f

    def close(self):
        self.cap.release()
