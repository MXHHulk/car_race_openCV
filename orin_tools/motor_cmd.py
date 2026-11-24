#!/usr/bin/env python3
"""Orin -> BMduino 馬達/舵機控制（依「UART 通訊協議書」CMD/STA + XOR8 checksum）

搭配 BMduino 韌體 bmduino_motor.ino。以固定頻率持續送 CMD，
BMduino 超過 500 ms 沒收到有效 CMD 會自動停車。

⚠ 第一次測試請把車架高，讓輪子離地！

用法：
  python3 motor_cmd.py --fwa 0 --bwl 20 --bwr 20 -t 2    # 定值跑 2 秒後停
  python3 motor_cmd.py --fwa 20 -t 1                     # 只轉前輪
  python3 motor_cmd.py --teleop                          # 鍵盤遙控（SSH 可用）
"""
import argparse
import curses
import locale
import time
import unicodedata

import serial


def checksum(payload: str) -> int:
    cs = 0
    for ch in payload.encode("ascii"):
        cs ^= ch
    return cs


def clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def diff_split(speed, fwa, fwa_max, gain):
    """後輪差速：前輪轉越多，外側輪越快、內側輪越慢（gain > 1 時內側輪會反轉），幫忙轉急彎。

    gain 0 = 兩輪同速；1 = 打到底時外側 2 倍、內側停；1.5 = 內側反轉一半。fwa 正 = 右轉（右輪是內側）。
    和 lane/controller.py 的 wheels() 是同一個公式，手動試出來的 gain 可以直接填進 config 的 diff_gain。
    """
    r = max(-1.0, min(1.0, fwa / fwa_max)) if fwa_max else 0.0
    return clamp(speed * (1 + gain * r), -100, 100), clamp(speed * (1 - gain * r), -100, 100)


class BMduinoLink:
    def __init__(self, port="/dev/ttyTHS1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0)
        self.ser.reset_input_buffer()
        self._buf = None
        self.sta = {}          # 最新 STA 欄位，例如 {"S1": -1}
        self.sta_time = 0.0
        self.good = self.bad = 0
        self.raw_bytes = 0     # 收到的原始 byte 數（判斷是否完全沒收到訊號）
        self.raw_head = b""    # 最前面收到的原始資料，除錯用

    def send(self, fwa, bwl, bwr):
        fwa, bwl, bwr = clamp(fwa, -90, 90), clamp(bwl, -100, 100), clamp(bwr, -100, 100)
        payload = f"CMD,fwa {fwa},bwl {bwl},bwr {bwr}"
        self.ser.write(f"<{payload},cs {checksum(payload):02X}>\n".encode("ascii"))

    def stop(self):
        for _ in range(3):
            self.send(0, 0, 0)
            time.sleep(0.02)

    def poll(self):
        """讀取並解析收到的 STA，回傳本次是否有新封包。"""
        got = False
        data = self.ser.read(512)
        self.raw_bytes += len(data)
        if len(self.raw_head) < 64:
            self.raw_head += data[:64 - len(self.raw_head)]
        for ch in data.decode("ascii", errors="replace"):
            if ch == "<":
                self._buf = ""
            elif self._buf is None:
                continue
            elif ch == ">":
                body, self._buf = self._buf, None
                if self._handle(body):
                    got = True
            elif len(self._buf) > 128:
                self._buf = None
            else:
                self._buf += ch
        return got

    def _handle(self, body):
        idx = body.rfind(",cs ")
        hexpart = body[idx + 4:] if idx >= 0 else ""
        if (idx < 0 or len(hexpart) != 2 or any(c not in "0123456789ABCDEF" for c in hexpart)
                or checksum(body[:idx]) != int(hexpart, 16)):
            self.bad += 1
            return False
        fields = body[:idx].split(",")
        if fields[0] != "STA":
            return False
        sta = {}
        for f in fields[1:]:
            k, _, v = f.partition(" ")
            try:
                sta[k] = int(v)
            except ValueError:
                pass
        self.sta, self.sta_time = sta, time.time()
        self.good += 1
        return True

    def close(self):
        self.stop()
        self.ser.close()


def run_fixed(link, fwa, bwl, bwr, duration, hz):
    print(f"送出 fwa={fwa} bwl={bwl} bwr={bwr}，{duration}s @ {hz}Hz（Ctrl+C 可中止）")
    period = 1.0 / hz
    t_end = time.time() + duration
    while time.time() < t_end:
        link.send(fwa, bwl, bwr)
        link.poll()
        time.sleep(period)
    link.stop()
    age = time.time() - link.sta_time if link.sta_time else None
    print(f"已停止。STA 有效 {link.good} / 無效 {link.bad}，最後 STA={link.sta}"
          + ("" if age is None else f"（{age:.2f}s 前）"))
    print(f"收到原始 bytes：{link.raw_bytes}")
    if not link.good:
        if link.raw_bytes == 0:
            print("警告：完全沒收到資料 → BMduino TX1(D21) → 74HC125 → Orin Pin 10 這段沒訊號，"
                  "或有其他程式正在讀 /dev/ttyTHS1（sudo fuser -v /dev/ttyTHS1）")
        else:
            print(f"警告：有收到資料但不是有效 STA → 檢查鮑率。最前面的資料：{link.raw_head!r}")


TELEOP_HELP = """\
W/S : 速度 +/- 5       A/D : 轉向 -/+ 5 度
Z/X : 差速增益 -/+ 0.25（轉彎時內側輪慢、外側輪快）
空白: 停車(速度歸 0)    C   : 前輪回正
Q   : 離開（會先停車）"""


def put(scr, y, x, text):
    """在視窗裡寫一行，超出視窗的部分截掉。curses 寫到視窗外（終端機太小、中文字算兩格）會丟
    addwstr() returned ERR，整個遙控程式就當掉。"""
    h, w = scr.getmaxyx()
    if y >= h:
        return
    out, used = "", x
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + cw >= w:
            break
        out += ch
        used += cw
    try:
        scr.addstr(y, x, out)
    except curses.error:
        pass


def run_teleop(link, hz, speed_max, fwa_max, gain=0.0):
    locale.setlocale(locale.LC_ALL, "")      # 中文字要用 UTF-8 顯示

    def ui(scr):
        curses.curs_set(0)
        scr.nodelay(True)
        speed, fwa, g = 0, 0, gain
        period = 1.0 / hz
        while True:
            k = scr.getch()
            if k != -1:
                c = chr(k).lower() if 0 <= k < 256 else ""
                if c == "w":
                    speed += 5
                elif c == "s":
                    speed -= 5
                elif c == "z":
                    g = max(0.0, g - 0.25)
                elif c == "x":
                    g = min(2.0, g + 0.25)
                elif c == "a":
                    fwa -= 5
                elif c == "d":
                    fwa += 5
                elif c == " ":
                    speed = 0
                elif c == "c":
                    fwa = 0
                elif c == "q":
                    break
                speed = clamp(speed, -speed_max, speed_max)
                fwa = clamp(fwa, -fwa_max, fwa_max)

            bwl, bwr = diff_split(speed, fwa, fwa_max, g)
            link.send(fwa, bwl, bwr)
            link.poll()
            age = time.time() - link.sta_time if link.sta_time else 99
            scr.erase()
            for i, line in enumerate(TELEOP_HELP.splitlines()):
                put(scr, i, 0, line)
            put(scr, 6, 0, f"速度 {speed:4d} %   轉向 fwa={fwa:4d} 度   差速增益 {g:.2f}")
            put(scr, 7, 0, f"實際送出：左後輪 bwl={bwl:4d} %   右後輪 bwr={bwr:4d} %")
            put(scr, 8, 0, f"STA={link.sta}  有效 {link.good} 無效 {link.bad}")
            put(scr, 9, 0, "連線正常" if age < 0.5 else "!! 沒收到 STA（BMduino 沒回應？）")
            scr.refresh()
            time.sleep(period)

    try:
        curses.wrapper(ui)
    finally:
        link.stop()
        print("已停車並離開。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default="/dev/ttyTHS1")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--hz", type=float, default=20.0, help="CMD 送出頻率")
    ap.add_argument("--fwa", type=int, default=0, help="前輪轉角 -90~90（負左正右）")
    ap.add_argument("--bwl", type=int, default=0, help="左後輪 -100~100 %%")
    ap.add_argument("--bwr", type=int, default=0, help="右後輪 -100~100 %%")
    ap.add_argument("-t", "--time", type=float, default=2.0, help="定值模式持續秒數")
    ap.add_argument("--teleop", action="store_true", help="鍵盤遙控模式")
    ap.add_argument("--speed-max", type=int, default=50, help="遙控速度上限 %%")
    ap.add_argument("--fwa-max", type=int, default=30, help="遙控轉角上限 度")
    ap.add_argument("--diff", type=float, default=0.0, help="遙控的起始差速增益（Z/X 可再調）")
    args = ap.parse_args()

    link = BMduinoLink(args.port, args.baud)
    try:
        if args.teleop:
            run_teleop(link, args.hz, args.speed_max, args.fwa_max, args.diff)
        else:
            run_fixed(link, args.fwa, args.bwl, args.bwr, args.time, args.hz)
    except KeyboardInterrupt:
        print("\n中止")
    finally:
        link.close()


if __name__ == "__main__":
    main()
