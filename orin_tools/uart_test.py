#!/usr/bin/env python3
"""Orin <-> BMduino UART 通訊測試（依「UART 通訊協議書」ASCII 封包 + XOR8 checksum）

安全：只送出 fwa 0, bwl 0, bwr 0（馬達停止），不會讓車子動。

用法：
  python3 uart_test.py --list                     # 列出可用序列埠
  python3 uart_test.py -p /dev/ttyTHS1 -b 115200  # 測試 10 秒
  python3 uart_test.py -p /dev/ttyTHS1 -b 460800 --no-cs
"""
import argparse
import glob
import time

import serial


def checksum(payload: str) -> int:
    cs = 0
    for ch in payload.encode("ascii"):
        cs ^= ch
    return cs


def build_packet(payload: str, use_cs: bool) -> bytes:
    if use_cs:
        payload = f"{payload},cs {checksum(payload):02X}"
    return f"<{payload}>\n".encode("ascii")


def verify(body: str):
    """回傳 (ok, 原因)。body 為 < 與 > 之間的內容。"""
    idx = body.rfind(",cs ")
    if idx < 0:
        return False, "缺少 cs"
    hexpart = body[idx + 4:]
    if len(hexpart) != 2 or not all(c in "0123456789ABCDEF" for c in hexpart):
        return False, f"cs 格式錯誤: {hexpart!r}"
    calc = checksum(body[:idx])
    if calc != int(hexpart, 16):
        return False, f"cs 不符 (收到 {hexpart}, 計算 {calc:02X})"
    return True, ""


def list_ports():
    pats = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyTHS*", "/dev/ttyS*"]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default="/dev/ttyTHS1")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("-t", "--time", type=float, default=10.0, help="測試秒數")
    ap.add_argument("--hz", type=float, default=10.0, help="CMD 送出頻率")
    ap.add_argument("--no-cs", action="store_true", help="不附加/不驗證 checksum")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return

    use_cs = not args.no_cs
    pkt = build_packet("CMD,fwa 0,bwl 0,bwr 0", use_cs)
    print(f"開啟 {args.port} @ {args.baud} 8N1，送出 {pkt!r} @ {args.hz} Hz，共 {args.time}s")

    ser = serial.Serial(args.port, args.baud, timeout=0.01)
    ser.reset_input_buffer()

    sent = good = bad = raw_bytes = 0
    buf = None  # None = 等待 '<'
    period = 1.0 / args.hz
    t_end = time.time() + args.time
    t_next = 0.0
    while time.time() < t_end:
        now = time.time()
        if now >= t_next:
            ser.write(pkt)
            sent += 1
            t_next = now + period

        data = ser.read(256)
        raw_bytes += len(data)
        for ch in data.decode("ascii", errors="replace"):
            if ch == "<":
                buf = ""
            elif buf is None:
                continue
            elif ch == ">":
                body, buf = buf, None
                ok, why = verify(body) if use_cs else (True, "")
                if ok:
                    good += 1
                    print(f"[OK ] <{body}>")
                else:
                    bad += 1
                    print(f"[BAD] <{body}>  {why}")
            elif len(buf) > 128:
                buf = None  # 超長丟棄
            else:
                buf += ch

    ser.write(pkt)  # 結束前再送一次停止
    ser.close()

    print("\n==== 結果 ====")
    print(f"送出 CMD: {sent}  收到原始 bytes: {raw_bytes}  有效封包: {good}  無效封包: {bad}")
    if good:
        print("PASS：BMduino 有回傳有效封包，通訊正常。")
    elif raw_bytes:
        print("FAIL：有收到資料但無有效封包 → 檢查鮑率、--no-cs 設定、封包格式是否一致。")
    else:
        print("FAIL：完全沒收到資料 → 檢查接線 TX/RX 是否交叉、GND 共地、序列埠名稱、BMduino 是否有回報 STA。")


if __name__ == "__main__":
    main()
