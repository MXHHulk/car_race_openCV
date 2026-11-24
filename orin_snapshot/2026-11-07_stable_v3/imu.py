"""IMU（GY-521 / MPU-6050）：量車身傾斜角和車頭轉角。

接線（Orin Nano 40-pin）：SDA 第 27 腳、SCL 第 28 腳（/dev/i2c-1）、VCC 第 17 腳（3.3 V）、GND 第 25 腳。

不用知道 IMU 怎麼裝：啟動時車子不動，先量 1 秒重力方向當「上」。
  - tilt：現在的重力方向和啟動時差幾度（上坡、下坡都是正值）
  - yaw：陀螺儀繞「上」軸轉了幾度，正 = 往左（和 yaw.py 相同）
  - pitch：車頭往上抬幾度（上坡正、下坡負），要知道 IMU 哪一軸朝車頭（forward，車上是 +x）
    run_20261103_184744：平地最多 +5.8°、上坡 +13.5 ~ +16°、下坡 −15 ~ −19°
啟動時車子要停在平地上，不然 tilt 的 0 度就不是平地。

單獨執行 `python3 imu.py` 會每 0.2 秒印一次，用來確認有讀到：把車子前後抬起 tilt 要變大，原地左轉 yaw 要變大。
"""
import fcntl
import math
import os
import struct
import threading
import time

I2C_SLAVE = 0x0703
PWR_MGMT_1, SMPLRT_DIV, CONFIG, GYRO_CONFIG, ACCEL_CONFIG, ACCEL_XOUT, WHO_AM_I = 0x6B, 0x19, 0x1A, 0x1B, 0x1C, 0x3B, 0x75
ACC_LSB, GYRO_LSB = 16384.0, 65.5   # ±2 g、±500 °/s


class IMU:
    def __init__(self, bus=1, addr=0x68, hz=200, calib_s=1.0, tilt_tau=0.3, forward="+x"):
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        # AD0 腳接高時位址是 0x69：設定的位址沒回應就試另一個
        for a in (addr, addr ^ 0x01):
            try:
                fcntl.ioctl(self.fd, I2C_SLAVE, a)
                who = self._read(WHO_AM_I, 1)[0]
                addr = a
                break
            except OSError:
                continue
        else:
            os.close(self.fd)
            raise OSError(f"/dev/i2c-{bus} 的 0x68、0x69 都沒有回應：用 `i2cdetect -y -r {bus}` 確認接線和匯流排編號")
        self._write(PWR_MGMT_1, 0x00)       # 叫醒
        time.sleep(0.1)
        self._write(SMPLRT_DIV, 0x04)       # 1 kHz / 5 = 200 Hz
        self._write(CONFIG, 0x03)           # 低通 44 Hz：濾掉馬達震動
        self._write(GYRO_CONFIG, 0x08)      # ±500 °/s
        self._write(ACCEL_CONFIG, 0x00)     # ±2 g
        print(f"[imu] /dev/i2c-{bus} 0x{addr:02x}，WHO_AM_I = 0x{who:02x}")
        self.period, self.calib_s, self.tilt_tau = 1.0 / hz, calib_s, tilt_tau
        self.fwd_i, self.fwd_s = "xyz".index(forward[-1]), (-1.0 if forward[0] == "-" else 1.0)
        self.pitch = 0.0
        self.up = None                      # 啟動時的重力方向（單位向量，感測器座標）
        self.bias = (0.0, 0.0, 0.0)         # 陀螺儀零點（°/s）
        self.acc = (0.0, 0.0, 0.0)          # 最新加速度（g，低通過）
        self.tilt = 0.0
        self.yaw = 0.0
        self.ok = False                     # 校正完成
        self.errors = 0
        self.lock = threading.Lock()
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _write(self, reg, val):
        os.write(self.fd, bytes([reg, val]))

    def _read(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, n)

    def _sample(self):
        ax, ay, az, _, gx, gy, gz = struct.unpack(">7h", self._read(ACCEL_XOUT, 14))
        return (ax / ACC_LSB, ay / ACC_LSB, az / ACC_LSB), (gx / GYRO_LSB, gy / GYRO_LSB, gz / GYRO_LSB)

    def _loop(self):
        # 校正：車子不動，平均重力方向和陀螺儀零點
        n, sa, sg = 0, [0.0] * 3, [0.0] * 3
        t_end = time.time() + self.calib_s
        while self._running and time.time() < t_end:
            try:
                a, g = self._sample()
            except OSError:
                self.errors += 1
                time.sleep(self.period)
                continue
            sa = [s + v for s, v in zip(sa, a)]
            sg = [s + v for s, v in zip(sg, g)]
            n += 1
            time.sleep(self.period)
        if not n:
            print("[imu] 讀不到資料，IMU 不使用")
            return
        norm = math.sqrt(sum(s * s for s in sa)) or 1.0
        with self.lock:
            self.up = tuple(s / norm for s in sa)
            self.bias = tuple(s / n for s in sg)
            self.acc = tuple(s / n for s in sa)
            self.ok = True
        print(f"[imu] 校正完成：重力 {norm / n:.2f} g，陀螺儀零點 {', '.join(f'{b:+.2f}' for b in self.bias)} °/s")
        t_prev = time.time()
        while self._running:
            try:
                a, g = self._sample()
            except OSError:
                self.errors += 1
                time.sleep(self.period)
                continue
            now = time.time()
            dt = min(0.05, now - t_prev)
            t_prev = now
            k = min(1.0, dt / self.tilt_tau)
            with self.lock:
                self.acc = tuple(p + (v - p) * k for p, v in zip(self.acc, a))
                # 繞「上」軸的角速度 = 車頭轉向（右手定則，往左轉為正）
                self.yaw += sum((v - b) * u for v, b, u in zip(g, self.bias, self.up)) * dt
                norm = math.sqrt(sum(v * v for v in self.acc)) or 1.0
                c = sum(v * u for v, u in zip(self.acc, self.up)) / norm
                self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                # 車頭抬起時重力讀值往車頭那一軸跑：和啟動時比多了幾度（IMU 裝得不平也沒關係）
                f = lambda v: math.degrees(math.asin(max(-1.0, min(1.0, self.fwd_s * v))))
                self.pitch = f(self.acc[self.fwd_i] / norm) - f(self.up[self.fwd_i])
            time.sleep(max(0.0, self.period - (time.time() - now)))

    def read(self):
        """回傳 (tilt 度, yaw 度, 加速度 (x, y, z) g, pitch 度)；還沒校正好時回傳 None。"""
        with self.lock:
            return (self.tilt, self.yaw, self.acc, self.pitch) if self.ok else None

    def close(self):
        self._running = False
        self._th.join(timeout=0.5)
        os.close(self.fd)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="確認 IMU 有讀到")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x68)
    args = ap.parse_args()
    imu = IMU(args.bus, args.addr)
    print("校正中，車子不要動…")
    try:
        while True:
            time.sleep(0.2)
            r = imu.read()
            if r:
                tilt, yaw, (ax, ay, az), pitch = r
                print(f"傾斜 {tilt:5.1f}°  俯仰 {pitch:+5.1f}°  車頭 {yaw:+7.1f}°  加速度 {ax:+.2f} {ay:+.2f} {az:+.2f} g  錯誤 {imu.errors}")
    except KeyboardInterrupt:
        pass
    imu.close()
