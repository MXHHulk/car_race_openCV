// 找出是什麼讓 Serial1 TX1(D21) 停止輸出。不驅動馬達。
// Serial1 每 100 ms 送一次 <STA,S1 -1,cs 34>，Orin 用 uart_test.py 計算收到幾個。
// 在 USB 序列埠監控視窗（115200，結尾字元選「沒有行結尾」或任一皆可）輸入：
//   3  pinMode + analogWrite(D8/D9, 0)          ← 測 PWM 腳
//   1  Wire.begin()                             ← 測 I2C 初始化
//   2  每 50 ms 寫一次 PCA9685（CH15，未使用）    ← 測 I2C 傳輸
//   4  重新呼叫 Serial1.begin()                  ← 看能不能把 TX1 搶回來
//   0  停止第 2 項的週期寫入
// 每次切換後，在 Orin 跑 python3 uart_test.py -t 3，記下「有效封包」數（正常約 30）。

#include <Wire.h>

const long BAUD = 115200;
const uint8_t PCA_ADDR = 0x40;

bool wireOn = false, i2cLoop = false, pwmOn = false;
unsigned long staCount = 0, rxBytes = 0, i2cCount = 0, i2cErr = 0;

uint8_t xorChecksum(const char* s, size_t n) {
  uint8_t cs = 0;
  for (size_t i = 0; i < n; i++) cs ^= (uint8_t)s[i];
  return cs;
}

void sendSTA() {
  const char* payload = "STA,S1 -1";
  char pkt[32];
  snprintf(pkt, sizeof(pkt), "<%s,cs %02X>\n", payload, xorChecksum(payload, strlen(payload)));
  Serial1.print(pkt);
  staCount++;
}

void pcaWriteCh15Off() {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x06 + 4 * 15);
  Wire.write(0); Wire.write(0); Wire.write(0); Wire.write(0x10);  // CH15 full OFF
  if (Wire.endTransmission() != 0) i2cErr++;
  i2cCount++;
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(BAUD);
  delay(500);
  Serial.println("UART bisect ready. 輸入 3 / 1 / 2 / 4 / 0");
}

void loop() {
  while (Serial1.available()) { Serial1.read(); rxBytes++; }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '3') {
      pinMode(8, OUTPUT); pinMode(9, OUTPUT);
      analogWrite(8, 0); analogWrite(9, 0);
      pwmOn = true;
      Serial.println(">> 3: analogWrite(D8/D9, 0) 完成");
    } else if (c == '1') {
      Wire.begin();
      wireOn = true;
      Serial.println(">> 1: Wire.begin() 完成");
    } else if (c == '2') {
      if (!wireOn) { Wire.begin(); wireOn = true; }
      i2cLoop = true;
      Serial.println(">> 2: 開始每 50 ms 寫 PCA9685");
    } else if (c == '0') {
      i2cLoop = false;
      Serial.println(">> 0: 停止 I2C 週期寫入");
    } else if (c == '4') {
      Serial1.begin(BAUD);
      Serial.println(">> 4: Serial1.begin() 重新初始化");
    }
  }

  unsigned long now = millis();
  static unsigned long tSta = 0, tI2c = 0, tDbg = 0;
  if (now - tSta >= 100) { tSta = now; sendSTA(); }
  if (i2cLoop && now - tI2c >= 50) { tI2c = now; pcaWriteCh15Off(); }
  if (now - tDbg >= 1000) {
    tDbg = now;
    Serial.print("pwm="); Serial.print(pwmOn);
    Serial.print(" wire="); Serial.print(wireOn);
    Serial.print(" i2cLoop="); Serial.print(i2cLoop);
    Serial.print("  sta="); Serial.print(staCount);
    Serial.print(" rx="); Serial.print(rxBytes);
    Serial.print(" i2c="); Serial.print(i2cCount);
    Serial.print(" i2cErr="); Serial.println(i2cErr);
  }
}
