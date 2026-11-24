// 步驟 2：I2C 掃描。找出 BMduino 的 Wire 上有哪些裝置。不驅動任何東西。
// PCA9685 正常時會看到 0x40（另外常會多一個 0x70，是 PCA9685 的 All Call 位址，也正常）。
// USB 序列埠 115200，每 3 秒掃一次。

#include <Wire.h>

void setup() {
  Serial.begin(115200);
  delay(1000);
  Wire.begin();
  Serial.println("I2C scan ready");
}

void loop() {
  int found = 0;
  Serial.println("---- 掃描中 ----");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("找到裝置 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) Serial.println("沒有找到任何 I2C 裝置");
  else { Serial.print("共 "); Serial.print(found); Serial.println(" 個"); }
  delay(3000);
}
