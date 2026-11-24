// PCA9685 通道掃描：找出 IN1~IN4 實際接在哪個通道。⚠ 車架高、輪子離地！
// ENA(D8)、ENB(D9) 一直保持 HIGH，PCA9685 CH1~CH15 輪流單獨輸出 HIGH 各 2 秒。
// 看序列埠(115200)目前是哪個 CH，記下「哪顆馬達、往哪個方向轉」。
// CH0 預設是前輪舵機，不掃（給舵機固定 HIGH 可能讓它打到底）。

#include <Wire.h>

const uint8_t PCA_ADDR = 0x40;
const uint8_t EN_A = 8, EN_B = 9;

void pcaWrite8(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(PCA_ADDR); Wire.write(reg); Wire.write(v); Wire.endTransmission();
}
void pcaDigital(uint8_t ch, bool high) {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x06 + 4 * ch);
  if (high) { Wire.write(0); Wire.write(0x10); Wire.write(0); Wire.write(0); }
  else      { Wire.write(0); Wire.write(0);    Wire.write(0); Wire.write(0x10); }
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  pinMode(EN_A, OUTPUT); pinMode(EN_B, OUTPUT);
  digitalWrite(EN_A, LOW); digitalWrite(EN_B, LOW);
  delay(1500);
  Wire.begin();
  Wire.beginTransmission(PCA_ADDR);
  Serial.println(Wire.endTransmission() == 0 ? "PCA9685 OK" : "ERROR: PCA9685 not found");
  pcaWrite8(0x00, 0x20);
  delay(1);
  pcaWrite8(0x01, 0x04);
  for (uint8_t ch = 1; ch <= 15; ch++) pcaDigital(ch, false);
  digitalWrite(EN_A, HIGH);
  digitalWrite(EN_B, HIGH);
  Serial.println("ENA=HIGH ENB=HIGH");
}

void loop() {
  for (uint8_t ch = 1; ch <= 15; ch++) {
    Serial.print(">> CH"); Serial.print(ch); Serial.println(" = HIGH");
    pcaDigital(ch, true);
    delay(2000);
    pcaDigital(ch, false);
    delay(1000);
  }
  Serial.println("== 一輪結束，5 秒後重來 ==");
  delay(5000);
}
