// 馬達接線診斷（不用 UART / Orin）。⚠ 車架高、輪子離地！
// 依序執行下列步驟，每步 3 秒，USB 序列埠(115200)會印出目前步驟，請邊看邊用電表量：
//   A1 ENA(D8)=HIGH(digitalWrite)  IN1=H IN2=L  → 馬達 A 應轉
//   A2 ENA(D8)=HIGH                IN1=L IN2=H  → 馬達 A 反轉
//   A3 ENA(D8)=analogWrite 200     IN1=H IN2=L  → 馬達 A 轉（測 PWM 是否有效）
//   B1~B3 同上，馬達 B（ENB=D9, IN3/IN4=CH6/CH7）；馬達 A 的 IN1/IN2=CH4/CH5
// 每步之間停 1 秒。

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

void allOff() {
  digitalWrite(EN_A, LOW); digitalWrite(EN_B, LOW);
  for (uint8_t ch = 4; ch <= 7; ch++) pcaDigital(ch, false);
}

void step(const char* name, uint8_t en, int pwm, uint8_t inA, uint8_t inB, bool fwd) {
  allOff();
  delay(1000);
  Serial.print(">> "); Serial.println(name);
  pcaDigital(inA, fwd);
  pcaDigital(inB, !fwd);
  if (pwm < 0) digitalWrite(en, HIGH); else analogWrite(en, pwm);
  delay(3000);
}

void setup() {
  Serial.begin(115200);
  pinMode(EN_A, OUTPUT); pinMode(EN_B, OUTPUT);
  digitalWrite(EN_A, LOW); digitalWrite(EN_B, LOW);
  delay(1500);
  Wire.begin();
  Wire.beginTransmission(PCA_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: PCA9685 not found at 0x40 -> IN1~IN4 無法控制，先查 SDA/SCL/VCC_LOGIC/GND");
  } else {
    Serial.println("PCA9685 OK");
  }
  pcaWrite8(0x00, 0x20);  // 喚醒 + auto-increment
  delay(1);
  pcaWrite8(0x01, 0x04);  // totem-pole
  allOff();
}

void loop() {
  step("A1 ENA=HIGH(digital) IN1=H IN2=L  馬達A 正轉?", EN_A, -1, 4, 5, true);
  step("A2 ENA=HIGH(digital) IN1=L IN2=H  馬達A 反轉?", EN_A, -1, 4, 5, false);
  step("A3 ENA=PWM 200       IN1=H IN2=L  馬達A 轉?",   EN_A, 200, 4, 5, true);
  step("B1 ENB=HIGH(digital) IN3=H IN4=L  馬達B 正轉?", EN_B, -1, 6, 7, true);
  step("B2 ENB=HIGH(digital) IN3=L IN4=H  馬達B 反轉?", EN_B, -1, 6, 7, false);
  step("B3 ENB=PWM 200       IN3=H IN4=L  馬達B 轉?",   EN_B, 200, 6, 7, true);
  allOff();
  Serial.println("== 一輪結束，5 秒後重來 ==");
  delay(5000);
}
