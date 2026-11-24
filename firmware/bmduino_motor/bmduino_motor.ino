// BMduino 馬達/舵機控制韌體（依「UART 通訊協議書」+ 附錄 B XOR8 checksum）
//
// 收：<CMD,fwa X,bwl Y,bwr Z,cs XX>\n   fwa -90~90 度(負左正右)，bwl/bwr -100~100 %(正前進)
// 回：<STA,S1 -1,cs XX>\n               每 100 ms 一次（尚未接 TOF，S1 = -1）
//
// 接線（依實車；通道與校正值取自已實測可動的 amr_ble_control2.ino v2.0）：
//   Serial1  D20 RX1 / D21 TX1  ↔ Orin UART1 (/dev/ttyTHS1)
//   I2C SDA/SCL → PCA9685 (0x40)
//   馬達 1：ENA ← 74LS14 ← D8，IN1/IN2 ← PCA9685 CH4/CH5，輸出 OUT1/OUT2
//   馬達 2：ENB ← 74LS14 ← D9，IN3/IN4 ← PCA9685 CH6/CH7，輸出 OUT3/OUT4
//   前輪舵機 → PCA9685 CH15（原本 CH0，2026-11-01 改接；機構零點 1520us，行程 1100~1920us，約 12us/度）
//
// 安全：上電馬達停止；超過 FAILSAFE_MS 沒收到有效 CMD → 兩顆馬達停止，舵機維持最後位置。

#include <Wire.h>

// ================= 依實車調整 =================
const long BAUD = 115200;                 // 與 Orin 端相同
const uint8_t PCA_ADDR = 0x40;

const uint32_t PCA_OSC_HZ = 25880000UL;   // PCA9685 振盪器實測值（標稱 25MHz）
const uint8_t SERVO_CH = 15;              // 前輪舵機接 PCA9685 哪個通道
const int SERVO_CENTER_US = 1520;         // 機構零點：輪子打直的脈寬（拆過舵臂要重調）
const float SERVO_US_PER_DEG = 12.0;      // 實測約 12us/度
// 轉向行程絕對上下限。原本 1100–1920us（約 -35°～+33°）是裝連桿後的干涉點，避免打到機構。
// 2026-10-28 為了過直角彎，放寬到舵機本身的驗證範圍 700–2100us（約 -68°～+48°）。
// ⚠ 超過干涉點時舵機會頂住機構：可能發燙、燒毀或打壞齒輪。實際轉多少由 Orin 端 fwa 決定，請從小角度慢慢加。
// 想恢復原本的安全範圍，改回 1100 / 1920 重新燒錄。
const int SERVO_US_MIN = 700;
const int SERVO_US_MAX = 2100;
const bool SERVO_INVERT = false;          // 左右轉相反就改 true

struct Motor {
  uint8_t enPin;   // BMduino 腳位（PWM 調速）
  uint8_t inA;     // PCA9685 通道（方向）
  uint8_t inB;
  bool invert;     // 給正值卻倒轉就改 true
};
// 極性取自藍芽韌體台架實測（馬達1 invert=true、馬達2 invert=false 時兩顆同向前進）。
// 哪一顆是左輪尚未確認：bwl 轉到右輪的話，把兩行對調。
Motor LEFT  = {8, 4, 5, true};            // 馬達 1
Motor RIGHT = {9, 6, 7, false};           // 馬達 2
const int MIN_DUTY = 0;                   // 馬達起動死區(0~255)，低速轉不動時調高，例如 60

const unsigned long FAILSAFE_MS = 500;    // 失聯保護
const unsigned long RX_TIMEOUT_MS = 50;   // 未完成封包逾時
const unsigned long STA_PERIOD_MS = 100;  // STA 回報週期
const bool DEBUG = true;                  // USB 序列埠印出狀態（115200）
// ==============================================

const int MAX_LEN = 64;
char buf[MAX_LEN + 1];
int len = -1;                 // -1 = 等待 '<'
unsigned long rxStart = 0;

int curFwa = 0, curL = 0, curR = 0;
bool motorsStopped = true;
unsigned long lastValidCmd = 0;
bool everValid = false;
unsigned long okCount = 0, badCount = 0, staCount = 0;

// ---------- PCA9685 ----------
// prescale = round(osc / (4096 * 50Hz)) - 1
const uint8_t pcaPrescale = (uint8_t)((PCA_OSC_HZ + 4096UL * 25) / (4096UL * 50) - 1);

void pcaWrite8(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(reg);
  Wire.write(v);
  Wire.endTransmission();
}

void pcaSet(uint8_t ch, uint16_t on, uint16_t off) {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x06 + 4 * ch);
  Wire.write(on & 0xFF);
  Wire.write(on >> 8);
  Wire.write(off & 0xFF);
  Wire.write(off >> 8);
  Wire.endTransmission();
}

void pcaDigital(uint8_t ch, bool high) {
  if (high) pcaSet(ch, 4096, 0);   // full ON
  else      pcaSet(ch, 0, 4096);   // full OFF
}

bool pcaInit() {
  Wire.begin();
  Wire.beginTransmission(PCA_ADDR);
  if (Wire.endTransmission() != 0) return false;
  pcaWrite8(0x00, 0x10);           // MODE1: sleep（才能改 prescale）
  pcaWrite8(0xFE, pcaPrescale);    // 25.88MHz / (4096 * 50Hz) - 1 ≈ 125
  pcaWrite8(0x00, 0x20);           // 喚醒 + auto-increment
  delay(1);
  pcaWrite8(0x01, 0x04);           // MODE2: totem-pole 輸出
  return true;
}

// ---------- 舵機 / 馬達 ----------
void setServo(int fwa) {
  if (SERVO_INVERT) fwa = -fwa;
  float us = SERVO_CENTER_US + fwa * SERVO_US_PER_DEG;
  us = constrain(us, SERVO_US_MIN, SERVO_US_MAX);
  // 一個 tick = (prescale+1) / 振盪器頻率
  uint16_t ticks = (uint16_t)(us * (float)PCA_OSC_HZ / (1000000.0 * (pcaPrescale + 1)) + 0.5);
  pcaSet(SERVO_CH, 0, ticks);
}

void setMotor(const Motor& m, int pct, int prevPct) {
  if (m.invert) { pct = -pct; prevPct = -prevPct; }
  pct = constrain(pct, -100, 100);
  if (pct == 0) {
    analogWrite(m.enPin, 0);
    pcaDigital(m.inA, false);
    pcaDigital(m.inB, false);
    return;
  }
  bool dirChanged = (pct > 0) != (prevPct > 0) || prevPct == 0;
  if (dirChanged) {
    analogWrite(m.enPin, 0);       // 先關 EN 再換方向
    pcaDigital(m.inA, pct > 0);
    pcaDigital(m.inB, pct < 0);
  }
  int duty = MIN_DUTY + (long)(255 - MIN_DUTY) * abs(pct) / 100;
  analogWrite(m.enPin, duty);
}

void applyCmd(int fwa, int l, int r) {
  if (fwa != curFwa) setServo(fwa);
  // 只在數值改變時才寫 PCA9685（停車時 curL/curR 已是 0），避免每個 CMD 都產生 I2C 傳輸
  if (l != curL) setMotor(LEFT, l, curL);
  if (r != curR) setMotor(RIGHT, r, curR);
  curFwa = fwa; curL = l; curR = r;
  motorsStopped = (l == 0 && r == 0);
}

void stopMotors() {
  setMotor(LEFT, 0, 0);
  setMotor(RIGHT, 0, 0);
  curL = curR = 0;
  motorsStopped = true;
}

// ---------- 封包 ----------
uint8_t xorChecksum(const char* s, size_t n) {
  uint8_t cs = 0;
  for (size_t i = 0; i < n; i++) cs ^= (uint8_t)s[i];
  return cs;
}

// 驗證 checksum；通過時回傳 ",cs " 的位置，否則 NULL
char* verify(char* body) {
  char* p = strstr(body, ",cs ");
  if (!p) return NULL;
  char* last;
  while ((last = strstr(p + 1, ",cs ")) != NULL) p = last;
  const char* hx = p + 4;
  if (strlen(hx) != 2 || !isxdigit(hx[0]) || !isxdigit(hx[1])) return NULL;
  if (islower(hx[0]) || islower(hx[1])) return NULL;   // 協議規定大寫
  uint8_t got = (uint8_t)strtol(hx, NULL, 16);
  return got == xorChecksum(body, p - body) ? p : NULL;
}

bool parseInt(const char* s, int& out) {
  if (*s == '\0') return false;
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0') return false;
  out = (int)v;
  return true;
}

// body 已去掉 ",cs XX"。三個欄位都要有、不可重複、需在範圍內，否則整包無效。
bool parseCMD(char* body, int& fwa, int& bwl, int& bwr) {
  if (strncmp(body, "CMD,", 4) != 0) return false;
  bool hasF = false, hasL = false, hasR = false;
  char* save;
  for (char* tok = strtok_r(body + 4, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    char* sp = strchr(tok, ' ');
    if (!sp) return false;
    *sp = '\0';
    int v;
    if (!parseInt(sp + 1, v)) return false;
    if (strcmp(tok, "fwa") == 0)      { if (hasF || v < -90 || v > 90) return false;   fwa = v; hasF = true; }
    else if (strcmp(tok, "bwl") == 0) { if (hasL || v < -100 || v > 100) return false; bwl = v; hasL = true; }
    else if (strcmp(tok, "bwr") == 0) { if (hasR || v < -100 || v > 100) return false; bwr = v; hasR = true; }
    // 未知欄位忽略
  }
  return hasF && hasL && hasR;
}

void handlePacket(char* body) {
  char* csPos = verify(body);
  int fwa, l, r;
  if (csPos) {
    *csPos = '\0';
    if (parseCMD(body, fwa, l, r)) {
      okCount++;
      lastValidCmd = millis();
      everValid = true;
      applyCmd(fwa, l, r);
      return;
    }
  }
  badCount++;
}

void sendSTA(int s1) {
  char payload[32], pkt[48];
  snprintf(payload, sizeof(payload), "STA,S1 %d", s1);
  snprintf(pkt, sizeof(pkt), "<%s,cs %02X>\n", payload, xorChecksum(payload, strlen(payload)));
  Serial1.print(pkt);
  staCount++;
}

void readUart() {
  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == '<') { len = 0; rxStart = millis(); continue; }
    if (len < 0) continue;
    if (c == '>') {
      buf[len] = '\0';
      len = -1;
      handlePacket(buf);
      continue;
    }
    if (len >= MAX_LEN) { len = -1; badCount++; continue; }
    buf[len++] = c;
  }
  if (len >= 0 && millis() - rxStart > RX_TIMEOUT_MS) { len = -1; badCount++; }
}

// ---------- 主程式 ----------
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEFT.enPin, OUTPUT);
  pinMode(RIGHT.enPin, OUTPUT);
  analogWrite(LEFT.enPin, 0);
  analogWrite(RIGHT.enPin, 0);

  Serial.begin(115200);

  if (!pcaInit()) {
    Serial.println("ERROR: PCA9685 not found at 0x40 (檢查 SDA/SCL/VCC_LOGIC)");
  }
  stopMotors();
  setServo(0);

  // Serial1 一定要在 Wire / analogWrite 之後才初始化：
  // D21 同時是 TX1 與 SCL1，先開 Serial1 再 Wire.begin() 時，TX1 會沒有輸出（Orin 收不到 STA）。
  Serial1.begin(BAUD);
  Serial.println("BMduino motor ready");
}

void loop() {
  readUart();
  unsigned long now = millis();

  // 失聯保護
  bool alive = everValid && (now - lastValidCmd <= FAILSAFE_MS);
  if (!alive && !motorsStopped) {
    stopMotors();
    if (DEBUG) Serial.println("FAILSAFE: no valid CMD, motors stopped");
  }
  digitalWrite(LED_BUILTIN, alive ? HIGH : LOW);

  static unsigned long tSta = 0;
  if (now - tSta >= STA_PERIOD_MS) {
    tSta = now;
    sendSTA(-1);
  }

  static unsigned long tDbg = 0;
  if (DEBUG && now - tDbg >= 500) {
    tDbg = now;
    Serial.print(alive ? "[RUN ] " : "[STOP] ");
    Serial.print("fwa="); Serial.print(curFwa);
    Serial.print(" bwl="); Serial.print(curL);
    Serial.print(" bwr="); Serial.print(curR);
    Serial.print("  ok="); Serial.print(okCount);
    Serial.print(" bad="); Serial.print(badCount);
    Serial.print(" sta="); Serial.println(staCount);
  }
}
