/*
 * AMR BLE 遙控韌體 — BMC77M001 + 雙驅動馬達 + 轉向舵機        [ v2.0 ]
 * ==================================================================
 * 版本紀錄
 *   v2.0  D11 改為 INPUT_PULLUP 讀取緊急開關，按下立刻進入失效保護。
 *         含 25ms 去彈跳與鎖定：按著期間移動封包一律忽略，放開後
 *         還要收到一個油門中立的封包才解除，避免鬆手瞬間暴衝
 *   v1.9  (作廢，不要照做) D11 曾設成輸出高電位當訊號源。按鈕另一端
 *         接 GND，按下去等於輸出腳直接對地短路，會燒腳位
 *   v1.8  轉向行程改為左右獨立的絕對上下限 1100-1920us。
 *         裝上連桿後兩側干涉點本來就不對稱，滿舵各自停在自己那一側
 *   v1.7  機構零點微調為 1520us（行程 980-2060us）
 *   v1.6  機構零點改為 1500us（行程 960-2040us）
 *   v1.5  機構零點定案 1540us（原 1400us 是舵機空載中位）。
 *         行程變成 1000-2080us，仍在舵機驗證範圍 700-2100us 內
 *   v1.4  中位微調步進放大：, . 為 20us（約1.7度），< > 為 100us（約8.3度）。
 *         原本 5us 只有 0.4 度，肉眼看不出來
 *   v1.3  未知按鍵改為印出提示，不再靜默忽略
 *   v1.2  新增 STEER_INVERT（左右相反，已設 1 修正）與中位微調 steerTrimUs，
 *         轉向改用「轉向語意」的偏移量表示，極性只在 steerApply() 翻一次
 *   v1.1  轉向行程改用角度設定 SERVO_MAX_DEG，目前設 +-45 度（860-1940us）
 *   v1.0  BLE 遙控首版
 *
 * 手機 App（掃描名稱 BMC_CAR）送 3-byte 移動封包，本韌體解碼後
 * 驅動兩顆驅動馬達與轉向舵機，並在斷線／逾時時自動煞車回正。
 *
 * 協定與失效保護沿用 Wai-0424/501b-ai-car-ble-control 的設計，
 * 但機構參數全部換成本車實測值（見下方「實測校正值」）。
 * ==================================================================
 * 接線
 *   BLE 模組 BMC77M001 -> BMduino SPI1 排針（Serial3）
 *     SPI1 VDD3 (跳帽設 5V) -- 模組 VDD   (pin4)
 *     SPI1 GND              -- 模組 GND   (pin5)
 *     SPI1 D27 / TX3        -- 模組 RX    (pin2)   <- 交叉
 *     SPI1 D28 / RX3        -- 模組 TX    (pin3)   <- 交叉
 *     SPI1 D26 / D29        空接；模組 pin1 NC 空接
 *     !! 模組的 BMCOM 接頭腳序（NC/RX/TX/VDD/GND）與 SPI1 排針不同，
 *        不可拿排線直接插，必須一根一根拉
 *     模組規格：工作電壓 3.3-5.5V，UART 邏輯跟著 VDD，預設 115200
 *
 *   BMduino D8 -> 74LS14 兩級反相 -> 驅動板 ENA   (馬達1 速度)
 *   BMduino D9 -> 74LS14 兩級反相 -> 驅動板 ENB   (馬達2 速度)
 *     緩衝器不能省：驅動板實測約需 3.8V 才是高電位，3.3V 直連推不動。
 *     但 74LS14 的 V_OH 典型僅 3.4V，踩在門檻上沒餘裕 -> 建議換 74HCT14
 *
 *   PCA9685 PWM4 -> IN1   PWM5 -> IN2      (馬達1 方向)
 *   PCA9685 PWM6 -> IN3   PWM7 -> IN4      (馬達2 方向)
 *   PCA9685 CH0  -> 轉向舵機訊號
 *   PCA9685 走 Wire1 (BMCOM1，跳帽 5V)，OE 必須接 GND
 *
 *   BMduino D11 -> 緊急開關 -> GND     (常開按鈕，按下導通)
 *     D11 設成 INPUT_PULLUP，高電位由內部上拉提供：
 *       沒按 = 開路 = HIGH      按下 = 接到 GND = LOW
 *     !! 這支腳絕對不能設成 OUTPUT 再輸出 HIGH —— 按鈕另一端是
 *        GND，按下去就是輸出級直接對地短路，按一次就可能燒掉
 *     按鈕不需要任何外部零件，兩條線直接拉即可
 *
 *   BMduino 5V/GND -> 驅動板邏輯電源
 *   12V 外部 -> VM（建議串 15A 保險絲）
 *   !! 舵機 V+ 走外部 6V，負極與 PCA9685 共地。
 *      接 BMduino 5V 會在堵轉時拖垮整條邏輯電源
 * ==================================================================
 * 實測校正值（不要換成別的專案的數字，兩者尺標差一倍以上）
 *   PCA9685 振盪器  25.88 MHz（標稱 25MHz，未校正會有 3.5% 脈寬誤差）
 *   舵機本體中位    1400 us（空載、拆掉舵臂時量的）
 *   機構零點        1520 us（裝上連桿後輪子打直的位置）
 *   舵機驗證極限    700 - 2100 us = +-60 度（絕對極限 500-2300us）
 *   轉向行程        1100 - 1920 us（左右不對稱：右 35 度、左 33 度）
 *   舵機斜率        約 12 us/度，線性無死區，左右對稱
 *   轉向極性        STEER_INVERT=0（搖桿與實際轉向相反，已修正）
 *   中位微調        拆過舵臂就要重調，用 < > , . 調到輪子打直
 *   馬達極性        mA.invert = true、mB.invert = false 時兩顆同向前進
 * ==================================================================
 * 移動封包協定（App -> 韌體，固定 3 bytes）
 *   byte0  'M' (0x4D)   封包標頭
 *   byte1  throttle     128=停止，>128 前進，<128 後退，離 128 愈遠愈快
 *   byte2  steer        128=置中，>128 右轉，<128 左轉
 *   App 以約 20Hz 持續發送（放開搖桿也送 128/128），封包本身兼作心跳
 *
 * 失效保護
 *   BLE 斷線事件            -> 立刻煞車 + 轉向回正
 *   逾時 COMMAND_TIMEOUT_MS -> 連線還在但沒收到封包，同樣煞車 + 回正
 *   緊急開關按下            -> 立刻煞車 + 回正，並鎖定遙控；放開後要
 *                              油門回中立才解除（見 estopLatch）
 *   死區 +-6                -> 手指沒完全放開時不會微幅前進或飄轉向
 * ==================================================================
 * 序列埠指令（台架用，不需要手機）
 *   ?  狀態      0 或空白鍵  急停
 *   E  切換封包回音（回音會讓 BLE 流量加倍，預設關閉）
 *   1  切換馬達1 極性    2  切換馬達2 極性
 *   , . 中位微調 20us    < > 粗調 100us    /  歸零
 * ==================================================================
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "BM7701-00-1.h"

#define FW_VERSION      "v2.0"

// ---- 緊急開關 ----------------------------------------------------------
// 常開按鈕，一端 D11 一端 GND。靠內部上拉提供高電位，
// 所以「沒按 = HIGH、按下 = LOW」。不可設成 OUTPUT（會對地短路）
#define ESTOP_PIN         11
#define ESTOP_ACTIVE_LOW  1     // 常開接地按鈕 = 1；若改用常閉按鈕改成 0
#define ESTOP_DEBOUNCE_MS 25    // 機械接點彈跳通常在 10ms 內

bool          estopPressed = false;  // 去彈跳後的按鈕實體狀態
bool          estopLatch   = false;  // 鎖定中：移動封包一律忽略
bool          estopRaw     = false;  // 最近一次原始讀值
unsigned long estopEdgeMs  = 0;      // 原始讀值上次變化的時間

// ---- PCA9685 -------------------------------------------------------------
#define PCA_ADDR        0x40
#define PCA_OSC_HZ      25880000UL   // 實測校正值
#define PCA_PWM_HZ      50

Adafruit_PWMServoDriver pwm(PCA_ADDR, Wire);
uint8_t pcaPrescale = 0;

// ---- BLE -----------------------------------------------------------------
BM7701_00_1 BC7701(&Serial3);

#define TX_POWER     0x0F
#define XTAL_CLOAD   0x04
#define ADV_MIN      100
#define ADV_MAX      100
#define CON_MIN      30
#define CON_MAX      30
#define CON_LATENCY  0
#define CON_TIMEOUT  300

uint8_t BDAddress[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
uint8_t BDName[]     = {'B', 'M', 'C', '_', 'C', 'A', 'R'};  // App 用這個字串比對
uint8_t Adata[]      = {0x02, 0x01, 0x06};
uint8_t Sdata[]      = {0x03, 0x02, 0x0f, 0x18};

bool     bleInitOK   = false;
bool     bleConnect  = false;
bool     bleConnIntv = false;
uint8_t  rxBuf[256]  = {0};
uint8_t  rxLen       = 0;
bool     echoBack    = false;      // 回音會讓 BLE 流量加倍，預設關

// ---- 移動封包 ------------------------------------------------------------
#define CMD_MOVE            'M'
#define THROTTLE_DEADZONE   6
#define STEER_DEADZONE      6
#define COMMAND_TIMEOUT_MS  300

unsigned long lastCommandMs   = 0;
bool          failsafeActive  = true;   // 開機即為安全狀態

// ---- 轉向舵機 ------------------------------------------------------------
#define SERVO_CH        0
// 機構零點：裝上舵臂與連桿後，輪子打直對應的脈寬。
// 注意這跟舵機本體的空載中位 1400us 是兩回事 —— 這個值由舵臂
// 鎖上去的角度決定，只要拆過舵臂就要重新校，用 < > , . 重調
#define SERVO_US_CTR    1520
#define SERVO_US_PER_DEG 12.0f       // 實測約 12us/度

// 轉向行程上下限（原始 us）。左右不對稱 —— 裝上連桿後兩側的干涉點
// 本來就不會對稱，所以直接給絕對值，不再用「中位 +- 對稱範圍」。
//   以中位 1520us 計：右滿舵 1100us（-420us 約 35 度）
//                     左滿舵 1920us（+400us 約 33 度）
// 舵機本身驗證極限 700-2100us，這裡收在裡面，留機構餘裕
#define SERVO_US_MIN    1100
#define SERVO_US_MAX    1920

uint16_t servoUs = SERVO_US_CTR;

// 轉向極性：搖桿/按鍵方向與實際轉向相反時設 1
#define STEER_INVERT    0

// 中位微調 (us)，正值代表中位往「右」移（轉向語意，不是原始 us）。
// 裝上舵臂與連桿之後，機構零點幾乎一定會偏離空載量到的 1400us。
// 用 , 和 . 即時微調，調到輪子打直之後把數字告訴我，再寫回 SERVO_US_CTR
#define STEER_TRIM_FINE   20    // , . 微調，約 1.7 度
#define STEER_TRIM_COARSE 100   // < > 粗調，約 8.3 度
int16_t steerTrimUs = 0;
// centerUs() 定義在下面「舵機底層」—— .ino 的自動原型會插在第一個函式
// 定義之前，若把函式放在 struct Motor 之上，Motor 的原型就會早於宣告


// ---- 驅動馬達 ------------------------------------------------------------
#define RAMP_STEP       5            // 每個斜坡週期變化多少 PWM
#define RAMP_MS         5            // 斜坡週期，0->255 約 255ms

struct Motor {
  uint8_t     enPin;
  uint8_t     chA, chB;              // PCA9685 上的兩支方向通道
  bool        invert;                // true = 這顆馬達的正反轉對調
  int8_t      dir;                   // 目前實際方向 -1 / 0 / 1
  int8_t      wantDir;               // 想去的方向，換向時先降速再套用
  int         speed;                 // 目前 PWM
  int         target;                // 斜坡目標 PWM
  int         pendingSpeed;          // 換向完成後要回到的速度
  const char *name;
};

// 台架實測：這組極性讓兩顆同向前進。裝上車體變成鏡像安裝後，
// 左右兩顆的 invert 會相反，用 1 / 2 鍵可即時切換
Motor mA = {8, 4, 5, true,  0, 0, 0, 0, 0, "馬達1"};
Motor mB = {9, 6, 7, false, 0, 0, 0, 0, 0, "馬達2"};

unsigned long rampTickMs = 0;
unsigned long hbMs       = 0;
unsigned long lastLogMs  = 0;

// ---- 舵機底層 ------------------------------------------------------------
// 自己換算 count，避免 writeMicroseconds 每次都去讀一次 prescale
uint16_t usToCount(uint16_t us) {
  return (uint16_t)((float)us * (float)PCA_OSC_HZ
                    / (1000000.0f * (float)(pcaPrescale + 1)) + 0.5f);
}

// 實際輸出的中位 us（把轉向語意的 trim 換算回原始 us）
uint16_t centerUs() {
  int16_t t = STEER_INVERT ? -steerTrimUs : steerTrimUs;
  return (uint16_t)((int)SERVO_US_CTR + t);
}

// 左右各自還有多少行程可用（轉向語意）。中位被 trim 移動時會跟著變
uint16_t steerRangeRight() {
  uint16_t c = centerUs();
  return STEER_INVERT ? (c - SERVO_US_MIN) : (SERVO_US_MAX - c);
}
uint16_t steerRangeLeft() {
  uint16_t c = centerUs();
  return STEER_INVERT ? (SERVO_US_MAX - c) : (c - SERVO_US_MIN);
}

// deflectUs 是「轉向語意」的偏移量：正 = 右轉、負 = 左轉。
// 極性只在這裡翻一次，上層完全不用管舵機實際往哪邊轉
void steerApply(int deflectUs) {
  int d = STEER_INVERT ? -deflectUs : deflectUs;
  long us = (long)centerUs() + d;
  if (us < SERVO_US_MIN) us = SERVO_US_MIN;
  if (us > SERVO_US_MAX) us = SERVO_US_MAX;
  servoUs = (uint16_t)us;
  pwm.setPWM(SERVO_CH, 0, usToCount(servoUs));
}

// 回報「轉向語意」的角度，正 = 右轉
float steerDeg() {
  int d = (int)servoUs - (int)centerUs();
  if (STEER_INVERT) d = -d;
  return d / SERVO_US_PER_DEG;
}

// ---- 馬達底層 ------------------------------------------------------------
// setPin 對 0 和 4095 有全關/全開的特例處理，方向腳要的就是這個
void applyDir(Motor &m, int8_t dir) {
  if (m.invert) dir = -dir;          // 只在這裡翻極性，m.dir 仍記錄邏輯方向

  if (dir > 0) {                     // H L 正轉
    pwm.setPin(m.chA, 4095);
    pwm.setPin(m.chB, 0);
  } else if (dir < 0) {              // L H 反轉
    pwm.setPin(m.chA, 0);
    pwm.setPin(m.chB, 4095);
  } else {                           // L L 煞車
    pwm.setPin(m.chA, 0);
    pwm.setPin(m.chB, 0);
  }
}

// 若正往反方向轉，先把目標設 0，等斜坡降到 0 之後才翻方向。
// 全速下直接反轉（plugging）電流會接近兩倍堵轉值
void motorCommand(Motor &m, int8_t dir, int speed) {
  m.wantDir      = dir;
  m.pendingSpeed = speed;

  if (dir == 0) { m.target = 0; return; }
  if (m.dir == dir) { m.target = speed; return; }

  if (m.speed == 0) {
    m.dir = dir;
    applyDir(m, dir);
    m.target = speed;
  } else {
    m.target = 0;
  }
}

void driveCommand(int8_t dir, int speed) {
  motorCommand(mA, dir, speed);
  motorCommand(mB, dir, speed);
}

void updateRamp(Motor &m) {
  if (m.speed != m.target) {
    if (m.speed < m.target) {
      m.speed += RAMP_STEP;
      if (m.speed > m.target) m.speed = m.target;
    } else {
      m.speed -= RAMP_STEP;
      if (m.speed < m.target) m.speed = m.target;
    }
    analogWrite(m.enPin, m.speed);
    return;
  }

  if (m.speed == 0 && m.dir != m.wantDir) {
    m.dir = m.wantDir;
    applyDir(m, m.dir);
    if (m.dir != 0) m.target = m.pendingSpeed;
  }
}

// 立刻停止，不走斜坡 —— 失效保護要的是馬上停，不是慢慢停
void hardStop(Motor &m) {
  analogWrite(m.enPin, 0);
  m.speed = 0; m.target = 0; m.dir = 0; m.wantDir = 0; m.pendingSpeed = 0;
  applyDir(m, 0);
}

void enterFailsafe(const __FlashStringHelper *why) {
  hardStop(mA);
  hardStop(mB);
  steerApply(0);
  failsafeActive = true;
  Serial.print(F("!! 失效保護：煞車 + 轉向回正 — "));
  Serial.println(why);
}

// ---- 緊急開關 ------------------------------------------------------------
// 去彈跳後才認定狀態改變。按下的瞬間就進失效保護，並且鎖住遙控 ——
// 光是進失效保護不夠，手機還在 20Hz 送封包，下一個封包就會把它解除
void updateEstop() {
  bool raw = (digitalRead(ESTOP_PIN) == (ESTOP_ACTIVE_LOW ? LOW : HIGH));
  unsigned long now = millis();

  if (raw != estopRaw) {                          // 剛變化，重新計時
    estopRaw    = raw;
    estopEdgeMs = now;
    return;
  }
  if (now - estopEdgeMs < ESTOP_DEBOUNCE_MS) return;   // 還沒穩定
  if (raw == estopPressed) return;                     // 狀態沒變

  estopPressed = raw;
  if (estopPressed) {
    estopLatch = true;
    enterFailsafe(F("緊急開關按下"));
  } else {
    Serial.println(F(">>> 緊急開關已放開 — 油門回中立後才恢復遙控"));
  }
}

// ---- 封包處理 ------------------------------------------------------------
void handleMovePacket(uint8_t throttleByte, uint8_t steerByte) {
  int t = (int)throttleByte - 128;

  // 急停鎖定中就什麼都不做。按著的時候永遠鎖住；放開之後還要先看到
  // 一個油門中立的封包才解除 —— 否則鬆手當下搖桿若還推著，車子會
  // 立刻竄出去，這正是急停最不該發生的事
  if (estopLatch) {
    if (!estopPressed && abs(t) <= THROTTLE_DEADZONE) {
      estopLatch = false;
      Serial.println(F(">>> 急停解除，恢復遙控"));
    }
    lastCommandMs = millis();
    return;
  }

  if (abs(t) <= THROTTLE_DEADZONE) {
    driveCommand(0, 0);
  } else {
    int speed = constrain(abs(t) * 255 / 127, 0, 255);
    driveCommand(t > 0 ? 1 : -1, speed);
  }

  int s = (int)steerByte - 128;
  if (abs(s) <= STEER_DEADZONE) {
    steerApply(0);
  } else {
    // 左右行程不對稱，各自用自己那一側的範圍換算，
    // 這樣兩邊滿舵都剛好停在 SERVO_US_MIN / SERVO_US_MAX
    long range = (s > 0) ? steerRangeRight() : steerRangeLeft();
    steerApply((int)((long)s * range / 127));
  }

  lastCommandMs  = millis();
  failsafeActive = false;
}

// ---- 狀態 ----------------------------------------------------------------
const __FlashStringHelper *dirText(int8_t d) {
  if (d > 0) return F("前進");
  if (d < 0) return F("後退");
  return F("停止");
}

void printStatus() {
  Serial.println(F("---------------- 狀態 ----------------"));
  Serial.print(F("BLE    init="));
  Serial.print(bleInitOK ? F("OK") : F("FAIL"));
  Serial.print(F("  連線="));
  Serial.print(bleConnect ? F("已連") : F("未連"));
  Serial.print(F("  失效保護="));
  Serial.println(failsafeActive ? F("作用中") : F("解除"));
  Serial.print(F("急停   按鈕="));
  Serial.print(estopPressed ? F("按下") : F("放開"));
  Serial.print(F("  鎖定="));
  Serial.println(estopLatch ? F("是") : F("否"));
  Serial.print(F("馬達1  "));
  Serial.print(dirText(mA.dir));
  Serial.print(F("  PWM "));
  Serial.print(mA.speed);
  Serial.print(F(" -> "));
  Serial.print(mA.target);
  Serial.print(F("   極性 "));
  Serial.println(mA.invert ? F("反轉") : F("正常"));
  Serial.print(F("馬達2  "));
  Serial.print(dirText(mB.dir));
  Serial.print(F("  PWM "));
  Serial.print(mB.speed);
  Serial.print(F(" -> "));
  Serial.print(mB.target);
  Serial.print(F("   極性 "));
  Serial.println(mB.invert ? F("反轉") : F("正常"));
  Serial.print(F("轉向   "));
  Serial.print(servoUs);
  Serial.print(F(" us  ("));
  Serial.print(steerDeg(), 1);
  Serial.println(F(" 度，正=右轉)"));
  Serial.print(F("中位微調 "));
  Serial.print(steerTrimUs);
  Serial.print(F(" us -> 實際中位 "));
  Serial.print(centerUs());
  Serial.println(F(" us"));
  Serial.print(F("封包回音 "));
  Serial.println(echoBack ? F("開") : F("關"));
  Serial.println(F("--------------------------------------"));
}

// 微調中位並立刻套用，方便一邊按一邊看輪子
void nudgeTrim(int16_t delta) {
  steerTrimUs += delta;
  steerApply(0);
  Serial.print(F(">>> 中位微調 "));
  Serial.print(steerTrimUs);
  Serial.print(F(" us -> 實際中位 "));
  Serial.print(centerUs());
  Serial.println(F(" us"));
}

void toggleInvert(Motor &m) {
  hardStop(m);
  m.invert = !m.invert;
  applyDir(m, 0);
  Serial.print(F(">>> "));
  Serial.print(m.name);
  Serial.print(F(" 極性 "));
  Serial.println(m.invert ? F("反轉") : F("正常"));
}

// ---- BLE 初始化 ----------------------------------------------------------
// 依序設定並開始廣播，任一步失敗就中止並印出是哪一步
bool bleInit() {
  BC7701.begin(BAUD_115200);
  Serial.println(F("BLE 初始化中 ..."));

  uint8_t step = 1;
  while (step != 10) {
    switch (step) {
      case 1: if (BC7701.setAddress(BDAddress)) step++;
              else { Serial.println(F("1 setAddress 失敗")); return false; } break;
      case 2: if (BC7701.setName(sizeof(BDName), BDName)) step++;
              else { Serial.println(F("2 setName 失敗")); return false; } break;
      case 3: if (BC7701.setAdvIntv(ADV_MIN / 0.625, ADV_MAX / 0.625, 7)) step++;
              else { Serial.println(F("3 setAdvIntv 失敗")); return false; } break;
      case 4: if (BC7701.setAdvData(APPEND_NAME, sizeof(Adata), Adata)) step++;
              else { Serial.println(F("4 setAdvData 失敗")); return false; } break;
      case 5: if (BC7701.setScanData(sizeof(Sdata), Sdata)) step++;
              else { Serial.println(F("5 setScanData 失敗")); return false; } break;
      case 6: if (BC7701.setTXpower(TX_POWER)) step++;
              else { Serial.println(F("6 setTXpower 失敗")); return false; } break;
      case 7: if (BC7701.setCrystalOffset(XTAL_CLOAD)) step++;
              else { Serial.println(F("7 setCrystalOffset 失敗")); return false; } break;
      case 8: if (BC7701.setFeature(FEATURE_DIR, AUTO_SEND_SATUS)) step++;
              else { Serial.println(F("8 setFeature 失敗")); return false; } break;
      case 9: if (BC7701.setAdvCtrl(ENABLE)) step++;
              else { Serial.println(F("9 setAdvCtrl 失敗")); return false; } break;
    }
  }
  return true;
}

// ---- 主程式 --------------------------------------------------------------
void setup() {
  delay(60);
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { }

  // 緊急開關最早設定。開機當下就按著的話直接以鎖定狀態啟動，
  // 這裡只設旗標不呼叫 enterFailsafe() —— PCA9685 還沒 begin()
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  delay(2);                                  // 等內部上拉把線拉穩
  estopPressed = (digitalRead(ESTOP_PIN) == (ESTOP_ACTIVE_LOW ? LOW : HIGH));
  estopRaw     = estopPressed;
  estopLatch   = estopPressed;
  estopEdgeMs  = millis();

  // 先把致能腳壓到 0 —— 開機瞬間不要讓馬達動
  pinMode(mA.enPin, OUTPUT);
  pinMode(mB.enPin, OUTPUT);
  analogWrite(mA.enPin, 0);
  analogWrite(mB.enPin, 0);

  pwm.begin();
  pwm.setOscillatorFrequency(PCA_OSC_HZ);   // 必須在 setPWMFreq 之前
  pwm.setPWMFreq(PCA_PWM_HZ);
  delay(10);
  pcaPrescale = pwm.readPrescale();

  applyDir(mA, 0);
  applyDir(mB, 0);
  steerApply(0);
  lastCommandMs  = millis();
  failsafeActive = true;

  Serial.println();
  Serial.print(F("AMR BLE 遙控韌體  "));
  Serial.println(F(FW_VERSION));
  Serial.print(F("PCA9685 振盪器 "));
  Serial.print(PCA_OSC_HZ / 1000000.0f, 2);
  Serial.print(F(" MHz  prescale "));
  Serial.println(pcaPrescale);
  Serial.println(F("馬達已煞車、轉向已回正"));
  Serial.print(F("緊急開關 D"));
  Serial.print(ESTOP_PIN);
  Serial.print(F(" (INPUT_PULLUP) 目前 "));
  Serial.println(estopPressed ? F("按下 — 已鎖定，放開後才能遙控")
                              : F("放開"));

  bleInitOK = bleInit();
  delay(650);

  if (bleInitOK) {
    Serial.println(F("BLE 就緒，廣播名稱 BMC_CAR，等待手機連線"));
  } else {
    Serial.println(F("BLE 初始化失敗 — 檢查 SPI1 接線（TX/RX 是否交叉）、"));
    Serial.println(F("                  跳帽電壓、模組供電"));
  }
  Serial.println(F("序列埠指令：? 狀態  0/空白 急停  E 回音  1/2 馬達極性"));
  rampTickMs = millis();
}

void loop() {
  unsigned long now = millis();

  updateEstop();

  // 馬達斜坡
  if (now - rampTickMs >= RAMP_MS) {
    rampTickMs = now;
    updateRamp(mA);
    updateRamp(mB);
  }

  // 看門狗：連著線但封包斷了，一樣要停
  if (bleConnect && !failsafeActive && now - lastCommandMs > COMMAND_TIMEOUT_MS) {
    enterFailsafe(F("逾時未收到移動封包"));
  }

  // 心跳，2 秒一次，別洗版
  if (now - hbMs >= 2000) {
    hbMs = now;
    Serial.print(F("[hb] BLE="));
    Serial.print(bleConnect ? F("已連") : F("未連"));
    Serial.print(F("  馬達 "));
    Serial.print(mA.speed);
    Serial.print(F("/"));
    Serial.print(mB.speed);
    Serial.print(F("  轉向 "));
    Serial.print(servoUs);
    Serial.println(F(" us"));
  }

  // BLE 事件
  uint8_t status = BC7701.receiveData(rxBuf, rxLen);
  if (status) {
    switch (status) {
      case API_CONNECTED:
        if (!bleConnect) {
          bleConnect    = true;
          lastCommandMs = now;
          Serial.println(F(">>> BLE 已連線"));
          if (!bleConnIntv) {
            BC7701.wakeUp();
            delay(30);
            if (BC7701.setConnIntv(CON_MIN / 1.25, CON_MAX / 1.25,
                                   CON_LATENCY, CON_TIMEOUT)) {
              bleConnIntv = true;
            }
          }
        }
        break;

      case API_DISCONNECTED:
        bleConnect  = false;
        bleConnIntv = false;
        enterFailsafe(F("BLE 斷線"));
        break;

      case DATA_RECEIVED:
        if (rxLen == 3 && rxBuf[0] == CMD_MOVE) {
          handleMovePacket(rxBuf[1], rxBuf[2]);
          // 20Hz 全印會洗版，限制成每 250ms 印一次
          if (now - lastLogMs >= 250) {
            lastLogMs = now;
            Serial.print(F("move  throttle="));
            Serial.print(rxBuf[1]);
            Serial.print(F("  steer="));
            Serial.print(rxBuf[2]);
            Serial.print(F("  -> "));
            Serial.print(servoUs);
            Serial.println(F(" us"));
          }
        } else {
          Serial.print(F("未知封包，長度 "));
          Serial.println(rxLen);
        }
        if (echoBack) BC7701.sendData(rxBuf, rxLen);
        break;

      case API_ERROR:
        break;
    }
  }

  // 序列埠指令（台架用）
  if (!Serial.available()) return;
  char cmd = Serial.read();
  if (cmd == '\n' || cmd == '\r') return;

  if (cmd == '0' || cmd == ' ') { enterFailsafe(F("手動急停")); return; }

  switch (toupper(cmd)) {
    case '?': printStatus(); break;
    case 'E':
      echoBack = !echoBack;
      Serial.print(F(">>> 封包回音 "));
      Serial.println(echoBack ? F("開") : F("關"));
      break;
    case '1': toggleInvert(mA); break;
    case '2': toggleInvert(mB); break;

    case ',': nudgeTrim(-STEER_TRIM_FINE);   break;
    case '.': nudgeTrim(+STEER_TRIM_FINE);   break;
    case '<': nudgeTrim(-STEER_TRIM_COARSE); break;
    case '>': nudgeTrim(+STEER_TRIM_COARSE); break;
    case '/': steerTrimUs = 0; nudgeTrim(0); break;

    default:
      // 靜默忽略會讓人分不出「按錯鍵」和「程式當掉」
      Serial.print(F("未知指令 '"));
      Serial.print(cmd);
      Serial.println(F("'  可用：? 0/空白 E 1 2 , . < > /"));
      Serial.println(F("  （轉向靠手機搖桿，A/D 是 amr_drive_test 才有的）"));
      break;
  }
}
