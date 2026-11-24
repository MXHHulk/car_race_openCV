// BMduino UART 通訊測試韌體（依「UART 通訊協議書」）
// Serial1 (D20 RX1 / D21 TX1) 接 Orin 40-pin UART；Serial (USB) 用於序列埠監控視窗除錯。
// 收到有效 CMD（checksum 正確）就回 <STA,S1 -1,cs XX>\n，並閃 LED。不驅動任何馬達。

const long BAUD = 115200;   // 要與 Orin 端 uart_test.py -b 相同
const int MAX_LEN = 64;

char buf[MAX_LEN + 1];
int len = -1;               // -1 = 等待 '<'
unsigned long okCount = 0, badCount = 0;

uint8_t xorChecksum(const char* s, size_t n) {
  uint8_t cs = 0;
  for (size_t i = 0; i < n; i++) cs ^= (uint8_t)s[i];
  return cs;
}

bool verify(const char* body) {
  const char* p = strstr(body, ",cs ");
  if (!p) return false;
  const char* last;
  while ((last = strstr(p + 1, ",cs ")) != NULL) p = last;  // 取最後一個
  if (strlen(p + 4) != 2) return false;
  uint8_t got = (uint8_t)strtol(p + 4, NULL, 16);
  return got == xorChecksum(body, p - body);
}

void sendSTA(int s1) {
  char payload[32], pkt[48];
  snprintf(payload, sizeof(payload), "STA,S1 %d", s1);
  snprintf(pkt, sizeof(pkt), "<%s,cs %02X>\n", payload, xorChecksum(payload, strlen(payload)));
  Serial1.print(pkt);       // 用 print，不用 println（避免 CRLF）
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(BAUD);
  Serial.println("BMduino UART test ready");
}

void loop() {
  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == '<') { len = 0; continue; }
    if (len < 0) continue;
    if (c == '>') {
      buf[len] = '\0';
      len = -1;
      if (strncmp(buf, "CMD,", 4) == 0 && verify(buf)) {
        okCount++;
        sendSTA(-1);
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      } else {
        badCount++;
      }
      Serial.print("RX <"); Serial.print(buf); Serial.print(">  ok=");
      Serial.print(okCount); Serial.print(" bad="); Serial.println(badCount);
      continue;
    }
    if (len >= MAX_LEN) { len = -1; continue; }  // 超長丟棄
    buf[len++] = c;
  }
}
