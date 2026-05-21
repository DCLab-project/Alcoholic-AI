#include <Arduino.h>

constexpr uint8_t PIR_PIN  = 2;
constexpr uint8_t TRIG_PIN = 9;
constexpr uint8_t ECHO_PIN = 10;

constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long PRINT_INTERVAL_MS = 200;

// 거리 기준값
// CLOSE_CM 이하이면 "물체가 가까이 있다"
// FAR_CM 이상이면 "물체가 사라졌다 / 멀어졌다"
constexpr float CLOSE_CM = 15.0f;
constexpr float FAR_CM   = 25.0f;

// HC-SR04 timeout
// 30000us는 대략 5m 정도까지 측정 가능
constexpr unsigned long PULSE_TIMEOUT_US = 30000;

// 초음파 값 튐 방지를 위한 샘플 개수
constexpr int SAMPLE_COUNT = 5;

// true:
// 물체가 다시 가까이 오면 PIR이 아직 HIGH여도 일단 0으로 리셋.
// PIR이 LOW로 내려간 뒤 다시 감지되면 1 출력.
//
// false:
// 물체가 가까이 돌아와도 PIR이 HIGH면 바로 1 출력.
constexpr bool RESET_TO_ZERO_ON_OBJECT_RETURN = true;

bool objectClose = true;       // 처음에는 물체가 가까이 있다고 가정
bool ignorePirUntilLow = false;

float readDistanceCmOnce() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, PULSE_TIMEOUT_US);

  if (duration == 0) {
    // Echo가 안 들어오면 물체가 없거나 너무 멀다고 판단
    return 999.0f;
  }

  // HC-SR04 거리 공식: cm = duration / 58.0
  return duration / 58.0f;
}

float readDistanceCmAverage() {
  float sum = 0.0f;
  int valid = 0;

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    float d = readDistanceCmOnce();

    if (d > 0.0f && d < 500.0f) {
      sum += d;
      valid++;
    }

    delay(10);
  }

  if (valid == 0) {
    return 999.0f;
  }

  return sum / valid;
}

void setup() {
  pinMode(PIR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  digitalWrite(TRIG_PIN, LOW);

  Serial.begin(SERIAL_BAUD);

  // PIR 센서는 전원 인가 후 안정화 시간이 필요할 수 있음.
  // 너무 길게 기다리기 싫으면 1000 정도로 줄여도 됨.
  delay(2000);
}

void loop() {
  float distanceCm = readDistanceCmAverage();
  int pirDetected = digitalRead(PIR_PIN);

  bool wasClose = objectClose;

  // 히스테리시스 적용
  // 가까운 상태에서 FAR_CM 이상이면 "사라짐"
  if (objectClose && distanceCm >= FAR_CM) {
    objectClose = false;
  }

  // 사라진 상태에서 CLOSE_CM 이하이면 "다시 가까워짐"
  if (!objectClose && distanceCm <= CLOSE_CM) {
    objectClose = true;
  }

  // 물체가 다시 가까이 온 순간 0으로 리셋하고 싶을 때
  if (RESET_TO_ZERO_ON_OBJECT_RETURN && !wasClose && objectClose) {
    ignorePirUntilLow = true;
  }

  int outputCode = 0;

  if (!objectClose) {
    // 초음파 앞 물체가 사라지면 2가 최우선
    outputCode = 2;
  } else {
    if (ignorePirUntilLow) {
      outputCode = 0;

      if (pirDetected == LOW) {
        ignorePirUntilLow = false;
      }
    } else if (pirDetected == HIGH) {
      outputCode = 1;
    } else {
      outputCode = 0;
    }
  }

  Serial.print(outputCode);
  Serial.print(',');
  Serial.print(pirDetected == HIGH ? 1 : 0);
  Serial.print(',');
  Serial.println(distanceCm, 2);

  delay(PRINT_INTERVAL_MS);
}
