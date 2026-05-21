# Alcoholic-AI Jetson Main Controller

Arduino 센서값으로 Jetson 내부 AI 모드를 전환하고, 식재료/주류 인식 결과를 BE API로 전송하는 런타임 배포 브랜치입니다.

이 브랜치는 학습 코드 전체가 아니라 시연 실행에 필요한 파일만 포함합니다.

## 동작 구조

```text
Arduino + PIR + HC-SR04
-> Jetson main_controller.py
-> ingredient / liquor classifier
-> BE API
```

Arduino가 보내는 센서 코드는 Jetson 내부 모드 전환에만 사용합니다.

| Arduino code | Jetson mode | 의미 |
| --- | --- | --- |
| `0` | `Sleep` | 평상시, 인식/POST 없음 |
| `1` | `Alcohol` | PIR 감지, 주류 인식 |
| `2` | `Ingredient` | 초음파 앞 물체 사라짐, 식재료 인식 |

## POST 기준

센서 이벤트:

- `0 / Sleep`은 BE에 POST하지 않습니다.
- `1 / Alcohol`, `2 / Ingredient`는 상태가 바뀔 때만 `/api/v1/sensors/events`로 전송합니다.

주류 인식:

- 상태 `1` 진입 시 5초 동안 liquor 모델의 top-1 결과를 투표합니다.
- 5초 동안 가장 많이 나온 class 1개만 `/api/v1/recognitions/liquor`로 POST합니다.
- 예: `red_wine` 30회, `sake` 10회, `soju` 2회면 `red_wine` 전송.

식재료 인식:

- 상태 `2`에서 top confidence가 85% 이상인 첫 결과만 `/api/v1/recognitions/ingredients`로 POST합니다.
- 같은 상태2 세션에서 반복 POST하지 않습니다.

식재료 이동 방향 표시:

- 상태 `2`에서만 OpenCV 움직임 추적을 켭니다.
- 화면 위쪽에서 나타나 아래쪽으로 사라지면 `input`을 표시합니다.
- 화면 아래쪽에서 나타나 위쪽으로 사라지면 `output`을 표시합니다.
- 이 방향 표시는 BE POST와 별개로 CV 화면에만 표시됩니다.

라벨 보정:

- `leek`는 BE canonical key인 `green_onion`으로 전송합니다.
- `whiskey`는 `whisky`로 전송합니다.

## 포함된 모델

```text
checkpoints/ingredient/
  public_local_30class_mobilenet_v3_large_ep10_best.pth
  public_local_30class_mobilenet_v3_large_ep10_meta.json

checkpoints/liquor/
  public_local_7class_liquor_mobilenet_v3_large_ep5_best.pth
  public_local_7class_liquor_mobilenet_v3_large_ep5_meta.json
```

## 실행 준비

Python 패키지:

```bash
pip install -r requirements-runtime.txt
```

Jetson에서는 generic CUDA wheel을 무작정 설치하지 말고, Jetson 환경에 맞는 `torch`, `torchvision`, `cv2`를 먼저 준비해야 합니다.

간단 확인:

```bash
python3 -c "import torch, torchvision, cv2; print(torch.__version__)"
```

Arduino 펌웨어 업로드:

```bash
cd src/arduino
platformio run --target upload
```

Arduino serial 출력 형식:

```text
state_code,pir,distance_cm
```

예:

```text
1,1,18.50
2,0,30.00
```

## 실행

BE 기본 주소는 `http://192.168.50.123:8000`입니다.

```bash
./run_main_controller.sh
```

스크립트 내용:

```bash
python3 src/common/main_controller.py \
  --serial-port /dev/ttyACM0 \
  --camera 0 \
  --enable-be-post \
  --be-base-url http://192.168.50.123:8000 \
  --pir-hold-seconds 5 \
  --ingredient-post-threshold 0.85 \
  --enable-ingredient-direction
```

OpenCV 화면 없이 FE 웹 UI만 띄우고 싶으면 직접 실행할 때 `--headless`를 추가하면 됩니다.

## 주요 파일

```text
run_main_controller.sh
src/common/main_controller.py
src/common/ingredient_models.py
src/common/project_paths.py
src/arduino/platformio.ini
src/arduino/src/main.cpp
checkpoints/
```

## BE API

센서 이벤트:

```http
POST /api/v1/sensors/events
```

인식 결과:

```http
POST /api/v1/recognitions/ingredients
POST /api/v1/recognitions/liquor
```

기본 source:

- `jetson-ingredient-classifier`
- `jetson-liquor-classifier`
- `arduino`
