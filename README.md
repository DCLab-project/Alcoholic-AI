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

- 상태 `2`에서 OpenCV 추적이 시작되면 ingredient 모델의 top-1 결과를 투표합니다.
- 추적 중 bbox 중심이 화면 위/아래 가장자리 15% 구간을 벗어나 충분히 보이는 프레임만 투표에 포함합니다.
- 그중 top-1 confidence가 50%를 넘은 예측만 투표에 포함합니다.
- 추적이 끝나면 투표에 포함된 class 중 가장 많이 나온 class 1개만 `/api/v1/recognitions/ingredients`로 POST합니다.
- 투표에 포함된 예측이 하나도 없으면 POST하지 않습니다.
- 예: `tofu` 20회, `onion` 5회, `broccoli` 2회면 `tofu` 전송.

식재료 vote 세부 기준:

- 추적 범위는 물체가 보이기 시작한 시점부터 사라질 때까지입니다.
- 방향 판정은 전체 추적 궤적을 사용합니다.
- class vote는 전체 추적 궤적 중 물체가 충분히 화면 안쪽에 들어온 프레임만 사용합니다.
- `--ingredient-vote-visible-margin 0.15`는 화면 위/아래 15% 가장자리 구간을 vote에서 제외한다는 뜻입니다.
- `--ingredient-vote-min-confidence 0.50`은 top-1 confidence가 50%를 초과한 예측만 vote에 넣는다는 뜻입니다.
- `50.0%`는 초과가 아니므로 vote에 포함되지 않고, `50.1%`부터 포함됩니다.

식재료 이동 방향 표시:

- 상태 `2`에서만 OpenCV 움직임 추적을 켭니다.
- 화면 위쪽에서 나타나 아래쪽으로 사라지면 `input`을 표시합니다.
- 화면 아래쪽에서 나타나 위쪽으로 사라지면 `output`을 표시합니다.
- 같은 추적 세션을 기준으로 ingredient recognition POST도 확정합니다.

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
  --enable-ingredient-direction \
  --ingredient-vote-min-confidence 0.50 \
  --ingredient-vote-visible-margin 0.15
```

OpenCV 화면 없이 FE 웹 UI만 띄우고 싶으면 직접 실행할 때 `--headless`를 추가하면 됩니다.

## 튜닝 팁

식재료가 잘 보이는데도 아래 로그가 반복되면 vote에 포함된 예측이 없다는 뜻입니다.

```text
[VOTE] ingredient skipped: max_conf=0.0% <= required=50.0%
```

이 경우 보통 bbox visible 조건이 너무 좁거나, confidence 기준이 아직 높은 상황입니다.

- 더 많은 프레임을 vote에 포함하려면 `--ingredient-vote-visible-margin`을 낮춥니다.
  - 예: `0.15 -> 0.10`
- confidence 기준을 더 낮추려면 `--ingredient-vote-min-confidence`를 낮춥니다.
  - 예: `0.50 -> 0.45`
- 움직이는 물체가 너무 작아서 추적이 끊기면 `--direction-min-area`를 낮춥니다.
  - 예: `1200 -> 800`

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
