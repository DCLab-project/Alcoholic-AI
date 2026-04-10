# Alcoholic-AI

## 1. 저장소 목적

이 저장소는 **술안주 추천 AI 냉장고 프로젝트의 AI 전용 저장소**입니다.  
현재 이 브랜치에는 기존 개인 실험 레포에서 검증한 **식재료 인식 베이스라인**이 이식되어 있으며, 이후 같은 형식으로 **주류 인식 모델**, **배포 최적화**, **Jetson 추론 자산**까지 확장하는 것을 목표로 합니다.

이 저장소가 담당하는 범위는 다음과 같습니다.

- 식재료 인식 모델 학습 및 평가
- 주류 카테고리 인식 모델 학습 및 평가
- 데이터셋 정리 및 라벨 관리
- Jetson 배포용 추론 스크립트 관리
- 추후 ONNX / TensorRT export 및 최적화

---

## 2. 현재 상태

현재 이 저장소에서 가장 먼저 완성되어 있는 모듈은 **식재료 인식 베이스라인**입니다.

현재 구현되어 있는 내용:

- 데이터셋 구조 확인 스크립트
- ingredient classification dataset build 스크립트
- Jetson 카메라 smoke test / preview / capture 스크립트
- 식재료 인식 학습 코드
- 모델 sweep 코드
- 단일 이미지 추론 코드
- webcam 실시간 추론 코드
- alcohol recognition용 라벨 및 설정 scaffold

아직 완전히 구현되지 않은 내용:

- 주류 인식 학습 파이프라인 end-to-end
- ONNX export
- TensorRT 변환
- 배포 전용 통합 inference runner

---

## 3. 중요한 카메라 가정

현재 카메라는 **냉장고 내부를 보는 카메라가 아닙니다.**

- 카메라는 냉장고 상단에 설치
- 카메라는 외부를 향함
- 사용자가 식재료나 주류를 카메라 앞에 보여주는 방식

즉 현재 baseline은  
**외부-facing 카메라 기반의 frame-level classification**에 가깝고,  
아직 inside-fridge multi-object detection 파이프라인은 아닙니다.

---

## 4. 현재 이식된 식재료 인식 베이스라인

### stable 19 클래스

- bacon
- bread
- broccoli
- butter
- carrot
- cheese
- chicken
- cucumber
- egg
- fish
- lettuce
- milk
- onion
- pepper
- potato
- sausage
- spinach
- tomato
- yogurt

### 사용된 데이터셋 크기

- train: 26693
- val: 6536
- test: 8628

---

## 5. 현재까지 확인된 최고 성능

### 최고 정확도 모델

- run: `stage1_efficientnet_b0_lr0p0003_bs16_ep5`
- model: `efficientnet_b0`
- learning rate: `3e-4`
- batch size: `16`
- epochs: `5`
- best val acc: `0.8141`
- test acc: `0.8719`

### 실전 배포 후보

- run: `stage1_mobilenet_v3_large_lr0p0003_bs16_ep5`
- model: `mobilenet_v3_large`
- learning rate: `3e-4`
- batch size: `16`
- epochs: `5`
- best val acc: `0.8023`
- test acc: `0.8666`

### 중요한 관찰

- stage2에서 10 epoch로 더 오래 학습한 결과가 5 epoch winner를 넘지 못했습니다.
- 현재 데이터셋/증강 기준으로는 **5 epoch 근처가 sweet spot**으로 보입니다.

상세 실험 결과는 아래 문서를 참고하면 됩니다.

- `docs/experiments/ingredient_baseline_results.md`

---

## 6. 저장소 구조

```text
Alcoholic-AI/
|-- README.md
|-- CONTRIBUTING.md
|-- .gitignore
|-- assets/
|   `-- labels/
|-- configs/
|-- docs/
|   |-- architecture/
|   |-- experiments/
|   `-- setup/
|-- requirements-host-cv.txt
|-- requirements-jetson-train.txt
|-- scripts/
|   |-- camera/
|   `-- data_prep/
`-- src/
    |-- alcohol_recognition/
    |-- common/
    `-- ingredient_recognition/
```

이 구조는 현재 실제로 돌아가던 ingredient baseline 코드를 크게 깨지 않으면서 팀 레포로 옮기기 위한 형태입니다.  
즉, 지금은 “예쁘게 완전 재구성”보다 **작동하던 것을 그대로 팀원이 이어서 쓸 수 있게 하는 것**을 우선했습니다.

---

## 7. 주요 파일 설명

### 식재료 인식 파이프라인

- `src/ingredient_recognition/train.py`
  - 식재료 인식 모델 학습
- `src/ingredient_recognition/sweep.py`
  - 여러 모델/파라미터 비교
- `src/ingredient_recognition/infer_image.py`
  - 단일 이미지 추론
- `src/ingredient_recognition/infer_webcam.py`
  - webcam 실시간 추론

### 공통 유틸

- `src/common/ingredient_models.py`
  - 모델 생성, transform, meta 로딩
- `src/common/project_paths.py`
  - 프로젝트 루트/상대 경로 처리

### 데이터셋 준비

- `scripts/data_prep/inspect_dataset.py`
- `scripts/data_prep/build_ingredient_dataset.py`

### 카메라 유틸

- `scripts/camera/cam_smoke_test.py`
- `scripts/camera/preview_cam.py`
- `scripts/camera/capture_dataset.py`

### 라벨 및 설정

- `assets/labels/ingredient_labels_stable19.txt`
- `assets/labels/ingredient_labels.txt`
- `assets/labels/alcohol_labels.txt`
- `configs/ingredient_stable19_baseline.yaml`
- `configs/alcohol_baseline.yaml`

---

## 8. Jetson 환경 문서

실제로 동작했던 Jetson 환경 정보는 아래 문서에 정리되어 있습니다.

- `docs/setup/README.md`
- `docs/setup/jetson_env_working.md`

특히 `jetson_env_working.md`에는 다음 정보가 들어 있습니다.

- Jetson release 정보
- Python / pip 버전
- torch / torchvision / numpy / Pillow / cv2 버전
- CUDA / cuDNN 정보
- OpenCV GStreamer 지원 여부
- 카메라 연결 상태 관련 메모

즉 팀원이 이 레포를 받아서 Jetson에서 이어서 작업할 때,  
환경 재현의 출발점으로 바로 사용할 수 있습니다.

---

## 9. 대표 실행 명령

### 데이터셋 구조 확인

```bash
python scripts/data_prep/inspect_dataset.py data/external/multi_class_food_image_dataset
python scripts/data_prep/inspect_dataset.py data/external/grocery_store_dataset
```

### ingredient dataset build

```bash
python scripts/data_prep/build_ingredient_dataset.py
```

### 최고 정확도 모델 기준 학습 예시

```bash
python src/ingredient_recognition/train.py \
  --model-name efficientnet_b0 \
  --lr 3e-4 \
  --batch-size 16 \
  --epochs 5 \
  --device cuda \
  --run-name stage1_efficientnet_b0_lr0p0003_bs16_ep5
```

### sweep 실행

```bash
python src/ingredient_recognition/sweep.py
```

### 단일 이미지 추론

```bash
python src/ingredient_recognition/infer_image.py \
  --image path/to/image.jpg \
  --meta checkpoints/ingredient/stage1_efficientnet_b0_lr0p0003_bs16_ep5_meta.json
```

### webcam 추론

```bash
python src/ingredient_recognition/infer_webcam.py \
  --meta checkpoints/ingredient/stage1_mobilenet_v3_large_lr0p0003_bs16_ep5_meta.json \
  --device cuda
```

---

## 10. 이 레포를 클론한 뒤 바로 알아야 할 점

이 저장소에는 **코드와 문서**는 들어 있지만, 아래는 Git에 포함되지 않습니다.

- raw dataset
- processed dataset
- 학습 체크포인트 `.pth`
- 실행 결과물
- logs

즉 팀원이 바로 완전히 같은 결과를 재현하려면, 추가로 아래가 필요합니다.

- source dataset 또는 processed dataset
- 체크포인트 (`*_best.pth`, `*_meta.json`)
- Jetson 런타임 환경 세팅

---

## 11. 다음 권장 작업

이 브랜치 이후 팀원이 이어서 하면 좋은 우선순위는 아래와 같습니다.

1. 실촬영 ingredient 데이터 보강
2. beef / pork 데이터 추가 후 ingredient 재학습
3. 같은 형식으로 alcohol recognition 학습/추론 추가
4. ONNX export 추가
5. TensorRT 변환 및 Jetson 추론 속도 비교

---

## 12. 한 줄 요약

이 저장소는 **기존에 Jetson에서 검증된 식재료 인식 AI baseline을 팀 AI 레포에 정착시킨 출발점**이며,  
이후 같은 패턴으로 **주류 인식, export, 배포 최적화**까지 확장하기 위한 기반 저장소입니다.
