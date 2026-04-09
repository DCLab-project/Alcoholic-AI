# AI

## 1. 저장소 목적
이 저장소는 **술안주 추천 AI 냉장고의 식재료/주류 인식 모델과 배포용 추론 자산**을 담당합니다.

프로젝트 계획서 기준으로 AI 파트는 다음 내용을 포함합니다.
- CNN 기반 식재료 인식 모델 학습
- CNN 기반 주류 인식 모델 학습
- 공개 데이터셋 + 실제 촬영 데이터셋 구축
- PyTorch -> ONNX -> TensorRT 최적화
- Jetson Orin Nano 배포용 추론 모듈 구성

## 2. 주요 역할
- 식재료 인식 모델 학습 및 평가
- 주류 카테고리 인식 모델 학습 및 평가
- 데이터셋 정제 및 증강
- 모델 export 및 최적화
- 배포용 inference 인터페이스 제공

## 3. 데이터셋 개요
계획서 기준 식재료 인식 모델은 **총 30개 식재료 클래스**를 대상으로 하며,
- 28개 클래스는 공개 데이터셋 사용
- 소고기, 돼지고기 2개 클래스는 실제 촬영 데이터로 보강
구조를 따릅니다.

또한 실제 촬영 데이터는 시점, 거리, 냉장고 내부 조명, 포장 상태 등을 반영해 수집합니다.

## 4. 권장 구조
```bash
datasets/
  raw/
  processed/
  annotations/
models/
training/
export/
inference/
configs/
notebooks/
scripts/
```

## 5. 개발 원칙
- 학습 코드와 추론 코드를 분리합니다.
- 실험성 코드와 배포 코드를 분리합니다.
- 모델 파일명에는 버전과 목적을 명시합니다.
- 데이터셋 변경 시 변경 이유를 기록합니다.

## 6. 배포 흐름
기본 흐름은 아래와 같습니다.
1. PyTorch 학습
2. ONNX 변환
3. TensorRT 엔진 최적화
4. Jetson 배포
5. 실시간 추론

## 7. 실행 방법
```bash
# example
pip install -r requirements.txt
python training/train.py
python export/export_onnx.py
python inference/run.py
```

## 8. 환경 변수/설정 예시
```bash
DATASET_ROOT=
MODEL_OUTPUT_DIR=
ONNX_OUTPUT_DIR=
TENSORRT_OUTPUT_DIR=
DEVICE=
```

## 9. AI에서 특히 중요하게 볼 것
- 데이터셋 버전 관리
- 클래스 정의 일관성
- 학습/검증/테스트 분리
- export 후 추론 결과 검증
- Jetson 환경에서의 최적화 여부
