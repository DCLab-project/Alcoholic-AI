# AI CONTRIBUTING

## 1. 실험 관리 원칙
- 모든 실험은 설정값과 결과를 함께 기록합니다.
- 모델 성능 비교 시 데이터셋 버전과 전처리 조건을 명시합니다.
- 노트북은 탐색용으로만 사용하고, 최종 로직은 스크립트로 정리합니다.

## 2. 모델/파일 네이밍 규칙
형식:
```bash
{task}_{model}_{version}
```

예시:
```bash
ingredient_mobilenet_v1
alcohol_mobilenet_v2
ingredient_mobilenet_v1.onnx
ingredient_mobilenet_v1.engine
```

## 3. 디렉토리 규칙
- `datasets/raw`: 원본 데이터
- `datasets/processed`: 전처리 결과
- `training`: 학습 코드
- `export`: ONNX/TensorRT 변환 코드
- `inference`: 배포 추론 코드
- `configs`: 실험 설정 파일

## 4. 데이터셋 관리 규칙
- 클래스 추가/삭제 시 문서 갱신 필수
- 라벨 체계 변경 시 changelog 기록
- 대용량 원본 데이터는 Git에 직접 올리지 않음
- 샘플 데이터만 저장소에 포함

## 5. 평가 규칙
최소한 아래를 확인합니다.
- 학습/검증/테스트 분리 여부
- 클래스 불균형 여부
- export 전후 결과 차이
- Jetson 추론 가능 여부

## 6. PR 추가 규칙
AI PR에는 가능하면 아래를 포함합니다.
- 데이터셋 버전
- 사용한 모델 구조
- 주요 하이퍼파라미터
- 주요 결과 지표
- export 여부 및 배포 영향

## 7. 금지 사항
- 실험 결과 없이 모델 파일만 업로드하지 않기
- raw data를 무분별하게 Git에 커밋하지 않기
- 학습 코드와 배포 코드를 한 파일에 섞지 않기
