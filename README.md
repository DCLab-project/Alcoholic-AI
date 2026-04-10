# Alcoholic-AI

Alcoholic-AI is the AI repository for the smart fridge project. This repository now contains the migrated ingredient-recognition baseline that was originally developed and validated on Jetson Orin Nano with a Logitech C920 USB camera.

The repository is intended to cover:

- ingredient recognition model training and evaluation
- alcohol category recognition model training and evaluation
- dataset preparation and label management
- Jetson deployment-oriented inference utilities
- future ONNX and TensorRT export workflows

## Current Status

The most complete implemented module in this repository today is the ingredient-recognition baseline.

Current implemented scope:

- dataset inspection scripts
- ingredient dataset build script
- Jetson camera smoke test, preview, and capture scripts
- ingredient training script
- ingredient experiment sweep script
- metadata-driven image inference
- metadata-driven webcam inference
- alcohol recognition scaffold and label definitions

Not fully implemented yet:

- alcohol model training pipeline end-to-end
- ONNX export
- TensorRT export
- unified deployment runner

## Important Camera Assumption

The current camera is external-facing, mounted above the fridge and pointing outward.

It is meant to observe:

- ingredients being inserted or removed in front of the fridge
- alcohol bottles or cans shown by the user in front of the camera

This is not currently an inside-fridge multi-object detection pipeline.

## Migrated Ingredient Baseline

Stable 19 ingredient classes:

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

Confirmed dataset size used for the current stable ingredient classifier:

- train: 26693
- val: 6536
- test: 8628

## Best Confirmed Results

Best accuracy model:

- run: `stage1_efficientnet_b0_lr0p0003_bs16_ep5`
- model: `efficientnet_b0`
- lr: `3e-4`
- batch size: `16`
- epochs: `5`
- best val acc: `0.8141`
- test acc: `0.8719`

Best deployment candidate:

- run: `stage1_mobilenet_v3_large_lr0p0003_bs16_ep5`
- model: `mobilenet_v3_large`
- lr: `3e-4`
- batch size: `16`
- epochs: `5`
- best val acc: `0.8023`
- test acc: `0.8666`

Important finding:

- longer 10-epoch stage-2 runs did not beat the best 5-epoch winner
- with the current data and augmentation setup, the strongest confirmed runs peaked around 5 epochs

Detailed experiment history is documented in `docs/experiments/ingredient_baseline_results.md`.

## Repository Layout

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

This structure preserves the currently working ingredient pipeline so the team can continue model improvement without first refactoring everything.

## Key Files

Ingredient pipeline:

- `src/ingredient_recognition/train.py`
- `src/ingredient_recognition/sweep.py`
- `src/ingredient_recognition/infer_image.py`
- `src/ingredient_recognition/infer_webcam.py`

Shared helpers:

- `src/common/ingredient_models.py`
- `src/common/project_paths.py`

Dataset scripts:

- `scripts/data_prep/inspect_dataset.py`
- `scripts/data_prep/build_ingredient_dataset.py`

Camera scripts:

- `scripts/camera/cam_smoke_test.py`
- `scripts/camera/preview_cam.py`
- `scripts/camera/capture_dataset.py`

Labels and configs:

- `assets/labels/ingredient_labels_stable19.txt`
- `assets/labels/ingredient_labels.txt`
- `assets/labels/alcohol_labels.txt`
- `configs/ingredient_stable19_baseline.yaml`
- `configs/alcohol_baseline.yaml`

## Setup Notes

Working Jetson environment details are documented in:

- `docs/setup/README.md`
- `docs/setup/jetson_env_working.md`

## Typical Commands

Inspect datasets:

```bash
python scripts/data_prep/inspect_dataset.py data/external/multi_class_food_image_dataset
python scripts/data_prep/inspect_dataset.py data/external/grocery_store_dataset
```

Build ingredient dataset:

```bash
python scripts/data_prep/build_ingredient_dataset.py
```

Train best-accuracy ingredient candidate:

```bash
python src/ingredient_recognition/train.py \
  --model-name efficientnet_b0 \
  --lr 3e-4 \
  --batch-size 16 \
  --epochs 5 \
  --device cuda \
  --run-name stage1_efficientnet_b0_lr0p0003_bs16_ep5
```

Run ingredient sweep:

```bash
python src/ingredient_recognition/sweep.py
```

Run image inference:

```bash
python src/ingredient_recognition/infer_image.py \
  --image path/to/image.jpg \
  --meta checkpoints/ingredient/stage1_efficientnet_b0_lr0p0003_bs16_ep5_meta.json
```

Run webcam inference:

```bash
python src/ingredient_recognition/infer_webcam.py \
  --meta checkpoints/ingredient/stage1_mobilenet_v3_large_lr0p0003_bs16_ep5_meta.json \
  --device cuda
```

## Next Recommended Work

1. continue improving ingredient recognition with real captured samples
2. add beef and pork data properly
3. implement alcohol training, evaluation, and webcam inference in the same style
4. add ONNX export and validate export parity
5. add TensorRT conversion and Jetson runtime benchmarking
