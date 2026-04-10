# Ingredient Baseline Results

This document summarizes the confirmed finished ingredient-recognition experiments preserved from the working Jetson pipeline and project notes.

## Dataset Used

- Stable classifier scope: 19 classes
- Train images: 26693
- Validation images: 6536
- Test images: 8628

Stable classes:

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

## Confirmed Stage 1 Runs

| Run Name | Model | LR | Batch Size | Epochs | Best Val Acc | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stage1_mobilenet_v3_small_lr0p001_bs32_ep5` | `mobilenet_v3_small` | `1e-3` | `32` | `5` | `0.7561` | `0.8209` | Lightweight stable baseline |
| `stage1_mobilenet_v3_small_lr0p0003_bs32_ep5` | `mobilenet_v3_small` | `3e-4` | `32` | `5` | `0.7557` | `0.8187` | Similar to small 1e-3 run |
| `stage1_mobilenet_v3_large_lr0p001_bs16_ep5` | `mobilenet_v3_large` | `1e-3` | `16` | `5` | `0.7844` | `0.8410` | Stronger than MobileNetV3-Small |
| `stage1_mobilenet_v3_large_lr0p0003_bs16_ep5` | `mobilenet_v3_large` | `3e-4` | `16` | `5` | `0.8023` | `0.8666` | Best deployment candidate |
| `stage1_efficientnet_b0_lr0p001_bs16_ep5` | `efficientnet_b0` | `1e-3` | `16` | `5` | `0.7827` | `0.8426` | Strong accuracy candidate |
| `stage1_efficientnet_b0_lr0p0003_bs16_ep5` | `efficientnet_b0` | `3e-4` | `16` | `5` | `0.8141` | `0.8719` | Best accuracy model |
| `stage1_resnet18_lr0p001_bs8_ep5` | `resnet18` | `1e-3` | `8` | `5` | `0.6538` | `0.7243` | Underperformed on this setup |
| `stage1_resnet18_lr0p0003_bs8_ep5` | `resnet18` | `3e-4` | `8` | `5` | `0.7541` | `0.8202` | Improved at lower LR but still behind top models |

## Confirmed Stage 2 Runs

| Run Name | Model | LR | Batch Size | Epochs | Best Val Acc | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stage2_efficientnet_b0_lr0p0003_bs16_ep10` | `efficientnet_b0` | `3e-4` | `16` | `10` | `0.8104` | `0.8682` | Strong but below best 5-epoch winner |
| `stage2_efficientnet_b0_lr0p00015_bs16_ep10` | `efficientnet_b0` | `1.5e-4` | `16` | `10` | `0.8084` | `0.8650` | Lower LR did not improve final result |
| `stage2_mobilenet_v3_large_lr0p0003_bs16_ep10` | `mobilenet_v3_large` | `3e-4` | `16` | `10` | `0.8062` | `0.8642` | Slightly below best stage-1 MobileNetV3-Large |

## Summary Findings

- Best accuracy model: `efficientnet_b0` with `lr=3e-4`, `batch_size=16`, `epochs=5`
- Best deployment candidate: `mobilenet_v3_large` with `lr=3e-4`, `batch_size=16`, `epochs=5`
- MobileNetV3-Small remained a useful lightweight baseline but was not the top performer
- ResNet18 was the weakest family in the confirmed Jetson runs
- Longer Stage 2 10-epoch runs did not outperform the best 5-epoch Stage 1 winner
- With the current dataset and augmentation regime, the strongest models appear to peak at about 5 epochs

## Practical Interpretation

For the current project stage, the most sensible operating rule is:

- use `efficientnet_b0` when pure classification accuracy matters most
- use `mobilenet_v3_large` when near-top accuracy and more deployment-friendly behavior are both important

If more data is added later, the safest next step is to retrain the current winner first, then run a smaller local retune around:

- `lr = 3e-4`
- `batch size = 16`
- `epochs = 5 to 8`
