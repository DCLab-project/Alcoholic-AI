# Working Jetson Environment

This file records the actual runtime information collected from the Jetson environment where the ingredient baseline was trained and validated.

## Jetson Runtime

- L4T / release string:
  - `# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025`
- Target device name reported by PyTorch: `Orin`

## Python Environment

- venv path: `~/venvs/ai-fridge-train`
- Python: `3.10.12`
- pip: `26.0.1`
- python executable: `/home/jetk/venvs/ai-fridge-train/bin/python`

## Confirmed Installed Packages

- `torch`: `2.5.0a0+872d972e41.nv24.08`
- `torchvision`: `0.20.1`
- `numpy`: `1.26.1`
- `Pillow`: `12.2.0`
- `cv2`: `4.5.4`

## Not Present In The Checked venv

At the time of collection, these were not installed in the checked training venv:

- `torchaudio`
- `PyYAML`
- `tqdm`
- `matplotlib`
- `onnx`
- `onnxruntime`

If later code depends on them, install them explicitly and re-verify the pipeline.

## CUDA / cuDNN

- `torch.cuda.is_available`: `True`
- `device_count`: `1`
- `device_name`: `Orin`
- `torch.version.cuda`: `12.6`
- `torch.backends.cudnn.version()`: `90300`

## OpenCV Build Facts

- `cv2`: `4.5.4`
- `GStreamer`: `YES (1.19.90)`
- `v4l/v4l2`: `YES (linux/videodev2.h)`

These details are important because the Jetson camera pipeline depended on a GStreamer-enabled OpenCV build.

## Camera State At Collection Time

At the moment this environment snapshot was collected, the Logitech C920 was unplugged, so `/dev/video*` device nodes were not present.

However, earlier development validation already confirmed:

- Logitech C920 worked as the project camera
- `/dev/video0` was the active capture path during successful runs
- `scripts/camera/cam_smoke_test.py` succeeded using OpenCV + GStreamer
