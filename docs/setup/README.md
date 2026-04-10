# Jetson Setup Notes

This document captures the current working setup pattern used for the migrated ingredient-recognition baseline.

## Hardware

- Device: Jetson Orin Nano
- Camera: Logitech C920 USB webcam
- Camera placement: external-facing, mounted above the fridge

## Camera Validation

Typical checks:

```bash
lsusb
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The project used OpenCV + GStreamer as the working Jetson capture path.

## Known Camera Issue

A real issue encountered during development was `/dev/video0` being occupied by another service or container. During debugging, a systemd service such as `edgeai-person.service` could hold the camera and had to be stopped.

Useful check:

```bash
fuser -v /dev/video0
```

## Python Environment Pattern

Recommended split:

- system OpenCV + GStreamer for camera integration
- training venv for PyTorch-based training and inference

Known working training venv:

```bash
~/venvs/ai-fridge-train
```

## OpenCV / NumPy Note

One real issue encountered was an ABI mismatch between system OpenCV and a user-installed NumPy 2.x package. The working recovery path was to align NumPy with the OpenCV build and avoid replacing Jetson OpenCV casually with an incompatible pip wheel.
