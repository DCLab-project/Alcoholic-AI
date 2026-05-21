#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 src/common/main_controller.py \
  --serial-port /dev/ttyACM0 \
  --camera 0 \
  --enable-be-post \
  --be-base-url http://192.168.50.123:8000 \
  --pir-hold-seconds 5 \
  --enable-ingredient-direction \
  --ingredient-vote-min-confidence 0.50 \
  --ingredient-vote-visible-margin 0.15
