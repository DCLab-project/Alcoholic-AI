import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.ingredient_models import build_webcam_transform, load_model_from_meta


DEFAULT_DEVICE_PATH = "/dev/video0"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_INFER_EVERY = 3
DEFAULT_CONF_THRESHOLD = 0.60
DEFAULT_TOPK = 3


def make_pipeline(device_path: str, width: int, height: int, fps: int):
    return (
        f"v4l2src device={device_path} ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true max-buffers=1"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True, help="path to *_meta.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera-device", type=str, default=DEFAULT_DEVICE_PATH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, labels, meta, meta_path, weight_path, image_size = load_model_from_meta(args.meta, device)
    tfm = build_webcam_transform(image_size=image_size)

    pipeline = make_pipeline(args.camera_device, args.width, args.height, args.fps)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] failed to open camera with pipeline: {pipeline}")

    print("[INFO] q=quit")
    print(f"[INFO] model_name = {meta['model_name']}")
    print(f"[INFO] run_name   = {meta['run_name']}")
    print(f"[INFO] meta       = {meta_path}")
    print(f"[INFO] weight     = {weight_path}")
    print(f"[INFO] camera     = {args.camera_device}")
    print(f"[INFO] infer every {args.infer_every} frames")
    print(f"[INFO] confidence threshold = {args.threshold}")

    prev_time = time.time()
    fps = 0.0
    frame_idx = 0

    last_topk = []
    last_pred_label = "uncertain"
    last_conf = 0.0
    last_infer_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] frame read failed")
            break

        frame_idx += 1

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        if frame_idx % max(args.infer_every, 1) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tfm(rgb).unsqueeze(0).to(device)

            t0 = time.time()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0]
                values, indices = torch.topk(probs, k=min(max(args.topk, 1), len(labels)))
            last_infer_ms = (time.time() - t0) * 1000.0

            last_topk = []
            for v, i in zip(values.tolist(), indices.tolist()):
                last_topk.append((labels[i], v))

            top1_label, top1_conf = last_topk[0]
            if top1_conf >= args.threshold:
                last_pred_label = top1_label
            else:
                last_pred_label = "uncertain"
            last_conf = top1_conf

        cv2.putText(frame, f"model: {meta['model_name']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
        cv2.putText(frame, f"run: {meta['run_name']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
        cv2.putText(frame, f"pred: {last_pred_label}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"conf: {last_conf:.3f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"fps: {fps:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"infer_ms: {last_infer_ms:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"threshold: {args.threshold:.2f}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(frame, "q=quit", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        y0 = 280
        for rank, (label, prob) in enumerate(last_topk, start=1):
            line = f"{rank}. {label}: {prob:.3f}"
            cv2.putText(frame, line, (20, y0 + (rank - 1) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)

        cv2.imshow("Ingredient Webcam Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
