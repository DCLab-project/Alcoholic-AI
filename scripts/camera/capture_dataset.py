import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.project_paths import RAW_DATA_DIR


BASE_DIR = RAW_DATA_DIR
SAVE_DIRS = {
    "1": "alcohol",
    "2": "ingredient",
    "3": "beef",
    "4": "pork",
}

DEVICE = "/dev/video0"
WIDTH = 640
HEIGHT = 480
FPS = 30


def make_pipeline():
    return (
        f"v4l2src device={DEVICE} ! "
        f"image/jpeg, width={WIDTH}, height={HEIGHT}, framerate={FPS}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true max-buffers=1"
    )


def ensure_dirs():
    for name in SAVE_DIRS.values():
        (BASE_DIR / name).mkdir(parents=True, exist_ok=True)


def count_existing_images(folder_path: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in exts)


def make_filename(label_name: str, folder_path: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index = count_existing_images(folder_path) + 1
    return f"{label_name}_{timestamp}_{index:05d}.jpg"


def draw_overlay(frame, current_label: str):
    text1 = f"current label: {current_label}"
    text2 = "keys: 1=alcohol  2=ingredient  3=beef  4=pork"
    text3 = "s=save  q=quit"

    cv2.putText(frame, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


def main():
    ensure_dirs()

    pipeline = make_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise SystemExit("[ERROR] failed to open camera")

    current_key = "1"
    current_label = SAVE_DIRS[current_key]

    print("[INFO] camera opened")
    print("[INFO] key 1 -> alcohol")
    print("[INFO] key 2 -> ingredient")
    print("[INFO] key 3 -> beef")
    print("[INFO] key 4 -> pork")
    print("[INFO] key s -> save image")
    print("[INFO] key q -> quit")
    print(f"[INFO] save base dir -> {BASE_DIR}")

    save_msg = ""
    save_msg_until = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] frame read failed")
            break

        raw_frame = frame.copy()
        display_frame = draw_overlay(frame, current_label)

        if save_msg and time.time() < save_msg_until:
            cv2.putText(display_frame, save_msg, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("AI Fridge Capture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key != 255 and chr(key) in SAVE_DIRS:
            current_key = chr(key)
            current_label = SAVE_DIRS[current_key]
            print(f"[INFO] current label changed -> {current_label}")

        elif key == ord("s"):
            folder_path = BASE_DIR / current_label
            filename = make_filename(current_label, folder_path)
            save_path = folder_path / filename
            ok = cv2.imwrite(str(save_path), raw_frame)
            if ok:
                save_msg = f"saved: {save_path}"
                save_msg_until = time.time() + 1.5
                print(f"[SAVED] {save_path}")
            else:
                print("[ERROR] failed to save image")

        elif key == ord("q"):
            print("[INFO] quit")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
