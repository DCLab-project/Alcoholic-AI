import cv2
import time


PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=640, height=480, framerate=30/1 ! "
    "jpegdec ! videoconvert ! appsink drop=true max-buffers=1"
)


def main():
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    print("isOpened =", cap.isOpened())

    if not cap.isOpened():
        raise SystemExit("[ERROR] failed to open camera")

    time.sleep(0.5)

    ok_count = 0
    last_shape = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_count += 1
            last_shape = frame.shape

    cap.release()

    print("ok_count =", ok_count)
    print("last_shape =", last_shape)


if __name__ == "__main__":
    main()
