import cv2
import time


DEVICE_INDEX = 0
WIDTH = 1280
HEIGHT = 720


def main():
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[ERROR] camera open failed")
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(1.0)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] opened camera: /dev/video{DEVICE_INDEX}")
    print(f"[INFO] actual resolution: {int(actual_w)}x{int(actual_h)}")
    print(f"[INFO] actual fps: {actual_fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] frame read failed")
            break

        cv2.imshow("C920 Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
