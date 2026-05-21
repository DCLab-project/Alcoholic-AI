from __future__ import annotations

import argparse
import json
import os
import queue
import select
import sys
import termios
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.project_paths import PROJECT_ROOT, resolve_project_path


DEFAULT_INGREDIENT_META = (
    "checkpoints/ingredient/public_local_30class_mobilenet_v3_large_ep10_meta.json"
)
DEFAULT_LIQUOR_META = (
    "checkpoints/liquor/public_local_7class_liquor_mobilenet_v3_large_ep5_meta.json"
)

MODE_IDLE = "idle"
MODE_LIQUOR = "liquor"
MODE_INGREDIENT = "ingredient"

STATE_LABELS = {
    0: "Sleep",
    1: "Alcohol",
    2: "Ingredient",
}

MODE_LABELS = {
    MODE_IDLE: "Sleep",
    MODE_LIQUOR: "Alcohol",
    MODE_INGREDIENT: "Ingredient",
}

SENSOR_RECOMMENDED_MODES = {
    0: "standby",
    1: "liquor_scan_ready",
    2: "ingredient_scan",
}

INGREDIENT_CANONICAL_KEYS = {
    "avocado",
    "beef",
    "bread",
    "broccoli",
    "butter",
    "cabbage",
    "carrot",
    "cheese",
    "chicken",
    "cucumber",
    "egg",
    "eggplant",
    "fish",
    "garlic",
    "ginger",
    "green_onion",
    "lemon",
    "lettuce",
    "milk",
    "mushroom",
    "onion",
    "pepper",
    "pork",
    "potato",
    "radish",
    "salmon",
    "sausage",
    "tofu",
    "tomato",
    "zucchini",
}

LIQUOR_CANONICAL_KEYS = {
    "soju",
    "beer",
    "red_wine",
    "white_wine",
    "sparkling_wine",
    "whisky",
    "sake",
}

INGREDIENT_LABEL_ALIASES = {
    "leek": "green_onion",
    "green onion": "green_onion",
}

LIQUOR_LABEL_ALIASES = {
    "whiskey": "whisky",
}


@dataclass
class RuntimeState:
    sensor_code: int = 0
    raw_sensor_code: int = 0
    pir: int | None = None
    distance_cm: float | None = None
    reading_seq: int = 0
    active_mode: str = MODE_IDLE
    pir_hold_until: float = 0.0
    pir_rearm_required: bool = False
    changed_at: float = field(default_factory=time.time)


@dataclass
class SensorReading:
    state_code: int
    pir: int | None = None
    distance_cm: float | None = None


@dataclass
class LoadedModel:
    name: str
    model: Any
    labels: list[str]
    transform: Any
    image_size: int
    meta_path: Path
    checkpoint_path: Path
    prob_queue: deque
    last_topk: list[tuple[str, float]] = field(default_factory=list)
    last_label: str = "waiting"
    last_conf: float = 0.0
    last_infer_ms: float = 0.0
    candidate_label: str | None = None
    stable_count: int = 0
    last_sent_label: str | None = None
    last_sent_time: float = 0.0
    vote_counts: Counter[str] = field(default_factory=Counter)
    vote_conf_sums: dict[str, float] = field(default_factory=dict)
    vote_started_at: float = 0.0
    vote_deadline: float = 0.0
    vote_finalized: bool = False
    vote_max_conf: float = 0.0


@dataclass
class DirectionTracker:
    enabled: bool
    min_area: float
    max_area_ratio: float
    top_zone: float
    bottom_zone: float
    min_travel: float
    lost_frames_required: int
    min_track_frames: int
    display_seconds: float
    subtractor: Any = None
    active: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)
    lost_frames: int = 0
    last_seen_t: float = 0.0
    last_event: str = ""
    event_until: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    started_this_frame: bool = False
    finished_this_frame: bool = False
    seen_this_frame: bool = False
    finished_direction: str = ""

    def reset(self, clear_event: bool = False) -> None:
        self.subtractor = None
        self.active = False
        self.points.clear()
        self.lost_frames = 0
        self.last_seen_t = 0.0
        self.bbox = None
        self.started_this_frame = False
        self.finished_this_frame = False
        self.seen_this_frame = False
        self.finished_direction = ""
        if clear_event:
            self.last_event = ""
            self.event_until = 0.0

    def current_event(self, now: float) -> str:
        if self.last_event and now <= self.event_until:
            return self.last_event
        return ""

    def is_visible_for_vote(self, frame_shape, margin_ratio: float) -> bool:
        if self.bbox is None:
            return False

        frame_h = frame_shape[0]
        margin = frame_h * min(max(margin_ratio, 0.0), 0.45)
        _x, y, _w, h = self.bbox
        center_y = y + h / 2.0
        return margin <= center_y <= (frame_h - margin)

    def update(self, frame, now: float) -> str:
        self.started_this_frame = False
        self.finished_this_frame = False
        self.seen_this_frame = False
        self.finished_direction = ""

        if not self.enabled:
            return ""

        import cv2

        if self.subtractor is None:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=80,
                varThreshold=32,
                detectShadows=False,
            )

        h, w = frame.shape[:2]
        fgmask = self.subtractor.apply(frame)
        _ret, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        detected = self.detect_motion_center(mask, frame_area=float(h * w))
        if detected is None:
            if self.active:
                self.lost_frames += 1
                if self.lost_frames >= max(1, self.lost_frames_required):
                    event = self.finalize_track(frame_height=h, now=now)
                    self.active = False
                    self.points.clear()
                    self.lost_frames = 0
                    self.bbox = None
                    self.finished_this_frame = True
                    self.finished_direction = event
            return self.current_event(now)

        center_x, center_y, bbox = detected
        if not self.active:
            self.active = True
            self.points.clear()
            self.lost_frames = 0
            self.started_this_frame = True

        self.points.append((center_x, center_y))
        self.last_seen_t = now
        self.lost_frames = 0
        self.bbox = bbox
        self.seen_this_frame = True
        return self.current_event(now)

    def detect_motion_center(
        self,
        mask,
        frame_area: float,
    ) -> tuple[float, float, tuple[int, int, int, int]] | None:
        import cv2

        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        max_area = frame_area * min(max(self.max_area_ratio, 0.01), 1.0)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > max_area:
                continue
            candidates.append((area, contour))

        if not candidates:
            return None

        _area, contour = max(candidates, key=lambda item: item[0])
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        return center_x, center_y, (x, y, w, h)

    def finalize_track(self, frame_height: int, now: float) -> str:
        if len(self.points) < max(1, self.min_track_frames):
            return ""

        sample_n = min(3, len(self.points))
        start_y = sum(point[1] for point in self.points[:sample_n]) / sample_n
        end_y = sum(point[1] for point in self.points[-sample_n:]) / sample_n
        travel = end_y - start_y

        top_limit = frame_height * min(max(self.top_zone, 0.0), 1.0)
        bottom_limit = frame_height * min(max(self.bottom_zone, 0.0), 1.0)
        min_pixels = frame_height * min(max(self.min_travel, 0.0), 1.0)

        event = ""
        if start_y <= top_limit and end_y >= bottom_limit and travel >= min_pixels:
            event = "input"
        elif start_y >= bottom_limit and end_y <= top_limit and -travel >= min_pixels:
            event = "output"

        if event:
            self.last_event = event
            self.event_until = now + max(0.1, self.display_seconds)
            print(
                f"[DIRECTION] {event} "
                f"start_y={start_y:.1f} end_y={end_y:.1f} travel={travel:.1f}"
            )
        return event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Arduino state codes and switch between preloaded liquor and "
            "ingredient webcam classifiers without restarting model processes."
        )
    )
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ingredient-meta", default=DEFAULT_INGREDIENT_META)
    parser.add_argument("--liquor-meta", default=DEFAULT_LIQUOR_META)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--infer-every", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--pir-hold-seconds",
        type=float,
        default=5.0,
        help=(
            "Hold state 1/Alcohol for this many seconds after one PIR event. "
            "During the hold, state 0/1 serial values are ignored and state 2 "
            "still overrides immediately. The liquor model votes during this window."
        ),
    )
    parser.add_argument("--overlay-topk", type=int, default=3)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--enable-be-post", action="store_true")
    parser.add_argument("--be-base-url", default="")
    parser.add_argument("--sensor-events-url", default="")
    parser.add_argument("--inventory-events-url", default="")
    parser.add_argument("--liquor-recognition-url", default="")
    parser.add_argument("--device-id", default="jetson-arduino-bridge")
    parser.add_argument("--sensor-source", default="arduino")
    parser.add_argument("--post-timeout", type=float, default=2.0)
    parser.add_argument(
        "--enable-ingredient-direction",
        action="store_true",
        help="Track vertical ingredient motion in state 2 and draw input/output on the CV window.",
    )
    parser.add_argument("--direction-min-area", type=float, default=1200.0)
    parser.add_argument("--direction-max-area-ratio", type=float, default=0.60)
    parser.add_argument("--direction-top-zone", type=float, default=0.35)
    parser.add_argument("--direction-bottom-zone", type=float, default=0.65)
    parser.add_argument("--direction-min-travel", type=float, default=0.25)
    parser.add_argument("--direction-lost-frames", type=int, default=8)
    parser.add_argument("--direction-min-track-frames", type=int, default=4)
    parser.add_argument("--direction-display-seconds", type=float, default=2.5)
    parser.add_argument(
        "--ingredient-vote-min-confidence",
        type=float,
        default=0.50,
        help=(
            "Include an ingredient prediction in the tracking vote only when "
            "its top-1 confidence is above this value."
        ),
    )
    parser.add_argument(
        "--ingredient-vote-visible-margin",
        type=float,
        default=0.15,
        help=(
            "Include an ingredient prediction in the tracking vote only when "
            "the tracked bbox is inside this top/bottom frame margin."
        ),
    )
    parser.add_argument("--stable-frames", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--recognition-cooldown", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-repeat-recognition", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scan-request-id", default="")
    parser.add_argument("--window-name", default="Arduino Sensor Model Runtime")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open an OpenCV display window; print predictions only.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Minimum seconds between prediction log lines in headless mode.",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Load both models and exit without opening serial or camera.",
    )
    return parser.parse_args()


def baud_to_constant(baud: int) -> int:
    candidates = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }
    if baud not in candidates:
        raise ValueError(f"Unsupported baud rate for termios helper: {baud}")
    return candidates[baud]


def configure_serial(fd: int, baud: int) -> None:
    attrs = termios.tcgetattr(fd)
    speed = baud_to_constant(baud)

    iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs
    iflag &= ~(termios.IGNBRK | termios.IXON | termios.IXOFF | termios.IXANY)
    oflag = 0
    lflag = 0
    cflag = (cflag & ~termios.CSIZE) | termios.CS8
    cflag |= termios.CLOCAL | termios.CREAD
    cflag &= ~(termios.PARENB | termios.CSTOPB | getattr(termios, "CRTSCTS", 0))
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 1

    attrs = [iflag, oflag, cflag, lflag, speed, speed, cc]
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def sensor_code_to_mode(code: int) -> str | None:
    if code == 0:
        return MODE_IDLE
    if code == 1:
        return MODE_LIQUOR
    if code == 2:
        return MODE_INGREDIENT
    return None


def sensor_code_to_label(code: int) -> str:
    return STATE_LABELS.get(code, f"Unknown({code})")


def mode_to_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def parse_sensor_line(text: str) -> SensorReading:
    stripped = text.strip()

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        return SensorReading(
            state_code=int(payload.get("state_code", payload.get("state", 0))),
            pir=_optional_int(payload.get("pir")),
            distance_cm=_optional_float(payload.get("ultrasonic_cm", payload.get("distance_cm"))),
        )

    parts = [part.strip() for part in stripped.split(",")]
    reading = SensorReading(state_code=int(parts[0]))

    if len(parts) >= 2 and parts[1] != "":
        reading.pir = int(parts[1])

    if len(parts) >= 3 and parts[2] != "":
        reading.distance_cm = float(parts[2])

    return reading


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def canonicalize_label(mode: str, label: str) -> str:
    key = label.strip().lower().replace(" ", "_")

    if mode == MODE_INGREDIENT:
        key = INGREDIENT_LABEL_ALIASES.get(key, key)
        return key

    if mode == MODE_LIQUOR:
        key = LIQUOR_LABEL_ALIASES.get(key, key)
        return key

    return key


def build_url(base_url: str, explicit_url: str, path: str) -> str:
    if explicit_url:
        return explicit_url
    if not base_url:
        return ""
    return base_url.rstrip("/") + path


def make_sensor_event_payload(state: RuntimeState, device_id: str, source: str) -> dict[str, Any]:
    code = state.sensor_code
    pir = state.pir

    if code == 0:
        door_open = False
        motion_detected = False
    elif code == 1:
        door_open = False
        motion_detected = True
    else:
        door_open = True
        motion_detected = bool(pir) if pir is not None else False

    raw: dict[str, Any] = {
        "state_code": state.raw_sensor_code,
    }
    if pir is not None:
        raw["pir"] = int(pir)
    if state.distance_cm is not None:
        raw["ultrasonic_cm"] = float(state.distance_cm)

    payload: dict[str, Any] = {
        "device_id": device_id,
        "door_open": door_open,
        "motion_detected": motion_detected,
        "source": source,
        "recommended_mode": SENSOR_RECOMMENDED_MODES.get(code, "standby"),
        "raw": raw,
    }

    if state.distance_cm is not None:
        payload["distance_cm"] = float(state.distance_cm)

    return payload


def make_recognition_payload(
    mode: str,
    label: str,
    confidence: float,
    scan_request_id: str,
) -> tuple[str, dict[str, Any]]:
    canonical_label = canonicalize_label(mode, label)

    if mode == MODE_INGREDIENT:
        if canonical_label not in INGREDIENT_CANONICAL_KEYS:
            print(f"[WARN] ingredient label is not in canonical list: {canonical_label}")
        payload = {
            "ingredient_name": canonical_label,
            "confidence": float(confidence),
            "source": "jetson-ingredient-classifier",
        }
        endpoint_kind = "ingredient"
    elif mode == MODE_LIQUOR:
        if canonical_label not in LIQUOR_CANONICAL_KEYS:
            print(f"[WARN] liquor label is not in canonical list: {canonical_label}")
        payload = {
            "liquor_name": canonical_label,
            "confidence": float(confidence),
            "source": "jetson-liquor-classifier",
        }
        endpoint_kind = "liquor"
    else:
        raise ValueError(f"Unsupported recognition mode: {mode}")

    if scan_request_id:
        payload["scan_request_id"] = scan_request_id

    return endpoint_kind, payload


def make_inventory_event_payload(
    label: str,
    direction: str,
    confidence: float,
) -> dict[str, Any] | None:
    canonical_label = canonicalize_label(MODE_INGREDIENT, label)
    if canonical_label not in INGREDIENT_CANONICAL_KEYS:
        print(f"[WARN] ingredient label is not in canonical list: {canonical_label}")

    if direction == "input":
        action = "add"
    elif direction == "output":
        action = "subtract"
    else:
        return None

    return {
        "ingredient_name": canonical_label,
        "action": action,
        "quantity": 1,
        "confidence": float(confidence),
        "source": "jetson-ingredient-tracker",
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[bool, int | None, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, exc.code, body
    except Exception as exc:
        return False, None, repr(exc)


def post_worker(
    post_queue: queue.Queue[tuple[str, str, dict[str, Any]]],
    stop_event: threading.Event,
    timeout: float,
) -> None:
    while not stop_event.is_set() or not post_queue.empty():
        try:
            kind, url, payload = post_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        success, status, body = post_json(url, payload, timeout=timeout)
        if success:
            print(f"[POST OK] {kind} status={status} payload={json.dumps(payload, ensure_ascii=False)}")
        else:
            print(
                f"[POST FAIL] {kind} status={status} "
                f"payload={json.dumps(payload, ensure_ascii=False)} body={body}"
            )
        post_queue.task_done()


def enqueue_post(
    post_queue: queue.Queue[tuple[str, str, dict[str, Any]]] | None,
    kind: str,
    url: str,
    payload: dict[str, Any],
) -> None:
    if post_queue is None or not url:
        return
    post_queue.put((kind, url, payload))


def expire_pir_hold_if_needed(state: RuntimeState, now: float) -> bool:
    if state.pir_hold_until <= 0.0 or now < state.pir_hold_until:
        return False

    state.pir_hold_until = 0.0

    if state.active_mode == MODE_LIQUOR and state.sensor_code == 1:
        state.sensor_code = 0
        state.active_mode = MODE_IDLE
        state.changed_at = now
        return True

    return False


def apply_sensor_reading(
    state: RuntimeState,
    reading: SensorReading,
    pir_hold_seconds: float,
    now: float,
) -> bool:
    code = reading.state_code
    old_code = state.sensor_code
    old_mode = state.active_mode
    old_hold_until = state.pir_hold_until

    state.raw_sensor_code = code
    state.pir = reading.pir
    state.distance_cm = reading.distance_cm
    state.reading_seq += 1

    # Ultrasonic disappearance is always highest priority, even during PIR hold.
    if code == 2:
        state.sensor_code = 2
        state.active_mode = MODE_INGREDIENT
        state.pir_hold_until = 0.0
        state.pir_rearm_required = False
        if old_code != state.sensor_code or old_mode != state.active_mode:
            state.changed_at = now
        return (
            old_code != state.sensor_code
            or old_mode != state.active_mode
            or old_hold_until != state.pir_hold_until
        )

    # During the PIR hold, ignore 0/1 completely.
    if state.pir_hold_until > now:
        return False

    expired = expire_pir_hold_if_needed(state, now)

    if code == 0:
        state.sensor_code = 0
        state.active_mode = MODE_IDLE
        state.pir_rearm_required = False
    elif code == 1:
        if state.pir_rearm_required:
            # Ignore repeated/continuous PIR=1 from the same already-handled event.
            return expired

        state.sensor_code = 1
        state.active_mode = MODE_LIQUOR
        state.pir_hold_until = now + max(0.0, pir_hold_seconds)
        state.pir_rearm_required = True
    else:
        return expired

    changed = old_code != state.sensor_code or old_mode != state.active_mode
    if changed:
        state.changed_at = now

    return changed or expired or old_hold_until != state.pir_hold_until


def apply_sensor_code(
    state: RuntimeState,
    code: int,
    pir_hold_seconds: float,
    now: float,
) -> bool:
    return apply_sensor_reading(
        state,
        SensorReading(state_code=code),
        pir_hold_seconds=pir_hold_seconds,
        now=now,
    )


def serial_reader(
    port: str,
    baud: int,
    state: RuntimeState,
    lock: threading.Lock,
    stop_event: threading.Event,
    pir_hold_seconds: float,
) -> None:
    while not stop_event.is_set():
        fd = -1
        try:
            fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            configure_serial(fd, baud)
            print(f"[SERIAL] connected: {port} @ {baud}")

            time.sleep(2.0)
            buffer = ""

            while not stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue

                data = os.read(fd, 64)
                if not data:
                    continue

                for ch in data.decode("utf-8", errors="ignore"):
                    if ch == "\n":
                        text = buffer.strip()
                        buffer = ""

                        if not text:
                            continue

                        try:
                            reading = parse_sensor_line(text)
                        except (ValueError, TypeError, json.JSONDecodeError) as exc:
                            print(f"[SERIAL] ignored malformed line: {text!r} ({exc})")
                            continue

                        if sensor_code_to_mode(reading.state_code) is None:
                            print(f"[SERIAL] ignored unknown state code: {reading.state_code}")
                            continue

                        with lock:
                            now = time.time()
                            changed = apply_sensor_reading(
                                state,
                                reading,
                                pir_hold_seconds=pir_hold_seconds,
                                now=now,
                            )
                            if changed:
                                hold_remaining = max(0.0, state.pir_hold_until - now)
                                hold_text = (
                                    f" hold={hold_remaining:.1f}s"
                                    if hold_remaining > 0.0
                                    else ""
                                )
                                print(
                                    f"[STATE] raw={sensor_code_to_label(reading.state_code)} "
                                    f"-> effective={sensor_code_to_label(state.sensor_code)}"
                                    f"{hold_text}"
                                )
                    elif ch != "\r":
                        buffer += ch
        except FileNotFoundError:
            print(f"[SERIAL] port not found: {port}; retrying...")
            time.sleep(1.0)
        except OSError as exc:
            print(f"[SERIAL] error: {exc}; retrying...")
            time.sleep(1.0)
        finally:
            if fd >= 0:
                os.close(fd)


def load_runtime_model(
    name: str,
    meta_path: str | Path,
    device: Any,
    smooth: int,
) -> LoadedModel:
    from src.common.ingredient_models import build_webcam_transform, load_model_from_meta

    print(f"[LOAD] {name}: {meta_path}")
    model, labels, _meta, resolved_meta, checkpoint_path, image_size = load_model_from_meta(
        meta_path,
        device,
    )
    transform = build_webcam_transform(image_size=image_size)

    print(f"[LOAD] {name}: classes={len(labels)} image_size={image_size}")
    print(f"[LOAD] {name}: checkpoint={checkpoint_path}")

    return LoadedModel(
        name=name,
        model=model,
        labels=labels,
        transform=transform,
        image_size=image_size,
        meta_path=resolved_meta,
        checkpoint_path=checkpoint_path,
        prob_queue=deque(maxlen=max(1, smooth)),
    )


def reset_runtime_model(runtime_model: LoadedModel) -> None:
    runtime_model.prob_queue.clear()
    runtime_model.last_topk = []
    runtime_model.last_label = "waiting"
    runtime_model.last_conf = 0.0
    runtime_model.last_infer_ms = 0.0
    runtime_model.candidate_label = None
    runtime_model.stable_count = 0
    runtime_model.last_sent_label = None
    runtime_model.last_sent_time = 0.0
    runtime_model.vote_counts.clear()
    runtime_model.vote_conf_sums.clear()
    runtime_model.vote_started_at = 0.0
    runtime_model.vote_deadline = 0.0
    runtime_model.vote_finalized = False
    runtime_model.vote_max_conf = 0.0


def start_liquor_vote(runtime_model: LoadedModel, now: float, deadline: float) -> None:
    runtime_model.vote_counts.clear()
    runtime_model.vote_conf_sums.clear()
    runtime_model.vote_started_at = now
    runtime_model.vote_deadline = deadline
    runtime_model.vote_finalized = False
    runtime_model.vote_max_conf = 0.0


def start_ingredient_vote(runtime_model: LoadedModel, now: float) -> None:
    runtime_model.vote_counts.clear()
    runtime_model.vote_conf_sums.clear()
    runtime_model.vote_started_at = now
    runtime_model.vote_deadline = 0.0
    runtime_model.vote_finalized = False
    runtime_model.vote_max_conf = 0.0
    runtime_model.last_sent_label = None
    runtime_model.last_sent_time = 0.0
    print("[VOTE] ingredient tracking started")


def record_top1_vote(
    runtime_model: LoadedModel,
    mode: str,
    min_confidence: float | None = None,
) -> bool:
    if not runtime_model.last_topk:
        return False

    top_label, top_conf = runtime_model.last_topk[0]
    runtime_model.vote_max_conf = max(runtime_model.vote_max_conf, top_conf)
    if min_confidence is not None and top_conf <= min_confidence:
        return False

    canonical_label = canonicalize_label(mode, top_label)
    runtime_model.vote_counts[canonical_label] += 1
    runtime_model.vote_conf_sums[canonical_label] = (
        runtime_model.vote_conf_sums.get(canonical_label, 0.0) + top_conf
    )
    return True


def record_liquor_vote(runtime_model: LoadedModel) -> None:
    record_top1_vote(runtime_model, MODE_LIQUOR)


def record_ingredient_vote(runtime_model: LoadedModel, min_confidence: float) -> None:
    record_top1_vote(runtime_model, MODE_INGREDIENT, min_confidence=min_confidence)


def vote_winner(runtime_model: LoadedModel) -> tuple[str, int, int, float] | None:
    total = sum(runtime_model.vote_counts.values())
    if total <= 0:
        return None

    def rank(item: tuple[str, int]) -> tuple[int, float, str]:
        label, count = item
        avg_conf = runtime_model.vote_conf_sums.get(label, 0.0) / max(1, count)
        return count, avg_conf, label

    winner_label, winner_count = max(runtime_model.vote_counts.items(), key=rank)
    avg_conf = runtime_model.vote_conf_sums.get(winner_label, 0.0) / max(1, winner_count)
    return winner_label, winner_count, total, avg_conf


def maybe_finalize_liquor_vote(
    runtime_model: LoadedModel,
    post_queue: queue.Queue[tuple[str, str, dict[str, Any]]] | None,
    liquor_url: str,
    scan_request_id: str,
    now: float,
) -> None:
    if runtime_model.vote_finalized:
        return
    if runtime_model.vote_deadline <= 0.0 or now < runtime_model.vote_deadline:
        return

    runtime_model.vote_finalized = True
    winner = vote_winner(runtime_model)
    if winner is None:
        print("[VOTE] liquor skipped: no predictions collected")
        return

    label, count, total, confidence = winner
    _endpoint_kind, payload = make_recognition_payload(
        mode=MODE_LIQUOR,
        label=label,
        confidence=confidence,
        scan_request_id=scan_request_id,
    )
    enqueue_post(post_queue, "recognition:liquor", liquor_url, payload)
    runtime_model.last_sent_label = label
    runtime_model.last_sent_time = now
    print(
        f"[VOTE] liquor winner={label} votes={count}/{total} "
        f"avg_conf={confidence * 100:.1f}%"
    )


def maybe_finalize_ingredient_inventory_event(
    runtime_model: LoadedModel,
    post_queue: queue.Queue[tuple[str, str, dict[str, Any]]] | None,
    inventory_url: str,
    direction: str,
    min_confidence: float,
    now: float,
) -> None:
    if runtime_model.vote_finalized:
        return

    runtime_model.vote_finalized = True
    if runtime_model.vote_max_conf <= min_confidence:
        print(
            f"[VOTE] ingredient skipped: max_conf={runtime_model.vote_max_conf * 100:.1f}% "
            f"<= required={min_confidence * 100:.1f}%"
        )
        return

    winner = vote_winner(runtime_model)
    if winner is None:
        print("[VOTE] ingredient skipped: no predictions collected")
        return

    label, count, total, confidence = winner
    payload = make_inventory_event_payload(
        label=label,
        direction=direction,
        confidence=confidence,
    )
    if payload is None:
        print(f"[VOTE] ingredient skipped: unclear direction={direction!r}")
        return

    enqueue_post(post_queue, "inventory:ingredient", inventory_url, payload)
    runtime_model.last_sent_label = label
    runtime_model.last_sent_time = now
    print(
        f"[VOTE] ingredient inventory={label} direction={direction} "
        f"action={payload['action']} votes={count}/{total} avg_conf={confidence * 100:.1f}%"
    )


def open_camera(camera: int, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_inference(
    runtime_model: LoadedModel,
    frame,
    device: Any,
    topk: int,
    threshold: float,
) -> None:
    import cv2
    import torch
    import torch.nn.functional as F

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = runtime_model.transform(rgb).unsqueeze(0).to(device)

    start = time.time()
    with torch.inference_mode():
        logits = runtime_model.model(tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu()
    runtime_model.last_infer_ms = (time.time() - start) * 1000.0

    runtime_model.prob_queue.append(probs)
    avg_probs = torch.stack(list(runtime_model.prob_queue), dim=0).mean(dim=0)

    k = min(max(1, topk), len(runtime_model.labels))
    values, indices = torch.topk(avg_probs, k=k)
    runtime_model.last_topk = [
        (runtime_model.labels[int(idx)], float(value))
        for value, idx in zip(values, indices)
    ]

    top_label, top_conf = runtime_model.last_topk[0]
    runtime_model.last_label = top_label if top_conf >= threshold else "uncertain"
    runtime_model.last_conf = top_conf


def shorten(text: str, max_chars: int = 34) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "."


def draw_overlay(
    frame,
    mode: str,
    code: int,
    models: dict[str, LoadedModel],
    fps: float,
    hold_remaining: float,
    overlay_topk: int,
    overlay_alpha: float,
) -> None:
    import cv2

    h, w = frame.shape[:2]
    mode_label = mode_to_label(mode)
    state_label = sensor_code_to_label(code)

    if mode == MODE_IDLE:
        lines = [
            f"Mode: {mode_label}",
            f"Sensor: {state_label}",
            "Inference paused",
            f"FPS: {fps:.1f}",
        ]
        color = (215, 215, 215)
    else:
        model = models[mode]
        hold_text = f" ({hold_remaining:.1f}s)" if mode == MODE_LIQUOR and hold_remaining > 0 else ""
        lines = [
            f"Mode: {mode_label}{hold_text}",
            f"Sensor: {state_label}",
            f"Pred: {shorten(model.last_label, 24)} {model.last_conf * 100:.1f}%",
            f"{model.last_infer_ms:.0f} ms | {fps:.1f} FPS",
        ]
        for rank, (label, prob) in enumerate(model.last_topk[: max(0, overlay_topk)], start=1):
            lines.append(f"{rank}. {shorten(label, 20)} {prob * 100:.1f}%")
        color = (80, 255, 120) if mode == MODE_INGREDIENT else (0, 220, 255)

    font_scale = 0.46
    thickness = 1
    line_h = 19
    pad_x = 10
    pad_y = 9
    text_width = 0
    for line in lines:
        (line_w, _line_h), _baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_width = max(text_width, line_w)

    panel_w = min(w - 12, max(210, text_width + pad_x * 2))
    panel_h = min(h - 12, pad_y * 2 + line_h * len(lines))
    x1, y1 = 8, 8
    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = min(max(overlay_alpha, 0.0), 1.0)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        overlay[y1:y2, x1:x2],
        alpha,
        frame[y1:y2, x1:x2],
        1.0 - alpha,
        0,
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, y1 + pad_y + 13 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_direction_label(frame, label: str) -> None:
    if not label:
        return

    import cv2

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.45
    thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = max(8, (w - text_w) // 2)
    y = min(h - 16, max(text_h + 16, int(h * 0.16)))

    color = (80, 255, 120) if label == "input" else (0, 220, 255)
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    import cv2
    import torch

    ingredient_meta = resolve_project_path(args.ingredient_meta)
    liquor_meta = resolve_project_path(args.liquor_meta)
    sensor_events_url = build_url(
        args.be_base_url,
        args.sensor_events_url,
        "/api/v1/sensors/events",
    )
    inventory_events_url = build_url(
        args.be_base_url,
        args.inventory_events_url,
        "/api/v1/inventory/events",
    )
    liquor_recognition_url = build_url(
        args.be_base_url,
        args.liquor_recognition_url,
        "/api/v1/recognitions/liquor",
    )

    if args.enable_be_post:
        missing = []
        if not sensor_events_url:
            missing.append("sensor events URL")
        if not inventory_events_url:
            missing.append("inventory events URL")
        if not liquor_recognition_url:
            missing.append("liquor recognition URL")
        if missing:
            raise SystemExit(
                "--enable-be-post requires --be-base-url or explicit endpoint URLs: "
                + ", ".join(missing)
            )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] project={PROJECT_ROOT}")
    print(f"[INFO] device={device}")

    models = {
        MODE_LIQUOR: load_runtime_model(MODE_LIQUOR, liquor_meta, device, args.smooth),
        MODE_INGREDIENT: load_runtime_model(MODE_INGREDIENT, ingredient_meta, device, args.smooth),
    }

    if args.warmup_only:
        print("[INFO] warmup complete")
        return

    post_queue: queue.Queue[tuple[str, str, dict[str, Any]]] | None = None
    post_stop_event = threading.Event()
    post_thread: threading.Thread | None = None
    if args.enable_be_post:
        post_queue = queue.Queue()
        post_thread = threading.Thread(
            target=post_worker,
            args=(post_queue, post_stop_event, args.post_timeout),
            daemon=True,
        )
        post_thread.start()
        print("[INFO] BE POST enabled")
        print(f"[INFO] sensor events URL: {sensor_events_url}")
        print(f"[INFO] inventory URL: {inventory_events_url}")
        print(f"[INFO] liquor URL: {liquor_recognition_url}")

    cap = open_camera(args.camera, args.width, args.height)
    state = RuntimeState()
    direction_tracker = DirectionTracker(
        enabled=args.enable_ingredient_direction,
        min_area=args.direction_min_area,
        max_area_ratio=args.direction_max_area_ratio,
        top_zone=args.direction_top_zone,
        bottom_zone=args.direction_bottom_zone,
        min_travel=args.direction_min_travel,
        lost_frames_required=args.direction_lost_frames,
        min_track_frames=args.direction_min_track_frames,
        display_seconds=args.direction_display_seconds,
    )
    lock = threading.Lock()
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=serial_reader,
        args=(args.serial_port, args.baud, state, lock, stop_event, args.pir_hold_seconds),
        daemon=True,
    )
    reader_thread.start()

    print("[INFO] state mapping: 0=Sleep, 1=Alcohol, 2=Ingredient")
    if args.enable_ingredient_direction:
        print(
            "[INFO] ingredient direction tracker enabled: "
            "top->bottom=input, bottom->top=output"
        )
    print("[INFO] q=quit" if not args.headless else "[INFO] Ctrl+C=quit")

    frame_idx = 0
    last_mode = MODE_IDLE
    last_log_t = 0.0
    last_sensor_posted_code: int | None = None
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] frame read failed")
                time.sleep(0.05)
                continue

            frame_idx += 1
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            with lock:
                expired = expire_pir_hold_if_needed(state, now)
                active_mode = state.active_mode
                sensor_code = state.sensor_code
                reading_seq = state.reading_seq
                pir_hold_until = state.pir_hold_until
                hold_remaining = max(0.0, state.pir_hold_until - now)
                sensor_payload = make_sensor_event_payload(
                    state,
                    device_id=args.device_id,
                    source=args.sensor_source,
                )

            sensor_changed = sensor_code != last_sensor_posted_code
            should_post_sensor = (
                args.enable_be_post
                and reading_seq > 0
                and sensor_changed
            )
            if should_post_sensor:
                if sensor_code != 0:
                    enqueue_post(post_queue, "sensor", sensor_events_url, sensor_payload)
                last_sensor_posted_code = sensor_code

            if active_mode != last_mode:
                if args.enable_be_post and last_mode == MODE_LIQUOR and active_mode != MODE_LIQUOR:
                    maybe_finalize_liquor_vote(
                        runtime_model=models[MODE_LIQUOR],
                        post_queue=post_queue,
                        liquor_url=liquor_recognition_url,
                        scan_request_id=args.scan_request_id,
                        now=now,
                    )
                if active_mode in models:
                    reset_runtime_model(models[active_mode])
                    if active_mode == MODE_LIQUOR:
                        start_liquor_vote(models[MODE_LIQUOR], now, pir_hold_until)
                if args.enable_ingredient_direction:
                    if last_mode == MODE_INGREDIENT and active_mode != MODE_INGREDIENT:
                        finalized_direction = direction_tracker.finalize_track(
                            frame_height=frame.shape[0],
                            now=now,
                        )
                        if args.enable_be_post:
                            maybe_finalize_ingredient_inventory_event(
                                runtime_model=models[MODE_INGREDIENT],
                                post_queue=post_queue,
                                inventory_url=inventory_events_url,
                                direction=finalized_direction,
                                min_confidence=args.ingredient_vote_min_confidence,
                                now=now,
                            )
                        direction_tracker.reset(clear_event=False)
                    else:
                        direction_tracker.reset(clear_event=True)
                last_mode = active_mode

            direction_label = direction_tracker.current_event(now)
            if args.enable_ingredient_direction and active_mode == MODE_INGREDIENT:
                direction_label = direction_tracker.update(frame, now)
                if direction_tracker.started_this_frame:
                    start_ingredient_vote(models[MODE_INGREDIENT], now)
                if direction_tracker.finished_this_frame and args.enable_be_post:
                    maybe_finalize_ingredient_inventory_event(
                        runtime_model=models[MODE_INGREDIENT],
                        post_queue=post_queue,
                        inventory_url=inventory_events_url,
                        direction=direction_tracker.finished_direction,
                        min_confidence=args.ingredient_vote_min_confidence,
                        now=now,
                    )

            should_infer = (
                active_mode in models
                and frame_idx % max(1, args.infer_every) == 0
            )

            if should_infer:
                run_inference(
                    models[active_mode],
                    frame,
                    device=device,
                    topk=args.topk,
                    threshold=args.threshold,
                )
                if args.enable_be_post:
                    if active_mode == MODE_LIQUOR:
                        record_liquor_vote(models[MODE_LIQUOR])
                    elif (
                        active_mode == MODE_INGREDIENT
                        and args.enable_ingredient_direction
                        and direction_tracker.active
                        and direction_tracker.is_visible_for_vote(
                            frame.shape,
                            args.ingredient_vote_visible_margin,
                        )
                    ):
                        record_ingredient_vote(
                            models[MODE_INGREDIENT],
                            min_confidence=args.ingredient_vote_min_confidence,
                        )

            if args.headless:
                if active_mode in models and now - last_log_t >= args.print_every:
                    model = models[active_mode]
                    print(
                        f"[PRED] mode={active_mode} code={sensor_code} "
                        f"state={sensor_code_to_label(sensor_code)} "
                        f"label={model.last_label} conf={model.last_conf:.3f} "
                        f"infer_ms={model.last_infer_ms:.1f}"
                    )
                    last_log_t = now
                continue

            display = frame.copy()
            draw_overlay(
                display,
                active_mode,
                sensor_code,
                models,
                fps,
                hold_remaining=hold_remaining,
                overlay_topk=args.overlay_topk,
                overlay_alpha=args.overlay_alpha,
            )
            draw_direction_label(display, direction_label)
            cv2.imshow(args.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        stop_event.set()
        post_stop_event.set()
        if post_queue is not None:
            post_queue.join()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
