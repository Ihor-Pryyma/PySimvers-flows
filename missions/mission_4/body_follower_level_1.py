from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from missions.common import SPEED, init_drone

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
WINDOW_NAME = "Mission 4 / Body Follower Level 1"
CAMERA_INDEX = 0
TEXT_COLOR = (0, 255, 0)
POSE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 200, 255)
CENTER_COLOR = (0, 0, 255)
ZONE_COLOR = (80, 220, 220)
LEFT_RIGHT_DEADZONE = (0.4, 0.6)
UP_DOWN_DEADZONE = (0.35, 0.65)
TARGET_SHOULDER_WIDTH = (0.18, 0.30)
TRACKED_LANDMARKS = (
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    23,  # left hip
    24,  # right hip
)
POSE_CONNECTIONS = (
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
)


def ensure_model_exists(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model file not found. Downloading to {model_path} ...")

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            model_path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            "Could not download the MediaPipe pose model.\n"
            f"Tried: {MODEL_URL}\n"
            f"Expected local path: {model_path}\n"
            "Download the file manually and place it there, then rerun the script.\n"
            f"Original error: {exc}"
        ) from exc

    return model_path


def create_detector(model_path: Path) -> vision.PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_body_metrics(
    detection_result: vision.PoseLandmarkerResult,
) -> tuple[float | None, float | None, float | None]:
    if not detection_result.pose_landmarks:
        return None, None, None

    landmarks = detection_result.pose_landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    body_center_x = (
        left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x
    ) / 4
    body_center_y = (
        left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y
    ) / 4
    shoulder_width = abs(right_shoulder.x - left_shoulder.x)

    return body_center_x, body_center_y, shoulder_width


def lateral_speed_for_center_x(body_center_x: float | None) -> int:
    if body_center_x is None:
        return 0
    if body_center_x < LEFT_RIGHT_DEADZONE[0]:
        return -SPEED
    if body_center_x > LEFT_RIGHT_DEADZONE[1]:
        return SPEED
    return 0


def vertical_speed_for_center_y(body_center_y: float | None) -> int:
    if body_center_y is None:
        return 0
    if body_center_y < UP_DOWN_DEADZONE[0]:
        return SPEED
    if body_center_y > UP_DOWN_DEADZONE[1]:
        return -SPEED
    return 0


def forward_speed_for_body_size(shoulder_width: float | None) -> int:
    if shoulder_width is None:
        return 0
    if shoulder_width < TARGET_SHOULDER_WIDTH[0]:
        return SPEED
    if shoulder_width > TARGET_SHOULDER_WIDTH[1]:
        return -SPEED
    return 0


def command_label(lr: int, fb: int, ud: int) -> str:
    parts: list[str] = []
    if lr < 0:
        parts.append("LEFT")
    elif lr > 0:
        parts.append("RIGHT")

    if fb > 0:
        parts.append("FORWARD")
    elif fb < 0:
        parts.append("BACK")

    if ud > 0:
        parts.append("UP")
    elif ud < 0:
        parts.append("DOWN")

    return " + ".join(parts) if parts else "HOLD"


def draw_overlay(
    frame: cv2.typing.MatLike,
    detection_result: vision.PoseLandmarkerResult,
    body_center_x: float | None,
    body_center_y: float | None,
    shoulder_width: float | None,
    command_text: str,
) -> cv2.typing.MatLike:
    annotated = frame.copy()
    height, width, _ = annotated.shape

    left_boundary = int(width * LEFT_RIGHT_DEADZONE[0])
    right_boundary = int(width * LEFT_RIGHT_DEADZONE[1])
    top_boundary = int(height * UP_DOWN_DEADZONE[0])
    bottom_boundary = int(height * UP_DOWN_DEADZONE[1])

    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (left_boundary, top_boundary),
        (right_boundary, bottom_boundary),
        ZONE_COLOR,
        -1,
    )
    annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)

    cv2.rectangle(
        annotated,
        (left_boundary, top_boundary),
        (right_boundary, bottom_boundary),
        (255, 255, 255),
        2,
    )

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        points = {
            idx: (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
            for idx in TRACKED_LANDMARKS
        }

        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(
                annotated,
                points[start_idx],
                points[end_idx],
                POSE_COLOR,
                2,
                cv2.LINE_AA,
            )

        for point in points.values():
            cv2.circle(annotated, point, 5, JOINT_COLOR, -1, cv2.LINE_AA)

    if body_center_x is not None and body_center_y is not None:
        center = (int(body_center_x * width), int(body_center_y * height))
        cv2.circle(annotated, center, 9, CENTER_COLOR, -1, cv2.LINE_AA)

    size_text = (
        f"Shoulder width: {shoulder_width:.2f}" if shoulder_width is not None else "No body"
    )
    cv2.putText(
        annotated,
        f"Detected: {command_text}",
        (10, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        size_text,
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    return annotated


def main() -> None:
    model_path = ensure_model_exists(MODEL_PATH)
    drone = init_drone()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        drone.shutdown()
        raise SystemExit(f"Could not open camera index {CAMERA_INDEX}.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with create_detector(model_path) as detector:
            while True:
                is_success, frame = camera.read()
                if not is_success or frame is None:
                    print("Failed to read a frame from camera 0.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)

                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                body_center_x, body_center_y, shoulder_width = extract_body_metrics(
                    detection_result
                )

                lr = lateral_speed_for_center_x(body_center_x)
                fb = forward_speed_for_body_size(shoulder_width)
                ud = vertical_speed_for_center_y(body_center_y)
                drone.send_rc_control(lr, fb, ud, 0)

                command_text = command_label(lr, fb, ud)
                annotated_frame = draw_overlay(
                    frame,
                    detection_result,
                    body_center_x,
                    body_center_y,
                    shoulder_width,
                    command_text,
                )
                cv2.putText(
                    annotated_frame,
                    "Center your body in the box. Press q or Esc to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        camera.release()
        cv2.destroyAllWindows()
        if drone.is_flying:
            drone.land()
        drone.shutdown()


if __name__ == "__main__":
    main()
