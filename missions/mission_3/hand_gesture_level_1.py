from __future__ import annotations

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from missions.common import SPEED, init_drone
from missions.hand_gesture_mediapipe import (
    HAND_CONNECTIONS,
    MODEL_PATH,
    create_detector,
    ensure_model_exists,
)

WINDOW_NAME = "Mission 3 / Hand Gesture Level 1"
CAMERA_INDEX = 0
TEXT_COLOR = (0, 255, 0)
LEFT_COLOR = (80, 80, 220)
CENTER_COLOR = (80, 220, 220)
RIGHT_COLOR = (80, 220, 80)


def get_hand_center_x(detection_result: vision.HandLandmarkerResult) -> float | None:
    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]
    x_coords = [landmark.x for landmark in hand_landmarks]
    return sum(x_coords) / len(x_coords)


def classify_horizontal_zone(hand_center_x: float | None) -> str | None:
    if hand_center_x is None:
        return None

    if hand_center_x < 0.35:
        return "left"
    if hand_center_x > 0.65:
        return "right"
    return None


def lateral_speed_for_command(command: str | None) -> int:
    if command == "left":
        return -SPEED
    if command == "right":
        return SPEED
    return 0


def draw_zone_overlay(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    height, width, _ = frame.shape
    left_boundary = int(width * 0.35)
    right_boundary = int(width * 0.65)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (left_boundary, height), LEFT_COLOR, -1)
    cv2.rectangle(
        overlay, (left_boundary, 0), (right_boundary, height), CENTER_COLOR, -1
    )
    cv2.rectangle(overlay, (right_boundary, 0), (width, height), RIGHT_COLOR, -1)
    annotated = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    cv2.line(annotated, (left_boundary, 0), (left_boundary, height), (255, 255, 255), 2)
    cv2.line(
        annotated, (right_boundary, 0), (right_boundary, height), (255, 255, 255), 2
    )

    cv2.putText(
        annotated,
        "LEFT",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "DEADZONE",
        (left_boundary + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "RIGHT",
        (right_boundary + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def draw_hand(
    frame: cv2.typing.MatLike, detection_result: vision.HandLandmarkerResult
) -> cv2.typing.MatLike:
    annotated = draw_zone_overlay(frame)
    height, width, _ = annotated.shape

    if not detection_result.hand_landmarks:
        return annotated

    hand_landmarks = detection_result.hand_landmarks[0]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            annotated,
            points[start_idx],
            points[end_idx],
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(annotated, point, 4, (0, 200, 255), -1, cv2.LINE_AA)

    center_x = int(sum(point[0] for point in points) / len(points))
    center_y = int(sum(point[1] for point in points) / len(points))
    cv2.circle(annotated, (center_x, center_y), 8, (0, 0, 255), -1, cv2.LINE_AA)

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
        with create_detector(model_path, 1) as detector:
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
                hand_center_x = get_hand_center_x(detection_result)
                command = classify_horizontal_zone(hand_center_x)
                drone.send_rc_control(lateral_speed_for_command(command), 0, 0, 0)

                annotated_frame = draw_hand(frame, detection_result)
                status_text = command.upper() if command else "DEADZONE"
                cv2.putText(
                    annotated_frame,
                    f"Detected: {status_text}",
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    "Move one hand left or right. Press q or Esc to quit",
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
