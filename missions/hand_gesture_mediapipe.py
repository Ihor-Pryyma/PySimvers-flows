from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
WINDOW_NAME = "MediaPipe Hand Detection"
TEXT_COLOR = (0, 255, 0)
CAMERA_INDEX = 0
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def create_detector(model_path: Path, num_hands: int) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


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
            "Could not download the MediaPipe hand model.\n"
            f"Tried: {MODEL_URL}\n"
            f"Expected local path: {model_path}\n"
            "Download the file manually and place it there, then rerun the script.\n"
            f"Original error: {exc}"
        ) from exc

    return model_path


def finger_is_extended(
    landmarks: list, tip_idx: int, pip_idx: int
) -> bool:
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def thumb_is_extended(
    landmarks: list, handedness: str
) -> bool:
    thumb_tip_x = landmarks[4].x
    thumb_ip_x = landmarks[3].x
    if handedness.lower() == "right":
        return thumb_tip_x < thumb_ip_x
    return thumb_tip_x > thumb_ip_x


def classify_gesture(
    landmarks: list, handedness: str
) -> str:
    thumb = thumb_is_extended(landmarks, handedness)
    index = finger_is_extended(landmarks, 8, 6)
    middle = finger_is_extended(landmarks, 12, 10)
    ring = finger_is_extended(landmarks, 16, 14)
    pinky = finger_is_extended(landmarks, 20, 18)

    if all([thumb, index, middle, ring, pinky]):
        return "Open palm"
    if not any([thumb, index, middle, ring, pinky]):
        return "Fist"
    if index and middle and not any([ring, pinky]):
        return "Peace"
    if index and not any([middle, ring, pinky]):
        return "Pointing"
    if thumb and not any([index, middle, ring, pinky]):
        return "Thumbs up"
    return "Hand detected"


def draw_landmarks_on_frame(
    frame: cv2.typing.MatLike, detection_result: vision.HandLandmarkerResult
) -> cv2.typing.MatLike:
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape

    for hand_landmarks, handedness_list in zip(
        detection_result.hand_landmarks, detection_result.handedness
    ):
        points = [
            (int(lm.x * width), int(lm.y * height))
            for lm in hand_landmarks
        ]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                annotated_frame,
                points[start_idx],
                points[end_idx],
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(annotated_frame, point, 4, (0, 200, 255), -1, cv2.LINE_AA)

        handedness = handedness_list[0].category_name
        gesture = classify_gesture(hand_landmarks, handedness)

        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = max(30, int(min(y_coords) * height) - 10)

        cv2.putText(
            annotated_frame,
            f"{handedness}: {gesture}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


def main() -> None:
    model_path = ensure_model_exists(MODEL_PATH)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {CAMERA_INDEX}.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with create_detector(model_path, 1) as detector:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.monotonic() * 1000)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            annotated_frame = draw_landmarks_on_frame(frame, detection_result)

            cv2.putText(
                annotated_frame,
                "Press q or Esc to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
