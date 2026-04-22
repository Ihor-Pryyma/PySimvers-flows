from __future__ import annotations

import time

import cv2
import numpy as np
from pysimverse import Drone
from cvzone.ColorModule import ColorFinder

from missions.common import MAX_RC, MIN_RC

WINDOW_NAME = "Mission 5 / Line Following Level 1"
TEXT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 255, 255)
ROI_TOP_RATIO = 0.25
MIN_CONTOUR_AREA = 900
LOST_LINE_TIMEOUT = 1.2
POST_TAKEOFF_SETTLE_TIME = 1.0

# Red wraps around the HSV hue range, so we combine two cvzone masks.
RED_RANGE_LOW = {
    "hmin": 0,
    "smin": 120,
    "vmin": 70,
    "hmax": 10,
    "smax": 255,
    "vmax": 255,
}
RED_RANGE_HIGH = {
    "hmin": 170,
    "smin": 120,
    "vmin": 70,
    "hmax": 179,
    "smax": 255,
    "vmax": 255,
}


def clamp_rc(value: float) -> int:
    return max(MIN_RC, min(MAX_RC, int(value)))


def build_red_mask(
    frame: cv2.typing.MatLike, color_finder: ColorFinder
) -> cv2.typing.MatLike:
    _, low_mask = color_finder.update(frame, RED_RANGE_LOW)
    _, high_mask = color_finder.update(frame, RED_RANGE_HIGH)
    mask = cv2.bitwise_or(low_mask, high_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_largest_line_contour(
    mask: cv2.typing.MatLike,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA
    ]
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    return contour, cv2.boundingRect(contour)


def band_center(
    mask: cv2.typing.MatLike, top: int, bottom: int
) -> tuple[int, int] | None:
    band = mask[top:bottom, :]
    moments = cv2.moments(band)
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"]) + top
    return center_x, center_y


def compute_strip_centers(mask: cv2.typing.MatLike) -> list[tuple[int, int]]:
    height = mask.shape[0]
    step = height // 3
    centers: list[tuple[int, int]] = []

    for idx in range(3):
        top = idx * step
        bottom = height if idx == 2 else (idx + 1) * step
        center = band_center(mask, top, bottom)
        if center is not None:
            centers.append(center)

    return centers


def line_controls(
    centers: list[tuple[int, int]],
    frame_width: int,
) -> tuple[int, int, int]:
    if not centers:
        return 0, 0, 0

    bottom_center = max(centers, key=lambda point: point[1])
    target_x = frame_width // 2
    lateral_error = (bottom_center[0] - target_x) / target_x

    heading_error = 0.0
    if len(centers) >= 2:
        top_center = min(centers, key=lambda point: point[1])
        heading_error = (top_center[0] - bottom_center[0]) / target_x

    lr = clamp_rc(lateral_error * 85)
    yaw = clamp_rc((lateral_error * 30) + (heading_error * 75))

    correction_strength = max(abs(lateral_error), abs(heading_error))
    if correction_strength < 0.08:
        fb = 100
    elif correction_strength < 0.18:
        fb = 85
    elif correction_strength < 0.30:
        fb = 65
    else:
        fb = 45

    return lr, fb, yaw


def draw_overlay(
    frame: cv2.typing.MatLike,
    roi_top: int,
    mask: cv2.typing.MatLike,
    contour: np.ndarray | None,
    bounding_box: tuple[int, int, int, int] | None,
    centers: list[tuple[int, int]],
    command_text: str,
    line_visible: bool,
) -> cv2.typing.MatLike:
    annotated = frame.copy()
    height, width, _ = annotated.shape

    cv2.rectangle(annotated, (0, roi_top), (width, height), (255, 255, 255), 2)
    cv2.line(annotated, (width // 2, roi_top), (width // 2, height), TARGET_COLOR, 1)

    if contour is not None:
        shifted = contour + np.array([[[0, roi_top]]])
        cv2.drawContours(annotated, [shifted], -1, LINE_COLOR, 3)

    if bounding_box is not None:
        x, y, w, h = bounding_box
        cv2.rectangle(
            annotated, (x, y + roi_top), (x + w, y + h + roi_top), (0, 255, 255), 2
        )

    shifted_centers = [(x, y + roi_top) for x, y in centers]
    for center in shifted_centers:
        cv2.circle(annotated, center, 7, (0, 255, 255), -1, cv2.LINE_AA)

    if len(shifted_centers) >= 2:
        for start, end in zip(shifted_centers, shifted_centers[1:]):
            cv2.line(annotated, start, end, (255, 200, 0), 2, cv2.LINE_AA)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.resize(mask_bgr, (width // 3, (height - roi_top) // 3))
    preview_h, preview_w = mask_bgr.shape[:2]
    annotated[10 : 10 + preview_h, width - preview_w - 10 : width - 10] = mask_bgr

    cv2.putText(
        annotated,
        f"Line: {'DETECTED' if line_visible else 'SEARCHING'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Command: {command_text}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Following the red line. Press q or Esc to quit",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    return annotated


def main() -> None:
    drone = Drone()
    drone.connect()
    time.sleep(1)
    drone.streamon()
    drone.take_off(takeoff_height=30)
    time.sleep(POST_TAKEOFF_SETTLE_TIME)
    color_finder = ColorFinder(False)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    last_seen_at = time.monotonic()
    last_turn_direction = 0
    line_seen_once = False

    try:
        while True:
            frame, is_success = drone.get_frame()
            if not is_success or frame is None:
                print("Failed to read a frame from the drone stream.")
                drone.send_rc_control(0, 0, 0, 0)
                continue

            height, width, _ = frame.shape
            roi_top = int(height * ROI_TOP_RATIO)
            roi = frame[roi_top:, :]

            mask = build_red_mask(roi, color_finder)
            contour, bounding_box = find_largest_line_contour(mask)
            contour_mask = np.zeros_like(mask)
            if contour is not None:
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            centers = compute_strip_centers(contour_mask)

            if contour is not None and centers:
                last_seen_at = time.monotonic()
                line_seen_once = True
                lr, fb, yaw = line_controls(centers, width)
                if yaw != 0:
                    last_turn_direction = 1 if yaw > 0 else -1
                command_text = f"lr={lr} fb={fb} yaw={yaw}"
            else:
                lost_for = time.monotonic() - last_seen_at
                if not line_seen_once:
                    lr, fb, yaw = 0, 0, 0
                elif lost_for < LOST_LINE_TIMEOUT:
                    lr, fb, yaw = 0, 35, 25 * last_turn_direction
                else:
                    lr, fb, yaw = 0, 0, 20 * last_turn_direction
                command_text = f"search lr={lr} fb={fb} yaw={yaw}"

            drone.send_rc_control(lr, fb, 0, yaw)
            annotated = draw_overlay(
                frame,
                roi_top,
                mask,
                contour,
                bounding_box,
                centers,
                command_text,
                contour is not None and bool(centers),
            )
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        cv2.destroyAllWindows()
        if drone.is_flying:
            drone.land()
        drone.shutdown()


if __name__ == "__main__":
    main()
