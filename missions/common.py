import time

import cv2

from datetime import datetime
from pathlib import Path

from pynput import keyboard

from pysimverse import Drone

MIN_RC = -100
MAX_RC = 100
SPEED = 100
ROTATION_SPEED = 3

CAPTUE_DIR = Path("missions/captures")

pressed_keys = set()


def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pressed_keys.add(key)


def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pressed_keys.discard(key)


listener = keyboard.Listener(on_press=on_press, on_release=on_release)


def init_drone():
    """Initialize the drone and take off."""
    CAPTUE_DIR.mkdir(parents=True, exist_ok=True)

    drone = Drone()
    drone.connect()
    time.sleep(1)
    drone.streamon()
    drone.take_off()
    listener.start()

    return drone


def keyboard_control(drone: Drone, frame) -> None:
    """Control the drone using keyboard input."""
    global pressed_keys
    try:
        lr = SPEED if "f" in pressed_keys else 0
        if "g" in pressed_keys:
            lr = -SPEED
        fb = SPEED if "w" in pressed_keys else 0
        if "s" in pressed_keys:
            fb = -SPEED
        ud = SPEED if "q" in pressed_keys else 0
        if "e" in pressed_keys:
            ud = -SPEED
        yaw = ROTATION_SPEED if "a" in pressed_keys else 0
        if "d" in pressed_keys:
            yaw = -ROTATION_SPEED

        if "z" in pressed_keys and frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = CAPTUE_DIR / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Captured image saved to {path}")

        if "x" in pressed_keys:
            if drone.is_flying:
                drone.land()
            drone.shutdown()

        drone.send_rc_control(lr, fb, ud, yaw)
        time.sleep(0.05)
    except KeyboardInterrupt:
        pass
