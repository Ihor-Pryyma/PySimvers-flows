# PySimVerse Demo

Demo missions for controlling a `pysimverse` drone with scripted routes, keyboard input, and simple computer-vision behaviors.

## Repository Layout

- `run_menu.py` discovers missions under `missions/` and launches them from a terminal menu.
- `missions/common.py` contains shared drone startup and keyboard RC helpers.
- `missions/mission_1/` contains scripted garage route levels.
- `missions/mission_2/` contains an image capture mission using the drone video stream.
- `missions/mission_3/` contains a hand-position follower mission.
- `missions/mission_4/` contains a body follower mission.
- `missions/mission_5/` contains a red-line following mission.
- `missions/hand_gesture_mediapipe.py` is a standalone webcam-based MediaPipe hand detection demo reused by mission 3.

## Requirements

- Python 3.10+
- A working `pysimverse` environment and simulator/backend
- Webcam access for missions 3 and 4
- Drone video stream access for missions 2 and 5

Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current Python dependencies:

- `pysimverse`
- `opencv-python`
- `pynput`
- `mediapipe`
- `cvzone`

## Running

Launch the mission picker:

```bash
python run_menu.py
```

Menu controls:

- `Up` / `Down`
- `j` / `k`
- `Enter` to select
- `q` or `Esc` to go back or quit

You can also run a mission directly:

```bash
python -m missions.mission_1.garage_level_1
python -m missions.mission_2.image_capture_level_1
python -m missions.mission_3.hand_gesture_level_1
python -m missions.mission_4.body_follower_level_1
python -m missions.mission_5.line_following_level_1
```

Run the standalone hand detection demo:

```bash
python -m missions.hand_gesture_mediapipe
```

## Missions

### Mission 1: Garage

Three scripted route demos:

- `garage_level_1`
- `garage_level_2`
- `garage_level_3`

These levels use hardcoded movement sequences and land automatically at the end.

### Mission 2: Image Capture

Shows the drone camera feed and allows manual control from the keyboard.

Keyboard controls in `missions/common.py`:

- `w` / `s` move forward / backward
- `f` / `g` move right / left
- `q` / `e` move up / down
- `a` / `d` rotate
- `z` save a frame to `missions/captures/`
- `x` land and shut down

### Mission 3: Hand Gesture Level 1

Uses a webcam plus MediaPipe hand landmarks. Moving one hand into the left or right zone sends lateral RC commands to the drone.

### Mission 4: Body Follower Level 1

Uses a webcam plus MediaPipe pose tracking. The drone adjusts left/right, forward/back, and up/down to keep the body centered and at a target size.

### Mission 5: Line Following Level 1

Uses the drone video stream to detect and follow a red line. The overlay shows the ROI, detected contour, tracked centers, and current RC command.

## MediaPipe Models

Mission 3, mission 4, and `missions.hand_gesture_mediapipe` expect local model files under `missions/models/`.

If a required model file is missing, the script attempts to download it automatically on first run:

- `missions/models/hand_landmarker.task`
- `missions/models/pose_landmarker.task`

If automatic download fails, place the model file at the expected path and rerun the script.

## Adding Missions

`run_menu.py` automatically discovers Python files inside `missions/<mission_name>/`.

To add a mission level:

1. Create a new module under an existing mission folder, or add a new mission folder under `missions/`.
2. Add a `main()` entry point.
3. Run `python run_menu.py`.

Mission and level names shown in the menu are generated from folder and file names by replacing underscores with spaces and title-casing them.

## Notes

- `run_menu.py` uses `curses`, so it works best in a Unix-like terminal.
- Most missions take off automatically and try to land and shut down in cleanup.
- Vision-based missions open OpenCV windows and quit with `q` or `Esc`.
