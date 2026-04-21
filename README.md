# PySimVerse Demo

Small demo project for flying a `pysimverse` drone through scripted examples and simple mission levels.

## What is in this repo

- `run_menu.py` - a terminal menu that discovers missions under `missions/` and runs the selected level.
- `missions/common.py` - shared drone initialization helper.
- `missions/mission_1/` - three garage mission levels with different hardcoded routes.

## Requirements

- Python 3.10+ recommended
- `pysimverse` installed in your environment
- A working PySimVerse simulator setup or compatible drone backend

This repository does not currently include a `requirements.txt` or `pyproject.toml`, so dependency installation is expected to be handled manually.

## Setup

Create and activate a virtual environment if needed, then install the required package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pysimverse
```

## Running the demos

Run the first scripted flight:

```bash
python first_flight.py
```

Run the mission picker:

```bash
python run_menu.py
```

Menu controls:

- `↑` / `↓` or `k` / `j` to move
- `Enter` to select
- `q` or `Esc` to go back or quit

Run the RC loop:

```bash
python rc_control.py
```

`rc_control.py` currently sends a constant forward command in an infinite loop. Stop it manually with `Ctrl+C` or terminate the process from your terminal.

## Mission layout

Mission levels are discovered automatically from Python files inside `missions/<mission_name>/`.

Current missions:

- `Mission 1 / Garage Level 1`
- `Mission 1 / Garage Level 2`
- `Mission 1 / Garage Level 3`

To add a new mission or level:

1. Create a new folder under `missions/`, or add a new `.py` file to an existing mission folder.
2. Add a `main()` function and an `if __name__ == "__main__": main()` entry point.
3. Launch `python run_menu.py` and the new level will appear automatically.

## Notes

- `run_menu.py` uses Python `curses`, so it works best in a Unix-like terminal environment.
- Mission scripts rely on `missions.common.init_drone()` which connects, takes off, and sets the speed to `100`.
- Flight paths and rotations are currently hardcoded for demo purposes.
