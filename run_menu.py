from __future__ import annotations

import curses
import runpy
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MISSIONS_DIR = ROOT / "missions"


@dataclass(frozen=True)
class MissionModule:
    mission_label: str
    level_label: str
    module_name: str


def prettify(name: str) -> str:
    """Makes a name more human-friendly by replacing underscores with spaces and capitalizing words."""
    return name.replace("_", " ").title()


def discover_missions() -> list[MissionModule]:
    """Scans the missions directory for Python files and returns a list of MissionModule instances."""
    modules: list[MissionModule] = []

    for mission_dir in sorted(MISSIONS_DIR.iterdir()):
        if not mission_dir.is_dir() or mission_dir.name.startswith("__"):
            continue

        mission_label = prettify(mission_dir.name)
        for level_file in sorted(mission_dir.glob("*.py")):
            if level_file.stem == "__init__":
                continue

            modules.append(
                MissionModule(
                    mission_label=mission_label,
                    level_label=prettify(level_file.stem),
                    module_name=".".join(
                        level_file.relative_to(ROOT).with_suffix("").parts
                    ),
                )
            )

    return modules


def draw_menu(
    stdscr: curses.window, title: str, options: list[str], selected_idx: int
) -> None:
    """Draws a simple menu with the given title and options, highlighting the selected index."""
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    title_x = max(0, (width - len(title)) // 2)
    stdscr.addstr(1, title_x, title, curses.A_BOLD)
    stdscr.addstr(3, 2, "Use ↑/↓ to move, Enter to select, q to go back/quit.")

    for idx, option in enumerate(options):
        y = 5 + idx
        if y >= height - 1:
            break

        mode = curses.A_REVERSE if idx == selected_idx else curses.A_NORMAL
        stdscr.addstr(y, 4, option[: max(1, width - 8)], mode)

    stdscr.refresh()


def pick_option(stdscr: curses.window, title: str, options: list[str]) -> int | None:
    """Shows a menu and allows the user to pick an option. Returns the index of the selected option or None if cancelled."""
    selected_idx = 0

    while True:
        draw_menu(stdscr, title, options, selected_idx)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")):
            selected_idx = (selected_idx - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected_idx = (selected_idx + 1) % len(options)
        elif key in (curses.KEY_ENTER, 10, 13):
            return selected_idx
        elif key in (ord("q"), 27):
            return None


def mission_menu(
    stdscr: curses.window, modules: list[MissionModule]
) -> MissionModule | None:
    """
    Gets the user to select a mission and level from the list of modules.
    Returns the selected MissionModule or None if cancelled.
    """
    mission_names = sorted({module.mission_label for module in modules})
    mission_idx = pick_option(stdscr, "Choose Mission", mission_names)
    if mission_idx is None:
        return None

    chosen_mission = mission_names[mission_idx]
    mission_modules = [
        module for module in modules if module.mission_label == chosen_mission
    ]
    level_names = [module.level_label for module in mission_modules]

    level_idx = pick_option(stdscr, f"{chosen_mission}: Choose Level", level_names)
    if level_idx is None:
        return mission_menu(stdscr, modules)

    return mission_modules[level_idx]


def run_selected_module(module: MissionModule) -> None:
    """Runs the selected mission module using runpy."""
    print(
        f"Running {module.mission_label} / {module.level_label} ({module.module_name})"
    )
    runpy.run_module(module.module_name, run_name="__main__")


def main() -> None:
    modules = discover_missions()
    if not modules:
        raise SystemExit("No mission modules were found in the missions directory.")

    selected = curses.wrapper(mission_menu, modules)
    if selected is None:
        print("No mission selected.")
        return

    run_selected_module(selected)


if __name__ == "__main__":
    main()
