from missions.common import init_drone


def main():
    drone = init_drone()
    drone.move_forward(300)
    drone.move_right(300)
    drone.land()


if __name__ == "__main__":
    main()
