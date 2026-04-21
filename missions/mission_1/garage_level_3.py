from missions.common import init_drone


def main():
    dron = init_drone()

    dron.rotate(-35)
    dron.move_forward(500)

    dron.rotate(155)
    dron.move_down(60)
    dron.move_forward(600)

    dron.land()


if __name__ == "__main__":
    main()
