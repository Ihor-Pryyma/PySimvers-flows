from missions.common import init_drone


def main():
    dron = init_drone()

    dron.rotate(-90)
    dron.move_forward(300)

    dron.rotate(155)
    dron.move_forward(400)

    dron.move_forward(380)

    dron.land()


if __name__ == "__main__":
    main()
