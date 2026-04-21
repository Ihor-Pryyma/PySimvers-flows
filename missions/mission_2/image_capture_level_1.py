from missions.common import init_drone, keyboard_control
import cv2


def main():
    drone = init_drone()

    try:
        while True:
            frame, is_success = drone.get_frame()
            if is_success or frame is not None:
                cv2.imshow("Drone Feed", frame)
            cv2.waitKey(1)
            keyboard_control(drone, frame)
    finally:
        cv2.destroyAllWindows()
        if drone.is_flying:
            drone.land()
        drone.shutdown()


if __name__ == "__main__":
    main()
