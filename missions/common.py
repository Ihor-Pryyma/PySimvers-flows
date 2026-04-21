from pysimverse import Drone


def init_drone():
    """Initialize the drone and take off."""
    drone = Drone()
    drone.connect()
    drone.take_off()
    drone.set_speed(100)
    return drone
