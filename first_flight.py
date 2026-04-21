import time

from pysimverse import Drone

drone = Drone()
drone.connect()
drone.take_off()

drone.set_speed(50)
drone.move_forward(80)
time.sleep(2)
drone.move_backward(50)
time.sleep(2)

drone.rotate(5)

drone.land()
time.sleep(1)
