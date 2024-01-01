""" dronetest.py
Test file for drone.
Tests all functions called by main.py
Procedure:
    * drone connection
    * drone takes off
    * drone moves left
    * drone moves right
    * drone moves up
    * drone moves down
    * drone lands
Caller files: None
Called files: None
"""

from djitellopy import Tello
import cv2
import time

def moveLeft(drone: Tello):
    """ Function moveLeft.
        Rotates the drone 90 degrees counter clockwise.
    """
    drone.rotate_counter_clockwise(90)

def moveRight(drone: Tello):
    """ Function moveRight.
        Rotates the drone 90 degrees clockwise.
    """
    drone.rotate_clockwise(90)
    
def moveUp(drone: Tello):
    """ Function moveUp.
        Flies drone 20cm up until MAX_HEIGHT.
        
        MAX_HEIGHT = 180cm
    """
    if (drone.get_height() < 180):
        drone.move_up(20)
    else:
        raise Exception("Drone maximum height!")

def moveDown(drone: Tello):
    """ Function moveDown.
        Flies drone 20cm down until MIN_HEIGHT.
        
        MIN_HEIGHT = 30cm
    """
    if (drone.get_height() > 30):
        drone.move_down(20)
    else:
        raise Exception("Drone minimum height!")


tello = Tello()
tello.connect()

print("Battery: ", tello.get_battery())

tello.takeoff()
height = tello.get_height()
print("Height: ", height)

time.sleep(5)

moveLeft(tello)
height = tello.get_height()
print("Height: ", height)

time.sleep(5)

moveRight(tello)
height = tello.get_height()
print("Height: ", height)

for _ in range (10):
    try:
        moveUp(tello)
    except Exception as maxHeightError:
        print(maxHeightError)
    height = tello.get_height()
    print("Height: ", height)
    time.sleep(5)
    
for _ in range (10):
    try:
        moveDown(tello)
    except Exception as minHeightError:
        print(minHeightError)
    height = tello.get_height()
    print("Height: ", height)
    time.sleep(5)

tello.land()
height = tello.get_height()
print("Height: ", height)