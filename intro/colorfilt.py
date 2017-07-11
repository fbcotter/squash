import numpy as np
import cv2
from imutils.video import VideoStream

# Create a class to connect to the camera
resolution = (640, 480)
cam = VideoStream(usePiCamera=False, resolution=resolution)
# Start a separate thread for grabbing the feed
cam.start()

while True:
    frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    val = 100
    lower_red = np.array([0,val,0])
    upper_red = np.array([20,255,255])
    lower_red2 = np.array([350,val,0])
    upper_red2 = np.array([360,255,255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()

