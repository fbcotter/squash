import cv2
import numpy as np

SENSITIVITY_VALUE = 20
BLUR_SIZE = 11
ESC_KEY = 27
theObject = np.zeros((2,), np.float32)
debugMode = False
vid_source = '../data/bouncingBall.mp4'


def searchForMotion(thresholdImage, frame):
    """
    Look in a difference image for contours.

    This function assumes that we have a binary image which has highlighted the
    difference between two images.
    """
    objDetected = False
    global debugMode

    # Find the contours of the difference Image
    im, contours, hierarchy = cv2.findContours(
        thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If the contours object is not empty, then we have found some objects
    if len(contours) > 0:
        objDetected = True
    else:
        objDetected = False

    if debugMode:
        frame = cv2.drawContours(frame, contours, -1, (0,0,255), 1)

    if objDetected:
        # Sort the contours by length - this puts the longest contour at the end
        contours = sorted(contours, key=lambda x: cv2.arcLength(x, False))

        # Naively assume that the largest contour is the one we are interested
        # in.
        largestContourVec = contours[-1]
        x, y, w, h = cv2.boundingRect(largestContourVec)
        xcentre = int(x + w/2)
        ycentre = int(y + h/2)

        # update the object's position
        theObject[0], theObject[1] = xcentre, ycentre

        # draw some crosshairs on the object
        #  frame = add_crosshairs(frame, xcentre, ycentre)
        #  frame = cv2.putText(
            #  frame,
            #  text="Tracking object at ({:.1f}, {:.1f})".format(xcentre, ycentre),
            #  org=(x, y), fontFace=1, fontScale=1, color=(255, 0, 0), thickness=2)

    return frame


def add_crosshairs(frame, x, y):
    color = (0, 255, 0)
    radius = 20
    thickness = 2
    frame = cv2.circle(frame, (x, y), radius, color, thickness)
    frame = cv2.line(frame, (x, y-radius-5), (x, y+radius+5), color, thickness)
    frame = cv2.line(frame, (x-radius-5, y), (x+radius+5, y), color, thickness)

    return frame


def main():
    exit = False
    pause = False
    trackingMode = False
    global debugMode, vid_source

    while (not exit):
        cap = cv2.VideoCapture(vid_source)
        if not cap.isOpened():
            raise IOError('Could not open and read video file')
        ret, frame = cap.read()

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < \
              cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            # Copy the old frame out and read a new one in
            frame2 = frame
            ret, frame = cap.read()
            frame1 = frame

            # Convert both to grayscale for processing
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculate the difference image
            differenceImage = cv2.absdiff(gray1, gray2)
            _, thresholdImage = cv2.threshold(
                differenceImage, SENSITIVITY_VALUE, 255, cv2.THRESH_BINARY)
            # Show the difference image and threshold image
            if debugMode:
                cv2.imshow("Difference Image", differenceImage)
                cv2.imshow("Threshold Image", thresholdImage)
            else:
                # Destroy the windows so we don't see them anymore
                cv2.destroyWindow("Difference Image")
                cv2.destroyWindow("Threshold Image")

            # Blur the image to get rid of the noise
            thresholdImage = cv2.blur(thresholdImage, (BLUR_SIZE, BLUR_SIZE))
            #  thresholdImage = cv2.GaussianBlur(
                    #  thresholdImage, (BLUR_SIZE, BLUR_SIZE), 2.0)
            _, thresholdImage = cv2.threshold(
                thresholdImage, SENSITIVITY_VALUE, 255, cv2.THRESH_BINARY)
            if debugMode:
                cv2.imshow("Threshold Final", thresholdImage)
            else:
                cv2.destroyWindow("Threshold Final")

            # Detect motion
            if trackingMode:
                frame1 = searchForMotion(thresholdImage, frame1)

            cv2.imshow('frame', frame1)

            key = cv2.waitKey(10) & 0xFF

            if key == ESC_KEY:
                # Mark the outer loop to terminate
                exit = True
                break
            elif key == ord('t'):
                trackingMode = not trackingMode
                if trackingMode:
                    print('Tracking turned on')
                else:
                    print('Tracking turned off')
            elif key == ord('d'):
                debugMode = not debugMode
            elif key == ord('p'):
                print('Code Paused. Press "p" to resume')
                pause = True
                while pause:
                    key = cv2.waitKey() & 0xFF
                    if key == ord('p'):
                        print('Code Resumed')
                        pause = False
                pass

        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
