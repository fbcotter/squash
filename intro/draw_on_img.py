import cv2
import numpy as np

img = cv2.imread('out.jpg', cv2.IMREAD_COLOR)

# Remember, opencv uses BGR not RGB
cv2.line(img, (0,0), (150,150), (255,255,255), 5)
cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)
cv2.circle(img, (100, 63), 55, (0,0,255), -1)
pts = np.array([[10,5], [20,30], [70,20], [50,10], [60,40]])
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Tuts!', (0,130), font, 1, (200, 200, 200), 2, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
