import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
PATH = os.path.dirname(os.path.realpath(__file__))

print(os.listdir(PATH))
img = cv2.imread(os.path.join(PATH,'out.jpg'), cv2.IMREAD_GRAYSCALE)
img2 = np.array(img)
print(img2.shape)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
