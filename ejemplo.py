import cv2 as cv
import numpy as np


img2 = np.ones((400,400), np.uint8)

img = cv.imread('ImagenPrueba.jpg', 0 )
print(img.shape)


cv.imshow('img2', img)
cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()
