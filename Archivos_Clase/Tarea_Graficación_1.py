import cv2 as cv
import numpy as np 

img = np.ones((400,400), np.uint8)*255
i=j=0
for i in range(180, 200):
    for j in range(180, 200):
        img[i,j]=180

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()