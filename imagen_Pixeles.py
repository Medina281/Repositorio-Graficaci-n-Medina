import cv2 as cv
import numpy as np

img = np.ones([400,400],np.unt8)*255
img[1,1]=0
cv.imgshow('imagen',img)
cv.waitkey(0)
cv.destroyAllWindow()
