import cv2 as cv

img = cv.imread('imagenPrueba.jpg')
ra = (0, 60, 60)
rb = 


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

