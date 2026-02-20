import cv2 as cv
import numpy as np 

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if(ret):
        r,g,b=cv.split(img)
        cv.imshow('Videor',r)
        cv.imshow('Videog',g)
        cv.imshow('Videob',b)
        cv.imshow('Video',img)
    else:
        print('No se pudo jefe ')
        break
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()