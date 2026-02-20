import cv2 as cv

img = cv.imread("C:\\Users\\death\\Downloads\\images.png")
cv.imshow('img',img)
cv.waitKey()
cv.destroyAllWindows()