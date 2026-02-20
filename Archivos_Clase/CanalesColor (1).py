import numpy as np
import cv2 as cv

img1 = cv.imread(r"C:\Users\death\Downloads\images.png")

if img1 is None:
    print("No se pudo cargar la imagen. Revisa la ruta.")
    exit()

x, y, z = img1.shape
print(x, y, z)

img2 = np.zeros((x, y), np.uint8)

b, g, r = cv.split(img1)


mr = cv.merge([img2, img2, r])  # Solo rojo
mg = cv.merge([img2, g, img2])  # Solo verde
mb = cv.merge([b, img2, img2])  # Solo azul


nueva  = cv.merge([r, g, b])
nueva2 = cv.merge([g, b, r])
nueva3 = cv.merge([b, r, g])


cv.imshow('Solo ROJO (mr)', mr)
cv.imshow('Solo VERDE (mg)', mg)
cv.imshow('Solo AZUL (mb)', mb)


cv.imshow('n1 (r,g,b)', nueva)
cv.imshow('n2 (g,b,r)', nueva2)
cv.imshow('n3 (b,r,g)', nueva3)

cv.waitKey(0)
cv.destroyAllWindows()