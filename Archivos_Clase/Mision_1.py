import cv2
import numpy as np

# Cargar la imagen proporcionada
img = cv2.imread('vehiculo.jpg')


alto, ancho = img.shape[:2]
tx, ty = 300, 200

# MÉTODO 1: MODO RAW (Slicing de NumPy)
# Lienzo
canvas_raw = np.zeros((alto, ancho, 3), dtype=np.uint8)



# Formula
canvas_raw[ty:alto, tx:ancho] = img[0:alto-ty, 0:ancho-tx]


# MÉTODO 2: MODO OPENCV (Matriz de Traslación)


M = np.float32([
    [1, 0, tx], # [1, 0, 300]
    [0, 1, ty]  # [0, 1, 200]
])


canvas_opencv = cv2.warpAffine(img, M, (ancho, alto))


cv2.imshow('Original - Error de Sensor', img)
cv2.imshow('Corregida - Metodo Raw', canvas_raw)
cv2.imshow('Corregida - Metodo OpenCV', canvas_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()