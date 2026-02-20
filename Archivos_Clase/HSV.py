import cv2
import numpy as np

# Leer la imagen
img = cv2.imread('frutas.png')

# Convertir la imagen al espacio de color HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

kernel = np.ones((5,5), np.uint8)

# Definir el rango inferior y superior para detectar verde
lower_green = np.array([15, 100, 100])  # Hue, Saturación, Brillo mínimos
upper_green = np.array([35, 255, 255])  # Hue, Saturación, Brillo máximos

# Crear una máscara que solo incluya los píxeles dentro del rango
mask = cv2.inRange(hsv, lower_green, upper_green)
mask_apertura = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(img, img, mask=mask)

# Mostrar la imagen original y la imagen con el color detectado
cv2.imshow("Imagen Original", img)
cv2.imshow("Color Detectado", result)
cv2.imshow("Mascara Blanco y Negro", mask)
cv2.imshow("Mascara Apertura", mask_apertura)
cv2.waitKey(0)
cv2.destroyAllWindows()