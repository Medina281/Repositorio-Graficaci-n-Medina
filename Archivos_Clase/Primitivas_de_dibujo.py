import cv2 as cv
import numpy as np 

# Crear una imagen blanca
img = np.ones((400, 800, 3), dtype=np.uint8) * 255

# =========================
# CARRO DE CARRERAS SIMPLE
# =========================

# Cuerpo principal (Rectángulo)
cv.rectangle(img, (200, 200), (600, 280), (0, 255, 255), -1)  # Amarillo

# Parte frontal (Rectángulo pequeño)
cv.rectangle(img, (600, 220), (700, 280), (0, 0, 255), -1)  # Rojo

# Ventana (Rectángulo)
cv.rectangle(img, (350, 150), (500, 200), (255, 0, 0), -1)  # Azul

# Ruedas (Círculos)
cv.circle(img, (300, 300), 50, (0, 0, 0), -1)  # Negra izquierda
cv.circle(img, (550, 300), 50, (0, 0, 0), -1)  # Negra derecha

# Centro de ruedas
cv.circle(img, (300, 300), 20, (255, 0, 0), -1)  # Azul
cv.circle(img, (550, 300), 20, (255, 0, 0), -1)  # Azul

# Líneas de velocidad
cv.line(img, (150, 230), (200, 230), (0, 0, 255), 5)
cv.line(img, (120, 250), (200, 250), (0, 0, 255), 5)

# Número del carro
cv.putText(img, "1", (380, 260), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

# Mostrar imagen
cv.imshow('Carro de carreras', img)

cv.waitKey(0)
cv.destroyAllWindows()  