import cv2
import numpy as np

# ==========================
# Cargar imagen
# ==========================

img = cv2.imread(r"C:\Users\ramjo\Downloads\m4_ruido.png")

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

# ==========================
# Convertir a HSV
# ==========================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ==========================
# Rango de color CYAN
# ==========================

bajo = np.array([80,100,100])
alto = np.array([100,255,255])

# ==========================
# Crear máscara
# ==========================

mascara = cv2.inRange(hsv,bajo,alto)

# ==========================
# Mostrar resultados
# ==========================

cv2.imshow("Imagen original",img)
cv2.imshow("Mascara CYAN",mascara)

cv2.imwrite("m4_mascara.png",mascara)

cv2.waitKey(0)
cv2.destroyAllWindows()
