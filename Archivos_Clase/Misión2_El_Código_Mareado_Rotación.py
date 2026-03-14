import cv2
import numpy as np
import math

# ===============================
# Cargar imagen (ruta en Downloads)
# ===============================
img = cv2.imread(r"C:\Users\ramjo\Downloads\qr_rotado.jpg")

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

# Tamaño de la imagen
h, w = img.shape[:2]
print("Tamaño de la imagen:", w, "x", h)

# Centro de la imagen
cx = w // 2
cy = h // 2

# El QR está rotado 45° antihorario, así que lo corregimos girando -45°
angulo = -45
theta = math.radians(angulo)

# ===============================
# METODO 1: RAW (manual)
# ===============================
resultado_raw = np.zeros((h, w, 3), dtype=np.uint8)

# Recorremos la imagen destino y calculamos de dónde viene el píxel
for y_prima in range(h):
    for x_prima in range(w):

        x = int((x_prima - cx) * math.cos(-theta) - (y_prima - cy) * math.sin(-theta) + cx)
        y = int((x_prima - cx) * math.sin(-theta) + (y_prima - cy) * math.cos(-theta) + cy)

        if 0 <= x < w and 0 <= y < h:
            resultado_raw[y_prima, x_prima] = img[y, x]

# ===============================
# METODO 2: OpenCV
# ===============================
M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)

resultado_opencv = cv2.warpAffine(img, M, (w, h))

# ===============================
# Guardar resultados
# ===============================
cv2.imwrite("m2_resultado_raw.png", resultado_raw)
cv2.imwrite("m2_resultado_opencv.png", resultado_opencv)

# ===============================
# Verificación con prints
# ===============================
print("Pixeles no negros original:", np.count_nonzero(img))
print("Pixeles no negros RAW:", np.count_nonzero(resultado_raw))
print("Pixeles no negros OpenCV:", np.count_nonzero(resultado_opencv))

# ===============================
# Mostrar imágenes
# ===============================
cv2.imshow("Original", img)
cv2.imshow("Resultado RAW", resultado_raw)
cv2.imshow("Resultado OpenCV", resultado_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()