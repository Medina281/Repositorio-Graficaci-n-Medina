import cv2
import numpy as np
import os

# Crear carpeta capturas si no existe
if not os.path.exists("capturas"):
    os.makedirs("capturas")

# Cargar imagen
img = cv2.imread("frutas.png")

if img is None:
    print("Error: No se encontró frutas.png")
    exit()

# Guardar imagen original
cv2.imwrite("capturas/original.png", img)

# Convertir a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Guardar visualización HSV
hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("capturas/hsv.png", hsv_vis)

# ----------- RANGO PARA ROJO (puedes cambiar luego) -----------

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = mask1 + mask2

cv2.imwrite("capturas/mask_rojo.png", mask)

# ----------- LIMPIEZA MORFOLÓGICA -----------

kernel = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

cv2.imwrite("capturas/mask_rojo_limpia.png", mask_clean)

# ----------- CONTEO DE REGIONES -----------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

areas_validas = []
for i in range(1, num_labels):  # 0 es fondo
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 300:
        areas_validas.append(area)

print("Regiones totales (incluyendo ruido):", num_labels - 1)
print("Frutas detectadas:", len(areas_validas))
print("Áreas aproximadas:", areas_validas)