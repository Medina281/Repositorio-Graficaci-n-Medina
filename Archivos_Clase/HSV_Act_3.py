import cv2
import numpy as np

# 1. Leer la imagen y generar la máscara (solo para obtenerla)
img = cv2.imread('frutas.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([20, 100, 100])
upper_green = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)

# 2. Limpieza de la máscara (apertura para eliminar ruido pequeño)
kernel = np.ones((5, 5), np.uint8)
mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 3. Análisis de regiones conectadas (SOLO usando la máscara)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask_limpia, connectivity=8
)

# 4. Filtrar regiones pequeñas (ruido)
area_minima = 500  # ajustar según tamaño de la imagen
areas_validas = []

for i in range(1, num_labels):  # ignorar el fondo (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= area_minima:
        areas_validas.append(area)

# 5. Reporte final
print("Número total de frutas detectadas:", len(areas_validas))

for idx, area in enumerate(areas_validas, start=1):
    print(f"Fruta {idx}: Área aproximada = {area} píxeles")

# 6. Mostrar solo las máscaras (opcional, pero permitido)
cv2.imshow("Mascara Original", mask)
cv2.imshow("Mascara Limpia", mask_limpia)
cv2.waitKey(0)
cv2.destroyAllWindows()
