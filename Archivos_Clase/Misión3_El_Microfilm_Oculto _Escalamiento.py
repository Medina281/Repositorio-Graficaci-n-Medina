import cv2
import numpy as np

# ===============================
# Cargar imagen
# ===============================
img = cv2.imread(r"C:\Users\ramjo\Downloads\microfilm.jpg")

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

h, w = img.shape[:2]
print("Tamaño de la imagen:", w, "x", h)

# ===============================
# Recorte central (200x200)
# ===============================
cx = w // 2
cy = h // 2

recorte = img[cy-100:cy+100, cx-100:cx+100]

# ===============================
# FACTOR DE ESCALA
# ===============================
Sx = 5
Sy = 5

nuevo_h = recorte.shape[0] * Sy
nuevo_w = recorte.shape[1] * Sx

# ===============================
# METODO 1: RAW (Vecino cercano)
# ===============================
resultado_raw = np.zeros((nuevo_h, nuevo_w, 3), dtype=np.uint8)

for y in range(nuevo_h):
    for x in range(nuevo_w):

        # coordenada original
        orig_x = int(x / Sx)
        orig_y = int(y / Sy)

        resultado_raw[y, x] = recorte[orig_y, orig_x]

# ===============================
# METODO 2: OpenCV
# ===============================
resultado_opencv = cv2.resize(
    recorte,
    None,
    fx=5,
    fy=5,
    interpolation=cv2.INTER_CUBIC
)

# ===============================
# Guardar resultados
# ===============================
cv2.imwrite("m3_recorte.png", recorte)
cv2.imwrite("m3_raw.png", resultado_raw)
cv2.imwrite("m3_opencv.png", resultado_opencv)

# ===============================
# Mostrar imágenes
# ===============================
cv2.imshow("Recorte original", recorte)
cv2.imshow("Escala RAW", resultado_raw)
cv2.imshow("Escala OpenCV", resultado_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()