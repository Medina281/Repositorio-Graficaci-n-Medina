import cv2
import numpy as np

img = cv2.imread(r"C:\Users\ramjo\Downloads\m1_oscura.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

h, w = img.shape
print("Tamaño:", w, "x", h)

# =========================
# MODO RAW
# =========================
resultado_raw = np.zeros((h, w), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        valor = int(img[y, x]) * 50
        if valor > 255:
            valor = 255
        resultado_raw[y, x] = valor

# =========================
# MODO OPENCV / NUMPY
# =========================
resultado_opencv = np.clip(img.astype(np.int32) * 50, 0, 255).astype(np.uint8)

cv2.imwrite("m1_raw.png", resultado_raw)
cv2.imwrite("m1_opencv.png", resultado_opencv)

cv2.imshow("Original", img)
cv2.imshow("Resultado RAW", resultado_raw)
cv2.imshow("Resultado OpenCV", resultado_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()