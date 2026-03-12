import cv2
import numpy as np

img = cv2.imread(r"C:\Users\ramjo\OneDrive\Documentos\Graficacion\Repositorio-Graficaci-n-Medina\Archivos_Clase\m1_oscura.png")

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

h, w = img.shape[:2]
print("Tamaño de la imagen:", w, "x", h)

tx = 300
ty = 200

resultado_raw = np.zeros((h, w, 3), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        nx = x + tx
        ny = y + ty

        if 0 <= nx < w and 0 <= ny < h:
            resultado_raw[ny, nx] = img[y, x]

M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

resultado_opencv = cv2.warpAffine(img, M, (w, h))

cv2.imwrite("m1_resultado_raw.png", resultado_raw)
cv2.imwrite("m1_resultado_opencv.png", resultado_opencv)

print("Pixeles no negros en original:", np.count_nonzero(img))
print("Pixeles no negros en RAW:", np.count_nonzero(resultado_raw))
print("Pixeles no negros en OpenCV:", np.count_nonzero(resultado_opencv))

original_clara = cv2.convertScaleAbs(img, alpha=8.0, beta=80)
raw_clara = cv2.convertScaleAbs(resultado_raw, alpha=8.0, beta=80)
opencv_clara = cv2.convertScaleAbs(resultado_opencv, alpha=8.0, beta=80)

cv2.imshow("Original", img)
cv2.imshow("Resultado RAW", resultado_raw)
cv2.imshow("Resultado OpenCV", resultado_opencv)

cv2.imshow("Original Clara", original_clara)
cv2.imshow("RAW Clara", raw_clara)
cv2.imshow("OpenCV Clara", opencv_clara)

cv2.waitKey(0)
cv2.destroyAllWindows()

