import cv2
import numpy as np

# lienzo azul oscuro BGR(50, 20, 20)
img = np.full((500, 500, 3), (50, 20, 20), dtype=np.uint8)

# círculo amarillo
cv2.circle(img, (250, 250), 100, (0, 255, 255), 3)

# rectángulo rojo sólido
cv2.rectangle(img, (200, 200), (300, 300), (0, 0, 255), -1)

# diagonales blancas
cv2.line(img, (0, 0), (499, 499), (255, 255, 255), 2)
cv2.line(img, (499, 0), (0, 499), (255, 255, 255), 2)

cv2.imwrite("m3_sello_forjado.png", img)

cv2.imshow("Sello", img)
cv2.waitKey(0)
cv2.destroyAllWindows()