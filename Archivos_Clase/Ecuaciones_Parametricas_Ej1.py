import numpy as np
import cv2

width, height = 1000, 1000
img = np.ones((height, width, 3), dtype=np.uint8) * 255
center_x, center_y = width // 2, height // 2

theta_increment = 0.01
theta = 0
max_theta = 2 * np.pi  # un ciclo (puedes subirlo a 8*np.pi para más densidad)

# Parámetros de la rosa
scale = 300
n = 5  # pétalos si n impar; si n par, son 2n pétalos

while True:
    for t in np.arange(0, theta, theta_increment):
        r = scale * np.cos(n * t)
        x = int(center_x + r * np.cos(t))
        y = int(center_y + r * np.sin(t))
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)

    cv2.imshow("Rose Curve", img)
    theta += theta_increment

    if theta >= max_theta:
        theta = 0
        img[:] = 255  # reinicia limpio para ver la animación repetirse

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()