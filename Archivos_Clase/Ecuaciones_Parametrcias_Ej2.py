import numpy as np
import cv2

width, height = 1000, 1000
img = np.ones((height, width, 3), dtype=np.uint8) * 255
center_x, center_y = width // 2, height // 2

theta_increment = 0.01
theta = 0
max_theta = 2 * np.pi

# Parámetros Lissajous
A = 320
B = 320
a = 3
b = 2
delta = np.pi / 2  # desfase

while True:
    for t in np.arange(0, theta, theta_increment):
        x = int(center_x + A * np.sin(a * t + delta))
        y = int(center_y + B * np.sin(b * t))
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)

    cv2.imshow("Lissajous", img)
    theta += theta_increment

    if theta >= max_theta:
        theta = 0
        img[:] = 255

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()