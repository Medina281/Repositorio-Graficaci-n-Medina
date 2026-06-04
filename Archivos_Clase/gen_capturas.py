"""
Script auxiliar: genera las capturas PNG de cada escena.
Correr desde la misma carpeta donde esta demo.py
"""
import numpy as np
import cv2
import math
import os
import time

exec(open("demo.py").read().split("if __name__")[0])

os.makedirs("renders", exist_ok=True)
rng = np.random.default_rng(123)
fire_state = {
    "heat": np.zeros((H, W), np.float32),
    "rng": np.random.default_rng(999),
}
bufA = np.zeros((H, W, 3), np.uint8)

for sc in range(6):
    render_scene(bufA, sc, sc * 10 + 4.0, rng, fire_state)
    frame = post_vignette(bufA, 0.72)
    frame = post_scanlines(frame, 0.16)
    frame = post_posterize(frame, 24)
    cv2.imwrite(f"renders/scene_{sc}.png", frame)
    print(f"scene_{sc}.png OK")

print("Listo, capturas generadas en renders/")
