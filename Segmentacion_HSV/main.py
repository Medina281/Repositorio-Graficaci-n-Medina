import cv2
import numpy as np
import os

# ===================== CONFIG =====================
IMG_PATH = "frutas.png"
OUT_DIR = "capturas"

KERNEL_SIZE = 5     # limpieza morfológica
MIN_AREA = 300      # filtro de ruido por área

# Rangos HSV base (puedes ajustar si hace falta)
# OpenCV: H 0-179, S 0-255, V 0-255
RANGES = {
    "rojo": [
        (np.array([0, 100, 100]),  np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([179, 255, 255])),
    ],
    "verde": [
        (np.array([35, 40, 40]), np.array([85, 255, 255])),
    ],
    "amarillo": [
        (np.array([20, 80, 80]), np.array([35, 255, 255])),
    ],
}
# ==================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_base_images(img_bgr, hsv):
    cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img_bgr)

    # solo para captura (visualización)
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, "hsv.png"), hsv_vis)

def build_mask(hsv, ranges):
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        m = cv2.inRange(hsv, low, high)
        mask_total = cv2.bitwise_or(mask_total, m)
    return mask_total

def clean_mask(mask):
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    # apertura: quita puntitos
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cierre: rellena huequitos
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def count_regions(mask_clean):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    areas_validas = []
    for label in range(1, num_labels):  # 0 es fondo
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_AREA:
            areas_validas.append(area)

    regiones_totales = num_labels - 1
    frutas_detectadas = len(areas_validas)
    return regiones_totales, frutas_detectadas, areas_validas

def main():
    ensure_dir(OUT_DIR)

    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"ERROR: No se encontró '{IMG_PATH}'. Debe estar junto a main.py")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_base_images(img, hsv)

    print("===== RESULTADOS (por color) =====")

    for color, ranges in RANGES.items():
        mask = build_mask(hsv, ranges)
        mask_clean = clean_mask(mask)

        cv2.imwrite(os.path.join(OUT_DIR, f"mask_{color}.png"), mask)
        cv2.imwrite(os.path.join(OUT_DIR, f"mask_{color}_limpia.png"), mask_clean)

        regiones_totales, frutas_detectadas, areas = count_regions(mask_clean)

        print(f"\nColor: {color}")
        print("Regiones totales (incluyendo ruido):", regiones_totales)
        print("Frutas detectadas (filtradas):", frutas_detectadas)
        print("Áreas aproximadas:", areas)

    print("\nListo. Revisa la carpeta 'capturas'.")

if __name__ == "__main__":
    main()