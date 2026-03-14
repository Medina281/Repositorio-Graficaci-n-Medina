import cv2
import numpy as np

# ===============================
# Cargar imágenes
# ===============================

mitad1 = cv2.imread(r"C:\Users\ramjo\Downloads\m2_mitad1.png")
mitad2 = cv2.imread(r"C:\Users\ramjo\Downloads\m2_mitad2.png")

if mitad1 is None or mitad2 is None:
    print("Error cargando imágenes")
    exit()

# ===============================
# Crear lienzo 400x400
# ===============================

lienzo = np.ones((400,400,3), dtype=np.uint8) * 255

# ===============================
# Mitad 1 -> mover al origen
# ===============================

h1, w1 = mitad1.shape[:2]

M = np.float32([
[1,0,0],
[0,1,0]
])

pieza1 = cv2.warpAffine(mitad1, M, (400,400))

lienzo[0:h1,0:w1] = mitad1

# ===============================
# Mitad 2 -> rotar 180 grados
# ===============================

h2, w2 = mitad2.shape[:2]

centro = (w2//2, h2//2)

M_rot = cv2.getRotationMatrix2D(centro,180,1)

mitad2_rotada = cv2.warpAffine(mitad2, M_rot,(w2,h2))

# colocar abajo

lienzo[200:200+h2,0:w2] = mitad2_rotada

# ===============================
# Mostrar resultado
# ===============================

cv2.imshow("Mitad1",mitad1)
cv2.imshow("Mitad2",mitad2)
cv2.imshow("QR reconstruido",lienzo)

cv2.imwrite("m2_qr_completo.png",lienzo)

cv2.waitKey(0)
cv2.destroyAllWindows()