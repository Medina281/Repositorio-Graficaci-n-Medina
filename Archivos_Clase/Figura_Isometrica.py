import cv2
import numpy as np


img = np.ones((600, 600, 3), dtype=np.uint8) * 255
color = (0, 0, 0)
grosor = 2


c_top = (220, 220, 220)
c_left = (170, 170, 170)
c_right = (120, 120, 120)

def dibujar_cara(puntos, col):
    pts = np.array(puntos, np.int32)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, color, grosor)


s = 60 
dx, dy = 52, 30 


def p(x, y, z):
    base_x, base_y = 300, 450 
    px = base_x + (x - y) * dx
    py = base_y - (x + y) * dy - (z * s)
    return (int(px), int(py))


for i in range(3):
    for j in range(3):
        # Caras visibles de la base
        dibujar_cara([p(i,j,1), p(i+1,j,1), p(i+1,j+1,1), p(i,j+1,1)], c_top) # Techo base
        if j == 0: # Frente derecho
            dibujar_cara([p(i,0,0), p(i+1,0,0), p(i+1,0,1), p(i,0,1)], c_right)
        if i == 0: # Frente izquierdo
            dibujar_cara([p(0,j,0), p(0,j,1), p(0,j+1,1), p(0,j+1,0)], c_left)


for i in range(3):
    for j in range(3):
        if i == 2 or j == 2: # Esta es la forma de "L" en la parte de atrás
            for k in range(1, 3): # Niveles 1 y 2
                # Solo dibujamos caras que quedan a la vista
                # Cara superior
                if k == 2:
                    dibujar_cara([p(i,j,3), p(i+1,j,3), p(i+1,j+1,3), p(i,j+1,3)], c_top)
                # Caras laterales si no tienen bloques enfrente
                if i == 2: # Lateral derecho
                    dibujar_cara([p(3,j,k), p(3,j+1,k), p(3,j+1,k+1), p(3,j,k+1)], c_right)
                if j == 0 and i >= 2: # Frente de la L
                    dibujar_cara([p(i,0,k), p(i+1,0,k), p(i+1,0,k+1), p(i,0,k+1)], c_right)
                if i == 0 and j >= 2: # Frente izquierdo de la L
                    dibujar_cara([p(0,j,k), p(0,j+1,k), p(0,j+1,k+1), p(0,j,k+1)], c_left)

cv2.imshow("Figura Final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()