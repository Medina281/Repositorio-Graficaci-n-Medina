import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from PIL import Image
import sys
import os
import math

# Variables globales
texture_id = None
window = None

def load_texture(path):
    global texture_id
    if not os.path.exists(path):
        print(f"\n[ERROR] No se encuentra el archivo: '{path}'")
        sys.exit()

    try:
        img = Image.open(path).convert("RGB")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.tobytes()

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        print("¡Textura cargada correctamente!")
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        sys.exit()

def init():
    glClearColor(0.5, 0.8, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST)         
    glEnable(GL_TEXTURE_2D)         

    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 800/600, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    load_texture("Pasto.jpg")  

def draw_sun():
    glDisable(GL_TEXTURE_2D)
    glPushMatrix()
    glTranslatef(-8, 10, -8) 
    glColor3f(1.0, 0.9, 0.0)
    glPointSize(50.0)
    glBegin(GL_POINTS)
    glVertex3f(0, 0, 0)
    glEnd()
    
    glColor3f(1.0, 0.7, 0.0)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for i in range(0, 360, 30):
        rad = math.radians(i)
        glVertex3f(0, 0, 0)
        glVertex3f(math.cos(rad) * 1.8, math.sin(rad) * 1.8, 0)
    glEnd()
    glPopMatrix()
    glEnable(GL_TEXTURE_2D)

def draw_house(x, z, width, height, depth, r, g, b):
    glDisable(GL_TEXTURE_2D)
    # Cuerpo de la casa
    glPushMatrix()
    glTranslatef(x, 0, z)
    glBegin(GL_QUADS)
    glColor3f(r, g, b)
    # Frente
    glVertex3f(-width, 0, depth); glVertex3f(width, 0, depth); glVertex3f(width, height, depth); glVertex3f(-width, height, depth)
    # Atrás
    glVertex3f(-width, 0, -depth); glVertex3f(width, 0, -depth); glVertex3f(width, height, -depth); glVertex3f(-width, height, -depth)
    # Izquierda
    glVertex3f(-width, 0, -depth); glVertex3f(-width, 0, depth); glVertex3f(-width, height, depth); glVertex3f(-width, height, -depth)
    # Derecha
    glVertex3f(width, 0, -depth); glVertex3f(width, 0, depth); glVertex3f(width, height, depth); glVertex3f(width, height, -depth)
    glEnd()

    # Techo
    glBegin(GL_TRIANGLES)
    glColor3f(0.8, 0.1, 0.1)
    glVertex3f(-width, height, depth);  glVertex3f(width, height, depth);  glVertex3f(0, height + 1.2, 0)
    glVertex3f(width, height, depth);   glVertex3f(width, height, -depth); glVertex3f(0, height + 1.2, 0)
    glVertex3f(width, height, -depth);  glVertex3f(-width, height, -depth);glVertex3f(0, height + 1.2, 0)
    glVertex3f(-width, height, -depth); glVertex3f(-width, height, depth); glVertex3f(0, height + 1.2, 0)
    glEnd()
    glPopMatrix()
    glEnable(GL_TEXTURE_2D)

def draw_street():
    glDisable(GL_TEXTURE_2D)
    glColor3f(0.2, 0.2, 0.2) # Gris oscuro para el asfalto
    glBegin(GL_QUADS)
    # La calle va a lo largo del eje X frente a las casas
    glVertex3f(-15, 0.01, 5) 
    glVertex3f(15, 0.01, 5)
    glVertex3f(15, 0.01, 2)
    glVertex3f(-15, 0.01, 2)
    glEnd()

    # Línea central de la calle (punteada)
    glColor3f(1.0, 1.0, 1.0)
    for i in range(-14, 15, 4):
        glBegin(GL_QUADS)
        glVertex3f(i, 0.02, 3.6)
        glVertex3f(i + 2, 0.02, 3.6)
        glVertex3f(i + 2, 0.02, 3.4)
        glVertex3f(i, 0.02, 3.4)
        glEnd()
    glEnable(GL_TEXTURE_2D)

def draw_ground():
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor3f(1, 1, 1) 
    glBegin(GL_QUADS)
    scale = 10.0 
    glTexCoord2f(0, 0);         glVertex3f(-20, 0, 20)
    glTexCoord2f(scale, 0);     glVertex3f(20, 0, 20)
    glTexCoord2f(scale, scale); glVertex3f(20, 0, -20)
    glTexCoord2f(0, scale);     glVertex3f(-20, 0, -20)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)

def draw_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Cámara más alejada para ver todo el vecindario
    gluLookAt(15, 10, 20, 0, 0, 0, 0, 1, 0)

    draw_ground()
    draw_street()
    
    # Casa 1 (La original)
    draw_house(0, 0, 1, 1, 1, 0.7, 0.4, 0.2)
    
    # Casa 2 (Más alta y de otro color)
    draw_house(-5, -2, 0.8, 3, 0.8, 0.5, 0.5, 0.8)
    
    # Casa 3 (Más ancha/larga)
    draw_house(6, -1, 2, 0.8, 1.2, 0.4, 0.6, 0.4)
    
    draw_sun()

def main():
    global window
    if not glfw.init():
        return

    window = glfw.create_window(1000, 700, "Vecindario OpenGL - Medina", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    init()

    while not glfw.window_should_close(window):
        draw_scene()
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()