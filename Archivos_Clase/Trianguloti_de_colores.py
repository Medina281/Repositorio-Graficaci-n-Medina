import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Fondo negro
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0, -3)  # Mover la cámara

    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0) # Rojo
    glVertex3f(-1.0, -1.0, 0.0)
    glColor3f(0.0, 1.0, 0.0) # Verde
    glVertex3f(1.0, -1.0, 0.0)
    glColor3f(0.0, 0.0, 1.0) # Azul (ajustado a 1.0)
    glVertex3f(0.0, 1.0, 0.0)
    glEnd()

    glutSwapBuffers() # ¡Importante! Debe estar dentro de display()

def main():
    glutInit(sys.argv)
    # Cambié GLUT_RGB por GLUT_RGBA que es más estándar
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    
    # Agregamos la 'b' para evitar el error de TypeError que te salió
    glutCreateWindow(b"Triangulo con GLUT y Python")
    
    init()
    glutDisplayFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    main()