import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# Ángulos de rotación
angle_x = 0
angle_y = 0
angle_z = 0

def draw_cube():
    glBegin(GL_QUADS)

    # Cara frontal
    glColor3f(1, 0, 0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    # Cara trasera
    glColor3f(0, 1, 0)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)

    # Cara izquierda
    glColor3f(0, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)

    # Cara derecha
    glColor3f(1, 1, 0)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)

    # Cara superior
    glColor3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)

    # Cara inferior
    glColor3f(0, 1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    glEnd()


def main():
    global angle_x, angle_y, angle_z

    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "3 Cubos", None, None)
    glfw.make_context_current(window)

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 800/600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 🔺 Cubo 1 (rotación en X)
        glPushMatrix()
        glTranslatef(-3, 0, -8)
        glRotatef(angle_x, 1, 0, 0)
        draw_cube()
        glPopMatrix()

        # 🔺 Cubo 2 (rotación en Y)
        glPushMatrix()
        glTranslatef(0, 0, -8)
        glRotatef(angle_y, 0, 1, 0)
        draw_cube()
        glPopMatrix()

        # 🔺 Cubo 3 (rotación en Z)
        glPushMatrix()
        glTranslatef(3, 0, -8)
        glRotatef(angle_z, 0, 0, 1)
        draw_cube()
        glPopMatrix()

        # 🔄 Actualizar ángulos
        angle_x += 0.5
        angle_y += 0.5
        angle_z += 0.5

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()