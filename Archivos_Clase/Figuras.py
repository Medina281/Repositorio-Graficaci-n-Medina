import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

rotation = 0.0

def draw_sphere(quadric, x, y, z, radius, color):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    gluSphere(quadric, radius, 30, 30)
    glPopMatrix()

def draw_eye(quadric):
    draw_sphere(quadric, 0.7, 0, 0, 0.54, (0.85, 0.67, 0.65))
    draw_sphere(quadric, 0.56, 0, 0, 0.6, (1, 1, 1))
    draw_sphere(quadric, 0.49, 0, 0, 0.55, (0.84, 0.85, 0.92))
    draw_sphere(quadric, 0.3, 0, 0, 0.4, (0, 0, 0))

def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)

    light_position = [1.0, 1.0, 1.0, 0.2]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

def main():
    global rotation

    if not glfw.init():
        print("Error al inicializar GLFW")
        return

    window = glfw.create_window(800, 600, "Dos Esferas Simples", None, None)

    if not window:
        glfw.terminate()
        print("Error al crear ventana")
        return

    glfw.make_context_current(window)

    glClearColor(0.54, 0.72, 0.84, 1.0)
    setup_lighting()

    quadric = gluNewQuadric()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 800 / 600, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -5)

        rotation += 0.05
        glRotatef(rotation, 0, 1, 0)

        draw_eye(quadric)

        glfw.swap_buffers(window)
        glfw.poll_events()

    gluDeleteQuadric(quadric)
    glfw.terminate()

if __name__ == "__main__":
    main()