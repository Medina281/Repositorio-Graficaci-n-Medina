import math
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# Variable global para rotación
rotation_angle = 0.0


def draw_sphere():
    """Esfera usando GLU"""
    glColor3f(1.0, 0.2, 0.2)
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.5, 32, 32)
    gluDeleteQuadric(quadric)


def draw_cube():
    """Cubo dibujado manualmente"""
    glColor3f(0.2, 1.0, 0.2)

    vertices = [
        [-0.4, -0.4, -0.4],
        [ 0.4, -0.4, -0.4],
        [ 0.4,  0.4, -0.4],
        [-0.4,  0.4, -0.4],
        [-0.4, -0.4,  0.4],
        [ 0.4, -0.4,  0.4],
        [ 0.4,  0.4,  0.4],
        [-0.4,  0.4,  0.4],
    ]

    faces = [
        ([0, 1, 2, 3], [0.0, 0.0, -1.0]),  # atrás
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),   # frente
        ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),  # izquierda
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),   # derecha
        ([3, 2, 6, 7], [0.0, 1.0, 0.0]),   # arriba
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),  # abajo
    ]

    glBegin(GL_QUADS)
    for face, normal in faces:
        glNormal3f(*normal)
        for index in face:
            glVertex3f(*vertices[index])
    glEnd()


def draw_cone():
    """Cono usando GLU"""
    glColor3f(0.2, 0.2, 1.0)
    quadric = gluNewQuadric()

    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quadric, 0.5, 0.0, 1.0, 32, 32)
    glPopMatrix()

    # Base del cono
    glPushMatrix()
    glRotatef(90, 1, 0, 0)
    gluDisk(quadric, 0.0, 0.5, 32, 1)
    glPopMatrix()

    gluDeleteQuadric(quadric)


def draw_torus():
    """Toro aproximado matemáticamente"""
    glColor3f(1.0, 1.0, 0.2)

    R = 0.45   # radio mayor
    r = 0.18   # radio menor
    nsides = 24
    rings = 36

    for i in range(rings):
        theta = (2.0 * math.pi * i) / rings
        theta_next = (2.0 * math.pi * (i + 1)) / rings

        glBegin(GL_QUAD_STRIP)
        for j in range(nsides + 1):
            phi = (2.0 * math.pi * j) / nsides

            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            cos_theta_next = math.cos(theta_next)
            sin_theta_next = math.sin(theta_next)

            x1 = (R + r * cos_phi) * cos_theta
            y1 = (R + r * cos_phi) * sin_theta
            z1 = r * sin_phi

            nx1 = cos_phi * cos_theta
            ny1 = cos_phi * sin_theta
            nz1 = sin_phi

            x2 = (R + r * cos_phi) * cos_theta_next
            y2 = (R + r * cos_phi) * sin_theta_next
            z2 = r * sin_phi

            nx2 = cos_phi * cos_theta_next
            ny2 = cos_phi * sin_theta_next
            nz2 = sin_phi

            glNormal3f(nx1, ny1, nz1)
            glVertex3f(x1, y1, z1)

            glNormal3f(nx2, ny2, nz2)
            glVertex3f(x2, y2, z2)
        glEnd()


def draw_teapot():
    """Reemplazo simple de tetera: esfera achatada con base"""
    glColor3f(1.0, 0.2, 1.0)
    quadric = gluNewQuadric()

    glPushMatrix()
    glScalef(1.0, 0.75, 1.0)
    gluSphere(quadric, 0.5, 32, 32)
    glPopMatrix()

    gluDeleteQuadric(quadric)


def draw_cylinder():
    """Cilindro usando GLU"""
    glColor3f(0.2, 1.0, 1.0)
    quadric = gluNewQuadric()

    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quadric, 0.4, 0.4, 1.0, 32, 32)
    glPopMatrix()

    # Tapa inferior
    glPushMatrix()
    glRotatef(90, 1, 0, 0)
    gluDisk(quadric, 0.0, 0.4, 32, 1)
    glPopMatrix()

    # Tapa superior
    glPushMatrix()
    glTranslatef(0.0, 1.0, 0.0)
    glRotatef(-90, 1, 0, 0)
    gluDisk(quadric, 0.0, 0.4, 32, 1)
    glPopMatrix()

    gluDeleteQuadric(quadric)


def draw_disk():
    """Disco usando GLU"""
    glColor3f(1.0, 0.5, 0.2)
    quadric = gluNewQuadric()
    gluDisk(quadric, 0.2, 0.6, 32, 32)
    gluDeleteQuadric(quadric)


def draw_octahedron():
    """Octaedro dibujado manualmente"""
    glColor3f(1.0, 0.5, 0.5)

    vertices = [
        ( 0.0,  0.6,  0.0),  # top
        ( 0.6,  0.0,  0.0),
        ( 0.0,  0.0,  0.6),
        (-0.6,  0.0,  0.0),
        ( 0.0,  0.0, -0.6),
        ( 0.0, -0.6,  0.0),  # bottom
    ]

    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 1),
        (5, 2, 1),
        (5, 3, 2),
        (5, 4, 3),
        (5, 1, 4),
    ]

    glBegin(GL_TRIANGLES)
    for a, b, c in faces:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]

        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az

        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length != 0:
            nx /= length
            ny /= length
            nz /= length

        glNormal3f(nx, ny, nz)
        glVertex3f(ax, ay, az)
        glVertex3f(bx, by, bz)
        glVertex3f(cx, cy, cz)
    glEnd()


def draw_tetrahedron():
    """Tetraedro dibujado manualmente"""
    glColor3f(0.5, 0.5, 1.0)

    vertices = [
        ( 0.0,  0.6,  0.0),
        (-0.5, -0.3,  0.5),
        ( 0.5, -0.3,  0.5),
        ( 0.0, -0.3, -0.6),
    ]

    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 1),
        (1, 3, 2),
    ]

    glBegin(GL_TRIANGLES)
    for a, b, c in faces:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]

        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az

        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length != 0:
            nx /= length
            ny /= length
            nz /= length

        glNormal3f(nx, ny, nz)
        glVertex3f(ax, ay, az)
        glVertex3f(bx, by, bz)
        glVertex3f(cx, cy, cz)
    glEnd()


def draw_icosahedron():
    """Reemplazo simple: esfera de baja resolución"""
    glColor3f(1.0, 1.0, 0.5)
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.55, 12, 12)
    gluDeleteQuadric(quadric)


def draw_dodecahedron():
    """Reemplazo simple: cubo escalado"""
    glColor3f(0.5, 1.0, 0.5)
    glPushMatrix()
    glRotatef(25, 1, 1, 0)
    glScalef(0.9, 0.9, 0.9)
    draw_cube()
    glPopMatrix()


def draw_partial_disk():
    """Disco parcial (sector circular)"""
    glColor3f(0.8, 0.3, 0.8)
    quadric = gluNewQuadric()
    gluPartialDisk(quadric, 0.2, 0.6, 32, 16, 0, 270)
    gluDeleteQuadric(quadric)


def setup_lighting():
    """Configura iluminación para ver mejor las figuras 3D"""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_NORMALIZE)

    light_position = [2.0, 2.0, 2.0, 1.0]
    light_ambient = [0.3, 0.3, 0.3, 1.0]
    light_diffuse = [1.0, 1.0, 1.0, 1.0]
    light_specular = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)


def draw_all_3d_shapes(window):
    """Dibuja todas las figuras 3D en una cuadrícula 4x3"""
    global rotation_angle

    shapes = [
        ("Esfera", draw_sphere),
        ("Cubo", draw_cube),
        ("Cono", draw_cone),
        ("Toroide", draw_torus),
        ("Tetera", draw_teapot),
        ("Cilindro", draw_cylinder),
        ("Disco", draw_disk),
        ("Dodecaedro", draw_dodecahedron),
        ("Octaedro", draw_octahedron),
        ("Tetraedro", draw_tetrahedron),
        ("Icosaedro", draw_icosahedron),
        ("Disco Parcial", draw_partial_disk),
    ]

    cols = 4
    rows = 3

    width, height = glfw.get_framebuffer_size(window)

    if height == 0:
        return

    cell_width = max(width // cols, 1)
    cell_height = max(height // rows, 1)

    for idx, (name, draw_func) in enumerate(shapes):
        col = idx % cols
        row = idx // cols

        x = col * cell_width
        y = height - (row + 1) * cell_height

        glViewport(x, y, cell_width, cell_height)

        # Proyección
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = cell_width / max(cell_height, 1)
        gluPerspective(45, aspect, 0.1, 50.0)

        # Vista
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)

        # Aplicar rotación global
        glRotatef(rotation_angle, 0, 1, 0)
        glRotatef(rotation_angle * 0.5, 1, 0, 0)

        glPushMatrix()
        draw_func()
        glPopMatrix()


def main():
    global rotation_angle

    if not glfw.init():
        print("No se pudo inicializar GLFW")
        return

    window = glfw.create_window(1600, 900, "Todas las Figuras 3D de OpenGL", None, None)
    if not window:
        glfw.terminate()
        print("No se pudo crear la ventana")
        return

    glfw.make_context_current(window)

    glClearColor(0.1, 0.1, 0.15, 1.0)
    setup_lighting()

    glEnable(GL_MULTISAMPLE)
    glShadeModel(GL_SMOOTH)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rotation_angle += 0.5
        if rotation_angle > 360:
            rotation_angle -= 360

        draw_all_3d_shapes(window)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()