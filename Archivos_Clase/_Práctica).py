#!/usr/bin/env python3
from __future__ import annotations

import math
import sys

import glfw
from OpenGL.GL import *
from OpenGL.GLU import (
    GLU_FILL,
    gluLookAt,
    gluNewQuadric,
    gluPerspective,
    gluQuadricDrawStyle,
    gluSphere,
)

WINDOW_TITLE = "Orbita Dual - 1/2/3 cambia modo"

INITIAL_MODE = 1

ORBIT_RADIUS = 5.0
CAM_DISTANCE = 6.0
ANGLE_SPEED = 1.0

# =========================
# MISION 3
# =========================
USE_LIGHTING = True

_quadric = None


def draw_sphere(radius: float = 1.0) -> None:
    global _quadric

    if _quadric is None:
        _quadric = gluNewQuadric()
        gluQuadricDrawStyle(_quadric, GLU_FILL)

    gluSphere(_quadric, radius, 40, 24)


def setup_basic_lighting():
    """
    Luz direccional.
    IMPORTANTE:
    La posición de la luz depende de la matriz MODELVIEW actual.
    """

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    ambient = [0.2, 0.2, 0.2, 1.0]
    diffuse = [1.0, 1.0, 1.0, 1.0]

    # Luz direccional
    light_position = [2.0, 2.0, 2.0, 0.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)

    # IMPORTANTE:
    # La luz se coloca en el espacio actual de cámara.
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)


# =========================================================
# MODO 1
# OBJETO ROTA
# =========================================================
def render_rotating_object(angle: float) -> None:

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Camara fija
    glTranslatef(0.0, 0.0, -CAM_DISTANCE)

    # Luz en coordenadas de camara
    if USE_LIGHTING:
        setup_basic_lighting()

    # El objeto rota
    glRotatef(angle, 0.0, 1.0, 0.0)

    glColor3f(0.3, 0.6, 1.0)

    draw_sphere(1.0)


# =========================================================
# MODO 2
# CAMARA ORBITA
# =========================================================
def render_orbiting_camera(angle: float) -> None:

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # La camara gira alrededor del objeto
    glRotatef(-angle, 0.0, 1.0, 0.0)

    glTranslatef(0.0, 0.0, -CAM_DISTANCE)

    # Luz despues de mover camara
    if USE_LIGHTING:
        setup_basic_lighting()

    glColor3f(1.0, 0.5, 0.3)

    draw_sphere(1.0)


# =========================================================
# VARIANTE B
# =========================================================
def render_orbiting_camera_variant_b(angle: float) -> None:

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -CAM_DISTANCE)

    glRotatef(angle, 0.0, 1.0, 0.0)

    if USE_LIGHTING:
        setup_basic_lighting()

    glColor3f(0.4, 1.0, 0.4)

    draw_sphere(1.0)


# =========================================================
# MODO 3 - gluLookAt
# =========================================================
def render_with_lookat(angle: float) -> None:

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    a = math.radians(angle)

    cam_x = ORBIT_RADIUS * math.sin(a)
    cam_z = ORBIT_RADIUS * math.cos(a)

    gluLookAt(
        cam_x, 0.0, cam_z,
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    )

    if USE_LIGHTING:
        setup_basic_lighting()

    glColor3f(1.0, 0.9, 0.3)

    draw_sphere(1.0)


def main() -> None:

    if not glfw.init():
        print("No se pudo iniciar GLFW", file=sys.stderr)
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(
        800,
        600,
        WINDOW_TITLE,
        None,
        None
    )

    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)

    glfw.swap_interval(1)

    mode = INITIAL_MODE

    def on_key(win, key, scancode, action, mods):

        nonlocal mode

        if action != glfw.PRESS:
            return

        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)

        elif key == glfw.KEY_1:
            mode = 1
            print("Modo 1: objeto rota")

        elif key == glfw.KEY_2:
            mode = 2
            print("Modo 2: camara orbita")

        elif key == glfw.KEY_3:
            mode = 3
            print("Modo 3: gluLookAt")

    glfw.set_key_callback(window, on_key)

    glEnable(GL_DEPTH_TEST)

    glClearColor(0.08, 0.08, 0.12, 1.0)

    angle = 0.0

    while not glfw.window_should_close(window):

        fb_w, fb_h = glfw.get_framebuffer_size(window)

        if fb_h <= 0:
            fb_h = 1

        glViewport(0, 0, fb_w, fb_h)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)

        glLoadIdentity()

        gluPerspective(
            50.0,
            fb_w / float(fb_h),
            0.1,
            100.0
        )

        if mode == 1:
            render_rotating_object(angle)

        elif mode == 2:
            render_orbiting_camera(angle)

        elif mode == 3:
            render_with_lookat(angle)

        angle += ANGLE_SPEED

        if angle >= 360.0:
            angle -= 360.0

        glfw.swap_buffers(window)

        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()