#!/usr/bin/env python3
"""
Realidad Aumentada — Metrópoli 3D completa sobre marcador ArUco + control gestual.

La ciudad aparece anclada al marcador ArUco impreso. Incluye parque lleno, estadio, avión, edificios variados y tráfico/personas animadas.
Gestos (cualquier mano):
  Mover mano              -> girar la ciudad (yaw / pitch)
  Pinch (índice + pulgar) -> zoom in / out

Controles teclado:
  ESC / Q  -> salir

Requisitos: opencv-python, PyOpenGL, glfw, numpy, mediapipe
"""

from __future__ import annotations
import math, os, sys, time
from pathlib import Path

import cv2
import glfw
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel",      "3")

# ── Configuración ArUco ──────────────────────────────────────────────────────
CAMERA_INDEX    = 0
MARKER_LENGTH_M = 0.10
ARUCO_DICT      = cv2.aruco.DICT_4X4_50
MARKER_ID       = 0
ZNear, ZFar     = 0.001, 200.0

SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_NPZ  = SCRIPT_DIR / "camera_ar.npz"
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

# ── Estado gestual ───────────────────────────────────────────────────────────
city_yaw   =   0.0
city_pitch =  30.0
city_zoom  =   4.0
_prev_index = None
_prev_pinch = None

# Tamaño base de la ciudad en realidad aumentada.
# Si todavía se ve pequeña, sube este valor a 0.006 o 0.008.
CITY_BASE_SCALE = 0.0045
MIN_CITY_ZOOM = 0.5
MAX_CITY_ZOOM = 25.0

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20),
]

# ── MediaPipe ────────────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

# ── Calibración ──────────────────────────────────────────────────────────────
def default_camera_matrix(w, h):
    f = float(max(w, h))
    return np.array([[f,0,w/2.],[0,f,h/2.],[0,0,1]], dtype=np.float64)

def load_calibration(w, h):
    if CALIB_NPZ.is_file():
        d = np.load(CALIB_NPZ)
        return d["camera_matrix"], d["dist_coeffs"]
    return default_camera_matrix(w, h), np.zeros((5,1), dtype=np.float64)

# ── ArUco ────────────────────────────────────────────────────────────────────
def make_aruco_detector():
    dic    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dic, params), dic
    return None, dic

def detect_marker(gray, detector, dictionary):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=cv2.aruco.DetectorParameters())
    if ids is None or len(ids) == 0:
        return None
    matches = np.where(ids.flatten() == MARKER_ID)[0]
    if len(matches) == 0:
        return None
    return corners[int(matches[0])]

def marker_object_points(side):
    s = side / 2.0
    return np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)

def estimate_pose(corners, K, dist):
    pts2d = np.asarray(
        corners[0] if corners.ndim == 3 else corners,
        dtype=np.float32).reshape(-1, 2)
    obj   = marker_object_points(MARKER_LENGTH_M)
    flags = (cv2.SOLVEPNP_IPPE_SQUARE
             if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
             else cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, pts2d, K, dist, flags=flags)
    if not ok:
        raise RuntimeError("solvePnP falló")
    return rvec, tvec

# ── Matrices OpenGL ───────────────────────────────────────────────────────────
def projection_from_k(K, w, h, znear, zfar):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0] =  2.*fx/w;  P[1,1] =  2.*fy/h
    P[0,2] =  (w-2.*cx)/w;  P[1,2] = (2.*cy-h)/h
    P[2,2] = -(zfar+znear)/(zfar-znear);  P[2,3] = -1.
    P[3,2] = -2.*zfar*znear/(zfar-znear)
    return P

def modelview_from_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    M    = np.eye(4, dtype=np.float64)
    M[:3,:3] = R;  M[:3,3] = tvec.flatten()
    return (np.diag([1.,-1.,-1.,1.]) @ M).T.astype(np.float32)

# ── Textura fondo webcam ──────────────────────────────────────────────────────
_tex_id = None; _tex_buf = None

def upload_frame_texture(frame_bgr, w, h):
    global _tex_id, _tex_buf
    rgb = cv2.flip(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 0)
    if _tex_buf is None or _tex_buf.shape[:2] != (h, w):
        _tex_buf = np.empty((h, w, 3), dtype=np.uint8)
    np.copyto(_tex_buf, rgb)
    if _tex_id is None:
        _tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,_tex_buf)

def draw_background_quad(w, h):
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0,w,0,h,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, _tex_id)
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(0,0);  glTexCoord2f(1,0); glVertex2f(w,0)
    glTexCoord2f(1,1); glVertex2f(w,h);  glTexCoord2f(0,1); glVertex2f(0,h)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST)

# ── Primitivas base ───────────────────────────────────────────
def draw_generic_cube(w, h, d, r, g, b):
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_QUADS)

    # Frente
    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)

    # Atrás
    glColor3f(r * 0.82, g * 0.82, b * 0.82)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(-w_h, h, -d_h)

    # Izquierda
    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(-w_h, h, d_h)
    glVertex3f(-w_h, h, -d_h)

    # Derecha
    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(w_h, h, -d_h)

    # Arriba
    glColor3f(r * 0.92, g * 0.92, b * 0.92)
    glVertex3f(-w_h, h, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)

    glEnd()


def draw_pyramid(w, h, d, r, g, b):
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_TRIANGLES)

    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glColor3f(r * 0.82, g * 0.82, b * 0.82)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(0, h, 0)

    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glEnd()


def draw_windows(w, h, d, rows, cols):
    w_half, d_half = w / 2.0, d / 2.0
    win_w = w / (cols * 2)
    win_h = h / (rows * 2)

    glColor3f(0.97, 0.95, 0.45)
    glBegin(GL_QUADS)
    z_f = d_half + 0.01

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 3 == 0:
                continue
            x = -w_half + (c * (w / cols)) + win_w / 2
            y = (r * (h / rows)) + win_h / 2
            glVertex3f(x, y, z_f)
            glVertex3f(x + win_w, y, z_f)
            glVertex3f(x + win_w, y + win_h, z_f)
            glVertex3f(x, y + win_h, z_f)

    glEnd()


# ── Casas / edificios ─────────────────────────────────────────
def draw_detailed_house(w, h, d, r, g, b):
    draw_generic_cube(w, h, d, r, g, b)

    # puerta
    glPushMatrix()
    glTranslatef(0, 0, d / 2 + 0.02)
    draw_generic_cube(w * 0.22, h * 0.55, 0.05, 0.35, 0.2, 0.1)
    glPopMatrix()

    # ventanas
    glPushMatrix()
    glTranslatef(0, 0.35, 0)
    draw_windows(w * 0.8, h * 0.55, d, 2, 2)
    glPopMatrix()

    # techo
    w_h, d_h = (w + 0.3) / 2.0, (d + 0.3) / 2.0
    roof_y = h + 0.9
    glBegin(GL_TRIANGLES)
    glColor3f(0.82, 0.28, 0.12)
    glVertex3f(-w_h, h, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(0, roof_y, 0)

    glVertex3f(-w_h, h, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(0, roof_y, 0)
    glEnd()


def draw_apartment_block(w, h, d, r, g, b):
    draw_generic_cube(w, h, d, r, g, b)
    draw_windows(w, h * 0.95, d, max(4, int(h // 1.6)), max(3, int(w // 1.3)))


def draw_school():
    draw_generic_cube(12.0, 5.0, 5.0, 0.72, 0.72, 0.74)
    glPushMatrix()
    glTranslatef(4.0, 0, 4.0)
    draw_generic_cube(4.0, 5.0, 4.0, 0.72, 0.72, 0.74)
    glPopMatrix()
    draw_windows(10.0, 4.0, 5.0, 3, 5)

    glPushMatrix()
    glTranslatef(0, 5.0, 2.0)
    draw_generic_cube(2.5, 1.0, 0.5, 0.6, 0.1, 0.1)
    glPopMatrix()


def draw_church():
    draw_generic_cube(6.0, 6.0, 10.0, 0.85, 0.82, 0.75)

    glPushMatrix()
    glTranslatef(0, 0, 4.0)
    draw_generic_cube(3.0, 11.0, 3.0, 0.75, 0.72, 0.65)
    glTranslatef(0, 11.0, 0)
    draw_pyramid(3.4, 3.0, 3.4, 0.3, 0.3, 0.35)
    glTranslatef(0, 3.0, 0)
    draw_generic_cube(0.2, 1.2, 0.2, 0.9, 0.8, 0.2)
    glTranslatef(0, 0.4, 0)
    draw_generic_cube(0.8, 0.2, 0.2, 0.9, 0.8, 0.2)
    glPopMatrix()


def draw_small_playground():
    glBegin(GL_QUADS)
    glColor3f(0.35, 0.65, 0.3)
    glVertex3f(-7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, 7.0)
    glVertex3f(-7.0, 0.02, 7.0)
    glEnd()

    # columpio
    glPushMatrix()
    glTranslatef(-3.5, 0, -2.0)
    draw_generic_cube(0.1, 2.0, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(4.0, 0, 0)
    draw_generic_cube(0.1, 2.0, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(-2.0, 2.0, 0)
    draw_generic_cube(4.2, 0.1, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(0, -1.3, 0)
    draw_generic_cube(1.0, 0.08, 0.4, 0.8, 0.2, 0.2)
    glPopMatrix()

    # resbaladilla
    glPushMatrix()
    glTranslatef(2.5, 0, -2.0)
    draw_generic_cube(0.8, 1.6, 0.8, 0.2, 0.4, 0.8)
    glPushMatrix()
    glTranslatef(0, 0.7, 1.2)
    glRotatef(30, 1, 0, 0)
    draw_generic_cube(0.7, 0.1, 2.2, 0.85, 0.85, 0.85)
    glPopMatrix()
    glPopMatrix()

    # pasamanos
    glPushMatrix()
    glTranslatef(0, 0, 3.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 0, -4.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 1.8, 2.0)
    draw_generic_cube(0.12, 0.1, 4.2, 0.8, 0.6, 0.1)
    glPopMatrix()


def draw_kiosk():
    glPushMatrix()
    draw_generic_cube(3.2, 0.6, 3.2, 0.5, 0.35, 0.25)

    for px, pz in [(-1.4, -1.4), (1.4, -1.4), (1.4, 1.4), (-1.4, 1.4)]:
        glPushMatrix()
        glTranslatef(px, 0.6, pz)
        draw_generic_cube(0.15, 2.2, 0.15, 0.82, 0.73, 0.55)
        glPopMatrix()

    glTranslatef(0, 2.8, 0)
    draw_pyramid(3.6, 1.5, 3.6, 0.7, 0.2, 0.2)
    glPopMatrix()


# ── Infraestructura ───────────────────────────────────────────
def draw_street_lamp():
    glPushMatrix()
    draw_generic_cube(0.2, 4.0, 0.2, 0.2, 0.2, 0.22)
    glTranslatef(0, 4.0, 0.35)
    draw_generic_cube(0.2, 0.15, 0.9, 0.2, 0.2, 0.22)
    glTranslatef(0, -0.05, 0.35)
    draw_generic_cube(0.45, 0.22, 0.45, 0.95, 0.95, 0.5)
    glPopMatrix()


def draw_traffic_light():
    glPushMatrix()
    draw_generic_cube(0.15, 3.5, 0.15, 0.15, 0.15, 0.15)
    glTranslatef(0, 3.5, 0)
    draw_generic_cube(0.4, 1.0, 0.4, 0.1, 0.1, 0.1)

    glTranslatef(0, 0.3, 0.21)
    draw_generic_cube(0.2, 0.18, 0.02, 0.9, 0.1, 0.1)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.2, 0.18, 0.02, 0.9, 0.8, 0.1)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.2, 0.18, 0.02, 0.1, 0.8, 0.1)
    glPopMatrix()


def draw_bench():
    glPushMatrix()
    draw_generic_cube(1.2, 0.12, 0.35, 0.45, 0.28, 0.1)
    glPushMatrix()
    glTranslatef(-0.45, -0.01, 0)
    draw_generic_cube(0.08, 0.45, 0.08, 0.2, 0.2, 0.2)
    glTranslatef(0.9, 0, 0)
    draw_generic_cube(0.08, 0.45, 0.08, 0.2, 0.2, 0.2)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0.32, -0.12)
    draw_generic_cube(1.2, 0.12, 0.15, 0.45, 0.28, 0.1)
    glPopMatrix()
    glPopMatrix()


# ── Objetos extra para llenar el parque ───────────────────────
def draw_trash_can():
    glPushMatrix()
    draw_generic_cube(0.45, 0.75, 0.45, 0.08, 0.22, 0.08)
    glTranslatef(0, 0.78, 0)
    draw_generic_cube(0.55, 0.08, 0.55, 0.05, 0.12, 0.05)
    glPopMatrix()


def draw_flower_patch():
    glPushMatrix()
    draw_generic_cube(1.25, 0.08, 1.25, 0.25, 0.15, 0.07)
    for x, z, r, g, b in [
        (-0.4, -0.35, 0.9, 0.1, 0.2), (0.0, -0.35, 0.9, 0.8, 0.1),
        (0.4, -0.35, 0.7, 0.2, 0.9), (-0.25, 0.1, 0.2, 0.6, 1.0),
        (0.25, 0.15, 1.0, 0.45, 0.1), (0.0, 0.45, 1.0, 1.0, 1.0)
    ]:
        glPushMatrix()
        glTranslatef(x, 0.08, z)
        draw_generic_cube(0.16, 0.16, 0.16, r, g, b)
        glPopMatrix()
    glPopMatrix()


def draw_picnic_table():
    glPushMatrix()
    draw_generic_cube(1.55, 0.12, 0.65, 0.45, 0.25, 0.08)
    glPushMatrix(); glTranslatef(0, -0.25, 0.42); draw_generic_cube(1.7, 0.1, 0.22, 0.45, 0.25, 0.08); glPopMatrix()
    glPushMatrix(); glTranslatef(0, -0.25, -0.42); draw_generic_cube(1.7, 0.1, 0.22, 0.45, 0.25, 0.08); glPopMatrix()
    for x in [-0.55, 0.55]:
        for z in [-0.2, 0.2]:
            glPushMatrix()
            glTranslatef(x, -0.65, z)
            draw_generic_cube(0.09, 0.65, 0.09, 0.2, 0.2, 0.2)
            glPopMatrix()
    glPopMatrix()


def draw_fountain(t=0.0):
    glPushMatrix()
    draw_generic_cube(3.0, 0.35, 3.0, 0.55, 0.55, 0.58)
    glTranslatef(0, 0.35, 0)
    draw_generic_cube(2.35, 0.12, 2.35, 0.1, 0.35, 0.8)
    glTranslatef(0, 0.12, 0)
    draw_generic_cube(0.55, 1.1, 0.55, 0.72, 0.72, 0.75)

    # chorros de agua animados
    for i, ang in enumerate([0, 90, 180, 270]):
        height = 0.7 + 0.35 * math.sin(t * 3.0 + i)
        glPushMatrix()
        glRotatef(ang, 0, 1, 0)
        glTranslatef(0, 1.0, 0.55)
        draw_generic_cube(0.08, height, 0.08, 0.25, 0.65, 1.0)
        glPopMatrix()
    glPopMatrix()


def draw_statue():
    glPushMatrix()
    draw_generic_cube(1.3, 0.45, 1.3, 0.52, 0.52, 0.55)
    glTranslatef(0, 0.45, 0)
    draw_generic_cube(0.55, 1.35, 0.45, 0.65, 0.65, 0.68)
    glTranslatef(0, 1.35, 0)
    draw_generic_cube(0.38, 0.38, 0.38, 0.68, 0.68, 0.7)
    glPopMatrix()


def draw_park_path():
    # Caminos claros dentro del área verde
    glColor3f(0.62, 0.52, 0.38)
    glBegin(GL_QUADS)
    glVertex3f(-76, 0.025, -3)
    glVertex3f(-24, 0.025, -3)
    glVertex3f(-24, 0.025, 3)
    glVertex3f(-76, 0.025, 3)

    glVertex3f(-53, 0.026, 68)
    glVertex3f(-47, 0.026, 68)
    glVertex3f(-47, 0.026, -68)
    glVertex3f(-53, 0.026, -68)
    glEnd()


def draw_pond():
    glPushMatrix()
    # Lago rectangular sencillo con borde de piedra
    draw_generic_cube(8.0, 0.08, 5.2, 0.45, 0.45, 0.42)
    glTranslatef(0, 0.09, 0)
    draw_generic_cube(7.0, 0.05, 4.2, 0.08, 0.35, 0.75)
    glPopMatrix()


def draw_umbrella_table():
    glPushMatrix()
    draw_generic_cube(1.0, 0.12, 1.0, 0.48, 0.28, 0.1)
    glPushMatrix(); glTranslatef(0, -0.55, 0); draw_generic_cube(0.12, 0.6, 0.12, 0.25, 0.25, 0.25); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0.1, 0); draw_generic_cube(0.08, 1.35, 0.08, 0.25, 0.25, 0.25); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 1.45, 0); draw_pyramid(2.2, 0.75, 2.2, 0.9, 0.15, 0.15); glPopMatrix()
    glPopMatrix()


def draw_hedge_maze_piece(w=4.0, d=0.7):
    draw_generic_cube(w, 0.75, d, 0.05, 0.35, 0.08)


def draw_soccer_stadium():
    glPushMatrix()
    # base del estadio
    draw_generic_cube(18.0, 0.35, 12.0, 0.52, 0.52, 0.55)
    glTranslatef(0, 0.38, 0)

    # cancha
    glBegin(GL_QUADS)
    glColor3f(0.12, 0.55, 0.16)
    glVertex3f(-7.2, 0.03, -4.2)
    glVertex3f(7.2, 0.03, -4.2)
    glVertex3f(7.2, 0.03, 4.2)
    glVertex3f(-7.2, 0.03, 4.2)
    glEnd()

    # líneas de cancha
    glLineWidth(2.0)
    glColor3f(1, 1, 1)
    glBegin(GL_LINES)
    glVertex3f(0, 0.08, -4.2); glVertex3f(0, 0.08, 4.2)
    glVertex3f(-7.2, 0.08, 0); glVertex3f(7.2, 0.08, 0)
    glEnd()

    # gradas laterales
    for z, rot in [(-5.2, 0), (5.2, 180)]:
        glPushMatrix()
        glTranslatef(0, 0.0, z)
        for i in range(3):
            glPushMatrix()
            glTranslatef(0, i * 0.45, i * 0.35 if z < 0 else -i * 0.35)
            draw_generic_cube(18.5 - i * 1.2, 0.28, 0.65, 0.72, 0.72, 0.74)
            glPopMatrix()
        glPopMatrix()

    # porterías
    for x in [-6.7, 6.7]:
        glPushMatrix()
        glTranslatef(x, 0.1, 0)
        draw_generic_cube(0.12, 1.0, 2.4, 0.95, 0.95, 0.95)
        glPopMatrix()

    # torres de luces
    for x, z in [(-8.5, -5.5), (8.5, -5.5), (-8.5, 5.5), (8.5, 5.5)]:
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_generic_cube(0.18, 3.2, 0.18, 0.2, 0.2, 0.22)
        glTranslatef(0, 3.2, 0)
        draw_generic_cube(0.7, 0.28, 0.25, 0.95, 0.95, 0.65)
        glPopMatrix()

    glPopMatrix()


def draw_airplane(t):
    # avión pequeño que cruza el cielo
    x = ((t * 5.0) % 170.0) - 85.0
    z = -55 + 18.0 * math.sin(t * 0.22)
    y = 28.0 + 3.0 * math.sin(t * 0.8)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(90, 0, 1, 0)
    draw_generic_cube(1.1, 0.55, 5.5, 0.88, 0.88, 0.9)
    glPushMatrix(); glTranslatef(0, 0.08, 0); draw_generic_cube(7.0, 0.15, 0.9, 0.78, 0.78, 0.82); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0.28, -2.45); draw_pyramid(1.3, 1.0, 1.0, 0.65, 0.1, 0.1); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0.05, 2.85); draw_pyramid(1.0, 0.65, 0.9, 0.9, 0.9, 0.92); glPopMatrix()
    glPopMatrix()


def draw_varied_building(w, h, d, r, g, b, style=0):
    # Edificios con estilos distintos para que la ciudad no se vea repetida.
    if style % 5 == 0:
        draw_apartment_block(w, h, d, r, g, b)
        glPushMatrix()
        glTranslatef(0, h, 0)
        draw_generic_cube(w * 0.7, h * 0.18, d * 0.7, r * 0.85, g * 0.85, b * 0.85)
        glPopMatrix()
    elif style % 5 == 1:
        draw_generic_cube(w, h, d, r, g, b)
        # franjas de vidrio verticales
        for ox in [-w * 0.25, 0, w * 0.25]:
            glPushMatrix()
            glTranslatef(ox, 0.1, d / 2 + 0.03)
            draw_generic_cube(w * 0.12, h * 0.88, 0.04, 0.15, 0.55, 0.85)
            glPopMatrix()
        glPushMatrix(); glTranslatef(0, h, 0); draw_generic_cube(w * 0.25, 2.0, d * 0.25, 0.08, 0.08, 0.1); glPopMatrix()
    elif style % 5 == 2:
        draw_apartment_block(w, h * 0.72, d, r, g, b)
        glPushMatrix(); glTranslatef(0, h * 0.72, 0); draw_apartment_block(w * 0.72, h * 0.28, d * 0.72, r * 0.9, g * 0.9, b * 0.9); glPopMatrix()
        glPushMatrix(); glTranslatef(0, h + 0.1, 0); draw_pyramid(w * 0.75, 1.6, d * 0.75, 0.25, 0.25, 0.3); glPopMatrix()
    elif style % 5 == 3:
        draw_generic_cube(w, h, d, r, g, b)
        draw_windows(w, h * 0.95, d, max(4, int(h // 2)), max(3, int(w // 1.4)))
        # anexo lateral
        glPushMatrix(); glTranslatef(w * 0.55, 0, 0); draw_generic_cube(w * 0.35, h * 0.58, d * 0.8, r * 0.75, g * 0.75, b * 0.75); glPopMatrix()
    else:
        draw_apartment_block(w, h, d, r, g, b)
        # techo tipo helipuerto
        glPushMatrix()
        glTranslatef(0, h + 0.03, 0)
        draw_generic_cube(w * 0.9, 0.12, d * 0.9, 0.12, 0.12, 0.13)
        glColor3f(0.95, 0.95, 0.95)
        glBegin(GL_LINES)
        glVertex3f(-w * 0.25, 0.18, 0); glVertex3f(w * 0.25, 0.18, 0)
        glVertex3f(0, 0.18, -d * 0.25); glVertex3f(0, 0.18, d * 0.25)
        glEnd()
        glPopMatrix()


def draw_extra_park_objects(t):
    # Parque más lleno: caminos, estadio, lago, mesas, flores, fuentes, bancas y decoración.
    draw_park_path()

    # Estadio sencillo en el área verde
    glPushMatrix()
    glTranslatef(-35, 0, 56)
    draw_soccer_stadium()
    glPopMatrix()

    # Lago pequeño
    glPushMatrix()
    glTranslatef(-68, 0, 22)
    draw_pond()
    glPopMatrix()

    # Bancas alrededor del parque
    for i, z in enumerate(range(-62, 63, 8)):
        glPushMatrix()
        glTranslatef(-74, 0.25, z)
        glRotatef(90, 0, 1, 0)
        draw_bench()
        glPopMatrix()

        glPushMatrix()
        glTranslatef(-26, 0.25, z + 3)
        glRotatef(-90, 0, 1, 0)
        draw_bench()
        glPopMatrix()

    # Mesas normales y con sombrilla
    for x, z in [(-69, -56), (-61, -48), (-36, -54), (-31, -40), (-70, 44), (-62, 54), (-55, 62), (-29, 40)]:
        glPushMatrix(); glTranslatef(x, 0, z); draw_picnic_table(); glPopMatrix()

    for x, z in [(-69, -12), (-62, 13), (-36, 14), (-31, -14), (-58, 28), (-45, 31), (-73, 58), (-49, -61)]:
        glPushMatrix(); glTranslatef(x, 0.65, z); draw_umbrella_table(); glPopMatrix()

    # Muchas flores para que no se vea vacío
    flower_positions = [
        (-67, -6), (-63, 6), (-58, -2), (-42, 6), (-37, -5), (-32, 4),
        (-70, 20), (-34, -20), (-52, 42), (-48, -44), (-74, 10), (-74, -18),
        (-65, 33), (-61, 38), (-56, 34), (-43, 36), (-38, 31), (-31, 25),
        (-72, -48), (-66, -39), (-55, -50), (-44, -58), (-30, -50), (-28, -28),
    ]
    for x, z in flower_positions:
        glPushMatrix(); glTranslatef(x, 0, z); draw_flower_patch(); glPopMatrix()

    # Botes de basura
    for x, z in [(-76, -36), (-76, 36), (-24, -36), (-24, 36), (-56, 0), (-44, 0), (-72, 62), (-28, 62), (-71, -62), (-29, -62)]:
        glPushMatrix(); glTranslatef(x, 0, z); draw_trash_can(); glPopMatrix()

    # Fuentes y estatua
    glPushMatrix(); glTranslatef(-58, 0, 38); draw_fountain(t); glPopMatrix()
    glPushMatrix(); glTranslatef(-39, 0, -35); draw_statue(); glPopMatrix()

    # Pequeño laberinto/jardín de setos
    for x, z, w, d, rot in [
        (-68, -30, 6, 0.7, 0), (-62, -26, 5, 0.7, 90), (-56, -30, 6, 0.7, 0),
        (-66, -36, 4, 0.7, 90), (-52, -36, 4, 0.7, 90), (-59, -40, 8, 0.7, 0),
    ]:
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(rot, 0, 1, 0)
        draw_hedge_maze_piece(w, d)
        glPopMatrix()


def draw_moving_park_people(t, colors_people):
    # Personas caminando dentro del parque en rutas circulares/rectangulares
    walkers = [
        (-58, -18, 6.5, 4.0, 0.0), (-41, 18, 7.0, 5.0, 1.1),
        (-66, 35, 5.5, 3.5, 2.2), (-35, -18, 5.0, 4.5, 3.0),
        (-62, -45, 4.5, 5.0, 0.6), (-43, 47, 5.8, 4.0, 1.7),
        (-70, 5, 4.0, 8.0, 2.8), (-30, 15, 3.5, 7.0, 3.6),
        (-55, 0, 7.5, 2.7, 4.4), (-50, -55, 8.0, 3.0, 5.1),
    ]
    for i, (cx, cz, rx, rz, phase) in enumerate(walkers):
        px = cx + math.sin(t * 0.65 + phase) * rx
        pz = cz + math.cos(t * 0.65 + phase) * rz
        angle = math.degrees(math.atan2(math.cos(t * 0.65 + phase) * rx,
                                        -math.sin(t * 0.65 + phase) * rz))
        glPushMatrix()
        glTranslatef(px, 0, pz)
        glRotatef(angle, 0, 1, 0)
        draw_person(*colors_people[i % len(colors_people)])
        glPopMatrix()


def draw_moving_city_people(t, colors_people):
    # Personas caminando por banquetas de la ciudad
    walkers = [
        (8, -12, 8, 0), (24, 19, 7, 1), (40, -28, 9, 2),
        (57, 36, 8, 3), (73, 5, 6, 4), (52, 54, 10, 5),
        (19, 40, 7, 6), (5, 27, 5, 7), (66, -5, 8, 8),
        (34, 8, 6, 9),
    ]
    for i, (cx, cz, dist, phase) in enumerate(walkers):
        offset = math.sin(t * 0.8 + phase) * dist
        # unos caminan sobre X y otros sobre Z
        if i % 2 == 0:
            px, pz = cx + offset, cz
            rot = 90 if math.cos(t * 0.8 + phase) > 0 else -90
        else:
            px, pz = cx, cz + offset
            rot = 0 if math.cos(t * 0.8 + phase) > 0 else 180
        glPushMatrix()
        glTranslatef(px, 0, pz)
        glRotatef(rot, 0, 1, 0)
        draw_person(*colors_people[(i + 1) % len(colors_people)])
        glPopMatrix()


def draw_moving_traffic(t):
    # Carros y motos animados sobre calles existentes
    car_colors = [
        (0.9, 0.1, 0.1), (0.1, 0.35, 0.9), (0.95, 0.85, 0.1),
        (0.1, 0.7, 0.25), (0.95, 0.95, 0.95), (0.9, 0.45, 0.05),
        (0.45, 0.1, 0.75), (0.05, 0.05, 0.05)
    ]
    roads_x = [-6, 10, 26, 42, 58, 74]
    roads_z = [-58, -42, -26, -10, 6, 22, 38, 54, 70]

    # Carros moviéndose verticalmente
    for i, x in enumerate(roads_x):
        z = ((t * (5.0 + i * 0.35) + i * 19) % 144) - 72
        glPushMatrix()
        glTranslatef(x - 0.65, 0.1, z)
        if i % 2 == 1:
            glRotatef(180, 0, 1, 0)
        draw_car(*car_colors[i % len(car_colors)])
        glPopMatrix()

    # Carros moviéndose horizontalmente
    for i, z in enumerate(roads_z[1:-1]):
        x = ((t * (4.7 + i * 0.25) + i * 15) % 96) - 18
        glPushMatrix()
        glTranslatef(x, 0.1, z + 0.65)
        glRotatef(90 if i % 2 == 0 else -90, 0, 1, 0)
        draw_car(*car_colors[(i + 3) % len(car_colors)])
        glPopMatrix()

    # Motos más rápidas
    for i, z in enumerate([-58, -26, 6, 38, 70]):
        x = ((t * (7.5 + i * 0.45) + i * 21) % 96) - 18
        glPushMatrix()
        glTranslatef(x, 0.15, z - 0.65)
        glRotatef(90, 0, 1, 0)
        draw_motorcycle(*car_colors[(i + 5) % len(car_colors)])
        glPopMatrix()

    # Camiones lentos
    for i, x in enumerate([-6, 42, 74]):
        z = ((t * (2.8 + i * 0.3) + i * 40) % 144) - 72
        glPushMatrix()
        glTranslatef(x + 0.7, 0.1, z)
        if i == 1:
            glRotatef(180, 0, 1, 0)
        draw_truck()
        glPopMatrix()


def draw_crosswalk(width=4.6, depth=3.0, stripes=6, horizontal=False):
    stripe_w = width / (stripes * 2)
    glColor3f(0.95, 0.95, 0.95)
    glBegin(GL_QUADS)
    for i in range(stripes):
        if not horizontal:
            x0 = -width / 2 + i * stripe_w * 2
            x1 = x0 + stripe_w
            glVertex3f(x0, 0.03, -depth / 2)
            glVertex3f(x1, 0.03, -depth / 2)
            glVertex3f(x1, 0.03, depth / 2)
            glVertex3f(x0, 0.03, depth / 2)
        else:
            z0 = -width / 2 + i * stripe_w * 2
            z1 = z0 + stripe_w
            glVertex3f(-depth / 2, 0.03, z0)
            glVertex3f(depth / 2, 0.03, z0)
            glVertex3f(depth / 2, 0.03, z1)
            glVertex3f(-depth / 2, 0.03, z1)
    glEnd()


# ── Personajes / mascotas ─────────────────────────────────────
def draw_person(r, g, b):
    glPushMatrix()

    # piernas
    glPushMatrix()
    glTranslatef(-0.06, 0, 0)
    draw_generic_cube(0.06, 0.38, 0.06, 0.12, 0.12, 0.15)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.06, 0, 0)
    draw_generic_cube(0.06, 0.38, 0.06, 0.12, 0.12, 0.15)
    glPopMatrix()

    # torso
    glPushMatrix()
    glTranslatef(0, 0.38, 0)
    draw_generic_cube(0.24, 0.34, 0.12, r, g, b)
    glPopMatrix()

    # brazos
    glPushMatrix()
    glTranslatef(-0.16, 0.42, 0)
    draw_generic_cube(0.05, 0.28, 0.05, 0.86, 0.73, 0.62)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.16, 0.42, 0)
    draw_generic_cube(0.05, 0.28, 0.05, 0.86, 0.73, 0.62)
    glPopMatrix()

    # cabeza
    glPushMatrix()
    glTranslatef(0, 0.72, 0)
    draw_generic_cube(0.18, 0.2, 0.18, 0.88, 0.75, 0.64)
    glPopMatrix()

    glPopMatrix()


def draw_dog():
    glPushMatrix()
    draw_generic_cube(0.3, 0.3, 0.6, 0.55, 0.27, 0.07)

    glPushMatrix()
    glTranslatef(-0.1, -0.15, 0.2)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(0.2, 0, 0)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(0, 0, -0.4)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(-0.2, 0, 0)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glPopMatrix()

    glTranslatef(0, 0.25, 0.25)
    draw_generic_cube(0.25, 0.25, 0.25, 0.55, 0.27, 0.07)
    glTranslatef(0, 0.05, -0.1)
    draw_generic_cube(0.32, 0.1, 0.1, 0.3, 0.15, 0.0)
    glPopMatrix()


# ── Vehículos ─────────────────────────────────────────────────
def draw_truck():
    glPushMatrix()
    draw_generic_cube(1.6, 1.8, 4.0, 0.85, 0.85, 0.85)
    glTranslatef(0, 0, 2.3)
    draw_generic_cube(1.5, 1.1, 1.2, 0.8, 0.1, 0.1)
    glPopMatrix()


def draw_motorcycle(r, g, b):
    glPushMatrix()
    draw_generic_cube(0.4, 0.45, 1.2, r, g, b)
    glTranslatef(0, -0.14, 0.45)
    draw_generic_cube(0.22, 0.26, 0.28, 0.05, 0.05, 0.05)
    glTranslatef(0, 0, -0.9)
    draw_generic_cube(0.22, 0.26, 0.28, 0.05, 0.05, 0.05)
    glPopMatrix()


def draw_car(r, g, b):
    glPushMatrix()
    draw_generic_cube(1.15, 0.42, 2.05, r, g, b)
    glPushMatrix()
    glTranslatef(0, 0.42, -0.1)
    draw_generic_cube(0.9, 0.35, 1.1, r * 0.6, g * 0.6, b * 0.6)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, -1.1)
    draw_generic_cube(1.25, 0.24, 0.38, 0.05, 0.05, 0.05)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, 1.1)
    draw_generic_cube(1.25, 0.24, 0.38, 0.05, 0.05, 0.05)
    glPopMatrix()

    glPopMatrix()


# ── Canchas / vegetación ──────────────────────────────────────
def draw_volleyball_court():
    glBegin(GL_QUADS)
    glColor3f(0.9, 0.5, 0.2)
    glVertex3f(-4.0, 0.02, -7.0)
    glVertex3f(4.0, 0.02, -7.0)
    glVertex3f(4.0, 0.02, 7.0)
    glVertex3f(-4.0, 0.02, 7.0)

    glColor3f(0.1, 0.4, 0.7)
    glVertex3f(-3.0, 0.025, -6.0)
    glVertex3f(3.0, 0.025, -6.0)
    glVertex3f(3.0, 0.025, 6.0)
    glVertex3f(-3.0, 0.025, 6.0)
    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(-3.0, 0.03, 0.0)
    glVertex3f(3.0, 0.03, 0.0)
    glEnd()

    draw_generic_cube(0.1, 2.2, 0.1, 0.6, 0.6, 0.6)
    glPushMatrix()
    glTranslatef(0, 1.3, 0)
    draw_generic_cube(6.0, 0.7, 0.02, 0.9, 0.9, 0.9)
    glPopMatrix()


def draw_basketball_court():
    glBegin(GL_QUADS)
    glColor3f(0.75, 0.52, 0.3)
    glVertex3f(-4.0, 0.02, -8.0)
    glVertex3f(4.0, 0.02, -8.0)
    glVertex3f(4.0, 0.02, 8.0)
    glVertex3f(-4.0, 0.02, 8.0)
    glEnd()

    glPushMatrix()
    glTranslatef(0, 0, -7.5)
    draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2)
    glTranslatef(0, 3.0, 0.2)
    draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0)
    glTranslatef(0, -0.3, 0.2)
    draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, 7.5)
    draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2)
    glTranslatef(0, 3.0, -0.2)
    draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0)
    glTranslatef(0, -0.3, -0.2)
    draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1)
    glPopMatrix()


def draw_tree_round():
    draw_generic_cube(0.25, 1.0, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 1.0, 0)
    draw_generic_cube(1.2, 1.1, 1.2, 0.2, 0.55, 0.2)
    glPopMatrix()


def draw_tree_square():
    draw_generic_cube(0.25, 0.8, 0.25, 0.35, 0.2, 0.1)
    glPushMatrix()
    glTranslatef(0, 0.8, 0)
    draw_generic_cube(0.9, 1.4, 0.9, 0.15, 0.45, 0.15)
    glPopMatrix()


def draw_tree_pine():
    draw_generic_cube(0.25, 0.7, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.6, 0)
    draw_pyramid(1.5, 1.0, 1.5, 0.1, 0.38, 0.15)
    glTranslatef(0, 0.6, 0)
    draw_pyramid(1.1, 0.9, 1.1, 0.12, 0.42, 0.18)
    glTranslatef(0, 0.5, 0)
    draw_pyramid(0.7, 0.7, 0.7, 0.15, 0.48, 0.22)
    glPopMatrix()


def draw_tree_double_sphere():
    draw_generic_cube(0.2, 1.8, 0.2, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.8, 0)
    draw_generic_cube(0.9, 0.7, 0.9, 0.25, 0.6, 0.25)
    glTranslatef(0, 0.7, 0)
    draw_generic_cube(0.7, 0.5, 0.7, 0.3, 0.65, 0.3)
    glPopMatrix()


# ── Dibujado auxiliar de calles ───────────────────────────────
def draw_lane_markers_vertical(x, z0, z1):
    glColor3f(0.95, 0.85, 0.2)
    glBegin(GL_QUADS)
    z = z0
    while z < z1:
        glVertex3f(x - 0.08, 0.025, z)
        glVertex3f(x + 0.08, 0.025, z)
        glVertex3f(x + 0.08, 0.025, z + 2.0)
        glVertex3f(x - 0.08, 0.025, z + 2.0)
        z += 4.0
    glEnd()


def draw_lane_markers_horizontal(z, x0, x1):
    glColor3f(0.95, 0.85, 0.2)
    glBegin(GL_QUADS)
    x = x0
    while x < x1:
        glVertex3f(x, 0.025, z - 0.08)
        glVertex3f(x + 2.0, 0.025, z - 0.08)
        glVertex3f(x + 2.0, 0.025, z + 0.08)
        glVertex3f(x, 0.025, z + 0.08)
        x += 4.0
    glEnd()


# ── Escenario principal ───────────────────────────────────────
def draw_scenery(t=None):
    if t is None:
        t = time.time()

    # Avión animado cruzando el cielo
    draw_airplane(t)

    # Terreno
    glBegin(GL_QUADS)
    glColor3f(0.14, 0.14, 0.15)
    glVertex3f(-85, -0.01, 85)
    glVertex3f(85, -0.01, 85)
    glVertex3f(85, -0.01, -85)
    glVertex3f(-85, -0.01, -85)
    glEnd()

    # Parque principal grande
    glBegin(GL_QUADS)
    glColor3f(0.18, 0.42, 0.22)
    glVertex3f(-78, 0.01, 72)
    glVertex3f(-22, 0.01, 72)
    glVertex3f(-22, 0.01, -72)
    glVertex3f(-78, 0.01, -72)
    glEnd()

    # Objetos extra para que el área verde se vea llena
    draw_extra_park_objects(t)

    # Árboles del parque
    for z in range(-66, 67, 7):
        glPushMatrix(); glTranslatef(-72, 0, z); draw_tree_round(); glPopMatrix()
        glPushMatrix(); glTranslatef(-64, 0, z); draw_tree_square(); glPopMatrix()

        if z < -12 or z > 12:
            glPushMatrix(); glTranslatef(-54, 0, z); draw_tree_pine(); glPopMatrix()
            glPushMatrix(); glTranslatef(-44, 0, z); draw_tree_double_sphere(); glPopMatrix()

    # Canchas
    glPushMatrix(); glTranslatef(-47, 0, -12); draw_volleyball_court(); glPopMatrix()
    glPushMatrix(); glTranslatef(-47, 0, 12); draw_basketball_court(); glPopMatrix()

    # Kiosko y bancas
    glPushMatrix(); glTranslatef(-58, 0, -32); draw_kiosk(); glPopMatrix()
    glPushMatrix(); glTranslatef(-64, 0, -27); draw_bench(); glPopMatrix()
    glPushMatrix(); glTranslatef(-51, 0, -27); draw_bench(); glRotatef(180, 0, 1, 0); draw_bench(); glPopMatrix()

    # Personas en parque: algunas quietas y varias con movimiento
    people_park = [(-60, -20), (-55, -35), (-46, -5), (-48, 22), (-61, 30)]
    colors_people = [
        (0.8, 0.2, 0.2), (0.2, 0.35, 0.85), (0.15, 0.7, 0.2),
        (0.85, 0.6, 0.15), (0.6, 0.2, 0.75)
    ]
    for i, (px, pz) in enumerate(people_park):
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(*colors_people[i % len(colors_people)])
        glPopMatrix()

    draw_moving_park_people(t, colors_people)

    # Escuela
    glPushMatrix()
    glTranslatef(54.0, 0, 56.0)
    draw_school()
    glPopMatrix()

    # Iglesia
    glPushMatrix()
    glTranslatef(67.0, 0, 18.0)
    draw_church()
    glPopMatrix()

    # Parque infantil
    glPushMatrix()
    glTranslatef(30.0, 0, -8.0)
    draw_small_playground()
    glTranslatef(-2.0, 0.3, 1.0)
    draw_dog()
    glPopMatrix()

    # Gente en el parque infantil
    for px, pz, cr, cg, cb in [
        (24, -6, 0.2, 0.4, 0.8),
        (28, 2, 0.8, 0.3, 0.2),
        (34, -4, 0.2, 0.75, 0.25),
        (36, 4, 0.7, 0.6, 0.15),
    ]:
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(cr, cg, cb)
        glPopMatrix()

    # Calles principales
    glColor3f(0.28, 0.28, 0.30)
    glBegin(GL_QUADS)

    roads_x = [-6, 10, 26, 42, 58, 74]
    roads_z = [-58, -42, -26, -10, 6, 22, 38, 54, 70]

    for x in roads_x:
        glVertex3f(x - 1.6, 0.015, 74)
        glVertex3f(x + 1.6, 0.015, 74)
        glVertex3f(x + 1.6, 0.015, -74)
        glVertex3f(x - 1.6, 0.015, -74)

    for z in roads_z:
        glVertex3f(-20, 0.015, z + 1.6)
        glVertex3f(78, 0.015, z + 1.6)
        glVertex3f(78, 0.015, z - 1.6)
        glVertex3f(-20, 0.015, z - 1.6)

    glEnd()

    # Rayas centrales
    for x in roads_x:
        draw_lane_markers_vertical(x, -72, 72)
    for z in roads_z:
        draw_lane_markers_horizontal(z, -18, 76)

    # Cruces peatonales
    for x, z in [(10, -10), (26, 22), (42, -26), (58, 38), (74, 6)]:
        glPushMatrix(); glTranslatef(x, 0, z); draw_crosswalk(); glPopMatrix()
        glPushMatrix(); glTranslatef(x, 0, z); draw_crosswalk(horizontal=True); glPopMatrix()

    # Postes de luz
    for x in [-12, 4, 20, 36, 52, 68]:
        for z in [-50, -18, 14, 46]:
            glPushMatrix()
            glTranslatef(x, 0, z)
            draw_street_lamp()
            glPopMatrix()

    # Semáforos
    for x, z in [(10, -10), (26, 22), (42, -26), (58, 38), (74, 6)]:
        glPushMatrix(); glTranslatef(x - 2.2, 0, z - 2.2); draw_traffic_light(); glPopMatrix()
        glPushMatrix(); glTranslatef(x + 2.2, 0, z + 2.2); draw_traffic_light(); glPopMatrix()

    # Más tráfico: carros
    cars = [
        (-6, -49, 0.85, 0.1, 0.1),
        (10, -33, 0.1, 0.3, 0.8),
        (26, -17, 0.9, 0.8, 0.1),
        (42, -1, 0.15, 0.6, 0.2),
        (58, 15, 0.95, 0.95, 0.95),
        (74, 31, 0.9, 0.4, 0.0),
        (10, 47, 0.4, 0.1, 0.6),
        (26, 63, 0.1, 0.1, 0.1),
        (42, -49, 0.5, 0.35, 0.05),
        (58, -33, 0.15, 0.75, 0.75),
        (74, -17, 0.85, 0.25, 0.5),
        (-6, 15, 0.2, 0.5, 0.9),
        (26, 31, 0.75, 0.2, 0.2),
        (58, 63, 0.2, 0.7, 0.25),
    ]
    for x, z, r, g, b in cars:
        glPushMatrix()
        glTranslatef(x, 0.1, z)
        draw_car(r, g, b)
        glPopMatrix()

    # Camiones
    for x, z in [(-6, -2), (42, 47), (74, -49)]:
        glPushMatrix()
        glTranslatef(x, 0.1, z)
        draw_truck()
        glPopMatrix()

    # Motos
    motos = [
        (10, -57, 0.1, 0.8, 0.8),
        (26, -41, 0.9, 0.1, 0.5),
        (42, -25, 0.1, 0.9, 0.2),
        (58, -9, 0.9, 0.55, 0.0),
        (74, 7, 0.5, 0.2, 0.9),
        (10, 23, 0.95, 0.2, 0.2),
        (26, 39, 0.2, 0.45, 0.95),
        (42, 55, 0.9, 0.85, 0.1)
    ]
    for x, z, r, g, b in motos:
        glPushMatrix()
        glTranslatef(x, 0.15, z)
        draw_motorcycle(r, g, b)
        glPopMatrix()

    # Tráfico con movimiento real
    draw_moving_traffic(t)

    # Distrito comercial / rascacielos
    skyscrapers = [
        (-4, -64, 4.5, 18.0, 4.5, 0.25, 0.35, 0.5),
        (12, -64, 4.0, 24.0, 4.0, 0.2, 0.2, 0.3),
        (28, -64, 5.0, 28.0, 5.0, 0.15, 0.4, 0.45),
        (44, -64, 4.2, 16.0, 4.2, 0.3, 0.3, 0.35),
        (60, -64, 4.4, 20.0, 4.4, 0.25, 0.28, 0.45),
        (-4, -48, 4.0, 15.0, 4.0, 0.35, 0.35, 0.4),
        (12, -48, 5.5, 34.0, 5.5, 0.1, 0.25, 0.5),
        (28, -48, 4.2, 21.0, 4.2, 0.3, 0.2, 0.2),
        (44, -48, 5.0, 26.0, 5.0, 0.2, 0.36, 0.3),
        (60, -48, 4.0, 17.0, 4.0, 0.4, 0.32, 0.32),
        (-4, -32, 3.8, 13.0, 3.8, 0.28, 0.32, 0.36),
        (12, -32, 5.0, 22.0, 5.0, 0.18, 0.24, 0.42),
    ]

    for i, (x, z, w, h, d, r, g, b) in enumerate(skyscrapers):
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_varied_building(w, h, d, r, g, b, i)
        glPopMatrix()

    # Edificios medianos con diseños variados
    medium_buildings = [
        (60, -16, 4.6, 10, 4.0, 0.62, 0.64, 0.68, 2),
        (44, 0, 4.2, 9, 4.8, 0.52, 0.58, 0.72, 3),
        (28, 16, 5.0, 11, 4.0, 0.66, 0.54, 0.50, 4),
        (12, 32, 4.0, 8, 5.0, 0.58, 0.68, 0.62, 5),
        (-4, 16, 4.8, 9, 4.2, 0.70, 0.58, 0.68, 6),
        (68, 52, 5.0, 12, 5.0, 0.45, 0.55, 0.70, 7),
        (36, 66, 4.5, 7, 4.5, 0.72, 0.62, 0.48, 8),
    ]
    for x, z, w, h, d, r, g, b, style in medium_buildings:
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_varied_building(w, h, d, r, g, b, style)
        glPopMatrix()

    # Zona residencial más grande y densa
    house_colors = [
        (0.85, 0.45, 0.45), (0.45, 0.65, 0.85), (0.55, 0.75, 0.55),
        (0.85, 0.80, 0.55), (0.75, 0.60, 0.80), (0.80, 0.80, 0.80),
        (0.95, 0.65, 0.45), (0.65, 0.85, 0.78)
    ]

    for hz in range(-2, 69, 7):
        for hx in range(-2, 69, 7):
            # deja libres las calles
            near_vertical = any(abs(hx - rx) < 4 for rx in roads_x)
            near_horizontal = any(abs(hz - rz) < 4 for rz in roads_z)
            if near_vertical or near_horizontal:
                continue

            index_seed = abs(int(hx * 17 + hz * 29))
            w_var = 2.3 + (index_seed % 3) * 0.45
            h_var = 2.0 + ((index_seed + 1) % 3) * 0.45
            d_var = 2.3 + ((index_seed + 2) % 3) * 0.45

            offset_x = ((index_seed % 5) - 2) * 0.18
            offset_z = (((index_seed + 2) % 5) - 2) * 0.18
            r_c, g_c, b_c = house_colors[index_seed % len(house_colors)]

            glPushMatrix()
            glTranslatef(hx + offset_x, 0, hz + offset_z)
            draw_detailed_house(w_var, h_var, d_var, r_c, g_c, b_c)
            glPopMatrix()

            # árbol o persona afuera de algunas casas
            if index_seed % 4 == 0:
                glPushMatrix()
                glTranslatef(hx + 1.7, 0, hz - 1.2)
                draw_tree_round()
                glPopMatrix()
            elif index_seed % 6 == 0:
                glPushMatrix()
                glTranslatef(hx - 1.4, 0, hz + 1.0)
                color = colors_people[index_seed % len(colors_people)]
                draw_person(*color)
                glPopMatrix()

    # Gente caminando en banquetas / ciudad
    city_people = [
        (8, -8), (14, -12), (24, 21), (31, 19), (40, -28), (48, -24),
        (57, 36), (63, 34), (73, 5), (69, 10), (52, 54), (36, 56),
        (19, 40), (5, 27)
    ]
    for i, (px, pz) in enumerate(city_people):
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(*colors_people[(i + 2) % len(colors_people)])
        glPopMatrix()

    # Más personas caminando con movimiento
    draw_moving_city_people(t, colors_people)

    # Perros extra
    for px, pz in [(18, 18), (62, 50)]:
        glPushMatrix()
        glTranslatef(px, 0.3, pz)
        draw_dog()
        glPopMatrix()


# ════════════════════════════════════════════════════════════════════════════
#  ILUMINACIÓN
# ════════════════════════════════════════════════════════════════════════════
def setup_lighting():
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, (.5,1.,.5,0.))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.,1.,.95,1.))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (.3,.3,.3,1.))
    glEnable(GL_NORMALIZE)

# ════════════════════════════════════════════════════════════════════════════
#  ESCENA 3D ANCLADA AL MARCADOR
# ════════════════════════════════════════════════════════════════════════════
def draw_scene_3d(rvec, tvec, K, w, h, t):
    P  = projection_from_k(K, w, h, ZNear, ZFar)
    MV = modelview_from_pose(rvec, tvec)

    glMatrixMode(GL_PROJECTION); glLoadMatrixf(P)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
    glMultMatrixf(MV)

    # Rotación + zoom gestual
    glRotatef(city_pitch, 1, 0, 0)
    glRotatef(city_yaw,   0, 1, 0)
    glScalef(city_zoom, city_zoom, city_zoom)

    # Escalar ciudad (~110 unidades) para que quepa sobre el marcador (~0.1 m)
    s = 0.00055
    glScalef(s, s, s)

    setup_lighting()
    draw_scenery(t)

# ════════════════════════════════════════════════════════════════════════════
#  OVERLAY MANO
# ════════════════════════════════════════════════════════════════════════════
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand_overlay(frame, hand_landmarks, fw, fh):
    kps = [(int(lm.x*fw), int(lm.y*fh)) for lm in hand_landmarks]
    for a,b in HAND_CONN:
        cv2.line(frame, kps[a], kps[b], (0,255,0), 2)
    for pt in kps:
        cv2.circle(frame, pt, 4, (0,120,255), cv2.FILLED)
    # Pinch línea
    cv2.line(frame, kps[4], kps[8], (0,255,255), 2)
    mid = ((kps[4][0]+kps[8][0])//2, (kps[4][1]+kps[8][1])//2)
    px  = int(math.hypot(kps[8][0]-kps[4][0], kps[8][1]-kps[4][1]))
    cv2.putText(frame, f"pinch:{px}px", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

# ════════════════════════════════════════════════════════════════════════════
#  GESTOS
# ════════════════════════════════════════════════════════════════════════════
def process_gestures(results, fw, fh):
    global city_yaw, city_pitch, city_zoom, _prev_index, _prev_pinch
    if not results or not results.hand_landmarks:
        _prev_index = None; _prev_pinch = None; return
    hand_lm = results.hand_landmarks[0]
    kps     = [(lm.x, lm.y) for lm in hand_lm]
    if len(kps) < 21: return
    index_tip = kps[8]; thumb_tip = kps[4]
    pinch = math.hypot(index_tip[0]-thumb_tip[0], index_tip[1]-thumb_tip[1])
    if _prev_index is not None:
        dx = (index_tip[0]-_prev_index[0])*250.
        dy = (index_tip[1]-_prev_index[1])*250.
        city_yaw = (city_yaw + dx * 1.8) % 360.0
        city_pitch = max(-85.0, min(85.0, city_pitch + dy * 1.2))
    if _prev_pinch is not None:
        # Pinch: separa índice y pulgar para acercar/agrandar.
        # Rango aumentado para que la ciudad pueda verse grande en pantalla.
        city_zoom = max(MIN_CITY_ZOOM, min(MAX_CITY_ZOOM, city_zoom-(pinch-_prev_pinch)*12.))
    _prev_index = index_tip; _prev_pinch = pinch

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: falta '{MODEL_PATH}'")
        print("Descárgalo de: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.", file=sys.stderr); sys.exit(1)

    ret, probe = cap.read()
    if not ret: sys.exit(1)
    cam_h, cam_w = probe.shape[:2]

    K, dist         = load_calibration(cam_w, cam_h)
    detector, dictn = make_aruco_detector()

    if not glfw.init(): sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(
        cam_w, cam_h,
        "RA Ciudad | mano=girar | pinch=zoom | ESC=salir",
        None, None)
    if not window:
        glfw.terminate(); sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    def on_key(win, key, _sc, action, _mods):
        if action==glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)

    glfw.set_key_callback(window, on_key)
    glEnable(GL_DEPTH_TEST)

    opts = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    with HandLandmarker.create_from_options(opts) as landmarker:
        while not glfw.window_should_close(window):
            ret, frame = cap.read()
            if not ret: continue

            h, w = frame.shape[:2]
            t    = glfw.get_time()

            # 1) Detectar ArUco en frame ORIGINAL para que el patron sea reconocible
            gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_orig = detect_marker(gray_orig, detector, dictn)

            # 2) Flip para display y mano
            frame = cv2.flip(frame, 1)

            # 3) Si hay marcador, espejear sus esquinas al frame flipado y calcular pose
            rvec, tvec = None, None
            if corners_orig is not None:
                # Espejear coordenada X de las esquinas: x_flip = w - x_orig
                corners_flip = corners_orig.copy()
                corners_flip[..., 0] = w - corners_orig[..., 0]
                rvec, tvec = estimate_pose(corners_flip, K, dist)

            # 4) Deteccion mano sobre frame flipado
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = landmarker.detect(mp_img)
            process_gestures(results, w, h)

            # 5) Overlay mano
            if results and results.hand_landmarks:
                draw_hand_overlay(frame, results.hand_landmarks[0], w, h)

            # 6) Render
            glViewport(0, 0, w, h)
            upload_frame_texture(frame, w, h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_background_quad(w, h)

            if rvec is not None:
                draw_scene_3d(rvec, tvec, K, w, h, t)

            glfw.swap_buffers(window)
            glfw.poll_events()

    cap.release()
    glfw.terminate()


if __name__ == "__main__":
    main()