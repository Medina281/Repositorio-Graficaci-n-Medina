#!/usr/bin/env python3
"""
Realidad Aumentada — Ciudad 3D animada sobre marcador ArUco + control gestual.
VERSIÓN MEJORADA v2.0

Mejoras respecto a v1:
  - Zoom MUCHO más potente (rango 0.1–15x, sensibilidad x8)
  - Personajes peatones caminando animados (8 personas, 4 rutas)
  - Cancha de fútbol con portería y líneas
  - Cancha de tenis con red
  - Puesto de chimichancas con SHREK de merolico
  - Más carritos y rutas vehiculares
  - Perros paseando con dueño

Gestos (mano izquierda):
  Mover mano              -> girar la ciudad (yaw / pitch)
  Pinch (índice + pulgar) -> zoom in / out (¡mucho más rápido!)

Controles teclado:
  ESC / Q  -> salir

Requisitos: opencv-python, PyOpenGL, glfw, numpy, mediapipe
"""

from __future__ import annotations
import math, os, sys
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
city_zoom  =   4.0   # zoom inicial más cercano (antes era 1.0)
_prev_index = None
_prev_pinch = None

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

# ════════════════════════════════════════════════════════════════════════════
#  PRIMITIVAS BASE
# ════════════════════════════════════════════════════════════════════════════
def draw_generic_cube(w, h, d, r, g, b):
    w_h, d_h = w/2., d/2.
    glBegin(GL_QUADS)
    glColor3f(r,g,b)
    glVertex3f(-w_h,0,d_h);  glVertex3f(w_h,0,d_h);  glVertex3f(w_h,h,d_h);  glVertex3f(-w_h,h,d_h)
    glColor3f(r*.8,g*.8,b*.8)
    glVertex3f(-w_h,0,-d_h); glVertex3f(w_h,0,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(-w_h,h,-d_h)
    glColor3f(r*.7,g*.7,b*.7)
    glVertex3f(-w_h,0,-d_h); glVertex3f(-w_h,0,d_h); glVertex3f(-w_h,h,d_h); glVertex3f(-w_h,h,-d_h)
    glVertex3f( w_h,0,-d_h); glVertex3f( w_h,0,d_h); glVertex3f( w_h,h,d_h); glVertex3f( w_h,h,-d_h)
    glColor3f(r*.9,g*.9,b*.9)
    glVertex3f(-w_h,h,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(w_h,h,d_h); glVertex3f(-w_h,h,d_h)
    glEnd()

def draw_pyramid(w, h, d, r, g, b):
    w_h, d_h = w/2., d/2.
    glBegin(GL_TRIANGLES)
    glColor3f(r,g,b)
    glVertex3f(-w_h,0,d_h);  glVertex3f(w_h,0,d_h);  glVertex3f(0,h,0)
    glColor3f(r*.8,g*.8,b*.8)
    glVertex3f(-w_h,0,-d_h); glVertex3f(w_h,0,-d_h); glVertex3f(0,h,0)
    glColor3f(r*.7,g*.7,b*.7)
    glVertex3f(-w_h,0,-d_h); glVertex3f(-w_h,0,d_h); glVertex3f(0,h,0)
    glVertex3f( w_h,0,-d_h); glVertex3f( w_h,0,d_h); glVertex3f(0,h,0)
    glEnd()

def draw_windows(w, h, d, rows, cols):
    w_half, d_half = w/2., d/2.
    win_w = w/(cols*2); win_h = h/(rows*2)
    glColor3f(0.95,0.95,0.4)
    glBegin(GL_QUADS)
    z_f = d_half+0.01
    for r in range(rows):
        for c in range(cols):
            if (r+c)%3==0: continue
            x = -w_half+(c*(w/cols))+win_w/2
            y = (r*(h/rows))+win_h/2
            glVertex3f(x,y,z_f); glVertex3f(x+win_w,y,z_f)
            glVertex3f(x+win_w,y+win_h,z_f); glVertex3f(x,y+win_h,z_f)
    glEnd()

# ════════════════════════════════════════════════════════════════════════════
#  PERSONAJE PEATÓN ANIMADO — NUEVO
# ════════════════════════════════════════════════════════════════════════════
def draw_pedestrian(t, phase=0.0, shirt_r=0.2, shirt_g=0.4, shirt_b=0.9,
                    pants_r=0.2, pants_g=0.2, pants_b=0.35,
                    skin_r=0.9, skin_g=0.7, skin_b=0.5):
    """
    Personaje humano estilizado con animación de caminar.
    phase: desplazamiento de fase para que no todos caminen igual.
    """
    cycle = math.sin(t * 4.0 + phase)
    leg_swing = cycle * 0.35      # ángulo de piernas
    arm_swing = -cycle * 0.3      # brazos van al revés que piernas

    # Cabeza
    glPushMatrix()
    glTranslatef(0, 3.0, 0)
    draw_generic_cube(0.5, 0.55, 0.5, skin_r, skin_g, skin_b)
    # Pelo
    draw_generic_cube(0.52, 0.15, 0.52, 0.3, 0.2, 0.1)
    glPopMatrix()

    # Cuerpo (torso)
    glPushMatrix()
    glTranslatef(0, 1.5, 0)
    draw_generic_cube(0.65, 1.4, 0.4, shirt_r, shirt_g, shirt_b)
    glPopMatrix()

    # Brazo izquierdo
    glPushMatrix()
    glTranslatef(-0.45, 2.6, 0)
    glRotatef(arm_swing * 57.3, 1, 0, 0)  # rad a grados
    glTranslatef(0, -0.5, 0)
    draw_generic_cube(0.22, 1.0, 0.22, shirt_r*.85, shirt_g*.85, shirt_b*.85)
    glTranslatef(0, -1.0, 0)
    draw_generic_cube(0.18, 0.6, 0.18, skin_r, skin_g, skin_b)
    glPopMatrix()

    # Brazo derecho
    glPushMatrix()
    glTranslatef(0.45, 2.6, 0)
    glRotatef(-arm_swing * 57.3, 1, 0, 0)
    glTranslatef(0, -0.5, 0)
    draw_generic_cube(0.22, 1.0, 0.22, shirt_r*.85, shirt_g*.85, shirt_b*.85)
    glTranslatef(0, -1.0, 0)
    draw_generic_cube(0.18, 0.6, 0.18, skin_r, skin_g, skin_b)
    glPopMatrix()

    # Pierna izquierda
    glPushMatrix()
    glTranslatef(-0.2, 1.5, 0)
    glRotatef(leg_swing * 57.3, 1, 0, 0)
    glTranslatef(0, -0.75, 0)
    draw_generic_cube(0.28, 1.5, 0.28, pants_r, pants_g, pants_b)
    glTranslatef(0, -1.5, 0)
    # pie
    draw_generic_cube(0.25, 0.2, 0.5, 0.1, 0.1, 0.1)
    glPopMatrix()

    # Pierna derecha
    glPushMatrix()
    glTranslatef(0.2, 1.5, 0)
    glRotatef(-leg_swing * 57.3, 1, 0, 0)
    glTranslatef(0, -0.75, 0)
    draw_generic_cube(0.28, 1.5, 0.28, pants_r, pants_g, pants_b)
    glTranslatef(0, -1.5, 0)
    draw_generic_cube(0.25, 0.2, 0.5, 0.1, 0.1, 0.1)
    glPopMatrix()

# ════════════════════════════════════════════════════════════════════════════
#  SHREK DE MEROLICO — NUEVO
# ════════════════════════════════════════════════════════════════════════════
def draw_shrek_vendor(t):
    """
    Figura estilo Shrek verde vendiendo chimichancas.
    Cabeza grande redonda, orejas de ogro, mandil de cocinero.
    """
    bounce = math.sin(t * 2.0) * 0.1   # leve bamboleo

    glPushMatrix()
    glTranslatef(0, bounce, 0)

    # Piernas gordas verdes
    glPushMatrix()
    glTranslatef(-0.35, 0, 0)
    draw_generic_cube(0.55, 1.6, 0.55, 0.22, 0.5, 0.12)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.5, 0.3, 0.6, 0.15, 0.35, 0.08)  # bota
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.35, 0, 0)
    draw_generic_cube(0.55, 1.6, 0.55, 0.22, 0.5, 0.12)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.5, 0.3, 0.6, 0.15, 0.35, 0.08)
    glPopMatrix()

    # Cuerpo gordo verde
    glPushMatrix()
    glTranslatef(0, 1.6, 0)
    draw_generic_cube(1.3, 1.8, 0.85, 0.2, 0.48, 0.1)
    glPopMatrix()

    # Mandil blanco/crema de cocinero
    glPushMatrix()
    glTranslatef(0, 2.0, 0.43)
    draw_generic_cube(1.0, 1.3, 0.05, 0.92, 0.88, 0.78)
    # Letrero del mandil
    glPushMatrix()
    glTranslatef(0, 0.2, 0.03)
    draw_generic_cube(0.85, 0.4, 0.02, 0.85, 0.15, 0.1)  # parche rojo
    glPopMatrix()
    glPopMatrix()

    # Brazo izquierdo con cucharón
    arm_wave = math.sin(t * 1.8) * 15.0
    glPushMatrix()
    glTranslatef(-0.8, 2.8, 0)
    glRotatef(-40 + arm_wave, 1, 0, 0)
    glRotatef(20, 0, 0, 1)
    draw_generic_cube(0.35, 1.1, 0.35, 0.2, 0.48, 0.1)
    # Cucharón (palo + cuchara)
    glPushMatrix()
    glTranslatef(0, -1.1, 0)
    draw_generic_cube(0.08, 1.4, 0.08, 0.6, 0.5, 0.3)  # palo
    glTranslatef(0, -1.6, 0)
    draw_generic_cube(0.4, 0.15, 0.4, 0.5, 0.5, 0.5)   # cuchara
    glPopMatrix()
    glPopMatrix()

    # Brazo derecho
    glPushMatrix()
    glTranslatef(0.8, 2.8, 0)
    glRotatef(-30, 1, 0, 0)
    glRotatef(-15, 0, 0, 1)
    draw_generic_cube(0.35, 1.1, 0.35, 0.2, 0.48, 0.1)
    glPopMatrix()

    # Cabeza GRANDE y redonda (ogro style)
    glPushMatrix()
    glTranslatef(0, 4.5, 0)
    # Cabeza principal grande
    draw_generic_cube(1.1, 1.0, 1.0, 0.22, 0.55, 0.12)
    # Mejillas gordas
    glPushMatrix()
    glTranslatef(-0.5, 0.1, 0.4)
    draw_generic_cube(0.35, 0.45, 0.35, 0.25, 0.58, 0.14)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.5, 0.1, 0.4)
    draw_generic_cube(0.35, 0.45, 0.35, 0.25, 0.58, 0.14)
    glPopMatrix()

    # Ojos negros
    glPushMatrix()
    glTranslatef(-0.3, 0.5, 0.51)
    draw_generic_cube(0.2, 0.18, 0.05, 0.05, 0.05, 0.05)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.3, 0.5, 0.51)
    draw_generic_cube(0.2, 0.18, 0.05, 0.05, 0.05, 0.05)
    glPopMatrix()

    # Nariz bulbosa
    glPushMatrix()
    glTranslatef(0, 0.2, 0.52)
    draw_generic_cube(0.32, 0.2, 0.25, 0.25, 0.58, 0.14)
    glPopMatrix()

    # Cejas fruncidas
    glPushMatrix()
    glTranslatef(-0.3, 0.75, 0.51)
    glRotatef(-10, 0, 0, 1)
    draw_generic_cube(0.28, 0.1, 0.04, 0.1, 0.25, 0.05)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.3, 0.75, 0.51)
    glRotatef(10, 0, 0, 1)
    draw_generic_cube(0.28, 0.1, 0.04, 0.1, 0.25, 0.05)
    glPopMatrix()

    # OREJAS DE OGRO (las famosas)
    glPushMatrix()
    glTranslatef(-0.58, 0.6, 0)
    draw_generic_cube(0.18, 0.6, 0.18, 0.2, 0.5, 0.1)
    glTranslatef(0, 0.6, 0)
    draw_pyramid(0.22, 0.35, 0.22, 0.2, 0.5, 0.1)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.58, 0.6, 0)
    draw_generic_cube(0.18, 0.6, 0.18, 0.2, 0.5, 0.1)
    glTranslatef(0, 0.6, 0)
    draw_pyramid(0.22, 0.35, 0.22, 0.2, 0.5, 0.1)
    glPopMatrix()

    # Sombrero de chef
    glPushMatrix()
    glTranslatef(0, 1.05, 0)
    draw_generic_cube(0.85, 0.12, 0.85, 0.92, 0.92, 0.88)  # ala
    glTranslatef(0, 0.12, 0)
    draw_generic_cube(0.55, 0.9, 0.55, 0.95, 0.95, 0.92)   # cuerpo gorro
    glPopMatrix()

    glPopMatrix()  # fin bounce
    glPopMatrix()  # fin shrek

# ════════════════════════════════════════════════════════════════════════════
#  PUESTO DE CHIMICHANCAS — NUEVO
# ════════════════════════════════════════════════════════════════════════════
def draw_chimichanga_stand():
    """
    Carrito/puesto de comida con toldo, parrilla y letreros.
    """
    # Base del carrito
    draw_generic_cube(4.0, 0.6, 2.5, 0.7, 0.5, 0.3)

    # Ruedas
    for wx in [-1.5, 1.5]:
        for wz in [-1.0, 1.0]:
            glPushMatrix()
            glTranslatef(wx, -0.4, wz)
            draw_generic_cube(0.4, 0.35, 0.4, 0.15, 0.15, 0.15)
            glPopMatrix()

    # Mostrador / parrilla
    glPushMatrix()
    glTranslatef(0, 0.6, 0)
    draw_generic_cube(3.8, 0.35, 2.2, 0.5, 0.35, 0.2)
    # Parrilla metálica
    glPushMatrix()
    glTranslatef(0, 0.35, 0)
    draw_generic_cube(2.0, 0.1, 1.0, 0.3, 0.3, 0.3)
    glPopMatrix()
    # Chimichancas en la parrilla (cubitos dorados)
    for cx in [-0.6, 0, 0.6]:
        glPushMatrix()
        glTranslatef(cx, 0.45, 0)
        draw_generic_cube(0.45, 0.2, 0.3, 0.85, 0.65, 0.2)  # doradito
        glPopMatrix()
    glPopMatrix()

    # Postes del toldo
    for px in [-1.8, 1.8]:
        glPushMatrix()
        glTranslatef(px, 0.6, 0)
        draw_generic_cube(0.12, 3.5, 0.12, 0.6, 0.4, 0.2)
        glPopMatrix()

    # Toldo bicolor rojo y blanco
    glPushMatrix()
    glTranslatef(0, 4.1, 0)
    draw_generic_cube(4.4, 0.18, 2.8, 0.85, 0.1, 0.1)   # rojo
    glPushMatrix()
    glTranslatef(0, 0.18, 0)
    draw_generic_cube(4.4, 0.12, 2.8, 0.95, 0.92, 0.88)  # blanco
    glPopMatrix()
    # Flecos del toldo
    for fx in range(-10, 11, 2):
        glPushMatrix()
        glTranslatef(fx*0.2, -0.1, 1.4)
        draw_generic_cube(0.12, 0.35, 0.08, 0.85, 0.1, 0.1)
        glPopMatrix()
    glPopMatrix()

    # Letrero: "CHIMICHANCAS"
    glPushMatrix()
    glTranslatef(0, 3.5, 1.45)
    draw_generic_cube(3.2, 0.7, 0.1, 0.9, 0.1, 0.05)  # fondo rojo
    # Decoración del letrero (rectángulos blancos simulando texto)
    for lx in [-1.1, -0.7, -0.3, 0.1, 0.5, 0.9]:
        glPushMatrix()
        glTranslatef(lx, 0.15, 0.06)
        draw_generic_cube(0.22, 0.3, 0.02, 0.95, 0.92, 0.88)
        glPopMatrix()
    glPopMatrix()

    # Letrero lateral "HOY ESPECIAL"
    glPushMatrix()
    glTranslatef(2.1, 1.8, 0)
    glRotatef(90, 0, 1, 0)
    draw_generic_cube(2.0, 0.6, 0.08, 0.95, 0.75, 0.1)  # amarillo
    glPopMatrix()

# ════════════════════════════════════════════════════════════════════════════
#  CANCHA DE FÚTBOL — NUEVA
# ════════════════════════════════════════════════════════════════════════════
def draw_soccer_field():
    """Cancha de fútbol con portería, círculo central y líneas."""
    # Pasto
    glBegin(GL_QUADS)
    glColor3f(0.18, 0.55, 0.18)
    glVertex3f(-12, 0.02, -18); glVertex3f(12, 0.02, -18)
    glVertex3f(12, 0.02, 18);   glVertex3f(-12, 0.02, 18)
    glEnd()

    # Pasto más claro intercalado (efecto real)
    glBegin(GL_QUADS)
    glColor3f(0.22, 0.62, 0.22)
    for row in range(-3, 3):
        z1 = row * 6.0
        z2 = z1 + 3.0
        glVertex3f(-12, 0.025, z1); glVertex3f(12, 0.025, z1)
        glVertex3f(12, 0.025, z2);  glVertex3f(-12, 0.025, z2)
    glEnd()

    # Línea central horizontal
    glColor3f(0.9, 0.9, 0.9)
    glBegin(GL_QUADS)
    glVertex3f(-12, 0.03, -0.15); glVertex3f(12, 0.03, -0.15)
    glVertex3f(12, 0.03,  0.15);  glVertex3f(-12, 0.03,  0.15)
    glEnd()

    # Líneas de borde
    for x1, x2, z1, z2 in [
        (-12, -11.85, -18, 18), (11.85, 12, -18, 18),  # laterales
        (-12, 12, -18, -17.85), (-12, 12, 17.85, 18),  # fondos
    ]:
        glBegin(GL_QUADS)
        glColor3f(0.9, 0.9, 0.9)
        glVertex3f(x1, 0.03, z1); glVertex3f(x2, 0.03, z1)
        glVertex3f(x2, 0.03, z2); glVertex3f(x1, 0.03, z2)
        glEnd()

    # Área chica y grande
    for z_mult in [1, -1]:
        glBegin(GL_QUADS)
        glColor3f(0.9, 0.9, 0.9)
        # Área grande
        z0 = z_mult * 18.0
        z1 = z_mult * 12.0
        glVertex3f(-6, 0.03, z0); glVertex3f(6, 0.03, z0)
        glVertex3f(6, 0.03, z1);  glVertex3f(-6, 0.03, z1)
        # bordes área
        for xa, xb, za, zb in [
            (-6.15, -5.85, z1, z0), (5.85, 6.15, z1, z0)]:
            glVertex3f(xa, 0.03, za); glVertex3f(xb, 0.03, za)
            glVertex3f(xb, 0.03, zb); glVertex3f(xa, 0.03, zb)
        glEnd()

    # Porterías (dos)
    for z_side in [1, -1]:
        glPushMatrix()
        glTranslatef(0, 0, z_side * 18.0)
        # Postes
        for px in [-2.2, 2.2]:
            glPushMatrix(); glTranslatef(px, 0, 0)
            draw_generic_cube(0.15, 2.4, 0.15, 0.92, 0.92, 0.92)
            glPopMatrix()
        # Larguero
        glPushMatrix(); glTranslatef(0, 2.4, 0)
        draw_generic_cube(4.55, 0.15, 0.15, 0.92, 0.92, 0.92)
        glPopMatrix()
        # Red de portería
        glPushMatrix(); glTranslatef(0, 1.2, -z_side * 0.8)
        draw_generic_cube(4.4, 2.4, 0.1, 0.88, 0.88, 0.88)
        glPopMatrix()
        glPopMatrix()

# ════════════════════════════════════════════════════════════════════════════
#  CANCHA DE TENIS — NUEVA
# ════════════════════════════════════════════════════════════════════════════
def draw_tennis_court():
    """Cancha de tenis con red, líneas y postes."""
    # Superficie azul
    glBegin(GL_QUADS)
    glColor3f(0.1, 0.35, 0.75)
    glVertex3f(-5, 0.02, -8); glVertex3f(5, 0.02, -8)
    glVertex3f(5, 0.02, 8);   glVertex3f(-5, 0.02, 8)
    glEnd()

    # Líneas blancas
    glColor3f(0.92, 0.92, 0.92)
    lines = [
        (-5, -4.85, -8, 8), (4.85, 5, -8, 8),  # laterales ext
        (-3, -2.85, -8, 8), (2.85, 3, -8, 8),  # laterales int
        (-5, 5, -8, -7.85), (-5, 5, 7.85, 8),  # fondos
        (-5, 5, -0.1, 0.1),                      # centro
    ]
    for x1,x2,z1,z2 in lines:
        glBegin(GL_QUADS)
        glVertex3f(x1, 0.03, z1); glVertex3f(x2, 0.03, z1)
        glVertex3f(x2, 0.03, z2); glVertex3f(x1, 0.03, z2)
        glEnd()

    # Postes de red
    for px in [-5.3, 5.3]:
        glPushMatrix(); glTranslatef(px, 0, 0)
        draw_generic_cube(0.12, 1.2, 0.12, 0.4, 0.35, 0.3)
        glPopMatrix()

    # Red
    glPushMatrix(); glTranslatef(0, 0.6, 0)
    draw_generic_cube(10.7, 0.8, 0.05, 0.88, 0.88, 0.88)
    glPopMatrix()

    # Soportes del borde de pista
    glBegin(GL_QUADS)
    glColor3f(0.6, 0.25, 0.1)
    glVertex3f(-5.5, 0, -8.5); glVertex3f(5.5, 0, -8.5)
    glVertex3f(5.5, 0, 8.5);   glVertex3f(-5.5, 0, 8.5)
    glEnd()

# ════════════════════════════════════════════════════════════════════════════
#  RESTO DE PRIMITIVAS ORIGINALES (sin cambios)
# ════════════════════════════════════════════════════════════════════════════
def draw_detailed_house(w, h, d, r, g, b):
    draw_generic_cube(w,h,d,r,g,b)
    glPushMatrix(); glTranslatef(0,0.3,0); draw_windows(w*.8,h*.6,d,2,2); glPopMatrix()
    w_h, d_h = (w+.2)/2., (d+.2)/2.; roof_y = h+.8
    glBegin(GL_TRIANGLES)
    glColor3f(.85,.35,.1)
    glVertex3f(-w_h,h,d_h); glVertex3f(w_h,h,d_h); glVertex3f(0,roof_y,0)
    glVertex3f(-w_h,h,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(0,roof_y,0)
    glEnd()

def draw_cafe():
    draw_generic_cube(4.,2.5,3.5,.75,.6,.45)
    glBegin(GL_QUADS)
    glColor3f(.9,.9,.5)
    glVertex3f(-1.5,.5,1.76); glVertex3f(1.5,.5,1.76); glVertex3f(1.5,1.8,1.76); glVertex3f(-1.5,1.8,1.76)
    glEnd()
    glPushMatrix(); glTranslatef(0,2.3,.3)
    draw_generic_cube(4.4,.2,4.,.4,.25,.15)
    glTranslatef(0,.5,1.6); draw_generic_cube(2.,.5,.1,.9,.85,.7)
    glPopMatrix()

def draw_traffic_light(is_green):
    draw_generic_cube(.15,3.5,.15,.25,.25,.25)
    glPushMatrix(); glTranslatef(0,3.5,0)
    draw_generic_cube(.4,1.,.4,.1,.1,.1)
    r_l = 1. if not is_green else .15
    g_l = 1. if is_green     else .15
    glPushMatrix(); glTranslatef(0,.25,.21);  draw_generic_cube(.2,.2,.02,r_l,0.,0.);  glPopMatrix()
    glPushMatrix(); glTranslatef(0,-.25,.21); draw_generic_cube(.2,.2,.02,0.,g_l,0.);  glPopMatrix()
    glPopMatrix()

def draw_pink_motorcycle():
    glPushMatrix()
    draw_generic_cube(.25,.4,1.,1.,.4,.7)
    glPushMatrix(); glTranslatef(0,-.05,.4);  draw_generic_cube(.12,.25,.25,.12,.12,.12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,-.05,-.4); draw_generic_cube(.12,.25,.25,.12,.12,.12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,.4,.1);    draw_generic_cube(.2,.12,.3,.2,.2,.2);      glPopMatrix()
    glPopMatrix()

def draw_school():
    draw_generic_cube(9.,4.5,4.5,.7,.7,.7)
    glPushMatrix(); glTranslatef(2.5,0,3.); draw_generic_cube(3.,4.5,3.,.65,.65,.65); glPopMatrix()
    draw_windows(8.,3.5,4.5,2,4)

def draw_church():
    draw_generic_cube(5.,5.5,8.,.85,.82,.75)
    glPushMatrix(); glTranslatef(0,0,3.2)
    draw_generic_cube(2.5,10.,2.5,.75,.72,.65)
    glTranslatef(0,10.,0); draw_pyramid(2.8,2.5,2.8,.3,.3,.35)
    glTranslatef(0,2.5,0); draw_generic_cube(.15,1.,.15,.9,.8,.2)
    glTranslatef(0,.3,0);  draw_generic_cube(.6,.15,.15,.9,.8,.2)
    glPopMatrix()

def draw_small_playground():
    glBegin(GL_QUADS)
    glColor3f(.35,.65,.3)
    glVertex3f(-4.5,.02,-4.5); glVertex3f(4.5,.02,-4.5)
    glVertex3f(4.5,.02,4.5);   glVertex3f(-4.5,.02,4.5)
    glEnd()
    glPushMatrix(); glTranslatef(-2.,0,-1.)
    draw_generic_cube(.1,1.8,.1,.2,.2,.2)
    glTranslatef(2.5,0,0); draw_generic_cube(.1,1.8,.1,.2,.2,.2)
    glTranslatef(-1.25,1.8,0); draw_generic_cube(2.7,.1,.1,.2,.2,.2)
    glTranslatef(0,-1.,0); draw_generic_cube(.8,.08,.3,.8,.2,.2)
    glPopMatrix()
    glPushMatrix(); glTranslatef(1.8,0,-1.5)
    draw_generic_cube(.6,1.4,.6,.2,.4,.8)
    glPushMatrix(); glTranslatef(0,.5,.8); glRotatef(30,1,0,0)
    draw_generic_cube(.5,.1,1.6,.8,.8,.8); glPopMatrix()
    glPopMatrix()

def draw_kiosk():
    glPushMatrix()
    draw_generic_cube(2.6,.5,2.6,.5,.35,.25)
    for sx in [-1.1,1.1]:
        for sz in [-1.1,1.1]:
            glPushMatrix(); glTranslatef(sx,.5,sz)
            draw_generic_cube(.12,1.8,.12,.8,.7,.5); glPopMatrix()
    glTranslatef(0,2.3,0); draw_pyramid(3.,1.2,3.,.7,.2,.2)
    glPopMatrix()

def draw_dog():
    glPushMatrix()
    draw_generic_cube(.25,.25,.5,.55,.27,.07)
    glPushMatrix()
    for ox in [-.08,.08]:
        for oz in [-.18,.18]:
            glPushMatrix(); glTranslatef(ox,-.12,oz)
            draw_generic_cube(.06,.15,.06,.4,.2,.0); glPopMatrix()
    glPopMatrix()
    glTranslatef(0,.2,.2); draw_generic_cube(.2,.2,.2,.55,.27,.07)
    glPopMatrix()

def draw_car(r, g, b):
    glPushMatrix()
    draw_generic_cube(1.,.38,1.8,r,g,b)
    glPushMatrix(); glTranslatef(0,.38,-.1)
    draw_generic_cube(.8,.32,1.,r*.6,g*.6,b*.6); glPopMatrix()
    glPopMatrix()

def draw_truck():
    glPushMatrix()
    draw_generic_cube(1.4,1.6,3.6,.85,.85,.85)
    glTranslatef(0,0,1.3); draw_generic_cube(1.3,1.,1.,.8,.1,.1)
    glPopMatrix()

def draw_volleyball_court():
    glBegin(GL_QUADS)
    glColor3f(.9,.5,.2)
    glVertex3f(-3.5,.02,-5.5); glVertex3f(3.5,.02,-5.5)
    glVertex3f(3.5,.02,5.5);   glVertex3f(-3.5,.02,5.5)
    glColor3f(.1,.4,.7)
    glVertex3f(-2.8,.025,-4.8); glVertex3f(2.8,.025,-4.8)
    glVertex3f(2.8,.025,4.8);   glVertex3f(-2.8,.025,4.8)
    glEnd()
    draw_generic_cube(.08,1.8,.08,.6,.6,.6)
    glPushMatrix(); glTranslatef(0,1.1,0); draw_generic_cube(5.4,.5,.02,.9,.9,.9); glPopMatrix()

def draw_basketball_court():
    glBegin(GL_QUADS)
    glColor3f(.75,.52,.3)
    glVertex3f(-3.5,.02,-6.); glVertex3f(3.5,.02,-6.)
    glVertex3f(3.5,.02,6.);   glVertex3f(-3.5,.02,6.)
    glEnd()
    glPushMatrix(); glTranslatef(0,0,-5.6)
    draw_generic_cube(.12,2.5,.12,.2,.2,.2)
    glTranslatef(0,2.5,.15); draw_generic_cube(1.5,.9,.04,1.,1.,1.); glPopMatrix()
    glPushMatrix(); glTranslatef(0,0,5.6)
    draw_generic_cube(.12,2.5,.12,.2,.2,.2)
    glTranslatef(0,2.5,-.15); draw_generic_cube(1.5,.9,.04,1.,1.,1.); glPopMatrix()

def draw_tree_round():
    draw_generic_cube(.2,.8,.2,.4,.25,.15)
    glPushMatrix(); glTranslatef(0,.8,0); draw_generic_cube(1.,.9,1.,.2,.55,.2); glPopMatrix()

def draw_tree_pine():
    draw_generic_cube(.2,.6,.2,.4,.25,.15)
    glPushMatrix()
    glTranslatef(0,.5,0); draw_pyramid(1.2,.8,1.2,.1,.38,.15)
    glTranslatef(0,.5,0); draw_pyramid(.9,.7,.9,.12,.42,.18)
    glPopMatrix()

def draw_animated_helicopter(t):
    glPushMatrix()
    draw_generic_cube(1.4,1.,3.,.2,.2,.8)
    glPushMatrix(); glTranslatef(0,.2,-2.); draw_generic_cube(.3,.3,1.5,.2,.2,.8); glPopMatrix()
    glPushMatrix(); glTranslatef(0,1.1,0); glRotatef(t*800,0,1,0)
    draw_generic_cube(4.,.04,.25,.9,.9,.9); glPopMatrix()
    glPopMatrix()

# ════════════════════════════════════════════════════════════════════════════
#  PEATONES ANIMADOS EN RUTAS — NUEVO
# ════════════════════════════════════════════════════════════════════════════
def draw_all_pedestrians(t):
    """
    8 peatones caminando por banquetas de la ciudad.
    Cada uno tiene ruta, colores y fase distintos.
    """
    # Ruta 1: banqueta norte-sur (x fijo, z oscila)
    pedestrians = [
        # (ruta_x, ruta_z_base, vel, phase, shirt_rgb, pants_rgb, scale, facing)
        # Banqueta izquierda, norte-sur
        (-19., 0., 8., 0.0,   (0.9,0.2,0.2), (0.2,0.2,0.4), 1.0, 0),
        (-19., 25., 8., 1.5,  (0.2,0.6,0.9), (0.1,0.15,0.3), 1.0, 0),
        (-19., -25., 6., 3.0, (0.9,0.8,0.2), (0.3,0.2,0.1), 0.9, 0),

        # Banqueta derecha, norte-sur
        (-11., 0., 7., 2.0,   (0.7,0.3,0.8), (0.2,0.2,0.2), 1.0, 1),
        (-11., 30., 9., 0.8,  (0.2,0.8,0.4), (0.15,0.3,0.15), 1.0, 1),

        # Banqueta este-oeste (z fijo, x varía)
        (0., 20., 6., 1.0,    (0.95,0.6,0.2), (0.25,0.2,0.2), 0.95, 2),
        (0., -25., 8., 2.5,   (0.3,0.3,0.9), (0.2,0.2,0.35), 1.05, 3),

        # Cerca del parque
        (-32., 2., 5., 0.5,   (0.8,0.8,0.8), (0.3,0.3,0.3), 0.85, 2),
    ]

    for i, ped in enumerate(pedestrians):
        rx, rz, vel, phase, shirt, pants, scale, facing = ped

        glPushMatrix()

        if facing == 0:   # norte-sur
            z_pos = rz + math.sin(t * vel * 0.1 + phase) * 30.0
            glTranslatef(rx, 0, z_pos)
            # Orientar hacia dirección de movimiento
            dz = math.cos(t * vel * 0.1 + phase)
            angle = 0 if dz >= 0 else 180
            glRotatef(angle, 0, 1, 0)
        elif facing == 1:  # norte-sur inverso
            z_pos = rz - math.sin(t * vel * 0.1 + phase) * 28.0
            glTranslatef(rx, 0, z_pos)
            dz = -math.cos(t * vel * 0.1 + phase)
            angle = 0 if dz >= 0 else 180
            glRotatef(angle, 0, 1, 0)
        elif facing == 2:  # este-oeste
            x_pos = rx + math.sin(t * vel * 0.1 + phase) * 20.0
            glTranslatef(x_pos, 0, rz)
            glRotatef(90, 0, 1, 0)
        else:              # este-oeste inverso
            x_pos = rx - math.sin(t * vel * 0.1 + phase) * 18.0
            glTranslatef(x_pos, 0, rz)
            glRotatef(-90, 0, 1, 0)

        glScalef(scale, scale, scale)
        draw_pedestrian(t, phase,
                        shirt_r=shirt[0], shirt_g=shirt[1], shirt_b=shirt[2],
                        pants_r=pants[0], pants_g=pants[1], pants_b=pants[2])
        glPopMatrix()

# ════════════════════════════════════════════════════════════════════════════
#  ESCENARIO COMPLETO
# ════════════════════════════════════════════════════════════════════════════
def draw_scenery(t):
    # Suelo
    glBegin(GL_QUADS)
    glColor3f(.14,.14,.15)
    glVertex3f(-55,-.01,55); glVertex3f(55,-.01,55)
    glVertex3f(55,-.01,-55); glVertex3f(-55,-.01,-55)
    glEnd()

    # Manzanas de asfalto
    manzanas = [
        (-32.,-30.,24.,24.),(-32.,5.,24.,20.),(-32.,38.,24.,22.),
        (0.,-30.,30.,24.),  (0.,5.,30.,20.),  (0.,38.,30.,22.),
        (35.,-30.,30.,24.), (35.,5.,30.,20.),  (35.,38.,30.,22.)
    ]
    glBegin(GL_QUADS)
    for mx,mz,mw,md in manzanas:
        glColor3f(.08,.08,.09)
        wh,dh = mw/2.,md/2.
        glVertex3f(mx-wh,.005,mz+dh); glVertex3f(mx+wh,.005,mz+dh)
        glVertex3f(mx+wh,.005,mz-dh); glVertex3f(mx-wh,.005,mz-dh)
    glEnd()

    # Semáforos animados
    traffic_phase   = int(t/6.)%2
    vertical_green  = (traffic_phase==0)
    horizontal_green= not vertical_green
    glPushMatrix(); glTranslatef(-17.,0,14.);  draw_traffic_light(vertical_green);  glPopMatrix()
    glPushMatrix(); glTranslatef(-13.,0,17.5); glRotatef(90,0,1,0); draw_traffic_light(horizontal_green); glPopMatrix()

    # ── PARQUE ───────────────────────────────────────────────────────────────
    glPushMatrix(); glTranslatef(-32.,0,5.)
    glBegin(GL_QUADS); glColor3f(.2,.45,.22)
    glVertex3f(-11.5,.01,9.5); glVertex3f(11.5,.01,9.5)
    glVertex3f(11.5,.01,-9.5); glVertex3f(-11.5,.01,-9.5); glEnd()
    glPushMatrix(); glTranslatef(-5.5,0,-3.);  draw_volleyball_court(); glPopMatrix()
    glPushMatrix(); glTranslatef(-5.5,0,4.5);  draw_small_playground(); glPopMatrix()
    glPushMatrix(); glTranslatef(-1.,.2,2.5);  draw_dog();              glPopMatrix()
    glPushMatrix(); glTranslatef(4.5,0,0.);    draw_kiosk();            glPopMatrix()
    for az in range(-8,9,4):
        glPushMatrix(); glTranslatef(-10.,0,az); draw_tree_round(); glPopMatrix()
        glPushMatrix(); glTranslatef(9.5,0,az);  draw_tree_pine();  glPopMatrix()
    glPopMatrix()

    # ── ZONA COMUNITARIA (con canchas nuevas) ─────────────────────────────
    glPushMatrix(); glTranslatef(0.,0,-30.)
    glPushMatrix(); glTranslatef(-7.,0,-4.); draw_school();            glPopMatrix()
    glPushMatrix(); glTranslatef(1.,0,2.);   draw_church();            glPopMatrix()
    glPushMatrix(); glTranslatef(9.,0,-2.);  draw_basketball_court();  glPopMatrix()
    # Cancha de fútbol NUEVA
    glPushMatrix(); glTranslatef(-20.,0,5.); draw_soccer_field();      glPopMatrix()
    glPopMatrix()

    # ── CANCHA DE TENIS (zona norte) ─────────────────────────────────────
    glPushMatrix(); glTranslatef(35.,0,15.); draw_tennis_court();      glPopMatrix()

    # ── PUESTO DE CHIMICHANCAS CON SHREK ──────────────────────────────────
    glPushMatrix(); glTranslatef(-5., 0., 20.)
    draw_chimichanga_stand()
    # Shrek parado al lado del puesto
    glPushMatrix(); glTranslatef(3.5, 0., 0.)
    glScalef(0.7, 0.7, 0.7)   # Shrek mediano (la ciudad tiene escala pequeña)
    draw_shrek_vendor(t)
    glPopMatrix()
    glPopMatrix()

    # ── RASCACIELOS ───────────────────────────────────────────────────────
    for x,z,w,h,d,r,g,b in [
        (-32.,-30.,4.5,18.,4.5,.25,.35,.5),(-25.,-26.,4.,14.,4.,.3,.3,.35),
        (0.,38.,5.,26.,5.,.15,.4,.45),(8.,42.,4.5,22.,4.5,.2,.2,.3),
        (-8.,35.,4.,17.,4.,.35,.35,.4),(35.,38.,5.5,29.,5.5,.1,.25,.5),
        (42.,44.,4.,20.,4.,.22,.45,.4),(28.,34.,4.2,15.,4.2,.4,.4,.45),
    ]:
        glPushMatrix(); glTranslatef(x,0,z)
        draw_generic_cube(w,h,d,r,g,b); draw_windows(w,h,d,int(h//1.5),4)
        glPopMatrix()

    # ── CASAS ─────────────────────────────────────────────────────────────
    house_positions = [
        (-38.,32.),(-32.,32.),(-26.,32.),(-35.,42.),(-29.,42.),
        (-10.,2.),(-4.,2.),(-10.,9.),(-4.,9.),
        (22.,1.),(28.,1.),(34.,1.),(25.,9.),(31.,9.),
        (24.,-35.),(30.,-35.),(38.,-35.),(32.,-25.)
    ]
    house_colors = [(.85,.45,.45),(.45,.65,.85),(.55,.75,.55),(.85,.80,.55),(.75,.60,.80)]
    for idx,(hx,hz) in enumerate(house_positions):
        rc,gc,bc = house_colors[idx%len(house_colors)]
        glPushMatrix(); glTranslatef(hx,0,hz)
        draw_detailed_house(2.4,2.,2.4,rc,gc,bc); glPopMatrix()

    glPushMatrix(); glTranslatef(4.,0,5.); draw_cafe(); glPopMatrix()

    # ── TRÁFICO VERTICAL (ampliado) ───────────────────────────────────────
    for idx, x_lane in enumerate([-15.,15.,49.]):
        sp = 9.+(idx%2)*3.
        # 3 carros por carril (antes 3, ahora 4 offsets)
        for off in [0., 50., 25., 75.]:
            z1 = -50.+((t*sp+off)%100.)
            if x_lane==-15. and not vertical_green and 4.<z1<16.: z1=4.
            # Variedad de colores de carros
            car_colors = [(.85,.1,.1), (.1,.5,.85), (.15,.7,.3), (.85,.75,.1)]
            r,g,b = car_colors[int(off/25) % len(car_colors)]
            glPushMatrix(); glTranslatef(x_lane-.7,.15,z1); draw_car(r,g,b); glPopMatrix()
        z2 = -50.+((t*sp+50.)%100.)
        glPushMatrix(); glTranslatef(x_lane+.7,.15,z2); draw_truck(); glPopMatrix()
        z3 = -50.+((t*sp+25.)%100.)
        glPushMatrix(); glTranslatef(x_lane,.15,z3); draw_pink_motorcycle(); glPopMatrix()

    # ── TRÁFICO HORIZONTAL (ampliado) ────────────────────────────────────
    for idx, z_lane in enumerate([-18.,16.,49.]):
        sp = 10.+(idx%2)*3.
        for off in [0., 50., 75., 33.]:
            x1 = -50.+((t*sp+off)%100.)
            if z_lane==16. and not horizontal_green and -27.<x1<-15.: x1=-27.
            car_colors2 = [(.9,.8,.1),(.5,.2,.8),(.9,.5,.2),(.2,.8,.6)]
            r,g,b = car_colors2[int(off/33) % len(car_colors2)]
            glPushMatrix(); glTranslatef(x1,.15,z_lane-.7); glRotatef(90,0,1,0); draw_car(r,g,b); glPopMatrix()

    # ── PEATONES ANIMADOS ─────────────────────────────────────────────────
    draw_all_pedestrians(t)

    # ── HELICÓPTERO ───────────────────────────────────────────────────────
    glPushMatrix()
    glRotatef(t*22,0,1,0); glTranslatef(15.,24.,10.)
    draw_animated_helicopter(t)
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

    glRotatef(city_pitch, 1, 0, 0)
    glRotatef(city_yaw,   0, 1, 0)
    glScalef(city_zoom, city_zoom, city_zoom)

    # Factor de escala: 0.003 hace la ciudad grande y visible sobre el marcador
    s = 0.003
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
    cv2.line(frame, kps[4], kps[8], (0,255,255), 2)
    mid = ((kps[4][0]+kps[8][0])//2, (kps[4][1]+kps[8][1])//2)
    px  = int(math.hypot(kps[8][0]-kps[4][0], kps[8][1]-kps[4][1]))
    cv2.putText(frame, f"pinch:{px}px | zoom:{city_zoom:.2f}x", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

# ════════════════════════════════════════════════════════════════════════════
#  GESTOS — ZOOM MEJORADO
# ════════════════════════════════════════════════════════════════════════════
def process_gestures(results, fw, fh):
    """
    MEJORA DE ZOOM:
    - Antes: sensibilidad x3, rango 0.2–5.0
    - Ahora: sensibilidad x8, rango 0.05–15.0
    - El delta se escala también por el zoom actual para que sea
      proporcional (más rápido al acercarse mucho).
    """
    global city_yaw, city_pitch, city_zoom, _prev_index, _prev_pinch
    if not results or not results.hand_landmarks:
        _prev_index = None; _prev_pinch = None; return
    hand_lm = results.hand_landmarks[0]
    kps     = [(lm.x, lm.y) for lm in hand_lm]
    if len(kps) < 21: return

    index_tip = kps[8]; thumb_tip = kps[4]
    pinch = math.hypot(index_tip[0]-thumb_tip[0], index_tip[1]-thumb_tip[1])

    # Rotación igual que antes
    if _prev_index is not None:
        dx = (index_tip[0]-_prev_index[0])*250.
        dy = (index_tip[1]-_prev_index[1])*250.
        city_yaw  += dx
        city_pitch = max(-85., min(85., city_pitch+dy))

    # ZOOM MEJORADO: sensibilidad x8, proporcional al zoom actual
    if _prev_pinch is not None:
        delta_pinch = pinch - _prev_pinch
        # Factor proporcional: más rápido cuando el zoom ya es grande
        zoom_factor = max(0.3, city_zoom * 0.5)
        city_zoom = max(0.05, min(15.0, city_zoom - delta_pinch * 8.0 * zoom_factor))

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
        "RA Ciudad v2 | mano=girar | pinch=zoom(0.05-15x) | ESC=salir",
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

            gray_orig    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_orig = detect_marker(gray_orig, detector, dictn)

            frame = cv2.flip(frame, 1)

            rvec, tvec = None, None
            if corners_orig is not None:
                corners_flip = corners_orig.copy()
                corners_flip[..., 0] = w - corners_orig[..., 0]
                rvec, tvec = estimate_pose(corners_flip, K, dist)

            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = landmarker.detect(mp_img)
            process_gestures(results, w, h)

            if results and results.hand_landmarks:
                draw_hand_overlay(frame, results.hand_landmarks[0], w, h)

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