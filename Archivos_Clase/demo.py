"""
Proyecto Final: Demo Procedural con OpenCV
Materia: Graficacion
Descripcion: Demo procedural de 60 segundos con 6 escenas, curvas parametricas,
             transformaciones afines y efectos de post-procesamiento.
"""

import time
import math
import numpy as np
import cv2

# ─────────────────────────────────────────
#  Configuracion global
# ─────────────────────────────────────────
W, H = 800, 600
FPS = 30
DURATION = 60.0          # 6 bloques x 10 s


# ─────────────────────────────────────────
#  Utilidades matematicas
# ─────────────────────────────────────────
def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(a, b, x):
    x = clamp01((x - a) / (b - a))
    return x * x * (3 - 2 * x)


def hsv_to_bgr(h, s, v):
    """Convierte HSV (h en [0,179], s,v en [0,255]) a BGR."""
    hsv = np.uint8([[[h % 180, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


# ─────────────────────────────────────────
#  Curvas parametricas → puntos para cv2.polylines
# ─────────────────────────────────────────
def make_curve(fx, fy, t0, t1, n, cx, cy, sx, sy):
    """
    Genera un arreglo de puntos para una curva parametrica.
    fx, fy: funciones en t
    (cx, cy): centro en pantalla
    (sx, sy): escala
    """
    ts = np.linspace(t0, t1, n, dtype=np.float32)
    xs = fx(ts) * sx + cx
    ys = fy(ts) * sy + cy
    return np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape((-1, 1, 2))


def curve_lissajous(a, b, delta, cx, cy, sx, sy):
    """Figura de Lissajous: x=sin(a*t+delta), y=sin(b*t)"""
    return make_curve(
        lambda t: np.sin(a * t + delta),
        lambda t: np.sin(b * t),
        0, 2 * math.pi, 900, cx, cy, sx, sy,
    )


def curve_rose(k, theta0, cx, cy, sx, sy):
    """Rosa polar: r=cos(k*theta), convertida a cartesiano."""
    return make_curve(
        lambda t: np.cos(k * t) * np.cos(t + theta0),
        lambda t: np.cos(k * t) * np.sin(t + theta0),
        0, 2 * math.pi, 1200, cx, cy, sx, sy,
    )


def curve_spirograph(R, r, d, phi, cx, cy, sx, sy):
    """
    Hipotrocoide (Spirograph):
    x = (R-r)cos(t) + d*cos((R-r)/r * t)
    y = (R-r)sin(t) - d*sin((R-r)/r * t)
    """
    w = (R - r) / r
    return make_curve(
        lambda t: (R - r) * np.cos(t) + d * np.cos(w * t + phi),
        lambda t: (R - r) * np.sin(t) - d * np.sin(w * t + phi),
        0, 14 * math.pi, 1600, cx, cy, sx, sy,
    )


def curve_lemniscata(a, theta0, cx, cy, sx, sy):
    """
    Lemniscata de Bernoulli: r^2 = a^2 * cos(2*theta)
    Solo existe para |cos(2t)| >= 0.
    """
    ts = np.linspace(0, 2 * math.pi, 1800, dtype=np.float32)
    r2 = np.cos(2 * ts)
    mask = r2 >= 0
    r = np.sqrt(np.where(mask, r2, 0)) * a
    xs = r * np.cos(ts + theta0) * sx + cx
    ys = r * np.sin(ts + theta0) * sy + cy
    pts = np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape((-1, 1, 2))
    return pts, mask


def curve_epicicloide(R, r, cx, cy, sx, sy):
    """
    Epicicloide: circulo de radio r rodando fuera de circulo de radio R.
    x = (R+r)cos(t) - r*cos((R+r)/r * t)
    y = (R+r)sin(t) - r*sin((R+r)/r * t)
    """
    return make_curve(
        lambda t: (R + r) * np.cos(t) - r * np.cos((R + r) / r * t),
        lambda t: (R + r) * np.sin(t) - r * np.sin((R + r) / r * t),
        0, 2 * math.pi * r, 2000, cx, cy, sx, sy,
    )


def curve_talini(a, cx, cy, sx, sy):
    """
    Curva de Talini (variante de astroide suavizada):
    x = cos^3(t), y = sin^3(t)  → astroide
    """
    return make_curve(
        lambda t: np.cos(t) ** 3,
        lambda t: np.sin(t) ** 3,
        0, 2 * math.pi, 800, cx, cy, sx * a, sy * a,
    )


# ─────────────────────────────────────────
#  Transformaciones afines explicitas
# ─────────────────────────────────────────
def affine_rotate(img, angle_deg, center=None):
    """Rotacion afin alrededor de center (default: centro de imagen)."""
    if center is None:
        center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def affine_shear(img, shx=0.0, shy=0.0):
    """
    Shear horizontal (shx) y vertical (shy).
    Matriz: [[1, shx, 0], [shy, 1, 0]]
    """
    M = np.float32([[1, shx, -shx * H / 2],
                    [shy, 1, -shy * W / 2]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def affine_mirror_x(img):
    """Espejo horizontal (flip en eje Y)."""
    return cv2.flip(img, 1)


def affine_scale_translate(img, sx, sy, tx, ty):
    """Escala + traslacion: M = [[sx,0,tx],[0,sy,ty]]"""
    M = np.float32([[sx, 0, tx], [0, sy, ty]])
    return cv2.warpAffine(img, M, (W, H))


# ─────────────────────────────────────────
#  Post-procesamiento (filtros)
# ─────────────────────────────────────────
def post_vignette(img, strength=0.7):
    """Viñeta oscura en los bordes."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    mask = np.clip(1.0 - strength * r2, 0.0, 1.0)
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)


def post_scanlines(img, strength=0.22):
    """Efecto de scanlines (lineas horizontales semitransparentes)."""
    out = img.astype(np.float32)
    y = np.arange(H, dtype=np.float32)
    m = 1.0 - strength * (0.5 + 0.5 * np.sin(2 * np.pi * y / 3.0))
    out *= m[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def post_posterize(img, q=32):
    """Posterizacion: reduce la cantidad de colores."""
    q = max(1, int(q))
    return ((img // q) * q).astype(np.uint8)


def post_chromatic_aberration(img, shift=3):
    """Aberracion cromatica: desplaza canal rojo y azul."""
    out = img.copy()
    out[:, shift:, 2] = img[:, :-shift, 2]   # rojo →
    out[:, :-shift, 0] = img[:, shift:, 0]   # azul ←
    return out


# ─────────────────────────────────────────
#  Fondos procedurales
# ─────────────────────────────────────────
def bg_gradient(img, t, hue0=10, hue1=140):
    """Degradado vertical animado en HSV."""
    hsv = np.zeros((H, W, 3), np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    hue = (hue0 + (hue1 - hue0) * ys + 10 * np.sin(t * 0.4 + ys * 2.0)).astype(np.float32)
    hsv[:, :, 0] = np.clip(hue, 0, 179).astype(np.uint8)[:, None]
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = (40 + 120 * (1 - ys)).astype(np.uint8)[:, None]
    img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def bg_stars(img, rng, n=400):
    """Estrellas deterministas."""
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    img[ys, xs] = (255, 255, 255)


# ─────────────────────────────────────────
#  ESCENAS
# ─────────────────────────────────────────

# ── Escena 0: Credits / Intro ────────────────────────────────────────────────
def scene_credits(img, t):
    """
    Escena de intro: fondo de estrellas + texto con fade-in.
    Curva: Lissajous decorativa en esquina.
    """
    bg_gradient(img, t, hue0=165, hue1=105)

    # Estrellas deterministas
    rng = np.random.default_rng(42)
    bg_stars(img, rng, 350)
    img[:] = cv2.GaussianBlur(img, (0, 0), 0.6)

    # Lissajous pequeño decorativo (esquina sup-der)
    pts = curve_lissajous(3, 2, math.pi / 2 + t * 0.3, 680, 100, 80, 60)
    col = hsv_to_bgr(int(20 + 20 * math.sin(t)), 200, 240)
    cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)

    # Texto principal con brillo pulsante
    alpha = 0.6 + 0.4 * math.sin(t * 1.5)
    overlay = img.copy()
    cv2.putText(overlay, "DEMO PROCEDURAL", (55, 220),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (240, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "Graficacion  |  OpenCV + Numpy", (55, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190, 210, 240), 2, cv2.LINE_AA)
    cv2.putText(overlay, "Curvas | Transformaciones | PostFX", (55, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 180, 210), 1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(img, 1 - alpha * 0.3, overlay, alpha * 0.3 + 0.7, 0)

    # Linea decorativa inferior
    cv2.line(img, (50, 360), (750, 360), (100, 120, 180), 1, cv2.LINE_AA)


# ── Escena 1: Lissajous animado ──────────────────────────────────────────────
def scene_lissajous(img, t):
    """
    Escena principal de curvas de Lissajous con parametros animados.
    Se dibujan varias figuras con diferentes (a,b).
    Transformacion: rotacion afin del frame completo.
    """
    bg_gradient(img, t, hue0=18, hue1=58)

    configs = [
        (3, 2, math.pi / 2),
        (5, 4, math.pi / 3 + t * 0.2),
        (3, 5, t * 0.15),
    ]
    hues = [20, 40, 60]
    scales = [(240, 180), (160, 120), (100, 80)]
    offsets = [(W // 2, H // 2 - 20), (180, 140), (620, 140)]

    for i, ((a, b, delta), hue, (sx, sy), (cx, cy)) in enumerate(
            zip(configs, hues, scales, offsets)):
        a_anim = a + 0.5 * math.sin(t * 0.5 + i)
        b_anim = b + 0.5 * math.cos(t * 0.7 + i)
        pts = curve_lissajous(a_anim, b_anim, delta, cx, cy, sx, sy)
        col = hsv_to_bgr(hue, 200, 235)
        thick = 2 if i == 0 else 1
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)

    # Titulo de escena
    cv2.putText(img, "Figuras de Lissajous", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "x=sin(at+d)  y=sin(bt)", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # TRANSFORMACION 1: Rotacion afin de un buffer auxiliar
    angle = 8 * math.sin(t * 0.4)
    img[:] = affine_rotate(img, angle)


# ── Escena 2: Rosa polar + Epicicloide ───────────────────────────────────────
def scene_rosa_epicicloide(img, t):
    """
    Combina rosa polar y epicicloide.
    Transformacion: shear horizontal animado.
    """
    bg_gradient(img, t, hue0=120, hue1=160)

    # Rosa polar k=5
    theta0 = t * 0.5
    pts_rosa = curve_rose(5, theta0, W // 2 - 130, H // 2, 200, 200)
    col_rosa = hsv_to_bgr(int(140 + 20 * math.sin(t * 0.5)), 210, 245)
    cv2.polylines(img, [pts_rosa], False, col_rosa, 2, cv2.LINE_AA)

    # Epicicloide (R=5, r=3) en la derecha
    pts_epi = curve_epicicloide(5, 3, W // 2 + 170, H // 2, 22, 22)
    col_epi = hsv_to_bgr(int(160 + 15 * math.cos(t * 0.7)), 190, 230)
    cv2.polylines(img, [pts_epi], False, col_epi, 2, cv2.LINE_AA)

    # Circulos pulsantes como "ritmo"
    for i in range(5):
        r = int(14 + 8 * math.sin(t * 2.2 + i * 1.2))
        xc = int(W * 0.12 + i * 130)
        cv2.circle(img, (xc, int(H * 0.88)), max(1, r), (200, 200, 200), 1, cv2.LINE_AA)

    # Elipse decorativa central
    axes = (int(60 + 20 * math.sin(t * 0.8)), int(35 + 10 * math.cos(t * 0.6)))
    cv2.ellipse(img, (W // 2, H // 2), axes, int(t * 30) % 360,
                0, 360, (180, 230, 180), 1, cv2.LINE_AA)

    cv2.putText(img, "Rosa polar  +  Epicicloide", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # TRANSFORMACION 2: Shear horizontal suave
    shx = 0.12 * math.sin(t * 0.5)
    img[:] = affine_shear(img, shx=shx, shy=0.0)


# ── Escena 3: Lemniscata + Astroide (transformaciones visibles) ───────────────
def scene_transformaciones(img, t):
    """
    Escena dedicada a mostrar transformaciones afines:
    - Espejo horizontal (flip)
    - Escala + traslacion animada
    - Shear vertical
    Se dibuja la lemniscata y la astroide pasando por cada transformacion.
    """
    bg_gradient(img, t, hue0=200, hue1=240)

    tmp = np.zeros_like(img)

    # Lemniscata
    a = 1.0
    pts_lem, mask = curve_lemniscata(a, t * 0.4, W // 2, H // 2, 230, 160)
    col_lem = hsv_to_bgr(int(200 + 20 * math.sin(t * 0.6)), 210, 245)
    cv2.polylines(tmp, [pts_lem], False, col_lem, 2, cv2.LINE_AA)

    # Astroide (curva de Talini)
    pts_ast = curve_talini(1.0, W // 2, H // 2, 150, 120)
    col_ast = hsv_to_bgr(int(220 + 20 * math.cos(t * 0.5)), 190, 230)
    cv2.polylines(tmp, [pts_ast], False, col_ast, 2, cv2.LINE_AA)

    # TRANSFORMACION 3: Espejo + combinar con original
    mirrored = affine_mirror_x(tmp)
    img[:] = cv2.addWeighted(tmp, 0.6, mirrored, 0.4, 0)

    # TRANSFORMACION 4: Escala animada + traslacion en un overlay
    scale = 0.55 + 0.2 * math.sin(t * 0.8)
    tx = 80 * math.cos(t * 0.5)
    ty = 40 * math.sin(t * 0.7)
    overlay = affine_scale_translate(tmp, scale, scale, tx, ty)
    col_overlay = hsv_to_bgr(int(30 + 20 * math.sin(t)), 180, 200)
    # Solo pinta donde haya píxeles (composicion por mascara)
    gray_ov = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    mask_ov = gray_ov > 10
    img[mask_ov] = cv2.addWeighted(img, 0.35, overlay, 0.65, 0)[mask_ov]

    # TRANSFORMACION 5: Shear vertical ligero
    shy = 0.08 * math.sin(t * 0.4 + 1.0)
    img[:] = affine_shear(img, shx=0.0, shy=shy)

    cv2.putText(img, "Lemniscata + Astroide", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "Espejo | Escala+Traslacion | Shear", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


# ── Escena 4: Spirograph + particulas ────────────────────────────────────────
def scene_spirograph(img, t, rng):
    """
    Hipotrocoide animada + campo de particulas.
    """
    bg_gradient(img, t, hue0=80, hue1=20)

    phi_anim = 0.4 * math.sin(t * 0.7)
    pts = curve_spirograph(8.0, 3.0, 5.0, phi_anim, W // 2, H // 2 - 20, 26, 26)
    col = hsv_to_bgr(int(10 + 140 * (0.5 + 0.5 * math.sin(t * 0.4))), 240, 240)
    cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)

    # Particulas flotando alrededor
    n = 600
    xs = rng.random(n) * W
    ys = rng.random(n) * H
    xs = (xs + 80 * np.sin(ys / 55.0 + t * 1.4)) % W
    ys = (ys + 60 * np.cos(xs / 70.0 + t * 1.1)) % H
    pcol = hsv_to_bgr(int(30 + 30 * math.sin(t * 0.9)), 200, 210)
    img[ys.astype(np.int32), xs.astype(np.int32)] = pcol

    img[:] = post_scanlines(img, 0.18)
    cv2.putText(img, "Hipotrocoide (Spirograph)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ── Escena 5: Fuego procedural (escena final) ─────────────────────────────────
def scene_fire(img, t, fire_state):
    """
    Simulacion de fuego con heatmap + paleta HSV.
    Aberracion cromatica como PostFX final.
    """
    heat = fire_state["heat"]
    rng = fire_state["rng"]

    heat[:] = (heat * 0.93).astype(np.float32)

    base_n = 1400
    xs = rng.integers(0, W, base_n)
    ys = rng.integers(int(H * 0.82), H, base_n)
    heat[ys, xs] += rng.random(base_n) * (0.8 + 0.6 * (0.5 + 0.5 * math.sin(t * 2.0)))

    heat[:] = cv2.GaussianBlur(heat, (0, 0), 2.2)
    heat[:-2, :] = heat[2:, :]
    heat[-2:, :] *= 0.0

    h = (20 - 20 * np.clip(heat, 0, 1)).astype(np.uint8)
    s = (220 - 80 * np.clip(heat, 0, 1)).astype(np.uint8)
    v = (60 + 195 * np.clip(heat, 0, 1)).astype(np.uint8)
    hsv = np.dstack([h, s, v]).astype(np.uint8)
    img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.rectangle(img, (0, int(H * 0.83)), (W, H), (10, 10, 10), -1)

    sparks = 160
    sx2 = rng.integers(0, W, sparks)
    sy2 = rng.integers(int(H * 0.55), int(H * 0.9), sparks)
    img[sy2, sx2] = (255, 255, 255)
    img[:] = cv2.GaussianBlur(img, (0, 0), 0.6)

    # Spirograph encima del fuego (composicion por capas)
    tmp = np.zeros_like(img)
    pts = curve_spirograph(6.0, 2.0, 4.0, t * 0.3, W // 2, H // 2 - 60, 20, 20)
    cv2.polylines(tmp, [pts], False, (255, 220, 100), 1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(img, 1.0, tmp, 0.5, 0)

    # PostFX: aberracion cromatica para dar efecto "calor"
    img[:] = post_chromatic_aberration(img, shift=int(3 + 2 * math.sin(t * 3)))

    cv2.putText(img, "Fuego Procedural + PostFX", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 220, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────
#  Dispatch de escenas
# ─────────────────────────────────────────
def render_scene(buf, scene_id, t, rng, fire_state):
    if scene_id == 0:
        scene_credits(buf, t)
    elif scene_id == 1:
        scene_lissajous(buf, t)
    elif scene_id == 2:
        scene_rosa_epicicloide(buf, t)
    elif scene_id == 3:
        scene_transformaciones(buf, t)
    elif scene_id == 4:
        scene_spirograph(buf, t, rng)
    else:
        scene_fire(buf, t, fire_state)


# ─────────────────────────────────────────
#  Timeline con transiciones (crossfade)
# ─────────────────────────────────────────
def timeline(t, rng, bufA, bufB, fire_state):
    """
    6 escenas (0..5), cada una de 10 segundos.
    Transicion crossfade de 1.2 s entre escenas consecutivas.
    Fade in/out global al inicio y final.
    """
    block = int(min(5, max(0, t // 10)))
    t_in = t - block * 10

    render_scene(bufA, block, t, rng, fire_state)
    frame = bufA.copy()

    # Crossfade en los últimos 1.2 s de cada bloque
    if block < 5 and t_in >= 8.8:
        render_scene(bufB, block + 1, t, rng, fire_state)
        a = smoothstep(8.8, 10.0, t_in)
        frame = cv2.addWeighted(bufA, 1 - a, bufB, a, 0)
        # Pequeno flash al corte
        flash = smoothstep(9.6, 10.0, t_in)
        if flash > 0:
            frame = cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.12 * flash, 0)

    # Fade in/out global
    fin = smoothstep(0.0, 1.5, t)
    fout = 1.0 - smoothstep(DURATION - 1.5, DURATION, t)
    f = fin * fout
    if f < 0.999:
        frame = (frame.astype(np.float32) * f).astype(np.uint8)

    return frame


# ─────────────────────────────────────────
#  Export a .mp4 con VideoWriter
# ─────────────────────────────────────────
def export_video(output_path="renders/demo_final.mp4"):
    import os
    os.makedirs("renders", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    rng = np.random.default_rng(123)
    bufA = np.zeros((H, W, 3), np.uint8)
    bufB = np.zeros((H, W, 3), np.uint8)
    fire_state = {
        "heat": np.zeros((H, W), np.float32),
        "rng": np.random.default_rng(999),
    }

    total_frames = int(DURATION * FPS)
    print(f"Exportando {total_frames} frames a {output_path} ...")
    t0 = time.perf_counter()

    for i in range(total_frames):
        t = i / FPS
        frame = timeline(t, rng, bufA, bufB, fire_state)
        frame = post_vignette(frame, 0.72)
        frame = post_scanlines(frame, 0.16)
        frame = post_posterize(frame, 24)
        writer.write(frame)
        if i % (FPS * 5) == 0:
            print(f"  {t:.0f}s / {DURATION:.0f}s")

    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"Listo. Tiempo de exportacion: {elapsed:.1f}s → {output_path}")


# ─────────────────────────────────────────
#  Main: preview en ventana
# ─────────────────────────────────────────
def main():
    rng = np.random.default_rng(123)
    bufA = np.zeros((H, W, 3), np.uint8)
    bufB = np.zeros((H, W, 3), np.uint8)
    fire_state = {
        "heat": np.zeros((H, W), np.float32),
        "rng": np.random.default_rng(999),
    }

    total_frames = int(DURATION * FPS)
    t0 = time.perf_counter()

    for i in range(total_frames):
        t = i / FPS
        frame = timeline(t, rng, bufA, bufB, fire_state)
        frame = post_vignette(frame, 0.72)
        frame = post_scanlines(frame, 0.16)
        frame = post_posterize(frame, 24)
        cv2.imshow("Proyecto Final — Demo Procedural", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC: salir
            break
        elif key == ord('e'):  # E: exportar video
            export_video()

    print(f"Tiempo total: {time.perf_counter() - t0:.1f}s")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_video()
    else:
        main()