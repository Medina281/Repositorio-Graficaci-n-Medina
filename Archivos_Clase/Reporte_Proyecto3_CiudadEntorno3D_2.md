# Proyecto 3: Ciudad Entorno 3D

**Materia:** Graficación  
**Grupo:** B  
**Integrantes:**
- David Emanuel Ordaz Amezcua
- Paola
- José Luis Medina Ramírez

---

## Objetivo de la Práctica

Desarrollar un entorno 3D interactivo que simule una metrópoli completa utilizando OpenGL, GLFW y MediaPipe. El proyecto tiene como finalidad implementar una ciudad con al menos 20 objetos móviles, controlar la cámara mediante el reconocimiento de landmarks de las manos con `gluLookAt`, y aplicar animaciones de transformación geométrica a los objetos del entorno.

---

## Capturas de Pantalla

> *A continuación se muestran capturas de la ciudad 3D en ejecución.*

![Vista general de la ciudad 3D](captura_ciudad.png)

*Vista aérea de la metrópoli completa con calles, edificios, parque y tráfico animado.*

---

## Descripción del Proyecto

El entorno representa una **metrópoli completa** compuesta por distintas zonas:

- **Zona residencial:** Casas con techo, puertas, ventanas y árboles variados.
- **Zona comercial / rascacielos:** Edificios altos con diferentes estilos arquitectónicos (vidrio, helipuerto, pisos escalonados, etc.).
- **Parque principal:** Área verde con canchas de voleibol y basquetbol, kiosko, fuente animada, estatua, lago, estadio de fútbol, mesas, bancas, flores y laberinto de setos.
- **Infraestructura vial:** Calles con rayas, cruces peatonales, semáforos y postes de luz.
- **Objetos especiales:** Escuela, iglesia y parque infantil.

---

## Objetos Móviles Implementados

El proyecto supera el requisito mínimo de 20 objetos móviles. A continuación se enlistan:

| # | Objeto | Tipo de movimiento |
|---|--------|--------------------|
| 1–6 | Autos (distintos colores) | Traslación sobre calles verticales |
| 7–14 | Autos en calles horizontales | Traslación continua en eje X |
| 15–19 | Motocicletas | Traslación rápida en calles |
| 20–22 | Camiones | Traslación lenta en calles principales |
| 23 | Avión | Trayectoria curva en el cielo (X y Z) |
| 24–33 | Personas en el parque | Movimiento circular/elíptico alrededor de puntos |
| 34–43 | Personas en la ciudad | Movimiento oscilatorio sobre banquetas |
| 44–47 | Chorros de la fuente | Animación vertical senoidal (escala Y) |

**Total: más de 44 objetos con movimiento activo.**

---

## Animaciones de Transformación Geométrica

Se implementaron las siguientes animaciones:

### 1. Traslación (vehículos y personas)
Los autos, motos, camiones y personas utilizan `glTranslatef` con posiciones calculadas en función del tiempo (`time.time()`), generando movimiento continuo sobre las calles y caminos del parque.

```python
z = ((t * 5.0 + i * 19) % 144) - 72
glTranslatef(x, 0.1, z)
```

### 2. Animación senoidal (fuente de agua)
Los chorros de la fuente varían su altura usando una función seno, logrando un efecto de agua en movimiento:

```python
height = 0.7 + 0.35 * math.sin(t * 3.0 + i)
draw_generic_cube(0.08, height, 0.08, 0.25, 0.65, 1.0)
```

### 3. Trayectoria compuesta del avión
El avión combina traslación en X, oscilación en Z y altura variable con seno:

```python
x = ((t * 5.0) % 170.0) - 85.0
z = -55 + 18.0 * math.sin(t * 0.22)
y = 28.0 + 3.0 * math.sin(t * 0.8)
```

### 4. Rotación de personas (orientación de caminata)
Cada persona se orienta en la dirección de su trayectoria usando `glRotatef` con un ángulo calculado con `atan2`:

```python
angle = math.degrees(math.atan2(math.cos(t * 0.65 + phase) * rx,
                                -math.sin(t * 0.65 + phase) * rz))
glRotatef(angle, 0, 1, 0)
```

---

## Control de Cámara con MediaPipe (gluLookAt)

Se utilizó la API de **MediaPipe HandLandmarker** para detectar los landmarks de ambas manos en tiempo real y controlar la cámara 3D.

### Mano Derecha — Rotación y Zoom

- El movimiento del dedo índice en X controla la rotación horizontal (`angle_y`), permitiendo giros de 360°.
- El movimiento en Y controla la inclinación vertical (`angle_x`), limitada entre -80° y 80°.
- La distancia de pinch (pulgar-índice) controla el zoom (`target_zoom`), con suavizado por interpolación lineal.

```python
angle_y += dx * 2.0
angle_y = angle_y % 360.0
angle_x = max(-80.0, min(80.0, angle_x + dy * 1.2))

calculated_zoom = -145.0 + (pinch / w) * 320.0
target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, calculated_zoom))
```

### Mano Izquierda — Paneo

- El desplazamiento del índice izquierdo genera paneo en X y Y de la cámara, limitado a ±40 y ±25 unidades respectivamente.

```python
pan_x += dx
pan_y -= dy
```

### Aplicación en gluLookAt

```python
gluLookAt(
    pan_x, cam_z * 0.62 + pan_y, cam_z,
    pan_x, 2.0 + pan_y, 0.0,
    0.0, 1.0, 0.0
)
glRotatef(angle_x, 1, 0, 0)
glRotatef(angle_y, 0, 1, 0)
```

---

## Tabla Comparativa de Resultados

| Criterio | Requerimiento | Implementado |
|----------|--------------|--------------|
| Objetos móviles | Mínimo 20 | ✅ Más de 44 |
| Control de cámara con manos | gluLookAt con landmarks | ✅ Rotación, zoom y paneo |
| Animación geométrica | Al menos una | ✅ Traslación, seno, rotación |
| Entorno 3D tipo ciudad/mundo | Sí | ✅ Metrópoli completa |
| Diversidad de objetos | Recomendado | ✅ Edificios, personas, vehículos, parque, estadio, iglesia, etc. |
| Detección de 2 manos | Opcional/deseable | ✅ Mano derecha y mano izquierda con funciones distintas |

---

## Preguntas de Análisis

**1. ¿Por qué se usa `gluLookAt` para controlar la cámara en lugar de mover directamente los objetos?**

`gluLookAt` define la posición y orientación del observador virtual de forma intuitiva (ojo, objetivo, vector arriba), lo que permite controlar la vista sin alterar las coordenadas de cada objeto en la escena. Mover todos los objetos individualmente sería computacionalmente costoso y conceptualmente incorrecto.

**2. ¿Qué ventaja tiene usar `time.time()` para las animaciones en lugar de un contador de frames?**

`time.time()` produce animaciones independientes de la velocidad de ejecución (FPS). Si el programa corre más lento o rápido en distintas máquinas, los objetos siempre se moverán a la misma velocidad real, garantizando consistencia visual.

**3. ¿Por qué se aplica suavizado al zoom con interpolación?**

```python
zoom = zoom * 0.84 + target_zoom * 0.16
```

Porque los gestos de la mano son inherentemente temblorosos. Sin suavizado, el zoom saltaría bruscamente entre valores. La interpolación actúa como un filtro paso-bajas que da sensación de inercia y suavidad al movimiento.

**4. ¿Qué limitaciones tiene el enfoque de primitivas `GL_QUADS` para construir la ciudad?**

No permite curvas reales (solo aproximaciones con muchos polígonos), no aplica iluminación física automática, y los modelos son de baja fidelidad visual. Sin embargo, es muy eficiente en rendimiento para escenas con cientos de objetos simultáneos como esta metrópoli.

---

## Conclusión Final

El proyecto logró construir una **metrópoli 3D interactiva y animada** que cumple y supera todos los requisitos establecidos. Se implementaron más de 44 objetos con movimiento continuo, incluyendo vehículos en calles, personas caminando, un avión cruzando el cielo y una fuente con animación senoidal.

El control mediante MediaPipe resultó funcional e intuitivo: la mano derecha permite rotar y hacer zoom sobre la ciudad, mientras que la mano izquierda panea la vista, todo en tiempo real sin teclado ni ratón.

El proyecto integra de forma efectiva conceptos de **transformaciones geométricas** (traslación, rotación, escala dinámica), **proyección en perspectiva** con `gluPerspective` y `gluLookAt`, y **visión por computadora** con detección de landmarks de manos, demostrando la convergencia entre graficación computacional y realidad aumentada.
