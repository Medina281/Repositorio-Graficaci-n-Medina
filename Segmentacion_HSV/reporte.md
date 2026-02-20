# Segmentación de Frutas usando Máscara HSV

## Nombre:
José Luis Medina Ramírez

## Materia:
Graficación

## Objetivo
Aplicar el modelo HSV para segmentar frutas por color, limpiar ruido y contar regiones conectadas.

---

# Actividad 1: Exploración HSV (Rojo)

## Capturas

Imagen original:
![](capturas/original.png)

Imagen en HSV:
![](capturas/hsv.png)

Máscara rojo:
![](capturas/mask_rojo.png)

Máscara rojo limpia:
![](capturas/mask_rojo_limpia.png)

---

## Resultados

Regiones totales (incluyendo ruido): 10  
Frutas detectadas: 8  

Áreas aproximadas:
- 3385
- 6407
- 5205
- 3138
- 1051
- 1053
- 5433
- 6505

---

## Reflexión

- Si el rango HSV es muy estrecho, partes de la fruta no se detectan.
- Si el rango es muy amplio, aparece ruido y se detectan objetos que no son frutas.

---

# Actividad 2: Limpieza de Ruido

## ¿Qué tipo de ruido aparece?
En la máscara sin limpiar aparecen pequeños puntos blancos aislados y algunas regiones pequeñas que no corresponden a frutas.

## ¿Por qué es necesario eliminarlo antes del conteo?
Porque el método de regiones conectadas puede contar esos puntos como si fueran frutas, aumentando incorrectamente el número de objetos detectados.

---

# Actividad 3: Conteo de Regiones

El conteo se realizó utilizando componentes conectadas sobre la máscara limpia.

**Resultados:**

- Regiones totales (incluyendo ruido): 10
- Frutas detectadas (filtradas por área): 8

Las áreas aproximadas de cada región válida fueron:

- 3385
- 6407
- 5205
- 3138
- 1051
- 1053
- 5433
- 6505

---

# Actividad 4: Comparación entre Colores

| Color     | Número Detectado | Observaciones |
|-----------|------------------:|--------------|
| Rojo      | 8  | Segmentación más estable y con menor ruido después de la limpieza. |
| Verde     | 25 | Detectó muchas regiones pequeñas, lo que indica mayor ruido en la máscara. |
| Amarillo  | 10 | Buena segmentación, aunque afectada ligeramente por variaciones de brillo. |

---

# Actividad 5: Análisis Crítico

1. **¿Por qué HSV es más adecuado que RGB?**  
   Porque HSV separa el tono (H) de la intensidad de luz (V), lo que facilita segmentar por color sin que la iluminación afecte tanto como en RGB.

2. **¿Cómo afecta la iluminación al canal V?**  
   La iluminación modifica el valor (V). Si hay sombras o luz intensa, el valor cambia y puede afectar la segmentación si el rango no está bien ajustado.

3. **¿Qué sucede si dos frutas tienen tonos similares?**  
   Pueden ser detectadas como una sola región o confundirse si el rango de tono es demasiado amplio.

4. **¿Qué limitaciones tiene la segmentación por color?**  
   Depende de la iluminación, sombras, reflejos y del fondo. También puede fallar si los colores son muy parecidos.

---

# Conclusión

En esta práctica se aplicó el modelo de color HSV para segmentar frutas mediante una máscara binaria. Se comprobó que el ajuste correcto del rango HSV es fundamental para obtener una segmentación precisa. Si el rango es demasiado estrecho, partes del objeto no se detectan; si es demasiado amplio, se introduce ruido. Las operaciones morfológicas permitieron mejorar la calidad de la máscara antes del conteo de regiones conectadas. Finalmente, se logró detectar 8 frutas rojas válidas, demostrando que HSV es una herramienta adecuada para segmentación por color, aunque depende de condiciones como iluminación y similitud de tonos.
