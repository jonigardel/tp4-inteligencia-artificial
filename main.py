"""TP4 - Inteligencia Artificial: transformada de Hough sobre la foto de un block de motor.

Uso:
    python main.py rectas      [imagen]   # punto 2: rectas (bordes del block)
    python main.py circulos    [imagen]   # punto 3: circunferencias (perforaciones)
    python main.py todo        [imagen]   # ambos

Los resultados (imágenes) se guardan en la carpeta resultados/.
"""
import argparse
import os
import time

from hough.dibujo import (VERDE, AMARILLO, acumulador_a_imagen, dibujar_circunferencia,
                          dibujar_cruz, dibujar_recta, dibujar_segmento)
from hough.imagen import a_escala_de_grises, detectar_bordes, guardar_imagen, leer_imagen_rgb
from hough.transformada import (acumulador_circunferencias, acumulador_rectas,
                                buscar_circunferencias, buscar_rectas, segmentos_de_recta)

IMAGEN_POR_DEFECTO = os.path.join("imagenes", "motor_block.png")
CARPETA_RESULTADOS = "resultados"

# Parámetros ajustados para la foto del block (ver README)
CANTIDAD_RECTAS = 15
VOTOS_MINIMOS_RECTA = 70
LARGO_MINIMO_SEGMENTO = 40
RADIOS_PERFORACIONES = range(5, 16)          # píxeles
CANTIDAD_CIRCUNFERENCIAS = 20
PUNTAJE_MINIMO_CIRCUNFERENCIA = 0.45          # fracción del perímetro con borde


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def preparar(ruta_imagen):
    """Etapas de bajo nivel: adquisición, preprocesamiento y extracción de bordes."""
    original = leer_imagen_rgb(ruta_imagen)
    inicio = time.perf_counter()
    bordes = detectar_bordes(a_escala_de_grises(original))
    alto, ancho = bordes.shape
    print(f"Imagen: {ruta_imagen}  ({ancho} x {alto} píxeles)")
    print(f"Píxeles de borde detectados: {int(bordes.sum())}  "
          f"({time.perf_counter() - inicio:.2f} s)")
    guardar_imagen(bordes * 255.0, os.path.join(CARPETA_RESULTADOS, "bordes.png"))
    return original, bordes


def ejecutar_rectas(original, bordes):
    titulo("Punto 2 - Transformada de Hough para rectas  (ρ = x·cos θ + y·sen θ)")
    inicio = time.perf_counter()
    acumulador, thetas, diagonal = acumulador_rectas(bordes)
    rectas = buscar_rectas(acumulador, thetas, diagonal, CANTIDAD_RECTAS, VOTOS_MINIMOS_RECTA,
                           vecindad_theta=6, vecindad_rho=12)
    print(f"Acumulador (θ, ρ): {acumulador.shape[0]} x {acumulador.shape[1]} celdas  "
          f"({time.perf_counter() - inicio:.2f} s)\n")

    rectas_completas = original.copy()
    segmentos_detectados = original.copy()
    print(f"{'#':>3} | {'θ (°)':>6} | {'ρ (px)':>7} | {'votos':>5} | segmentos visibles")
    print("-" * 60)
    for numero, recta in enumerate(rectas, start=1):
        dibujar_recta(rectas_completas, recta.theta_grados, recta.rho, grosor=1)
        segmentos = segmentos_de_recta(bordes, recta, largo_minimo=LARGO_MINIMO_SEGMENTO)
        for s in segmentos:
            dibujar_segmento(segmentos_detectados, s.x1, s.y1, s.x2, s.y2)
        largos = ", ".join(f"{s.largo:.0f}" for s in segmentos) or "—"
        print(f"{numero:>3} | {recta.theta_grados:>6.1f} | {recta.rho:>7.1f} | {recta.votos:>5} | {largos} px")

    guardar_imagen(acumulador_a_imagen(acumulador.T), os.path.join(CARPETA_RESULTADOS, "rectas_acumulador.png"))
    guardar_imagen(rectas_completas, os.path.join(CARPETA_RESULTADOS, "rectas_completas.png"))
    guardar_imagen(segmentos_detectados, os.path.join(CARPETA_RESULTADOS, "rectas_salida.png"))
    print(f"\nRectas detectadas: {len(rectas)}. Imágenes guardadas en {CARPETA_RESULTADOS}/")


def ejecutar_circulos(original, bordes):
    titulo("Punto 3 - Transformada de Hough para circunferencias  ((x−xo)² + (y−yo)² = R²)")
    inicio = time.perf_counter()
    circunferencias = buscar_circunferencias(bordes, RADIOS_PERFORACIONES, CANTIDAD_CIRCUNFERENCIAS,
                                             PUNTAJE_MINIMO_CIRCUNFERENCIA)
    print(f"Radios evaluados: {RADIOS_PERFORACIONES.start} a {RADIOS_PERFORACIONES.stop - 1} px  "
          f"({time.perf_counter() - inicio:.2f} s)\n")

    salida = original.copy()
    print(f"{'#':>3} | {'xo':>4} | {'yo':>4} | {'R':>3} | {'votos':>5} | perímetro con borde")
    print("-" * 60)
    for numero, c in enumerate(circunferencias, start=1):
        dibujar_circunferencia(salida, c.xo, c.yo, c.radio, color=VERDE)
        dibujar_cruz(salida, c.xo, c.yo, tamanio=3, color=AMARILLO)
        print(f"{numero:>3} | {c.xo:>4} | {c.yo:>4} | {c.radio:>3} | {c.votos:>5} | {c.puntaje:.0%}")

    if circunferencias:
        mejor = circunferencias[0]
        acumulador, _ = acumulador_circunferencias(bordes, mejor.radio)
        guardar_imagen(acumulador_a_imagen(acumulador),
                       os.path.join(CARPETA_RESULTADOS, f"circulos_acumulador_R{mejor.radio}.png"))
        print(f"\nCentro mejor puntuado: X = {mejor.xo}, Y = {mejor.yo} (R = {mejor.radio} px)")
    guardar_imagen(salida, os.path.join(CARPETA_RESULTADOS, "circulos_salida.png"))
    print(f"Circunferencias detectadas: {len(circunferencias)}. Imágenes guardadas en {CARPETA_RESULTADOS}/")


def main():
    parser = argparse.ArgumentParser(description="Transformada de Hough sobre la foto de un block de motor")
    parser.add_argument("modo", choices=["rectas", "circulos", "todo"])
    parser.add_argument("imagen", nargs="?", default=IMAGEN_POR_DEFECTO)
    args = parser.parse_args()

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    original, bordes = preparar(args.imagen)
    if args.modo in ("rectas", "todo"):
        ejecutar_rectas(original, bordes)
    if args.modo in ("circulos", "todo"):
        ejecutar_circulos(original, bordes)


if __name__ == "__main__":
    main()
