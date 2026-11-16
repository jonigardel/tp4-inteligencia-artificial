"""Transformada de Hough para rectas y circunferencias, programada a mano.

Ambas siguen el mismo esquema: cada píxel de borde "vota" en un espacio de
parámetros (acumulador), y las acumulaciones de votos delatan las figuras.
"""
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Rectas: ρ = x·cos θ + y·sen θ
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recta:
    theta_grados: float
    rho: float
    votos: int


def acumulador_rectas(bordes: np.ndarray, paso_theta: float = 1.0):
    """Cada píxel de borde (x, y) traza en el plano (θ, ρ) la sinusoide
    ρ = x·cos θ + y·sen θ. Se usa la forma polar porque en (m, b) las rectas
    verticales tienen pendiente indeterminada."""
    alto, ancho = bordes.shape
    diagonal = int(np.ceil(np.hypot(alto, ancho)))
    thetas = np.deg2rad(np.arange(0.0, 180.0, paso_theta))
    acumulador = np.zeros((len(thetas), 2 * diagonal + 1), dtype=np.int64)
    ys, xs = np.nonzero(bordes)
    for indice, theta in enumerate(thetas):
        rhos = np.round(xs * np.cos(theta) + ys * np.sin(theta)).astype(int) + diagonal
        np.add.at(acumulador[indice], rhos, 1)
    return acumulador, np.rad2deg(thetas), diagonal


def buscar_rectas(acumulador, thetas_grados, diagonal, cantidad: int,
                  votos_minimos: int, vecindad_theta: int = 10, vecindad_rho: int = 15):
    """Toma los máximos del acumulador y anula su vecindad para no repetir
    la misma recta con parámetros casi iguales."""
    trabajo = acumulador.copy()
    rectas = []
    while len(rectas) < cantidad:
        i, j = np.unravel_index(np.argmax(trabajo), trabajo.shape)
        votos = int(trabajo[i, j])
        if votos < votos_minimos:
            break
        rectas.append(Recta(float(thetas_grados[i]), float(j - diagonal), votos))
        filas = trabajo.shape[0]
        for k in range(i - vecindad_theta, i + vecindad_theta + 1):
            # θ es periódico, pero al cruzar 0° ↔ 180° la misma recta cambia el signo de ρ
            cruza = k < 0 or k >= filas
            columna = (2 * diagonal - j) if cruza else j
            trabajo[k % filas, max(0, columna - vecindad_rho):columna + vecindad_rho + 1] = 0
    return rectas


# ---------------------------------------------------------------------------
# Circunferencias: (x − xo)² + (y − yo)² = R²
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Circunferencia:
    xo: int
    yo: int
    radio: int
    votos: int
    puntaje: float          # votos / perímetro: fracción de la circunferencia con borde


def acumulador_circunferencias(bordes: np.ndarray, radio: int, pasos_angulo: int = 180):
    """Con R conocido el espacio de parámetros es el plano (xo, yo): cada
    píxel de borde vota todos los centros posibles, que forman una
    circunferencia de radio R alrededor de él (figura 3 del enunciado)."""
    alto, ancho = bordes.shape
    acumulador = np.zeros((alto, ancho), dtype=np.int64)
    ys, xs = np.nonzero(bordes)
    angulos = np.linspace(0, 2 * np.pi, pasos_angulo, endpoint=False)
    desplazamientos = {(int(round(radio * np.cos(a))), int(round(radio * np.sin(a)))) for a in angulos}
    for dx, dy in desplazamientos:                 # sin repetir el mismo desplazamiento
        cx, cy = xs + dx, ys + dy
        dentro = (cx >= 0) & (cx < ancho) & (cy >= 0) & (cy < alto)
        np.add.at(acumulador, (cy[dentro], cx[dentro]), 1)
    return acumulador, len(desplazamientos)


def buscar_circunferencias(bordes: np.ndarray, radios, cantidad: int, puntaje_minimo: float,
                           separacion_minima: int = 10):
    """Repite la votación para cada radio candidato y normaliza los votos por
    la cantidad de puntos de cada circunferencia, para comparar radios distintos."""
    candidatos = []
    for radio in radios:
        acumulador, puntos = acumulador_circunferencias(bordes, radio)
        trabajo = acumulador.copy()
        for _ in range(cantidad):
            yo, xo = np.unravel_index(np.argmax(trabajo), trabajo.shape)
            votos = int(trabajo[yo, xo])
            if votos / puntos < puntaje_minimo:
                break
            candidatos.append(Circunferencia(int(xo), int(yo), int(radio), votos, votos / puntos))
            trabajo[max(0, yo - separacion_minima):yo + separacion_minima + 1,
                    max(0, xo - separacion_minima):xo + separacion_minima + 1] = 0
    candidatos.sort(key=lambda c: c.puntaje, reverse=True)
    elegidos = []
    for c in candidatos:                           # descarta duplicados de radios vecinos
        if all(np.hypot(c.xo - e.xo, c.yo - e.yo) > separacion_minima or abs(c.radio - e.radio) > 3
               for e in elegidos):
            elegidos.append(c)
        if len(elegidos) == cantidad:
            break
    return elegidos


# ---------------------------------------------------------------------------
# De la recta infinita al segmento visible
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segmento:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def largo(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))


def puntos_de_recta(recta: Recta, alto: int, ancho: int):
    """Puntos de la recta dentro de la imagen, avanzando de a un píxel
    sobre el eje en que la recta crece más rápido."""
    theta = np.deg2rad(recta.theta_grados)
    seno, coseno = np.sin(theta), np.cos(theta)
    if abs(seno) > abs(coseno):
        xs = np.arange(ancho, dtype=float)
        ys = (recta.rho - xs * coseno) / seno
    else:
        ys = np.arange(alto, dtype=float)
        xs = (recta.rho - ys * seno) / coseno
    dentro = (np.round(xs) >= 0) & (np.round(xs) < ancho) & (np.round(ys) >= 0) & (np.round(ys) < alto)
    return xs[dentro], ys[dentro]


def segmentos_de_recta(bordes: np.ndarray, recta: Recta, largo_minimo: int = 40,
                       hueco_maximo: int = 6, tolerancia: int = 2):
    """La transformada entrega rectas infinitas: se recorren y se conservan los
    tramos donde realmente hay borde, tolerando pequeños huecos."""
    alto, ancho = bordes.shape
    cercanos = _dilatar(bordes, tolerancia)
    xs, ys = puntos_de_recta(recta, alto, ancho)
    apoyo = cercanos[np.round(ys).astype(int), np.round(xs).astype(int)]
    segmentos, inicio, ultimo = [], None, None
    for k, tiene_borde in enumerate(apoyo):
        if tiene_borde:
            if inicio is None:
                inicio = k
            elif k - ultimo > hueco_maximo:
                segmentos.append((inicio, ultimo))
                inicio = k
            ultimo = k
    if inicio is not None:
        segmentos.append((inicio, ultimo))
    resultado = [Segmento(xs[a], ys[a], xs[b], ys[b]) for a, b in segmentos]
    return [s for s in resultado if s.largo >= largo_minimo]


def _dilatar(mascara: np.ndarray, radio: int) -> np.ndarray:
    ampliada = np.pad(mascara, radio)
    resultado = np.zeros_like(mascara)
    for dy in range(-radio, radio + 1):
        for dx in range(-radio, radio + 1):
            resultado |= ampliada[radio + dy:radio + dy + mascara.shape[0],
                                  radio + dx:radio + dx + mascara.shape[1]]
    return resultado
