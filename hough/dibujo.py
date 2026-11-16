"""Transformada inversa: las figuras detectadas se dibujan sobre la imagen original."""
import numpy as np

ROJO = (230, 30, 30)
VERDE = (40, 220, 60)
AMARILLO = (255, 220, 0)


def _pintar(imagen, xs, ys, color, grosor=1):
    alto, ancho = imagen.shape[:2]
    for dy in range(-grosor + 1, grosor):
        for dx in range(-grosor + 1, grosor):
            px, py = np.round(xs).astype(int) + dx, np.round(ys).astype(int) + dy
            dentro = (px >= 0) & (px < ancho) & (py >= 0) & (py < alto)
            imagen[py[dentro], px[dentro]] = color


def dibujar_recta(imagen, theta_grados, rho, color=ROJO, grosor=2, mascara=None):
    """Recorre la recta ρ = x·cos θ + y·sen θ despejando y (o x, si es casi vertical).
    Si se pasa una máscara, solo se pinta donde la recta tiene apoyo en bordes."""
    alto, ancho = imagen.shape[:2]
    theta = np.deg2rad(theta_grados)
    seno, coseno = np.sin(theta), np.cos(theta)
    if abs(seno) > abs(coseno):
        xs = np.arange(ancho, dtype=float)
        ys = (rho - xs * coseno) / seno
    else:
        ys = np.arange(alto, dtype=float)
        xs = (rho - ys * seno) / coseno
    if mascara is not None:
        px, py = np.round(xs).astype(int), np.round(ys).astype(int)
        dentro = (px >= 0) & (px < ancho) & (py >= 0) & (py < alto)
        xs, ys = xs[dentro], ys[dentro]
        apoyo = mascara[py[dentro], px[dentro]]
        xs, ys = xs[apoyo], ys[apoyo]
    _pintar(imagen, xs, ys, color, grosor)


def dibujar_circunferencia(imagen, xo, yo, radio, color=VERDE, grosor=2):
    angulos = np.linspace(0, 2 * np.pi, max(60, int(8 * radio)), endpoint=False)
    _pintar(imagen, xo + radio * np.cos(angulos), yo + radio * np.sin(angulos), color, grosor)


def dibujar_cruz(imagen, xo, yo, tamanio=6, color=AMARILLO):
    eje = np.arange(-tamanio, tamanio + 1, dtype=float)
    _pintar(imagen, xo + eje, np.full_like(eje, yo), color, 2)
    _pintar(imagen, np.full_like(eje, xo), yo + eje, color, 2)


def acumulador_a_imagen(acumulador):
    """Mapa de calor simple del acumulador: más votos, más claro."""
    normalizado = acumulador / (acumulador.max() or 1)
    valores = (np.sqrt(normalizado) * 255)
    return np.stack([valores, valores * 0.6, valores * 0.2], axis=-1)


def dibujar_segmento(imagen, x1, y1, x2, y2, color=ROJO, grosor=2):
    pasos = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    _pintar(imagen, np.linspace(x1, x2, pasos), np.linspace(y1, y2, pasos), color, grosor)
