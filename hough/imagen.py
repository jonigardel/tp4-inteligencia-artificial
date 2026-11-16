"""Lectura, escritura y preprocesamiento de imágenes (etapas de bajo nivel).

Pillow se usa solo para leer y guardar archivos. Todo el procesamiento
(escala de grises, suavizado, gradientes y bordes) está programado acá.
"""
import numpy as np
from PIL import Image


def leer_imagen_rgb(ruta: str) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float64)


def guardar_imagen(matriz: np.ndarray, ruta: str) -> None:
    Image.fromarray(np.clip(matriz, 0, 255).astype(np.uint8)).save(ruta)


def a_escala_de_grises(rgb: np.ndarray) -> np.ndarray:
    """Luminancia: 0,299·R + 0,587·G + 0,114·B."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def convolucionar(imagen: np.ndarray, nucleo: np.ndarray) -> np.ndarray:
    """Convolución 2D (filtrado espacial) con relleno por borde replicado."""
    alto_n, ancho_n = nucleo.shape
    pad_y, pad_x = alto_n // 2, ancho_n // 2
    ampliada = np.pad(imagen, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    resultado = np.zeros_like(imagen)
    nucleo_invertido = nucleo[::-1, ::-1]
    for i in range(alto_n):
        for j in range(ancho_n):
            resultado += nucleo_invertido[i, j] * ampliada[i:i + imagen.shape[0], j:j + imagen.shape[1]]
    return resultado


def nucleo_gaussiano(tamanio: int = 5, sigma: float = 1.2) -> np.ndarray:
    centro = tamanio // 2
    y, x = np.mgrid[-centro:centro + 1, -centro:centro + 1]
    nucleo = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return nucleo / nucleo.sum()


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def detectar_bordes(gris: np.ndarray, umbral_bajo: float = 0.10, umbral_alto: float = 0.25,
                    sigma: float = 1.2) -> np.ndarray:
    """Detector de bordes al estilo Canny, programado paso a paso.

    1. Suavizado gaussiano para reducir el ruido.
    2. Gradiente con Sobel: magnitud y dirección.
    3. Supresión de no máximos: el borde queda de un píxel de ancho.
    4. Doble umbral con histéresis: los bordes débiles se conservan
       solo si están conectados a un borde fuerte.
    Los umbrales son fracciones de la magnitud máxima del gradiente.
    """
    suavizada = convolucionar(gris, nucleo_gaussiano(5, sigma))
    gx = convolucionar(suavizada, SOBEL_X)
    gy = convolucionar(suavizada, SOBEL_Y)
    magnitud = np.hypot(gx, gy)
    magnitud /= magnitud.max() or 1.0

    adelgazada = _suprimir_no_maximos(magnitud, np.arctan2(gy, gx))
    fuertes = adelgazada >= umbral_alto
    debiles = (adelgazada >= umbral_bajo) & ~fuertes
    return _histeresis(fuertes, debiles)


def _suprimir_no_maximos(magnitud, direccion):
    angulo = (np.rad2deg(direccion) + 180) % 180
    sector = np.zeros(angulo.shape, dtype=int)            # 0°: vecinos izquierda/derecha
    sector[(angulo >= 22.5) & (angulo < 67.5)] = 1          # 45°
    sector[(angulo >= 67.5) & (angulo < 112.5)] = 2         # 90°: vecinos arriba/abajo
    sector[(angulo >= 112.5) & (angulo < 157.5)] = 3        # 135°
    desplazamientos = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)),
                       2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    ampliada = np.pad(magnitud, 1)
    alto, ancho = magnitud.shape
    resultado = np.zeros_like(magnitud)
    for numero, ((dy1, dx1), (dy2, dx2)) in desplazamientos.items():
        vecino1 = ampliada[1 + dy1:1 + dy1 + alto, 1 + dx1:1 + dx1 + ancho]
        vecino2 = ampliada[1 + dy2:1 + dy2 + alto, 1 + dx2:1 + dx2 + ancho]
        es_maximo = (sector == numero) & (magnitud >= vecino1) & (magnitud >= vecino2)
        resultado[es_maximo] = magnitud[es_maximo]
    return resultado


def _histeresis(fuertes, debiles):
    bordes = fuertes.copy()
    while True:
        ampliada = np.pad(bordes, 1)
        vecindad = np.zeros_like(bordes)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                vecindad |= ampliada[1 + dy:1 + dy + bordes.shape[0], 1 + dx:1 + dx + bordes.shape[1]]
        nuevos = debiles & vecindad & ~bordes
        if not nuevos.any():
            return bordes
        bordes |= nuevos
