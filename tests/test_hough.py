"""Pruebas sobre imágenes sintéticas con resultado conocido.  Ejecutar: python -m unittest -v"""
import unittest

import numpy as np

from hough.imagen import detectar_bordes
from hough.transformada import (Recta, acumulador_rectas, buscar_circunferencias, buscar_rectas,
                                segmentos_de_recta)


def imagen_con_rectangulo():
    imagen = np.zeros((120, 160))
    imagen[30:90, 40:120] = 255.0          # lados: x = 40 y x = 119, y = 30 y y = 89
    return imagen


class TestDeteccionDeBordes(unittest.TestCase):

    def test_imagen_uniforme_no_tiene_bordes(self):
        self.assertFalse(detectar_bordes(np.full((50, 50), 128.0)).any())

    def test_bordes_del_rectangulo_quedan_sobre_su_contorno(self):
        bordes = detectar_bordes(imagen_con_rectangulo())
        ys, xs = np.nonzero(bordes)
        self.assertTrue(bordes.any())
        cerca_de_un_lado = (np.minimum(abs(xs - 40), abs(xs - 119)) <= 2) | \
                           (np.minimum(abs(ys - 30), abs(ys - 89)) <= 2)
        self.assertTrue(cerca_de_un_lado.all())


class TestRectas(unittest.TestCase):

    def test_detecta_las_rectas_verticales_que_fallan_en_el_plano_m_b(self):
        bordes = detectar_bordes(imagen_con_rectangulo())
        acumulador, thetas, diagonal = acumulador_rectas(bordes)
        rectas = buscar_rectas(acumulador, thetas, diagonal, cantidad=4, votos_minimos=20)
        verticales = [r for r in rectas if r.theta_grados in (0.0, 1.0, 179.0)]
        self.assertEqual(len(verticales), 2)
        self.assertEqual(sorted(round(abs(r.rho)) // 10 for r in verticales), [3, 11])

    def test_recta_diagonal_conocida(self):
        bordes = np.zeros((100, 100), dtype=bool)
        for x in range(100):
            bordes[x, x] = True                 # y = x  ->  θ = 135°, ρ = 0
        acumulador, thetas, diagonal = acumulador_rectas(bordes)
        recta = buscar_rectas(acumulador, thetas, diagonal, cantidad=1, votos_minimos=10)[0]
        self.assertEqual(recta.theta_grados, 135.0)
        self.assertEqual(recta.rho, 0.0)

    def test_segmento_recorta_la_recta_al_tramo_con_borde(self):
        bordes = np.zeros((100, 200), dtype=bool)
        bordes[50, 60:140] = True               # tramo horizontal de 80 px
        segmentos = segmentos_de_recta(bordes, Recta(90.0, 50.0, 80), largo_minimo=10)
        self.assertEqual(len(segmentos), 1)
        self.assertAlmostEqual(segmentos[0].largo, 79, delta=5)   # ±2 px de tolerancia en cada extremo


class TestCircunferencias(unittest.TestCase):

    def test_encuentra_centro_y_radio_de_un_disco(self):
        y, x = np.mgrid[0:120, 0:120]
        imagen = np.where((x - 70) ** 2 + (y - 50) ** 2 <= 15 ** 2, 255.0, 0.0)
        bordes = detectar_bordes(imagen)
        mejor = buscar_circunferencias(bordes, range(10, 21), cantidad=1, puntaje_minimo=0.3)[0]
        self.assertLessEqual(abs(mejor.xo - 70), 1)
        self.assertLessEqual(abs(mejor.yo - 50), 1)
        self.assertLessEqual(abs(mejor.radio - 15), 1)

    def test_una_elipse_junta_menos_votos_que_un_circulo(self):
        y, x = np.mgrid[0:140, 0:140]
        circulo = np.where((x - 70) ** 2 + (y - 70) ** 2 <= 30 ** 2, 255.0, 0.0)
        elipse = np.where(((x - 70) / 30) ** 2 + ((y - 70) / 18) ** 2 <= 1, 255.0, 0.0)
        puntaje = lambda img: buscar_circunferencias(detectar_bordes(img), [30], 1, 0.0)[0].puntaje
        self.assertGreater(puntaje(circulo), 2 * puntaje(elipse))


if __name__ == "__main__":
    unittest.main()
