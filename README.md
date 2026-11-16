# TP4 – Transformada de Hough para rectas y circunferencias

Prototipo del Trabajo Práctico 4 de **Inteligencia Artificial** (Universidad Siglo 21).

Aplica la transformada de Hough sobre la foto real de un block de motor para detectar
**rectas** (bordes y aristas del block) y **circunferencias** (perforaciones), como base
para que el robot de la línea de montaje localice el punto de montaje.

## Imagen utilizada

`imagenes/motor_block.png` – block de motor V8.
Fuente: x-engineer.org, *What is the engine block* – https://x-engineer.org/engine-block/

## Qué está programado a mano

Para no operar "cajas negras", las librerías se limitan a la infraestructura:

| Librería | Uso |
|---|---|
| Pillow | Solo leer y guardar archivos de imagen |
| NumPy | Solo matrices y operaciones vectorizadas |

Todo el procesamiento está implementado en el proyecto:

1. **Escala de grises**: luminancia 0,299·R + 0,587·G + 0,114·B.
2. **Detección de bordes al estilo Canny**: suavizado gaussiano, gradiente de Sobel,
   supresión de no máximos y doble umbral con histéresis.
3. **Hough para rectas**: acumulador en el plano polar (θ, ρ) con ρ = x·cos θ + y·sen θ.
   Se usa la forma polar porque en (m, b) las rectas verticales tienen pendiente indeterminada.
4. **Hough para circunferencias**: con radio R fijo, cada píxel de borde vota una
   circunferencia de radio R en el plano de centros (xo, yo). Se repite para un rango de radios
   y los votos se normalizan por el perímetro.
5. **Búsqueda de máximos** con supresión de vecindad y **transformada inversa**: las figuras
   detectadas se dibujan sobre la foto original. Las rectas se recortan a los tramos con borde.

## Estructura

```
├── main.py                  # Programa único: modo rectas, circulos o todo
├── hough/
│   ├── imagen.py            # Lectura/escritura, grises, convolución, detector de bordes
│   ├── transformada.py      # Acumuladores, búsqueda de máximos y segmentos
│   └── dibujo.py            # Dibujo de rectas, segmentos, circunferencias y acumuladores
├── imagenes/motor_block.png # Foto de entrada
├── resultados/              # Imágenes generadas
└── tests/test_hough.py      # Pruebas con imágenes sintéticas de resultado conocido
```

## Ejecución

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
python main.py rectas        # punto 2
python main.py circulos      # punto 3
python main.py todo          # ambos
python main.py rectas otra_foto.jpg   # con otra imagen
python -m unittest -v        # pruebas
```

## Resultados generados

| Archivo | Contenido |
|---|---|
| `bordes.png` | Bordes detectados (entrada de ambas transformadas) |
| `rectas_salida.png` | Foto con los segmentos de recta detectados |
| `rectas_completas.png` | Foto con las rectas infinitas que entrega la transformada |
| `rectas_acumulador.png` | Plano (θ, ρ): cada punto de borde es una sinusoide |
| `circulos_salida.png` | Foto con las circunferencias detectadas y sus centros |
| `circulos_acumulador_R*.png` | Plano (xo, yo) para el radio de la mejor circunferencia |

## Autor

Jonathan Iván Gardel – Licenciatura en Informática
