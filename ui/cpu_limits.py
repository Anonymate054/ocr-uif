"""
ui/cpu_limits.py
================
Limita los hilos de las librerías numéricas al 70 % de los núcleos para no
congelar el resto del equipo. Debe importarse antes que numpy u onnxruntime,
porque esas librerías leen estas variables al cargarse.

CORES_LIMIT es también el valor por defecto de la barra de núcleos de la UI,
que decide cuántos PDFs se leen en paralelo (de 1 a AVAILABLE_CORES).
"""

import os

AVAILABLE_CORES = os.cpu_count() or 2
CORES_LIMIT = max(1, int(AVAILABLE_CORES * 0.70))


def clamp_cores(cores) -> int:
    """Núcleos pedidos, acotados a [1, AVAILABLE_CORES]; None → CORES_LIMIT."""
    if cores is None:
        return CORES_LIMIT
    return max(1, min(int(cores), AVAILABLE_CORES))


for _var in (
    "OMP_NUM_THREADS",
    "ONNXRUNTIME_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    # setdefault: respeta un valor puesto a mano por quien ejecuta la app
    os.environ.setdefault(_var, str(CORES_LIMIT))
