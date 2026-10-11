"""
ui/cpu_limits.py
================
Limita los hilos de las librerías numéricas al 70 % de los núcleos para no
congelar el resto del equipo. Debe importarse antes que numpy u onnxruntime,
porque esas librerías leen estas variables al cargarse.

La barra de núcleos de la UI decide cuántos PDFs se leen en paralelo: va de
1 a AVAILABLE_CORES y por defecto usa todos los disponibles (DEFAULT_CORES).
"""

import os


def _available_cores() -> int:
    """Núcleos que este proceso puede usar de verdad (respeta la afinidad de
    CPU en Python 3.13+); si no se pueden leer, el total del equipo."""
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count or os.cpu_count() or 2


AVAILABLE_CORES = _available_cores()
DEFAULT_CORES = AVAILABLE_CORES
CORES_LIMIT = max(1, int(AVAILABLE_CORES * 0.70))


def clamp_cores(cores) -> int:
    """Núcleos pedidos, acotados a [1, AVAILABLE_CORES]; None → DEFAULT_CORES."""
    if cores is None:
        return DEFAULT_CORES
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
