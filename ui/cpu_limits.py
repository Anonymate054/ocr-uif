"""
ui/cpu_limits.py
================
Limita los hilos de las librerías numéricas al 70 % de los núcleos para no
congelar el resto del equipo. Debe importarse antes que numpy u onnxruntime,
porque esas librerías leen estas variables al cargarse.
"""

import os

CORES_LIMIT = max(1, int((os.cpu_count() or 2) * 0.70))

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
