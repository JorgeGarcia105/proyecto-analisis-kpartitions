"""
exec_k.py
---------
Punto de entrada para GeoMIP k-particiones (k in {3, 4, 5}).
Ejecuta KPartitionGeometricSIA sobre N10A, N15A, N20A, N22A
y guarda resultados en Excel.

Uso (desde Method2_Dynamic_Programming_Reformulation/):
    uv run python ..\\..\\..\\..\\KParticiones\\GeoMIP_K\\exec_k.py --k 3
    uv run python ..\\..\\..\\..\\KParticiones\\GeoMIP_K\\exec_k.py --k 4
    uv run python ..\\..\\..\\..\\KParticiones\\GeoMIP_K\\exec_k.py --k 5

Salida:
    GeoMIP/results/resultados_Geometric_k{k}.xlsx
"""

import sys
import argparse
import time
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

# Raiz del proyecto: sube desde KParticiones/GeoMIP_K/ hasta proyecto-analisis-kpartitions/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raiz de Method2 — de ahi resuelve todos los imports src.*
METHOD2_ROOT = (
    PROJECT_ROOT
    / "GeoMIP"
    / "src"
    / "Method2_Dynamic_Programming_Reformulation"
)

if str(METHOD2_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD2_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from KParticiones.GeoMIP_K.src.strategies.geometric_k import KPartitionGeometricSIA

GEOMIP_ROOT = PROJECT_ROOT / "GeoMIP"
SAMPLES_DIR = GEOMIP_ROOT / "data" / "samples"
RESULTS_DIR = GEOMIP_ROOT / "results"

# sistema, estado_inicial, condicion, alcance, mecanismo
SISTEMAS = [
    ("N10A", "1000000000",             "1111111111",           "1111111111",           "1111111111"),
    ("N15A", "100000000000000",        "111111111111111",      "111111111111111",      "111111111111111"),
    ("N20A", "10000000000000000000",   "11111111111111111111", "11111111111111111111", "11111111111111111111"),
    ("N22A", "1000000000000000000000", "1111111111111111111111","1111111111111111111111","1111111111111111111111"),
]

TIMEOUT_SEG = 3600


def _worker(estado_ini, condicion, alcance, mecanismo, tpm, k, cola):
    try:
        gestor = Manager(estado_inicial=estado_ini)
        analizador = KPartitionGeometricSIA(gestor, k=k)
        t0 = time.perf_counter()
        sol = analizador.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        t_total = time.perf_counter() - t0
        cola.put({
            "particion": str(sol.particion),
            "perdida": str(sol.perdida).replace(".", ","),
            "tiempo_s": str(round(t_total, 4)).replace(".", ","),
            "error": None,
        })
    except Exception as e:
        cola.put({"particion": None, "perdida": None, "tiempo_s": None, "error": str(e)})


def ejecutar_sistema(nombre, estado_ini, condicion, alcance, mecanismo, k):
    tpm_path = SAMPLES_DIR / f"{nombre}.csv"
    if not tpm_path.exists():
        print(f"  [!] No se encontro: {tpm_path}")
        return {
            "sistema": nombre, "k": k,
            "perdida": None, "tiempo_s": None, "particion": None,
            "error": f"Archivo no encontrado: {tpm_path}",
        }

    tpm = np.genfromtxt(tpm_path, delimiter=",")

    cola = multiprocessing.Queue()
    proceso = multiprocessing.Process(
        target=_worker,
        args=(estado_ini, condicion, alcance, mecanismo, tpm, k, cola),
    )
    proceso.start()
    proceso.join(timeout=TIMEOUT_SEG)

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        return {
            "sistema": nombre, "k": k,
            "perdida": None, "tiempo_s": None, "particion": None,
            "error": "Timeout",
        }

    resultado = cola.get() if not cola.empty() else {
        "particion": None, "perdida": None, "tiempo_s": None, "error": "Sin resultado en cola"
    }
    return {"sistema": nombre, "k": k, **resultado}


def main():
    parser = argparse.ArgumentParser(description="GeoMIP k-particiones")
    parser.add_argument(
        "--k", type=int, choices=[3, 4, 5], default=3,
        help="Numero de particiones: 3, 4 o 5 (default: 3)"
    )
    args = parser.parse_args()
    k = args.k

    aplicacion.profiler_habilitado = False

    print(f"\n{'='*55}")
    print(f"  GeoMIP k-particiones  (k={k})")
    print(f"{'='*55}")

    resultados = []
    for nombre, estado_ini, condicion, alcance, mecanismo in SISTEMAS:
        print(f"\n  [{nombre}] ejecutando con k={k} ...")
        r = ejecutar_sistema(nombre, estado_ini, condicion, alcance, mecanismo, k)
        resultados.append(r)
        if r["error"]:
            print(f"  [{nombre}] ERROR: {r['error']}")
        else:
            print(f"  [{nombre}] perdida={r['perdida']}  tiempo={r['tiempo_s']}s")

    df = pd.DataFrame(resultados, columns=[
        "sistema", "k", "perdida", "tiempo_s", "particion", "error"
    ])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    salida = RESULTS_DIR / f"resultados_Geometric_k{k}.xlsx"
    df.to_excel(salida, index=False)

    print(f"\n  Resultados guardados en: {salida}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
