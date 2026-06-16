"""
exec_k.py
---------
Punto de entrada para QNodes k-particiones (k in {3, 4, 5}).
Ejecuta QNodesK sobre los subsistemas de prueba de N10A y guarda resultados
en Excel.

Uso desde la raiz del repositorio:
    uv run python KParticiones/QNodes_K/exec_k.py --k 3

Uso desde la carpeta KParticiones/QNodes_K/:
    uv run python exec_k.py --k 3

Salida:
    QNodes/results/resultados_QNodes_k{k}.xlsx
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Raiz del repositorio: sube desde KParticiones/QNodes_K/ hasta proyecto-analisis-kpartitions/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# QNodes contiene el paquete src.* que usa QNodesK.
QNODES_ROOT = PROJECT_ROOT / "QNodes"

# QNodes_K también tiene un directorio src/, pero solo contiene estrategias.
# Lo retiramos de sys.path para evitar que opaque al paquete src de QNodes.
LOCAL_EXEC_ROOT = Path(__file__).resolve().parent
for candidate in (str(LOCAL_EXEC_ROOT), str(LOCAL_EXEC_ROOT / "src")):
    while candidate in sys.path:
        sys.path.remove(candidate)

if str(QNODES_ROOT) not in sys.path:
    sys.path.insert(0, str(QNODES_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from KParticiones.QNodes_K.src.strategies.q_nodes_k import QNodesK

SAMPLES_DIR = PROJECT_ROOT / "GeoMIP" / "data" / "samples"
RESULTS_DIR = PROJECT_ROOT / "QNodes" / "results"

# sistema, estado_inicial, condicion, alcance, mecanismo
SISTEMAS = {
    "N10A": {
        "estado_inicial": "1000000000",
        "condicion": "1111111111",
        "pruebas": [
            ("ABCDEFGHIJ", "ABCDEFGHIJ"),
            ("ABCDEFGHIJ", "ABCDEFGHI"),
            ("ABCDEFGHIJ", "BCDEFGHIJ"),
            ("ABCDEFGHIJ", "BCDEFGHI"),
            ("ABCDEFGHIJ", "ABDEGHJ"),
            ("ABCDEFGHIJ", "ACEGI"),
            ("ABCDEFGHIJ", "BDFHJ"),
            ("ABCDEFGHI", "ABCDEFGHIJ"),
            ("ABCDEFGHI", "ABCDEFGHI"),
            ("ABCDEFGHI", "BCDEFGHIJ"),
            ("ABCDEFGHI", "BCDEFGHI"),
            ("ABCDEFGHI", "ABDEGHJ"),
            ("ABCDEFGHI", "ACEGI"),
            ("ABCDEFGHI", "BDFHJ"),
            ("BCDEFGHIJ", "ABCDEFGHIJ"),
        ],
    },
        "N15A": {
            "estado_inicial": "1" + "0" * 14,
            "condicion": "1" * 15,
            "pruebas": [
                ("ABCDEFGHIJKLMNO", "ABCDEFGHIJKLMNO"),
                ("ABCDEFGHIJKLMNO", "ABCDEFGHIJKLMN"),
                ("ABCDEFGHIJKLMNO", "BCDEFGHIJKLMNO"),
                ("ABCDEFGHIJKLMNO", "BCDEFGHIJKLMN"),
                ("ABCDEFGHIJKLMNO", "ABDEFGHIJKLNO"),
                ("ABCDEFGHIJKLMNO", "ACEGIKM") ,
                ("ABCDEFGHIJKLMNO", "BDFHJLN"),
                ("ABCDEFGHIJKLMN", "ABCDEFGHIJKLMNO"),
                ("ABCDEFGHIJKLMN", "ABCDEFGHIJKLMN"),
                ("ABCDEFGHIJKLMN", "BCDEFGHIJKLMNO"),
                ("ABCDEFGHIJKLMN", "BCDEFGHIJKLMN"),
                ("ABCDEFGHIJKLMN", "ABDEFGHIJKLNO"),
                ("ABCDEFGHIJKLMN", "ACEGIKM"),
                ("ABCDEFGHIJKLMN", "BDFHJLN"),
                ("BCDEFGHIJKLMNO", "ABCDEFGHIJKLMNO"),
            ],
        },
        "N20A": {
            "estado_inicial": "1" + "0" * 19,
            "condicion": "1" * 20,
            "pruebas": [
                ("ABCDEFGHIJKLMNOPQRST", "ABCDEFGHIJKLMNOPQRST"),
                ("ABCDEFGHIJKLMNOPQRST", "ABCDEFGHIJKLMNOPQRS"),
                ("ABCDEFGHIJKLMNOPQRST", "BCDEFGHIJKLMNOPQRST"),
                ("ABCDEFGHIJKLMNOPQRST", "BCDEFGHIJKLMNOPQRS"),
                ("ABCDEFGHIJKLMNOPQRST", "ABDEFGHIJKLMNOPQRT"),
                ("ABCDEFGHIJKLMNOPQRST", "ACEGIKMOQ"),
                ("ABCDEFGHIJKLMNOPQRST", "BDFHJLNPRT"),
                ("ABCDEFGHIJKLMNOPQRS", "ABCDEFGHIJKLMNOPQRST"),
                ("ABCDEFGHIJKLMNOPQRS", "ABCDEFGHIJKLMNOPQRS"),
                ("ABCDEFGHIJKLMNOPQRS", "BCDEFGHIJKLMNOPQRST"),
                ("ABCDEFGHIJKLMNOPQRS", "BCDEFGHIJKLMNOPQRS"),
                ("ABCDEFGHIJKLMNOPQRS", "ABDEFGHIJKLMNOPQRT"),
                ("ABCDEFGHIJKLMNOPQRS", "ACEGIKMOQ"),
                ("ABCDEFGHIJKLMNOPQRS", "BDFHJLNPRT"),
                ("BCDEFGHIJKLMNOPQRST", "ABCDEFGHIJKLMNOPQRST"),
            ],
        },
}

TIMEOUT_SEG = 3600


def _worker(estado_ini, condicion, alcance, mecanismo, tpm, k, cola):
    try:
        gestor = Manager(estado_inicial=estado_ini)
        analizador = QNodesK(tpm, k=k)
        t0 = time.perf_counter()
        sol = analizador.aplicar_estrategia(estado_ini, condicion, alcance, mecanismo)
        t_total = time.perf_counter() - t0
        cola.put(
            {
                "particion": str(sol.particion),
                "perdida": str(sol.perdida).replace(".", ","),
                "tiempo_s": str(round(t_total, 4)).replace(".", ","),
                "error": None,
            }
        )
    except Exception as e:
        cola.put({"particion": None, "perdida": None, "tiempo_s": None, "error": str(e)})


def letras_a_bits(letras: str, nodos: list) -> str:
    bits = ["0"] * len(nodos)
    for c in letras.upper():
        if c in nodos:
            bits[nodos.index(c)] = "1"
    return "".join(bits)


def ejecutar_sistema(nombre, estado_ini, condicion, alcance_letras, mecanismo_letras, k):
    tpm_path = SAMPLES_DIR / f"{nombre}.csv"
    if not tpm_path.exists():
        print(f"  [!] No se encontro: {tpm_path}")
        return {
            "sistema": nombre,
            "k": k,
            "perdida": None,
            "tiempo_s": None,
            "particion": None,
            "error": f"Archivo no encontrado: {tpm_path}",
        }

    tpm = np.genfromtxt(tpm_path, delimiter=",")
    # Derivar la lista de nodos a partir del largo de estado_inicial
    import string
    nodos = list(string.ascii_uppercase[: len(estado_ini)])
    alcance = letras_a_bits(alcance_letras, nodos)
    mecanismo = letras_a_bits(mecanismo_letras, nodos)

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
            "sistema": nombre,
            "k": k,
            "perdida": None,
            "tiempo_s": None,
            "particion": None,
            "error": "Timeout",
        }

    resultado = cola.get() if not cola.empty() else {
        "particion": None,
        "perdida": None,
        "tiempo_s": None,
        "error": "Sin resultado en cola",
    }
    return {"sistema": nombre, "k": k, **resultado}


def ejecutar_k(nombre_sistema: str, k: int) -> list[dict]:
    cfg = SISTEMAS[nombre_sistema]
    resultados = []

    print(f"\n{'='*55}")
    print(f"  QNodes k-particiones  (k={k})")
    print(f"{'='*55}")

    for idx, (alcance_l, mecanismo_l) in enumerate(cfg["pruebas"], start=1):
        print(f"\n  [{nombre_sistema}] prueba {idx:02d} ejecutando con k={k} ...")
        r = ejecutar_sistema(
            nombre_sistema,
            cfg["estado_inicial"],
            cfg["condicion"],
            alcance_l,
            mecanismo_l,
            k,
        )
        r["prueba"] = idx
        r["alcance"] = alcance_l
        r["mecanismo"] = mecanismo_l
        resultados.append(r)
        if r["error"]:
            print(f"  [{nombre_sistema}] prueba {idx:02d} ERROR: {r['error']}")
        else:
            print(f"  [{nombre_sistema}] prueba {idx:02d} perdida={r['perdida']}  tiempo={r['tiempo_s']}s")

    return resultados


def main():
    parser = argparse.ArgumentParser(description="QNodes k-particiones")
    parser.add_argument(
        "--k",
        type=int,
        choices=[3, 4, 5],
        default=None,
        help="Numero de particiones: 3, 4 o 5 (opcional si usas --ks)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="*",
        choices=[3, 4, 5],
        default=None,
        help="Lista de particiones a ejecutar, por ejemplo: --ks 3 4 5",
    )
    parser.add_argument(
        "--sistema",
        type=str,
        choices=list(SISTEMAS.keys()),
        default="N10A",
        help="Sistema a procesar: N10A, N15A o N20A (default: N10A)",
    )
    args = parser.parse_args()

    aplicacion.desactivar_profiling()
    aplicacion.set_pagina_red_muestra("A")

    if args.ks:
        ks = args.ks
    elif args.k is not None:
        ks = [args.k]
    else:
        ks = [3, 4, 5]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for k in ks:
        resultados = ejecutar_k(args.sistema, k)
        df = pd.DataFrame(
            resultados,
            columns=[
                "sistema",
                "prueba",
                "alcance",
                "mecanismo",
                "k",
                "perdida",
                "tiempo_s",
                "particion",
                "error",
            ],
        )

        salida = RESULTS_DIR / f"resultados_QNodes_k{k}_{args.sistema}.xlsx"
        df.to_excel(salida, index=False)

        print(f"\n  Resultados guardados en: {salida}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()