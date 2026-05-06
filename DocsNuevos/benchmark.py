"""
benchmark.py — Evaluación de GeometricSIA / KPartitionGeometricSIA
contra PyPhi (referencia exhaustiva), según métricas del Cap. 5.2.2.

Uso:
    python benchmark.py --excel tests/PruebasIniciales.xlsx --k 2
    python benchmark.py --excel tests/PruebasIniciales.xlsx --k 3 --sheet 0

Métricas calculadas (Tabla 5.1 del documento):
    - Tasa de acierto exacto       (>90% = Excelente)
    - Error relativo en Phi        (<1%  = Excelente)
    - Distancia de Jaccard         (<0.1 = Excelente)
    - Speedup relativo vs PyPhi
    - Escalabilidad O(n·2^n)
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Métricas (independientes del framework)
# ---------------------------------------------------------------------------

def tasa_acierto(resultados: List[Dict]) -> float:
    """Porcentaje de casos donde la bipartición coincide exactamente con PyPhi."""
    if not resultados:
        return 0.0
    aciertos = sum(1 for r in resultados if r.get("acierto_exacto", False))
    return aciertos / len(resultados) * 100


def error_relativo_phi(phi_optimo: float, phi_encontrado: float) -> float:
    """E_rel = |phi_opt - phi_enc| / phi_opt  (0 si phi_opt == 0)."""
    if phi_optimo == 0.0:
        return 0.0 if phi_encontrado == 0.0 else float("inf")
    return abs(phi_optimo - phi_encontrado) / phi_optimo


def distancia_jaccard(
    particion_a: Tuple[frozenset, frozenset],
    particion_b: Tuple[frozenset, frozenset],
) -> float:
    """
    Distancia de Jaccard entre dos biparticiones.
    Cada bipartición es una tupla de dos frozensets de nodos.
    Se toma el mínimo sobre las dos alineaciones posibles (A↔A, A↔B).
    """
    def jaccard_sets(s1: frozenset, s2: frozenset) -> float:
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return 1.0 - (inter / union if union > 0 else 1.0)

    # Alineación directa
    d_dir = (
        jaccard_sets(particion_a[0], particion_b[0])
        + jaccard_sets(particion_a[1], particion_b[1])
    ) / 2

    # Alineación cruzada
    d_cruz = (
        jaccard_sets(particion_a[0], particion_b[1])
        + jaccard_sets(particion_a[1], particion_b[0])
    ) / 2

    return min(d_dir, d_cruz)


def clasificar_calidad(
    tasa: float, err_rel: float, jaccard: float
) -> str:
    """Tabla 5.1 del documento."""
    if tasa > 90 and err_rel < 0.01 and jaccard < 0.1:
        return "Excelente"
    if tasa > 80 and err_rel < 0.05 and jaccard < 0.2:
        return "Bueno"
    if tasa > 70 and err_rel < 0.10 and jaccard < 0.3:
        return "Aceptable"
    return "Insuficiente"


def speedup(t_pyphi: float, t_geometric: float) -> float:
    """S_rel = T_pyphi / T_geometric."""
    if t_geometric == 0.0:
        return float("inf")
    return t_pyphi / t_geometric


# ---------------------------------------------------------------------------
# Runner principal (usa las clases del proyecto si están disponibles)
# ---------------------------------------------------------------------------

def _intentar_importar_estrategias():
    """Importa las estrategias del proyecto. Devuelve None si no están disponibles."""
    try:
        from src.controllers.strategies.geometric_k import KPartitionGeometricSIA
        from src.controllers.strategies.phi import Phi
        return KPartitionGeometricSIA, Phi
    except ImportError:
        return None, None


def ejecutar_caso(
    gestor,
    fila: pd.Series,
    k: int,
    KPartitionGeometricSIA,
    Phi,
) -> Dict:
    """Ejecuta un caso de prueba y devuelve métricas."""
    resultado = {
        "caso": fila.get("caso", "?"),
        "n_vars": fila.get("n_vars", 0),
        "phi_pyphi": None,
        "phi_geometric": None,
        "t_pyphi": None,
        "t_geometric": None,
        "acierto_exacto": False,
        "error_relativo": None,
        "jaccard": None,
        "calidad": None,
        "speedup": None,
    }

    condicion = str(fila.get("condicion", ""))
    alcance = str(fila.get("alcance", ""))
    mecanismo = str(fila.get("mecanismo", ""))
    tpm_path = fila.get("tpm_path", "")

    try:
        tpm = np.loadtxt(tpm_path, delimiter=",")
    except Exception as e:
        resultado["error"] = f"No se pudo cargar TPM: {e}"
        return resultado

    # --- PyPhi (referencia) ---
    try:
        phi_inst = Phi(gestor)
        t0 = time.perf_counter()
        sol_phi = phi_inst.aplicar_estrategia(condicion, alcance, mecanismo)
        resultado["t_pyphi"] = time.perf_counter() - t0
        resultado["phi_pyphi"] = float(sol_phi.perdida)
        part_phi = _extraer_particion(sol_phi.particion)
    except Exception as e:
        resultado["error_pyphi"] = str(e)
        part_phi = None

    # --- Geométrico ---
    try:
        geo_inst = KPartitionGeometricSIA(gestor, k=k)
        t0 = time.perf_counter()
        sol_geo = geo_inst.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        resultado["t_geometric"] = time.perf_counter() - t0
        resultado["phi_geometric"] = float(sol_geo.perdida)
        part_geo = _extraer_particion(sol_geo.particion)
    except Exception as e:
        resultado["error_geometric"] = str(e)
        part_geo = None

    # --- Métricas ---
    if resultado["phi_pyphi"] is not None and resultado["phi_geometric"] is not None:
        resultado["error_relativo"] = error_relativo_phi(
            resultado["phi_pyphi"], resultado["phi_geometric"]
        )

    if part_phi is not None and part_geo is not None:
        resultado["jaccard"] = distancia_jaccard(part_phi, part_geo)
        resultado["acierto_exacto"] = resultado["jaccard"] < 1e-6

    if resultado["t_pyphi"] and resultado["t_geometric"]:
        resultado["speedup"] = speedup(resultado["t_pyphi"], resultado["t_geometric"])

    return resultado


def _extraer_particion(fmt_particion) -> Optional[Tuple[frozenset, frozenset]]:
    """Convierte el formato de partición del proyecto a (frozenset, frozenset)."""
    try:
        if hasattr(fmt_particion, "parte1") and hasattr(fmt_particion, "parte2"):
            return frozenset(fmt_particion.parte1), frozenset(fmt_particion.parte2)
        if isinstance(fmt_particion, (list, tuple)) and len(fmt_particion) == 2:
            return frozenset(fmt_particion[0]), frozenset(fmt_particion[1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def generar_reporte(resultados: List[Dict], k: int, output_path: Path):
    """Genera un Excel con los resultados y un resumen de métricas."""
    df = pd.DataFrame(resultados)

    # Resumen global
    casos_validos = [r for r in resultados if r.get("phi_pyphi") is not None
                     and r.get("phi_geometric") is not None]

    if casos_validos:
        tasa = tasa_acierto(casos_validos)
        err_medio = np.mean([r["error_relativo"] for r in casos_validos
                             if r.get("error_relativo") is not None])
        jacc_medio = np.mean([r["jaccard"] for r in casos_validos
                              if r.get("jaccard") is not None])
        sp_medio = np.mean([r["speedup"] for r in casos_validos
                            if r.get("speedup") is not None])
        calidad = clasificar_calidad(tasa, err_medio, jacc_medio)
    else:
        tasa = err_medio = jacc_medio = sp_medio = 0.0
        calidad = "Sin datos"

    resumen = pd.DataFrame([{
        "k": k,
        "casos_evaluados": len(casos_validos),
        "tasa_acierto_%": round(tasa, 2),
        "error_relativo_medio": round(err_medio, 4),
        "jaccard_medio": round(jacc_medio, 4),
        "speedup_medio": round(sp_medio, 2),
        "calidad": calidad,
    }])

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="Detalle", index=False)
        resumen.to_excel(writer, sheet_name="Resumen", index=False)

    print(f"\n{'='*55}")
    print(f"  REPORTE BENCHMARK  (k={k})")
    print(f"{'='*55}")
    print(f"  Casos evaluados  : {len(casos_validos)}")
    print(f"  Tasa de acierto  : {tasa:.1f}%")
    print(f"  Error relativo Φ : {err_medio:.4f}")
    print(f"  Jaccard medio    : {jacc_medio:.4f}")
    print(f"  Speedup vs PyPhi : {sp_medio:.1f}x")
    print(f"  Calidad global   : {calidad}")
    print(f"{'='*55}")
    print(f"  Reporte guardado : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GeometricSIA vs PyPhi"
    )
    parser.add_argument(
        "--excel", default="tests/PruebasIniciales.xlsx",
        help="Ruta al Excel de pruebas"
    )
    parser.add_argument(
        "--sheet", type=int, default=0,
        help="Índice de hoja del Excel (default: 0)"
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="Número de particiones (default: 2)"
    )
    parser.add_argument(
        "--output", default="results/benchmark_results.xlsx",
        help="Archivo de salida Excel"
    )
    args = parser.parse_args()

    KPartitionGeometricSIA, Phi = _intentar_importar_estrategias()
    if KPartitionGeometricSIA is None:
        print("⚠️  No se pudieron importar las estrategias del proyecto.")
        print("   Ejecuta desde la raíz del proyecto con: uv run python benchmark.py ...")
        return

    try:
        df = pd.read_excel(args.excel, sheet_name=args.sheet)
    except Exception as e:
        print(f"Error leyendo Excel: {e}")
        return

    # El gestor se construye desde la primera fila (configuración base)
    # Ajustar según la estructura real del proyecto
    try:
        from src.controllers.manager import Manager
        gestor = Manager()  # Ajustar si Manager requiere argumentos
    except Exception as e:
        print(f"Error construyendo Manager: {e}")
        return

    resultados = []
    for _, fila in df.iterrows():
        print(f"  Procesando caso: {fila.get('caso', '?')} ...")
        r = ejecutar_caso(gestor, fila, args.k, KPartitionGeometricSIA, Phi)
        resultados.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generar_reporte(resultados, args.k, Path(args.output))


if __name__ == "__main__":
    main()
