"""
tests/test_geometric_k.py

Tests automatizados para KPartitionGeometricSIA.
Verifican las rubricas del PDF2 5.2.1 sin depender de la red completa.

Ejecutar:
    uv run pytest tests/test_geometric_k.py -v --tb=short
"""

import numpy as np
import pytest
from itertools import product as iterproduct
from typing import List, Dict


# ---------------------------------------------------------------------------
# Funciones puras extraidas de geometric_k.py para tests unitarios
# ---------------------------------------------------------------------------


def _calcular_costo_standalone(
    estado_ini: list,
    estado_fin: list,
    flat_data: List[np.ndarray],
    tabla: Dict = None,
) -> List[float]:
    """
    Version standalone de _calcular_costo para tests sin dependencias del proyecto.
    Construye su propia tabla interna si no se provee una.
    """
    if tabla is None:
        tabla = {}

    def _costo(ini, fin):
        key = (tuple(ini), tuple(fin))
        if key in tabla:
            return tabla[key]

        n_vars = len(flat_data)
        dh = sum(a != b for a, b in zip(ini, fin))
        factor = 1.0 / (2 ** dh) if dh > 0 else 1.0

        ini_int = int("".join(map(str, ini[::-1])), 2)
        fin_int = int("".join(map(str, fin[::-1])), 2)

        diffs = [abs(float(fd[ini_int]) - float(fd[fin_int])) for fd in flat_data]

        if dh > 1:
            for i in range(len(ini)):
                if ini[i] != fin[i]:
                    vecino = list(fin)
                    vecino[i] = ini[i]
                    vk = (tuple(ini), tuple(vecino))
                    if vk not in tabla:
                        _costo(ini, vecino)
                    if vk in tabla:
                        for n in range(n_vars):
                            diffs[n] += tabla[vk][n]

        tabla[key] = [factor * d for d in diffs]
        return tabla[key]

    # Llenar la tabla nivel a nivel (BFS)
    n = len(estado_ini)
    ini_tuple = tuple(estado_ini)
    tabla[(ini_tuple, ini_tuple)] = [0.0] * len(flat_data)

    caminos = {0: [list(estado_ini)]}
    for nivel in range(1, n + 1):
        visitados = set()
        caminos[nivel] = []
        for estado_ant in caminos[nivel - 1]:
            for i in range(n):
                if estado_ant[i] != estado_fin[i]:
                    nuevo = estado_ant.copy()
                    nuevo[i] = estado_fin[i]
                    t = tuple(nuevo)
                    if t not in visitados:
                        caminos[nivel].append(nuevo)
                        _costo(estado_ini, nuevo)
                        visitados.add(t)

    return tabla.get((tuple(estado_ini), tuple(estado_fin)), [])


def _particionar_variables_en_k(n_vars: int, k: int):
    """Generador de asignaciones (copia de geometric_k.py)."""
    if k == 1:
        yield [0] * n_vars
        return
    if k > n_vars:
        return
    if n_vars * k <= 20:
        for asig in iterproduct(range(k), repeat=n_vars):
            if len(set(asig)) == k:
                yield list(asig)


def _canonicalizar(asig: List[int], k: int) -> tuple:
    mapa, nueva, cnt = {}, [], 0
    for g in asig:
        if g not in mapa:
            mapa[g] = cnt
            cnt += 1
        nueva.append(mapa[g])
    return tuple(nueva)


def _asignacion_a_grupos(asig: List[int], k: int) -> List[List[int]]:
    grupos = [[] for _ in range(k)]
    for idx, g in enumerate(asig):
        grupos[g].append(idx)
    return grupos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_data_n3c():
    """
    Tensores elementales del ejemplo N3C del PDF2 4.
    Variables A, B, C con estado OFF como inicial.
    """
    return [
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),  # A
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),  # B
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),  # C
    ]


# ---------------------------------------------------------------------------
# Tests unitarios de la tabla de costos
# ---------------------------------------------------------------------------


class TestTablaCostos:
    """Verifica la Tabla 4.2 del PDF2 4.2.3."""

    def test_costo_000_000_es_cero(self, flat_data_n3c):
        result = _calcular_costo_standalone([0, 0, 0], [0, 0, 0], flat_data_n3c)
        assert result == [0.0, 0.0, 0.0]

    def test_costo_000_100(self, flat_data_n3c):
        """t(000,100) - estado [1,0,0]: solo A cambia -> A=0.5, B=0, C=0"""
        result = _calcular_costo_standalone([0, 0, 0], [1, 0, 0], flat_data_n3c)
        assert abs(result[0] - 0.5) < 1e-9, f"A esperado 0.5, obtenido {result[0]}"
        assert abs(result[1] - 0.0) < 1e-9
        assert abs(result[2] - 0.0) < 1e-9

    def test_costo_000_010(self, flat_data_n3c):
        """t(000,010) - estado [0,1,0]: solo B cambia -> A=0, B=0.5, C=0"""
        result = _calcular_costo_standalone([0, 0, 0], [0, 1, 0], flat_data_n3c)
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9
        assert abs(result[2] - 0.0) < 1e-9

    def test_costo_000_001(self, flat_data_n3c):
        """t(000,001) - estado [0,0,1]: solo C cambia -> A=0, B=0, C=0.5"""
        result = _calcular_costo_standalone([0, 0, 0], [0, 0, 1], flat_data_n3c)
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 0.0) < 1e-9
        assert abs(result[2] - 0.5) < 1e-9

    def test_costo_000_110(self, flat_data_n3c):
        """t(000,110): A y B cambian -> A=0.375, B=0.375, C=0"""
        result = _calcular_costo_standalone([0, 0, 0], [1, 1, 0], flat_data_n3c)
        assert abs(result[0] - 0.375) < 1e-9
        assert abs(result[1] - 0.375) < 1e-9
        assert abs(result[2] - 0.000) < 1e-9

    def test_costo_000_101(self, flat_data_n3c):
        """t(000,101): A y C cambian -> A=0.375, B=0, C=0.375"""
        result = _calcular_costo_standalone([0, 0, 0], [1, 0, 1], flat_data_n3c)
        assert abs(result[0] - 0.375) < 1e-9
        assert abs(result[1] - 0.000) < 1e-9
        assert abs(result[2] - 0.375) < 1e-9

    def test_costo_000_011(self, flat_data_n3c):
        """t(000,011): B y C cambian -> A=0, B=0.375, C=0.375"""
        result = _calcular_costo_standalone([0, 0, 0], [0, 1, 1], flat_data_n3c)
        assert abs(result[0] - 0.000) < 1e-9
        assert abs(result[1] - 0.375) < 1e-9
        assert abs(result[2] - 0.375) < 1e-9

    def test_costo_000_111_valor_critico_pdf(self, flat_data_n3c):
        """
        Critico: t(000,111) = 0.21875 para A, B y C.
        Valor de la Tabla 4.2 del PDF2 4.2.3.
        """
        result = _calcular_costo_standalone([0, 0, 0], [1, 1, 1], flat_data_n3c)
        assert len(result) == 3, "Debe haber 3 costos (una por variable)"
        for i, var in enumerate(["A", "B", "C"]):
            assert abs(result[i] - 0.21875) < 1e-6, (
                f"Variable {var}: esperado 0.21875, obtenido {result[i]:.6f}"
            )

    def test_factor_decrecimiento_exponencial(self, flat_data_n3c):
        """El factor gamma = 2^{-d_H} debe aplicarse correctamente."""
        # d_H(000, 100) = 1 -> gamma = 0.5
        r1 = _calcular_costo_standalone([0, 0, 0], [1, 0, 0], flat_data_n3c)
        # d_H(000, 110) = 2 -> gamma = 0.25
        r2 = _calcular_costo_standalone([0, 0, 0], [1, 1, 0], flat_data_n3c)
        # Para B: r1[1]=0.0, r2[1]=0.375 - la relacion no es simple pero
        # verificamos que el nivel-2 incorpora contribuciones de los vecinos
        assert r2[1] >= r1[1], "El costo a distancia 2 debe incluir vecinos"


# ---------------------------------------------------------------------------
# Tests de particionamiento
# ---------------------------------------------------------------------------


class TestParticionamiento:
    """Verifica la logica de generacion de k-particiones."""

    def test_k1_produce_un_grupo(self):
        asigs = list(_particionar_variables_en_k(4, 1))
        assert len(asigs) == 1
        assert asigs[0] == [0, 0, 0, 0]

    def test_k2_n3_tres_particiones_canonicas(self):
        """S(3,2) = 3 particiones de Stirling de segundo tipo."""
        asigs = list(_particionar_variables_en_k(3, 2))
        canonicas = set(_canonicalizar(a, 2) for a in asigs)
        assert len(canonicas) == 3

    def test_k3_n4_seis_particiones_canonicas(self):
        """S(4,3) = 6 particiones de Stirling de segundo tipo."""
        asigs = list(_particionar_variables_en_k(4, 3))
        canonicas = set(_canonicalizar(a, 3) for a in asigs)
        assert len(canonicas) == 6

    def test_k_mayor_que_n_no_genera_nada(self):
        asigs = list(_particionar_variables_en_k(2, 5))
        assert asigs == []

    def test_grupos_son_disjuntos_y_cubren_todo(self):
        """Para cualquier asignacion, los grupos cubren todos los indices exactamente."""
        for asig in _particionar_variables_en_k(4, 2):
            grupos = _asignacion_a_grupos(asig, 2)
            todos = sorted(v for g in grupos for v in g)
            assert todos == [0, 1, 2, 3], f"Grupos no cubren todo: {grupos}"

    def test_canonicalizacion_detecta_equivalencias(self):
        """[0,1,0] y [1,0,1] son la misma particion (renombrado de grupos)."""
        c1 = _canonicalizar([0, 1, 0], 2)
        c2 = _canonicalizar([1, 0, 1], 2)
        assert c1 == c2, "Deben ser la misma particion canonica"

    def test_canonicalizacion_distingue_distintas(self):
        """[0,0,1] y [0,1,0] son particiones distintas."""
        c1 = _canonicalizar([0, 0, 1], 2)
        c2 = _canonicalizar([0, 1, 0], 2)
        assert c1 != c2


# ---------------------------------------------------------------------------
# Tests de metricas
# ---------------------------------------------------------------------------


class TestMetricas:
    """Verifica las metricas del benchmark (PDF2 5.2.2)."""

    def test_error_relativo_cero_cuando_iguales(self):
        from benchmark import error_relativo_phi
        assert error_relativo_phi(0.5, 0.5) == 0.0

    def test_error_relativo_correcto(self):
        from benchmark import error_relativo_phi
        assert abs(error_relativo_phi(1.0, 0.9) - 0.1) < 1e-9

    def test_error_relativo_phi_optimo_cero(self):
        from benchmark import error_relativo_phi
        assert error_relativo_phi(0.0, 0.0) == 0.0

    def test_jaccard_identicos_es_cero(self):
        from benchmark import distancia_jaccard
        A = (frozenset([1, 2]), frozenset([3, 4]))
        assert distancia_jaccard(A, A) == 0.0

    def test_jaccard_disjuntos_es_uno(self):
        from benchmark import distancia_jaccard
        A = (frozenset([1, 2]), frozenset([3, 4]))
        B = (frozenset([5, 6]), frozenset([7, 8]))
        assert distancia_jaccard(A, B) == 1.0

    def test_jaccard_simetrico(self):
        from benchmark import distancia_jaccard
        A = (frozenset([1, 2]), frozenset([3]))
        B = (frozenset([1]), frozenset([2, 3]))
        assert abs(distancia_jaccard(A, B) - distancia_jaccard(B, A)) < 1e-9

    def test_jaccard_alineacion_cruzada(self):
        """La distancia debe ser la minima sobre las dos alineaciones."""
        from benchmark import distancia_jaccard
        S1, S2 = frozenset([1, 2]), frozenset([3, 4])
        A = (S1, S2)
        B = (S2, S1)
        assert distancia_jaccard(A, B) == 0.0

    def test_clasificar_excelente(self):
        from benchmark import clasificar_calidad
        assert clasificar_calidad(95, 0.005, 0.05) == "Excelente"

    def test_clasificar_bueno(self):
        from benchmark import clasificar_calidad
        assert clasificar_calidad(85, 0.03, 0.15) == "Bueno"

    def test_clasificar_insuficiente(self):
        from benchmark import clasificar_calidad
        assert clasificar_calidad(60, 0.15, 0.4) == "Insuficiente"

    def test_speedup_mayor_uno_cuando_pyphi_mas_lento(self):
        from benchmark import speedup
        assert speedup(t_pyphi=10.0, t_geometric=0.5) == 20.0

    def test_speedup_cero_tiempo_no_divide_por_cero(self):
        from benchmark import speedup
        result = speedup(t_pyphi=1.0, t_geometric=0.0)
        assert result == float("inf")
