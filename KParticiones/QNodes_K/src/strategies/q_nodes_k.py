"""
q_nodes_k.py
------------
Extension del algoritmo QNodes para k-particiones con k in {3, 4, 5}.

Relacion con el proyecto base
------------------------------
- Hereda de SIA (mismo patron que QNodes en q_nodes.py).
- Reutiliza sin modificar: funcion_submodular(), definir_clave(),
  sia_preparar_subsistema(), System.bipartir(), emd_efecto().
- El algoritmo Q original corre (2n - 2) fases para k=2.
  Esta extension aplica la Opcion B: corre (2n - k) fases y
  detiene cuando quedan exactamente (k-1) super-nodos colapsados,
  produciendo k grupos finales.

Archivos agregados/modificados en el proyecto base
----------------------------------------------------
- QNodes/src/constants/models.py : se agregaron QNODES_K_LABEL,
                                   QNODES_K_STRAREGY_TAG, QNODES_K_ANALYSIS_TAG
- QNodes/src/funcs/format.py     : se agrego fmt_kparticion_q()

Restricciones del enunciado respetadas
----------------------------------------
- Compatibilidad con arquitectura existente (hereda SIA de QNodes).
- funcion_submodular() copiada sin cambios — mismo comportamiento.
- definir_clave() copiada sin cambios.
- La evaluacion final de las k partes usa bipartir() x k + reconstruccion,
  igual que geometric_k.py.
- Un archivo parametrico para k in {3, 4, 5}.
- NO modifica q_nodes.py.
"""

import time
from typing import Union, List, Tuple, Optional

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_kparticion_q
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_K_ANALYSIS_TAG,
    QNODES_K_LABEL,
    QNODES_K_STRAREGY_TAG,
)
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion


class QNodesK(SIA):
    """
    Extension del algoritmo QNodes para k-particiones (k in {3, 4, 5}).

    Fundamento — Opcion B (criterio de parada modificado)
    -------------------------------------------------------
    El algoritmo Q original ejecuta (2n - 2) fases hasta colapsar
    todos los vertices en 2 super-nodos (biparticion).

    Para k partes se ejecutan (2n - k) fases, deteniendo cuando
    quedan exactamente (k-1) super-nodos. Al finalizar, vertices
    contiene k grupos que forman la k-particion.

    La funcion_submodular guia el crecimiento greedy de omega
    exactamente igual que en k=2 — no se altera.

    La evaluacion final de los k grupos usa bipartir() x k con
    reconstruccion en orden original de indices_ncubos, identica
    a geometric_k.py, garantizando la misma semantica de EMD.

    Parametro k
    -----------
    k in {3, 4, 5}. Para k=2 usar QNodes original.
    """

    def __init__(self, tpm: np.ndarray, k: int = 3):
        super().__init__(tpm)
        if k < 3 or k > 5:
            raise ValueError(f"k debe estar en {{3, 4, 5}}, se recibio k={k}")
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.k = k
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set = set()
        self.clave_submodular = [], []
        self.memoria_delta: dict = {}
        self.memoria_grupo_candidato: dict = {}
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.logger = SafeLogger(QNODES_K_STRAREGY_TAG)

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos
        )

        self.indices_alcance  = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)

        k_grupos = self._algorithm_k(vertices)
        perdida, dist_particion = self._evaluar_k_particion(k_grupos)

        key = tuple(tuple(_aplanar_grupo(g)) for g in k_grupos)
        self.memoria_grupo_candidato[key] = (perdida, dist_particion)

        partes_fmt = [_aplanar_grupo(g) for g in k_grupos]
        fmt_mip = fmt_kparticion_q(partes_fmt)

        return Solution(
            estrategia=QNODES_K_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ------------------------------------------------------------------
    # Algoritmo Q modificado — criterio de parada para k partes
    # ------------------------------------------------------------------

    @profile(context={TYPE_TAG: QNODES_K_ANALYSIS_TAG})
    def _algorithm_k(self, vertices: list) -> list:
        """
        Ejecuta el algoritmo Q con criterio de parada para k partes.

        En k=2 original: corre len(vertices)-1 fases → 1 super-nodo.
        Aqui: corre len(vertices)-(k-1) fases → k-1 super-nodos
        que junto con el nodo raiz forman k grupos.

        El ciclo interno (omega/delta) es identico al original.
        Solo cambia el numero de fases del bucle externo.
        """
        # Iterar hasta reducir el conjunto de vértices a k grupos
        if len(vertices) <= self.k:
            return vertices

        while len(vertices) > self.k:
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = list(vertices[1:])

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for _ in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip = 0

                for idx_d in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_delta = self.funcion_submodular(
                        deltas_ciclo[idx_d], omegas_ciclo
                    )
                    ganancia = emd_union - emd_delta

                    if ganancia < emd_local:
                        if emd_delta == INT_ZERO:
                            # Particion trivial: registrar en memoria.
                            # Para k=2 mantenemos el comportamiento original
                            # y retornamos; para k>2 no cortamos temprano.
                            clave = (
                                tuple(deltas_ciclo[idx_d])
                                if isinstance(deltas_ciclo[idx_d], list)
                                else (deltas_ciclo[idx_d],)
                            )
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_particion_candidata,
                            )
                            if self.k == 2:
                                return vertices
                        emd_local = ganancia
                        indice_mip = idx_d
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            # Colapsar el ultimo par en un super-nodo
            ultimo_omega = omegas_ciclo[LAST_IDX]
            ultimo_delta = deltas_ciclo[LAST_IDX]

            par = (
                [ultimo_omega] if isinstance(ultimo_omega, tuple) else ultimo_omega
            ) + (
                [ultimo_delta] if isinstance(ultimo_delta, tuple) else ultimo_delta
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par)
            vertices = omegas_ciclo

        # Cuando el bucle finaliza, `vertices` tiene tamaño == k
        return vertices

    # ------------------------------------------------------------------
    # Evaluacion final de las k partes
    # ------------------------------------------------------------------

    def _evaluar_k_particion(
        self, k_grupos: list
    ) -> Tuple[float, np.ndarray]:
        """
        Evalua la k-particion calculando la perdida EMD.

        Para cada grupo_i:
            futuros_i  = indices con tiempo == EFFECT
            presentes_i = indices con tiempo == ACTUAL
            dist_i = bipartir(futuros_i, presentes_i).distribucion_marginal()

        Reconstruye dist_particion en el orden original de indices_ncubos
        (igual que geometric_k.py) y calcula emd_efecto.
        """
        idxs = self.sia_subsistema.indices_ncubos
        n_futuros = len(idxs)

        # Mapa: indice_real_futuro -> (grupo_i, posicion_en_dist_i)
        futuro_a_grupo: dict = {}
        partes_arrays: List[Tuple[np.ndarray, np.ndarray]] = []

        for i, grupo in enumerate(k_grupos):
            nodos = _aplanar_grupo(grupo)
            futuros_i  = [idx for t, idx in nodos if t == EFFECT]
            presentes_i = [idx for t, idx in nodos if t == ACTUAL]

            for pos, f in enumerate(futuros_i):
                futuro_a_grupo[int(f)] = (i, pos)

            partes_arrays.append((
                np.array(futuros_i,  dtype=np.int8),
                np.array(presentes_i, dtype=np.int8),
            ))

        dists: List[Optional[np.ndarray]] = [None] * len(k_grupos)
        for i, (fut_arr, pres_arr) in enumerate(partes_arrays):
            if len(fut_arr) == 0:
                continue
            try:
                parte_sys = self.sia_subsistema.bipartir(fut_arr, pres_arr)
                dists[i] = parte_sys.distribucion_marginal()
            except Exception:
                dists[i] = None

        # Reconstruir en orden original
        dist_particion = np.zeros(n_futuros, dtype=np.float32)
        for l, f_real in enumerate(idxs):
            if int(f_real) not in futuro_a_grupo:
                continue
            gi, pos_i = futuro_a_grupo[int(f_real)]
            if dists[gi] is not None and pos_i < len(dists[gi]):
                dist_particion[l] = dists[gi][pos_i]

        perdida = emd_efecto(dist_particion, self.sia_dists_marginales)
        return float(perdida), dist_particion

    # ------------------------------------------------------------------
    # funcion_submodular — copiada sin cambios de q_nodes.py
    # ------------------------------------------------------------------

    def funcion_submodular(
        self,
        deltas: Union[tuple, list],
        omegas: list,
    ):
        """
        Copiada sin modificaciones de QNodes.funcion_submodular().
        Evalua el impacto de combinar delta con omega usando EMD.
        """
        self.clave_submodular = [], []

        clave_delta_actual, clave_delta_efecto = self.definir_clave(deltas)
        clave_delta = tuple(clave_delta_actual), tuple(clave_delta_efecto)

        idxs_alcance_delta  = self.clave_submodular[EFFECT]
        dims_mecanismo_delta = self.clave_submodular[ACTUAL]

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta,  dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = emd_delta, vector_delta_marginal
        else:
            emd_delta, vector_delta_marginal = self.memoria_delta[clave_delta]

        for omega in omegas:
            self.definir_clave(omega)

        idxs_alcance_union  = self.clave_submodular[EFFECT]
        dims_mecanismo_union = self.clave_submodular[ACTUAL]

        particion_union = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_union,  dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)

        return emd_union, emd_delta, vector_delta_marginal

    # ------------------------------------------------------------------
    # definir_clave — copiada sin cambios de q_nodes.py
    # ------------------------------------------------------------------

    def definir_clave(
        self,
        conjunto: Union[tuple, list],
    ):
        """Copiada sin modificaciones de QNodes.definir_clave()."""
        if isinstance(conjunto, tuple) and len(conjunto) == 2 and isinstance(conjunto[0], (int, np.integer)):
            tiempo, indice = conjunto
            self.clave_submodular[tiempo].append(indice)
        else:
            for item in conjunto:
                self.definir_clave(item)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()
        return self.clave_submodular

    def nodes_complement(self, nodes):
        return list(set(self.vertices) - set(nodes))


# ------------------------------------------------------------------
# Utilidad: aplanar grupos anidados a lista de (tiempo, indice)
# ------------------------------------------------------------------

def _aplanar_grupo(grupo) -> List[Tuple[int, int]]:
    """
    Aplana recursivamente un super-nodo (que puede ser lista de listas
    de tuplas) a una lista plana de (tiempo, indice).
    """
    resultado = []
    if isinstance(grupo, tuple) and len(grupo) == 2 and isinstance(grupo[0], (int, np.integer)):
        resultado.append(grupo)
    elif isinstance(grupo, (list, tuple)):
        for item in grupo:
            resultado.extend(_aplanar_grupo(item))
    return resultado
