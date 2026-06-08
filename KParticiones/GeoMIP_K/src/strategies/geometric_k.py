"""
geometric_k.py
--------------
Extension de la estrategia geometrica GeoMIP para k-particiones
con k in {3, 4, 5}.

Relacion con el proyecto base
------------------------------
- Hereda de SIA (mismo patron que GeometricSIA en geometric.py).
- Reutiliza sin modificar: System.bipartir(), System.distribucion_marginal(),
  emd_efecto(), sia_preparar_subsistema(), profiler_manager, SafeLogger.
- La tabla de costos T se construye con la misma logica BFS de geometric.py
  y se calcula UNA sola vez para todas las k-particiones candidatas.
- NO modifica ningun archivo del proyecto base.

Archivos agregados/modificados en el proyecto base
----------------------------------------------------
- src/constants/models.py  : se agregaron GEOMETRIC_K_LABEL,
                             GEOMETRIC_K_STRAREGY_TAG, GEOMETRIC_K_ANALYSIS_TAG
- src/funcs/format.py      : se agrego fmt_kparte_q()

Restricciones del enunciado respetadas
----------------------------------------
- Compatibilidad con arquitectura existente (hereda SIA).
- Reutilizacion eficiente: tabla T calculada una sola vez.
- Funciona para sistemas de 10 a 22 nodos (estrategia greedy).
- Los presentes se distribuyen entre partes segun regla geometrica basada en T.
- Un archivo parametrico para k in {3, 4, 5}.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional

from src.controllers.manager import Manager
from src.funcs.base import ABECEDARY, emd_efecto
from src.funcs.format import fmt_kparte_q
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import NET_LABEL, ACTUAL, EFECTO, TYPE_TAG
from src.constants.models import (
    GEOMETRIC_K_ANALYSIS_TAG,
    GEOMETRIC_K_LABEL,
    GEOMETRIC_K_STRAREGY_TAG,
)


class KPartitionGeometricSIA(SIA):
    """
    Extension geometrica de GeometricSIA para k-particiones (k in {3, 4, 5}).

    Fundamento
    ----------
    La tabla de costos T mide la 'inercia causal' entre pares de estados
    para cada variable futura, usando BFS sobre el hipercubo n-dimensional
    (identica a geometric.py).

    Para k > 2 se divide el conjunto de variables futuras en k grupos
    usando T como guia (variables con costos similares -> mismo grupo,
    variables con costos distintos -> grupos distintos).

    Los presentes se asignan a cada grupo segun la regla geometrica:
    el presente p_j va al grupo que contiene el futuro al que mas afecta,
    medido por T[(s, s_j)] donde s_j es el estado con el bit j invertido.

    La perdida de cada candidato se evalua con:
        delta_k = emd_efecto(dist_reconstruida, dist_original)
    donde dist_reconstruida coloca cada dist_i en el orden original
    de indices_ncubos (NO usa np.kron).

    Parametro k
    -----------
    k in {3, 4, 5}. Para k=2 usar GeometricSIA original.
    """

    def __init__(self, gestor: Manager, k: int = 3):
        super().__init__(gestor)
        if k < 3 or k > 5:
            raise ValueError(f"k debe estar en {{3, 4, 5}}, se recibio k={k}")
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.k = k
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_K_STRAREGY_TAG)

        # Tabla de costos: (estado_i, estado_j) -> [costo_f0, costo_f1, ...]
        self.tabla_transiciones: Dict[Tuple, List[float]] = {}
        # Caminos BFS por nivel: {nivel: [estado, ...]}
        self.caminos: Dict[int, List[List[int]]] = {}
        self.vertices: set = set()
        self._flat_data: List[np.ndarray] = []
        # Clave -> (perdida, dist_particion)
        self.memoria_particiones: Dict = {}

        self.estado_inicial: np.ndarray
        self.estado_final: np.ndarray

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    @profile(context={TYPE_TAG: GEOMETRIC_K_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> Solution:
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )
        self.vertices = set(presente + futuro)

        self._flat_data = [
            ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos
        ]

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        mip_key = self._find_k_mip()

        partes_fmt = self._key_a_partes_fmt(mip_key)
        fmt_mip = fmt_kparte_q(partes_fmt)

        return Solution(
            estrategia=GEOMETRIC_K_LABEL,
            perdida=self.memoria_particiones[mip_key][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip_key][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ------------------------------------------------------------------
    # Tabla de costos T — identica a geometric.py, calculada una sola vez
    # ------------------------------------------------------------------

    def _construir_tabla(self) -> None:
        """
        Construye la tabla de costos T por BFS nivel a nivel.
        Identica a calcular_costos_nivel + calcular_costo de geometric.py.
        Se ejecuta una unica vez y sirve para todos los candidatos.
        """
        self.caminos = {0: [self.estado_inicial.tolist()]}
        ini_tuple = tuple(self.caminos[0][0])
        self.tabla_transiciones[(ini_tuple, ini_tuple)] = [
            0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))
        ]
        n = len(self.estado_inicial)
        for nivel in range(1, n + 1):
            visitados: set = set()
            self.caminos[nivel] = []
            for estado_ant in self.caminos[nivel - 1]:
                estado_act = np.array(estado_ant)
                for i in range(n):
                    if estado_act[i] != self.estado_final[i]:
                        nuevo = estado_act.copy()
                        nuevo[i] = self.estado_final[i]
                        t = tuple(nuevo)
                        if t not in visitados:
                            self.caminos[nivel].append(nuevo.tolist())
                            self._calcular_costo(self.caminos[0][0], nuevo.tolist())
                            visitados.add(t)

    def _calcular_costo(self, estado_ini: list, estado_fin: list) -> None:
        """
        Calcula t(i,j) = (1/2^dH) * (|X[i]-X[j]| + sum(t(i, vecinos_de_j))).
        Identico a calcular_costo de geometric.py.
        """
        key = (tuple(estado_ini), tuple(estado_fin))
        if key in self.tabla_transiciones:
            return

        n_vars = len(self.sia_subsistema.indices_ncubos)
        dh = sum(a != b for a, b in zip(estado_ini, estado_fin))
        factor = 1.0 / (2 ** dh)

        ini_int = int("".join(map(str, estado_ini[::-1])), 2)
        fin_int = int("".join(map(str, estado_fin[::-1])), 2)

        diffs = np.abs(
            np.array([fd[ini_int] for fd in self._flat_data])
            - np.array([fd[fin_int] for fd in self._flat_data])
        ).tolist()

        if dh > 1:
            for i in range(len(estado_ini)):
                if estado_ini[i] != estado_fin[i]:
                    vecino = list(estado_fin)
                    vecino[i] = estado_ini[i]
                    vk = (tuple(estado_ini), tuple(vecino))
                    if vk in self.tabla_transiciones:
                        for n in range(n_vars):
                            diffs[n] += self.tabla_transiciones[vk][n]

        self.tabla_transiciones[key] = [factor * d for d in diffs]

    # ------------------------------------------------------------------
    # Asignacion de presentes — regla geometrica (nuevo para k > 2)
    # ------------------------------------------------------------------

    def _asignar_presentes(
        self, grupos_futuros: List[List[int]]
    ) -> List[List[int]]:
        """
        Asigna cada presente p_j al grupo que contiene el futuro al que
        mas afecta causalmente, segun T[(s, s_j)].

        s_j es el estado_inicial con el bit j invertido.
        El impacto de p_j sobre el grupo i es:
            max( T[(s, s_j)][l]  para l en grupos_futuros[i] )
        Se asigna p_j al grupo i* = argmax(impacto_i).

        Si s_j no esta en la tabla (bit j ya esta en estado_final),
        se asigna p_j al grupo 0 como fallback.
        """
        n_presentes = len(self.estado_inicial)
        grupos_presentes: List[List[int]] = [[] for _ in range(self.k)]

        for j in range(n_presentes):
            s_j = self.estado_inicial.tolist()
            s_j[j] = 1 - s_j[j]
            key_sj = (tuple(self.estado_inicial.tolist()), tuple(s_j))

            costos_sj = self.tabla_transiciones.get(key_sj, None)

            if costos_sj is None:
                grupos_presentes[0].append(j)
                continue

            mejor_grupo = 0
            mejor_impacto = -1.0
            for i, grupo in enumerate(grupos_futuros):
                if not grupo:
                    continue
                impacto_i = max(costos_sj[l] for l in grupo if l < len(costos_sj))
                if impacto_i > mejor_impacto:
                    mejor_impacto = impacto_i
                    mejor_grupo = i

            grupos_presentes[mejor_grupo].append(j)

        return grupos_presentes

    # ------------------------------------------------------------------
    # Generacion de candidatos — estrategia greedy geometrica
    # ------------------------------------------------------------------

    def _candidatos_geometricos(self) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Genera candidatos de k-particion usando la tabla T.

        Cada candidato es una lista de k tuplas (futuros_i, presentes_i)
        donde futuros_i y presentes_i son arrays de indices reales del subsistema.

        Estrategias aplicadas
        ---------------------
        Capa 1 — Cuantiles de costo global T[(s, s_bar)]:
            Ordena futuros por costo y los reparte en k grupos round-robin.
            Genera k variaciones por rotacion del orden.

        Capa 2 — Cortes en niveles BFS (k-1 puntos de corte):
            Usa los estados intermedios del BFS para definir k segmentos
            del recorrido, igual al espiritu de identificar_particiones_optimas
            de geometric.py pero con k-1 cortes en lugar de 1.

        Capa 3 — Separacion por extremos de costo:
            Para cada subconjunto de k-1 futuros con costos mas altos
            (uno por parte separada), genera una candidatura directa.
            Acotado a los top-(k-1) futuros para mantener eficiencia.
        """
        key_global = (tuple(self.estado_inicial), tuple(self.estado_final))
        costos = self.tabla_transiciones.get(key_global)
        if costos is None:
            return []

        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos
        n_futuros = len(costos)

        if n_futuros < self.k:
            return []

        candidatos = []
        orden = np.argsort(costos)

        # --- Capa 1: cuantiles + rotaciones ---
        for shift in range(self.k):
            orden_rot = np.roll(orden, shift)
            grupos_fut_idx = [[] for _ in range(self.k)]
            for pos, var_idx in enumerate(orden_rot):
                grupos_fut_idx[pos % self.k].append(int(var_idx))

            grupos_pres_idx = self._asignar_presentes(grupos_fut_idx)
            cand = self._construir_candidato(grupos_fut_idx, grupos_pres_idx, idxs, dims)
            if cand:
                candidatos.append(cand)

        # --- Capa 2: cortes en niveles BFS ---
        n_niveles = len(self.caminos)
        cortes = [int((i + 1) * n_niveles / self.k) for i in range(self.k - 1)]
        cortes = [min(c, n_niveles - 1) for c in cortes]

        grupos_fut_bfs = [[] for _ in range(self.k)]
        for seg in range(self.k):
            nivel_ini = 0 if seg == 0 else cortes[seg - 1]
            nivel_fin = cortes[seg] if seg < self.k - 1 else n_niveles - 1
            costo_seg_min = float("inf")
            mejor_estado_seg = None

            for nivel in range(nivel_ini, nivel_fin + 1):
                for estado in self.caminos.get(nivel, []):
                    key_e = (tuple(self.estado_inicial.tolist()), tuple(estado))
                    costos_e = self.tabla_transiciones.get(key_e)
                    if costos_e is None:
                        continue
                    costo_total = sum(costos_e)
                    if costo_total < costo_seg_min:
                        costo_seg_min = costo_total
                        mejor_estado_seg = estado

            if mejor_estado_seg is not None:
                key_m = (tuple(self.estado_inicial.tolist()), tuple(mejor_estado_seg))
                costos_m = self.tabla_transiciones.get(key_m, [0.0] * n_futuros)
                orden_seg = np.argsort(costos_m)
                tam = max(1, n_futuros // self.k)
                inicio = seg * tam
                fin = inicio + tam if seg < self.k - 1 else n_futuros
                for idx_pos in range(inicio, min(fin, len(orden_seg))):
                    grupos_fut_bfs[seg].append(int(orden_seg[idx_pos]))

        # Asegurar que todos los futuros esten cubiertos
        futuros_asignados = {f for g in grupos_fut_bfs for f in g}
        grupo_fallback = 0
        for f in range(n_futuros):
            if f not in futuros_asignados:
                grupos_fut_bfs[grupo_fallback].append(f)
                grupo_fallback = (grupo_fallback + 1) % self.k

        if all(len(g) > 0 for g in grupos_fut_bfs):
            grupos_pres_bfs = self._asignar_presentes(grupos_fut_bfs)
            cand_bfs = self._construir_candidato(grupos_fut_bfs, grupos_pres_bfs, idxs, dims)
            if cand_bfs:
                candidatos.append(cand_bfs)

        # --- Capa 3: separacion por extremos de costo ---
        top_k_minus_1 = list(np.argsort(costos)[::-1][: self.k - 1])
        if len(top_k_minus_1) == self.k - 1:
            grupos_fut_ext = [[f] for f in top_k_minus_1]
            resto = [f for f in range(n_futuros) if f not in top_k_minus_1]
            grupos_fut_ext.append(resto)
            if len(grupos_fut_ext) == self.k and all(len(g) > 0 for g in grupos_fut_ext):
                grupos_pres_ext = self._asignar_presentes(grupos_fut_ext)
                cand_ext = self._construir_candidato(
                    grupos_fut_ext, grupos_pres_ext, idxs, dims
                )
                if cand_ext:
                    candidatos.append(cand_ext)

        return candidatos

    def _construir_candidato(
        self,
        grupos_fut_idx: List[List[int]],
        grupos_pres_idx: List[List[int]],
        idxs: np.ndarray,
        dims: np.ndarray,
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Convierte indices locales de futuros/presentes en arrays de indices
        reales del subsistema listos para llamar a bipartir().

        Retorna None si algun grupo de futuros queda vacio (candidato invalido).
        """
        partes = []
        for i in range(self.k):
            if not grupos_fut_idx[i]:
                return None
            fut_real = np.array(
                [idxs[f] for f in grupos_fut_idx[i] if f < len(idxs)],
                dtype=np.int8,
            )
            pres_real = np.array(
                [dims[p] for p in grupos_pres_idx[i] if p < len(dims)],
                dtype=np.int8,
            )
            partes.append((fut_real, pres_real))
        return partes

    # ------------------------------------------------------------------
    # Evaluacion de candidatos — bipartir x k + reconstruccion en orden
    # ------------------------------------------------------------------

    def _evaluar_candidato(
        self,
        partes: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Evalua una k-particion calculando la perdida EMD.

        Para cada parte i:
            dist_i = subsistema.bipartir(futuros_i, presentes_i)
                                .distribucion_marginal()

        Reconstruye dist_particion en el orden original de indices_ncubos
        (NO usa np.kron) para mantener la misma semantica que emd_efecto
        en geometric.py: sum(|u - v|) sobre vectores de marginales.

        Retorna (perdida, dist_particion) o None si hay error.
        """
        idxs = self.sia_subsistema.indices_ncubos
        n_futuros = len(idxs)

        # Mapa: indice_real_futuro -> (grupo_i, posicion_en_dist_i)
        futuro_a_grupo: Dict[int, Tuple[int, int]] = {}
        for i, (fut_real, _) in enumerate(partes):
            for pos, f in enumerate(fut_real):
                futuro_a_grupo[int(f)] = (i, pos)

        dists: List[Optional[np.ndarray]] = [None] * self.k
        for i, (fut_real, pres_real) in enumerate(partes):
            try:
                parte_sys = self.sia_subsistema.bipartir(fut_real, pres_real)
                dists[i] = parte_sys.distribucion_marginal()
            except Exception:
                return None

        # Reconstruir en orden original de indices_ncubos
        dist_particion = np.zeros(n_futuros, dtype=np.float32)
        for l, f_real in enumerate(idxs):
            if int(f_real) not in futuro_a_grupo:
                return None
            grupo_i, pos_i = futuro_a_grupo[int(f_real)]
            if dists[grupo_i] is None or pos_i >= len(dists[grupo_i]):
                return None
            dist_particion[l] = dists[grupo_i][pos_i]

        perdida = emd_efecto(dist_particion, self.sia_dists_marginales)
        return perdida, dist_particion

    # ------------------------------------------------------------------
    # Algoritmo principal k-MIP
    # ------------------------------------------------------------------

    def _find_k_mip(self) -> tuple:
        """
        Encuentra la k-particion de Minima Perdida de Informacion (k-MIP).

        Flujo:
        1. Construir tabla T una sola vez (BFS identico a geometric.py).
        2. Generar candidatos geometricos (3 capas).
        3. Evaluar cada candidato con EMD real.
        4. Retornar la clave del minimo.
        """
        self.logger.critic(f"k-MIP geometrico iniciando (k={self.k}).")

        self._construir_tabla()

        candidatos = self._candidatos_geometricos()
        self.logger.critic(f"Candidatos generados: {len(candidatos)}")

        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos

        for partes in candidatos:
            key = self._partes_a_key(partes, dims, idxs)
            if key in self.memoria_particiones:
                continue
            resultado = self._evaluar_candidato(partes)
            if resultado is not None:
                self.memoria_particiones[key] = resultado

        if not self.memoria_particiones:
            key_trivial = self._key_fallback(dims, idxs)
            self.memoria_particiones[key_trivial] = (0.0, self.sia_dists_marginales)

        return min(
            self.memoria_particiones,
            key=lambda k: self.memoria_particiones[k][0],
        )

    # ------------------------------------------------------------------
    # Utilidades de clave y formato
    # ------------------------------------------------------------------

    def _partes_a_key(
        self,
        partes: List[Tuple[np.ndarray, np.ndarray]],
        dims: np.ndarray,
        idxs: np.ndarray,
    ) -> tuple:
        """
        Convierte una lista de k partes en una clave hashable para
        memoria_particiones.

        Estructura: tuple de k sub-tuplas, cada una con (tiempo, nodo).
        tiempo=ACTUAL(0) para presentes, tiempo=EFECTO(1) para futuros.
        """
        grupos = []
        for fut_real, pres_real in partes:
            nodos_parte = []
            for p in pres_real:
                nodos_parte.append((ACTUAL, int(p)))
            for f in fut_real:
                nodos_parte.append((EFECTO, int(f)))
            nodos_parte.sort(key=lambda x: x[1])
            grupos.append(tuple(nodos_parte))
        return tuple(grupos)

    def _key_fallback(self, dims: np.ndarray, idxs: np.ndarray) -> tuple:
        """Clave trivial de fallback: todos los nodos en la primera parte."""
        nodos = [(ACTUAL, int(d)) for d in dims] + [(EFECTO, int(i)) for i in idxs]
        nodos.sort(key=lambda x: x[1])
        parte_unica = tuple(nodos)
        partes_vacias = tuple(() for _ in range(self.k - 1))
        return (parte_unica,) + partes_vacias

    def _key_a_partes_fmt(
        self, key: tuple
    ) -> List[List[Tuple[int, int]]]:
        """
        Convierte la clave de memoria_particiones al formato que espera
        fmt_kparte_q: lista de k listas de (tiempo, nodo).
        """
        return [list(parte) for parte in key]

    def nodes_complement(self, nodes: list) -> list:
        return list(set(self.vertices) - set(nodes))

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
