import time
import numpy as np
from itertools import product as iterproduct
from typing import List, Dict, Tuple, Optional

from src.controllers.manager import Manager
from src.funcs.base import ABECEDARY, emd_efecto
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import NET_LABEL, ACTUAL, EFECTO, TYPE_TAG
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)


# ---------------------------------------------------------------------------
# Helpers de agrupamiento basados en la tabla de costos T
# ---------------------------------------------------------------------------


def _particionar_variables_en_k(n_vars: int, k: int):
    """
    Genera todas las formas de asignar `n_vars` variables a `k` grupos,
    con la restriccion de que cada grupo tenga al menos 1 elemento.

    Devuelve listas de asignaciones: cada asignacion es un array de longitud
    n_vars donde asignacion[i] = grupo al que pertenece la variable i (0..k-1).

    Para n_vars grandes esto puede ser costoso; se aplica un umbral.
    """
    if k == 1:
        yield [0] * n_vars
        return
    if k > n_vars:
        return

    # Numeros de Stirling de segundo tipo serian el conteo exacto, pero
    # generamos directamente con producto cartesiano y filtramos.
    # Para eficiencia, si n_vars * k > 20 usamos heuristica greedy en su lugar.
    if n_vars * k <= 20:
        for asig in iterproduct(range(k), repeat=n_vars):
            if len(set(asig)) == k:
                yield list(asig)
    else:
        # senal de que el llamador debe usar heuristica
        return


def _asignacion_a_grupos(asig: List[int], k: int) -> List[List[int]]:
    """Convierte array de asignaciones en lista de k grupos de indices."""
    grupos = [[] for _ in range(k)]
    for idx, g in enumerate(asig):
        grupos[g].append(idx)
    return grupos


def _costo_total_biparticion_grupos(
    tabla: Dict, estado_inicial: tuple, estado_final: tuple,
    grupos_futuros: List[List[int]], n_vars: int
) -> float:
    """
    Suma los costos de transicion de las variables asignadas a cada grupo
    respecto al estado inicial -> final, usando la tabla T ya calculada.

    Un buen agrupamiento tendra costos bajos DENTRO de cada grupo y altos
    ENTRE grupos distintos, reflejando independencia causal.
    """
    key = (estado_inicial, estado_final)
    if key not in tabla:
        return float("inf")
    costos = tabla[key]

    total = 0.0
    for grupo in grupos_futuros:
        # Penalizacion intra-grupo: queremos que variables del mismo grupo
        # tengan costos similares (baja discrepancia entre si indica
        # dependencia causal -> deben estar juntas).
        vals = [costos[i] for i in grupo if i < len(costos)]
        if len(vals) > 1:
            # Varianza intra-grupo: bajo = variables similares = buen grupo
            arr = np.array(vals, dtype=float)
            total += float(np.var(arr))
        # Bono si costo es cero (independencia perfecta detectada)
        total -= sum(1.0 for v in vals if v == 0.0) * 0.01
    return total


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------


class KPartitionGeometricSIA(SIA):
    """
    Extension geometrica de GeometricSIA para k-particiones.

    Idea central
    ------------
    La tabla de costos T (heredada del enfoque de biparticion) mide la
    "inercia causal" entre pares de estados para cada variable futura.
    Para k > 2, agrupamos las variables (presentes y futuras) en k partes
    usando esa tabla como guia, minimizando la varianza intra-grupo (variables
    causalmente dependientes -> mismo grupo) y maximizando la discrepancia
    inter-grupo (variables causalmente independientes -> grupos distintos).

    Despues de identificar candidatos geometricamente, evaluamos cada
    candidato con la EMD real (igual que en el caso k=2) y devolvemos
    la k-particion con menor perdida.

    Parametro k
    -----------
    Se puede pasar `k` al constructor. Por defecto k=2 (compatible con
    GeometricSIA). Para k>2, si el numero de variables es pequeno se hace
    busqueda exacta; si es grande se usa heuristica greedy basada en T.
    """

    def __init__(self, gestor: Manager, k: int = 2):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.k = k
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)

        # Tabla de costos: (estado_i, estado_j) -> [costo_var0, costo_var1, ...]
        self.tabla_transiciones: Dict[Tuple, List[float]] = {}
        self.vertices: set
        self._flat_data: List[np.ndarray] = []
        self.memoria_particiones: Dict = {}

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )
        self.vertices = set(presente + futuro)

        # Aplanar ncubos para acceso O(1) por indice de estado
        self._flat_data = [
            ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos
        ]

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        mip_key = self._find_k_mip()

        fmt_mip = fmt_biparte_q(
            list(mip_key), self.nodes_complement(list(mip_key))
        )
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=self.memoria_particiones[mip_key][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip_key][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ------------------------------------------------------------------
    # Construccion de la tabla de costos (igual a GeometricSIA)
    # ------------------------------------------------------------------

    def _construir_tabla(self):
        """Construye la tabla de costos T mediante BFS nivel a nivel."""
        caminos: Dict[int, List[List[int]]] = {
            0: [self.estado_inicial.tolist()]
        }
        ini_tuple = tuple(caminos[0][0])
        self.tabla_transiciones[(ini_tuple, ini_tuple)] = [
            0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))
        ]
        n = len(self.estado_inicial)
        for nivel in range(1, n + 1):
            visitados: set = set()
            caminos[nivel] = []
            for estado_ant in caminos[nivel - 1]:
                estado_act = np.array(estado_ant)
                for i in range(n):
                    if estado_act[i] != self.estado_final[i]:
                        nuevo = estado_act.copy()
                        nuevo[i] = self.estado_final[i]
                        t = tuple(nuevo)
                        if t not in visitados:
                            caminos[nivel].append(nuevo.tolist())
                            self._calcular_costo(
                                caminos[0][0], nuevo.tolist()
                            )
                            visitados.add(t)
        return caminos

    def _calcular_costo(self, estado_ini: list, estado_fin: list):
        """Calcula recursivamente t(i,j) con factor gamma = 2^{-d_H}."""
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
    # Identificacion de candidatos para k-particiones
    # ------------------------------------------------------------------

    def _candidatos_geometricos(self, k: int) -> List[List[List[int]]]:
        """
        Genera candidatos de k-particion usando la tabla T.

        Cada candidato es una lista de k grupos, donde cada grupo contiene
        indices de variables futuras (y opcionalmente presentes).

        Estrategia
        ----------
        1. Obtenemos el vector de costos del estado_inicial -> estado_final.
        2. Ordenamos variables por costo (menor costo = mayor dependencia causal).
        3. Asignamos variables a k grupos por ranking: las de costo similar
           van juntas (agrupamiento greedy por quantiles de costo).
        4. Completamos con presentes: todos los presentes van en todos los grupos
           (como en la biparticion original) o los distribuimos segun la misma
           logica de costos si hay suficientes presentes.

        Para sistemas pequenos (n_vars * k <= 20) probamos todas las asignaciones.
        """
        key_global = (tuple(self.estado_inicial), tuple(self.estado_final))
        costos = self.tabla_transiciones.get(key_global, None)
        if costos is None:
            return []

        n_vars = len(costos)
        n_presentes = len(self.estado_inicial)

        candidatos = []

        # ---- Caso exacto (sistemas pequenos) ----
        if n_vars * k <= 20:
            asignaciones_vistas = set()
            for asig in _particionar_variables_en_k(n_vars, k):
                # Canonicalizar para no repetir permutaciones de grupos
                canon = self._canonicalizar(asig, k)
                if canon in asignaciones_vistas:
                    continue
                asignaciones_vistas.add(canon)
                grupos_fut = _asignacion_a_grupos(asig, k)
                # Presentes: todos al primer grupo (como en biparticion base)
                grupos_pres = [list(range(n_presentes))] + [[] for _ in range(k - 1)]
                candidatos.append([grupos_pres[i] + grupos_fut[i] for i in range(k)])
            return candidatos

        # ---- Caso heuristico greedy (sistemas grandes) ----
        return self._candidatos_greedy(costos, n_vars, n_presentes, k)

    def _candidatos_greedy(
        self,
        costos: List[float],
        n_vars: int,
        n_presentes: int,
        k: int,
    ) -> List[List[List[int]]]:
        """
        Genera candidatos por agrupamiento greedy basado en costos T.

        Estrategia 1 - Quantiles:
            Divide variables ordenadas por costo en k franjas iguales.

        Estrategia 2 - Complemento:
            Para cada par (i, j) de variables con bajo costo mutuo,
            genera una biparticion que las separa (igual que GeometricSIA base)
            y extiende a k grupos con las restantes.

        Estrategia 3 - Espejo:
            Usa la mitad del recorrido BFS (como en GeometricSIA) para generar
            grupos adicionales.
        """
        candidatos = []
        orden = np.argsort(costos)  # menor costo -> mayor dependencia

        # Estrategia 1: quantiles de costo
        grupos_q = [[] for _ in range(k)]
        for pos, var_idx in enumerate(orden):
            grupos_q[pos % k].append(int(var_idx))
        presentes_base = list(range(n_presentes))
        grupos_con_pres = [[presentes_base if g == 0 else []][0] + grupos_q[g]
                           for g in range(k)]
        candidatos.append(grupos_con_pres)

        # Estrategia 2: para k=2 compatible con GeometricSIA base
        if k == 2:
            for idx in range(n_vars):
                futuros_g1 = [i for i in range(n_vars) if i != idx]
                futuros_g2 = [idx]
                candidatos.append([
                    presentes_base + futuros_g1,
                    futuros_g2,
                ])

        # Estrategia 3: rotacion de grupos
        for shift in range(1, min(k, n_vars)):
            orden_rot = np.roll(orden, shift)
            grupos_r = [[] for _ in range(k)]
            for pos, var_idx in enumerate(orden_rot):
                grupos_r[pos % k].append(int(var_idx))
            cand = [[presentes_base if g == 0 else []][0] + grupos_r[g]
                    for g in range(k)]
            candidatos.append(cand)

        return candidatos

    @staticmethod
    def _canonicalizar(asig: List[int], k: int) -> tuple:
        """
        Normaliza una asignacion de grupos para detectar permutaciones repetidas.
        El primer grupo que aparece se renombra a 0, el segundo a 1, etc.
        """
        mapa = {}
        nueva = []
        contador = 0
        for g in asig:
            if g not in mapa:
                mapa[g] = contador
                contador += 1
            nueva.append(mapa[g])
        return tuple(nueva)

    # ------------------------------------------------------------------
    # Evaluacion real de candidatos (EMD)
    # ------------------------------------------------------------------

    def _evaluar_candidato(
        self,
        grupos_vars: List[List[int]],
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Para una k-particion dada como lista de grupos de indices de variables,
        calcula la EMD entre la distribucion marginal reconstruida y el original.

        Para k=2 reproduce exactamente el comportamiento de GeometricSIA.
        Para k>2, combina las distribuciones marginales de cada parte mediante
        el producto tensorial, igual a como se define delta en el documento.

        Devuelve (emd, dist_particionada) o None si el candidato es invalido.
        """
        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos

        # Separar indices presentes (ACTUAL) y futuros (EFECTO) por grupo
        # Convencion: los indices < len(dims) son presentes, >= son futuros
        n_pres = len(dims)

        futuros_por_grupo = []
        presentes_por_grupo = []
        for grupo in grupos_vars:
            fut = [v for v in grupo if v >= n_pres]
            pres = [v for v in grupo if v < n_pres]
            futuros_por_grupo.append(fut)
            presentes_por_grupo.append(pres)

        # Fallback: si no hay separacion explicita de presentes, asignamos
        # todos los presentes al primer grupo (compatible con base)
        if all(len(p) == 0 for p in presentes_por_grupo):
            presentes_por_grupo[0] = list(range(n_pres))

        # Calcular distribucion marginal para cada parte
        dists = []
        for g in range(len(grupos_vars)):
            idx_fut = np.array(
                [idxs[v - n_pres] for v in futuros_por_grupo[g]
                 if (v - n_pres) < len(idxs)],
                dtype=np.int8,
            )
            idx_pres = np.array(
                [dims[v] for v in presentes_por_grupo[g]
                 if v < len(dims)],
                dtype=np.int8,
            )
            if len(idx_fut) == 0:
                continue
            try:
                parte = self.sia_subsistema.bipartir(idx_fut, idx_pres)
                dists.append(parte.distribucion_marginal())
            except Exception:
                return None

        if not dists:
            return None

        # Para k=2: usar la primera dist directamente (igual que base)
        # Para k>2: producto tensorial de distribuciones
        if len(dists) == 1:
            dist_final = dists[0]
        else:
            dist_final = dists[0]
            for d in dists[1:]:
                dist_final = np.kron(dist_final, d)
            # Normalizar para que sea distribucion de probabilidad valida
            s = dist_final.sum()
            if s > 0:
                dist_final = dist_final / s

        emd = emd_efecto(dist_final, self.sia_dists_marginales)
        return emd, dist_final

    # ------------------------------------------------------------------
    # Algoritmo principal de k-MIP
    # ------------------------------------------------------------------

    def _find_k_mip(self):
        """
        Encuentra la k-particion de Minima Perdida de Informacion (k-MIP).

        Flujo:
        1. Construir tabla de costos T (BFS modificado).
        2. Generar candidatos geometricos de k grupos.
        3. Evaluar cada candidato con EMD real.
        4. Retornar la clave del minimo.
        """
        self.logger.critic(f"k-MIP geometrico iniciando (k={self.k}).")

        # 1. Construir tabla T
        self._construir_tabla()

        # 2. Generar candidatos
        candidatos = self._candidatos_geometricos(self.k)
        self.logger.critic(f"Candidatos generados: {len(candidatos)}")

        # 3. Evaluar y almacenar
        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos

        for grupos in candidatos:
            # Construir clave compatible con memoria_particiones
            key_parts = []
            for grupo in grupos:
                for var in grupo:
                    if var < len(dims):
                        key_parts.append((ACTUAL, dims[var]))
                    else:
                        adj = var - len(dims)
                        if adj < len(idxs):
                            key_parts.append((EFECTO, idxs[adj]))
            key = tuple(key_parts)

            if key in self.memoria_particiones:
                continue

            resultado = self._evaluar_candidato(grupos)
            if resultado is not None:
                self.memoria_particiones[key] = resultado

        if not self.memoria_particiones:
            # Fallback: biparticion trivial
            key_trivial = tuple(
                [(ACTUAL, d) for d in dims] + [(EFECTO, i) for i in idxs]
            )
            self.memoria_particiones[key_trivial] = (0.0, self.sia_dists_marginales)

        return min(
            self.memoria_particiones,
            key=lambda k: self.memoria_particiones[k][0],
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def nodes_complement(self, nodes: list):
        return list(set(self.vertices) - set(nodes))

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
