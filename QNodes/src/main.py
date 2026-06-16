from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.force import BruteForce
from src.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada"""

    # ABCD #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "1111111111"
    mecanismo =      "0111111111"
    
    #010101010101010101010
    #101010101010101010101

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()
    
    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    ### analizador_bf = BruteForce(mpt)

    ### Ejemplo de solución mediante módulo de Q-Nodes ###
    analizador_qn = QNodes(mpt)

    sia_cero = analizador_qn.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_cero)
