# Proyecto-20261

Este repositorio contiene tres implementaciones principales para el analisis de MIP/IIT:

1. `QNodes` (base clasica, antes referida como Proyecto-2025A)
2. `GeoMIP/src/Method2_Dynamic_Programming_Reformulation`

## Requisitos

- Linux (probado en Ubuntu)
- Python 3.11+ (hay entornos locales con 3.12)
- `uv` instalado

Instalacion de `uv` (si no lo tienes):

```bash
pip install uv
```

## Estructura Rapida

- `QNodes/`: ejecucion directa de un caso de prueba (`exec.py`).
- `GeoMIP/src/Method1_GPU_Accelerated/`: procesamiento por lotes desde Excel.
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`: procesamiento por lotes desde Excel.
- `GeoMIP/data/samples/`: datasets TPM `N*.csv` usados por Method1/Method2.
- `GeoMIP/results/`: archivos Excel de entrada/salida para Method1/Method2.
- `DocsNuevos/`: mejoras y scripts auxiliares (k-particiones, benchmark, tests).

## 1) Ejecutar QNodes

### Dependencias

Desde `QNodes/`:

```bash
cd QNodes
uv sync
```

### Ejecucion

```bash
uv run exec.py
```

### Que hace

- Carga una red desde `QNodes/src/.samples/` (segun el estado inicial y pagina configurada).
- Ejecuta estrategia `BruteForce` desde `QNodes/src/main.py`.
- Imprime la solucion en consola.

### Ajustes comunes

Edita `QNodes/src/main.py`:

- `estado_inicial`
- `condiciones`
- `alcance`
- `mecanismo`

Si termina muy rapido, no necesariamente es error: puede ser un caso pequeno o corte temprano cuando `phi = 0`.

## 3) Ejecutar Method2_Dynamic_Programming_Reformulation

### Dependencias

Desde `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`:

```bash
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv sync
```

### Ejecucion

```bash
uv run exec.py
```

### Entrada por defecto

- Excel entrada: `GeoMIP/results/Pruebas_Metodo2.xlsx`
- Hoja usada actualmente: indice `8`
- Columna subsistema: `B`

### Salida por defecto

- Excel salida: `GeoMIP/results/resultados_Geometric.xlsx`

## 4) k-particiones y benchmark (DocsNuevos)

Se integro una extension de GeometricSIA para k-particiones y un benchmark
contra PyPhi.

Archivos clave:

- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/geometric_k.py`
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/benchmark.py`
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/tests/test_geometric_k.py`

### Ejecutar benchmark

Desde `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`:

```bash
uv run python benchmark.py --excel ../../results/Pruebas_Metodo2.xlsx --k 2
```

### Ejecutar tests

```bash
uv run pytest tests/test_geometric_k.py -v --tb=short
```

### Excel de ejemplo

- Archivo: `DocsNuevos/Ejemplos.xlsx`
- Se usa como referencia de formato para el benchmark.
