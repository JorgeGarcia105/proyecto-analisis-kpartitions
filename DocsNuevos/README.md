# DocsNuevos

Este directorio contiene mejoras y material de apoyo para el proyecto.

## Contenido

- `geometric_k.py`: extension de GeometricSIA para k-particiones.
- `benchmark.py`: compara GeometricSIA/KPartitionGeometricSIA vs PyPhi.
- `test_geometric_k.py`: tests unitarios de costos, particiones y metricas.
- `Ejemplos.xlsx`: Excel de ejemplo proporcionado por la profesora.

## Integracion en el proyecto

Las versiones integradas estan en:

- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/geometric_k.py`
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/benchmark.py`
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/tests/test_geometric_k.py`

## Uso rapido

Desde `GeoMIP/src/Method2_Dynamic_Programming_Reformulation`:

```bash
uv run python benchmark.py --excel ../../results/Pruebas_Metodo2.xlsx --k 2
```

```bash
uv run pytest tests/test_geometric_k.py -v --tb=short
```

## Excel de ejemplo

El archivo `Ejemplos.xlsx` se usa como referencia de formato de entrada.
Si el benchmark necesita leer columnas especificas, ajusta el Excel o el
script para alinear nombres como `caso`, `condicion`, `alcance`, `mecanismo`
 y `tpm_path`.
