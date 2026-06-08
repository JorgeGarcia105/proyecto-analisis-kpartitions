import argparse
from src.models.base.application import aplicacion
from src.main import iniciar


def main():
    parser = argparse.ArgumentParser(description="Ejecutar GeometricSIA")
    parser.add_argument(
        "--k", type=int, default=2,
        help="Numero de particiones: 2 (biparticion, default) o 3 (triparticion)"
    )
    args = parser.parse_args()

    aplicacion.profiler_habilitado = True
    iniciar(k=args.k)


if __name__ == "__main__":
    main()
