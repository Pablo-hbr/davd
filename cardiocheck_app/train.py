"""Pipeline de entrenamiento de CardioCheck.

Uso (desde la carpeta del proyecto, con el entorno virtual activo):

    python train.py            # ETL + comparación de modelos + entrenamiento
    python train.py --rapido   # sin la comparación por validación cruzada

Pasos: CSV -> limpieza -> SQLite -> lectura desde la BD -> evaluación en test ->
modelos finales (joblib) + métricas (JSON) que consume la app.
"""

from __future__ import annotations

import argparse
import sys
import time

from src import config, etl, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de CardioCheck")
    parser.add_argument("--rapido", action="store_true", help="omite la comparación LR / RF / GB")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos legibles en la consola de Windows
    inicio = time.time()

    print(f"1) ETL: {config.RAW_CSV.name} -> {config.DB_PATH}")
    datos = etl.ejecutar_etl()
    print(f"   {len(datos)} filas cargadas en la tabla '{config.TABLA_DATOS}'")
    print(f"   Prevalencia de evento coronario a 10 años: {datos[config.TARGET].mean():.1%}")

    print("2) Entrenamiento y evaluación")
    modelos, metricas = model.entrenar(datos, comparar=not args.rapido)

    if "comparacion_modelos" in metricas:
        print("   Comparación (validación cruzada sobre train, todas las variables):")
        for fila in metricas["comparacion_modelos"]:
            print(f"     {fila['modelo']:22s} AUC = {fila['auc_media']:.3f} ± {fila['auc_std']:.3f}")

    print("   Regresión logística por variante (test):")
    for variante, m in metricas["variantes"].items():
        mod = m["umbrales"]["moderado"]
        print(
            f"     {variante:11s} AUC test = {m['auc']:.3f} | AUC CV = {m['auc_cv_media']:.3f} ± {m['auc_cv_std']:.3f}"
            f" | Brier = {m['brier']:.4f} "
            f"(baseline {m['brier_baseline']:.4f}) | recall con umbral {mod['umbral']:.0%} = {mod['recall']:.2f}"
        )

    model.guardar(modelos, metricas)
    print(f"3) Guardado: {config.MODELOS_PATH.name} y {config.METRICAS_PATH.name} ({time.time() - inicio:.1f} s)")


if __name__ == "__main__":
    main()
