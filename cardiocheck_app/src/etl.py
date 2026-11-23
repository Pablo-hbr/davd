"""ETL: lectura del CSV de Framingham, limpieza y carga en la base de datos SQLite.

Flujo: CSV crudo -> limpiar() -> tabla `framingham` en SQLite -> el entrenamiento
lee siempre desde la base de datos.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src import config


def leer_csv(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Lee el CSV original de Framingham.

    Parameters:
        path: ruta al fichero CSV.
    Output:
        DataFrame con los datos crudos.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de datos: {path}")
    return pd.read_csv(path)


def diagnostico(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por columna: tipo, nulos, % de nulos, nº de valores distintos, mínimo y máximo."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "pct_nulos": (100 * df.isna().mean()).round(2),
            "valores_distintos": df.nunique(),
            "min": df.min(numeric_only=True),
            "max": df.max(numeric_only=True),
        }
    )


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza con sentido clínico.

    - cigsPerDay nulo: en el dataset solo ocurre en fumadores, se imputa con la
      mediana de los fumadores (los no fumadores tienen 0).
    - BPMeds nulo en personas sin hipertensión: 0 (nadie toma medicación para la
      tensión sin ser hipertenso en estos datos). El resto de nulos se imputan
      dentro del pipeline del modelo, con estadísticos del train.
    - Se descartan education, glucose y heartRate: la app no los pregunta.
    """
    out = df.copy()

    mediana_cigs = out.loc[out["currentSmoker"] == 1, "cigsPerDay"].median()
    out.loc[out["currentSmoker"] == 0, "cigsPerDay"] = 0
    out.loc[(out["currentSmoker"] == 1) & out["cigsPerDay"].isna(), "cigsPerDay"] = mediana_cigs

    out.loc[(out["prevalentHyp"] == 0) & out["BPMeds"].isna(), "BPMeds"] = 0

    columnas = config.VARIANTES["completo"] + [config.TARGET]
    return out[columnas].reset_index(drop=True)


def cargar_en_bd(df: pd.DataFrame, db_path: Path = config.DB_PATH, tabla: str = config.TABLA_DATOS) -> None:
    """Guarda el DataFrame en una tabla de SQLite (la reemplaza si ya existe)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(tabla, conn, if_exists="replace", index=False)


def leer_de_bd(db_path: Path = config.DB_PATH, tabla: str = config.TABLA_DATOS) -> pd.DataFrame:
    """Lee una tabla completa de la base de datos."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {tabla}", conn)


def ejecutar_etl(csv_path: Path = config.RAW_CSV, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """CSV -> limpieza -> base de datos. Devuelve los datos limpios tal y como quedan en la BD."""
    limpio = limpiar(leer_csv(csv_path))
    cargar_en_bd(limpio, db_path)
    return leer_de_bd(db_path)
