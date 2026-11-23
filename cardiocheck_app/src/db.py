"""Persistencia de las encuestas que rellenan los usuarios de la app (SQLite).

No se guarda ningún dato identificativo: solo las respuestas al formulario,
la predicción y la fecha.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

COLUMNAS_ENCUESTA = {
    "fecha": "TEXT",
    "age": "REAL",
    "male": "INTEGER",
    "currentSmoker": "INTEGER",
    "cigsPerDay": "REAL",
    "BPMeds": "INTEGER",
    "prevalentStroke": "INTEGER",
    "prevalentHyp": "INTEGER",
    "diabetes": "INTEGER",
    "BMI": "REAL",
    "sysBP": "REAL",
    "diaBP": "REAL",
    "totChol": "REAL",
    "variante": "TEXT",
    "probabilidad": "REAL",
    "nivel": "TEXT",
}


def crear_tabla_encuestas(db_path: Path = config.DB_PATH) -> None:
    """Crea la tabla de encuestas si no existe."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    columnas = ", ".join(f"{nombre} {tipo}" for nombre, tipo in COLUMNAS_ENCUESTA.items())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {config.TABLA_ENCUESTAS} "
            f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {columnas})"
        )


def guardar_encuesta(
    datos: dict, variante: str, probabilidad: float, nivel: str, db_path: Path = config.DB_PATH
) -> int:
    """Guarda una encuesta con su predicción. Devuelve el id asignado."""
    crear_tabla_encuestas(db_path)
    registro = {col: datos.get(col) for col in COLUMNAS_ENCUESTA}
    registro.update(
        fecha=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        variante=variante,
        probabilidad=float(probabilidad),
        nivel=nivel,
    )
    nombres = ", ".join(registro)
    marcas = ", ".join("?" for _ in registro)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            f"INSERT INTO {config.TABLA_ENCUESTAS} ({nombres}) VALUES ({marcas})",
            list(registro.values()),
        )
        return int(cursor.lastrowid)


def leer_encuestas(db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Devuelve todas las encuestas guardadas (DataFrame vacío si no hay ninguna)."""
    crear_tabla_encuestas(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {config.TABLA_ENCUESTAS} ORDER BY id", conn)
