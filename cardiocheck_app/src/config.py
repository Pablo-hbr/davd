"""Configuración central de CardioCheck: rutas, variables del modelo y umbrales.

Todo lo que se comparte entre el ETL, el entrenamiento y la app vive aquí para
que no haya "números mágicos" repartidos por el código.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas (siempre relativas al proyecto, nunca absolutas del autor) ---------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = BASE_DIR / "data" / "raw" / "framingham.csv"
DB_PATH = Path(os.environ.get("CARDIOCHECK_DB", BASE_DIR / "data" / "cardiocheck.db"))
MODELS_DIR = BASE_DIR / "models"
MODELOS_PATH = MODELS_DIR / "cardiocheck_modelos.joblib"
METRICAS_PATH = MODELS_DIR / "metricas.json"

TABLA_DATOS = "framingham"
TABLA_ENCUESTAS = "encuestas"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Variables -------------------------------------------------------------------
TARGET = "TenYearCHD"

# Variables que la app siempre pregunta (se responden sin pruebas médicas).
VARS_BASE = [
    "age",
    "male",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "BMI",
]
# Variables opcionales: solo si el usuario conoce sus cifras.
VARS_TENSION = ["sysBP", "diaBP"]
VARS_COLESTEROL = ["totChol"]

# Un modelo por combinación de campos opcionales: así nunca hay que "inventar"
# (imputar) la tensión o el colesterol de un usuario.
VARIANTES = {
    "base": VARS_BASE,
    "tension": VARS_BASE + VARS_TENSION,
    "colesterol": VARS_BASE + VARS_COLESTEROL,
    "completo": VARS_BASE + VARS_TENSION + VARS_COLESTEROL,
}
DESCRIPCION_VARIANTES = {
    "base": "datos básicos",
    "tension": "datos básicos + tensión arterial",
    "colesterol": "datos básicos + colesterol",
    "completo": "datos básicos + tensión arterial + colesterol",
}

# Nombres legibles y agrupación de variables para explicar cada predicción.
NOMBRES_VARIABLES = {
    "age": "Edad",
    "male": "Sexo (hombre)",
    "currentSmoker": "Fumador",
    "cigsPerDay": "Cigarrillos al día",
    "BPMeds": "Medicación para la tensión",
    "prevalentStroke": "Ictus previo",
    "prevalentHyp": "Hipertensión diagnosticada",
    "diabetes": "Diabetes",
    "BMI": "Índice de masa corporal",
    "sysBP": "Tensión sistólica",
    "diaBP": "Tensión diastólica",
    "totChol": "Colesterol total",
}
GRUPOS_FACTORES = {
    "age": "Edad",
    "male": "Sexo",
    "currentSmoker": "Tabaco",
    "cigsPerDay": "Tabaco",
    "BPMeds": "Hipertensión",
    "prevalentHyp": "Hipertensión",
    "sysBP": "Tensión medida",
    "diaBP": "Tensión medida",
    "totChol": "Colesterol",
    "diabetes": "Diabetes",
    "prevalentStroke": "Ictus previo",
    "BMI": "Peso (IMC)",
}

# --- Niveles de riesgo a 10 años (cortes clásicos de las escalas Framingham) -----
UMBRAL_MODERADO = 0.10
UMBRAL_ALTO = 0.20
COLORES_NIVEL = {"bajo": "#2e7d32", "moderado": "#ef8f00", "alto": "#c62828"}

# Rango de edad presente en los datos de entrenamiento.
EDAD_MIN_MODELO = 32
EDAD_MAX_MODELO = 70
