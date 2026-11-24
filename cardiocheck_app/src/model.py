"""Entrenamiento, evaluación e inferencia del modelo de riesgo cardiovascular.

Decisiones (justificadas en el README):
- Regresión logística: mismo AUC que Random Forest / Gradient Boosting en estos
  datos, probabilidades bien calibradas y cada predicción se explica con sus
  coeficientes.
- Sin class_weight: la app muestra una probabilidad, y re-ponderar las clases la
  infla artificialmente. El desbalanceo se gestiona con los umbrales de nivel.
- Un modelo por combinación de campos opcionales (tensión / colesterol) en vez
  de imputar los valores que el usuario no conoce.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


# --- Construcción de modelos ------------------------------------------------------
def pipeline_logistica() -> Pipeline:
    """Imputación (mediana del train) + escalado + regresión logística."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def modelos_candidatos() -> dict:
    """Modelos que se comparan en validación cruzada."""
    return {
        "Regresión logística": pipeline_logistica(),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20, random_state=config.RANDOM_STATE, n_jobs=-1
            ),
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.03, max_depth=3, random_state=config.RANDOM_STATE
        ),
    }


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición train/test estratificada y reproducible."""
    return train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=config.RANDOM_STATE
    )


# --- Evaluación ---------------------------------------------------------------------
def comparar_modelos(train: pd.DataFrame, variables: list[str], n_splits: int = 5) -> pd.DataFrame:
    """Compara los modelos candidatos con validación cruzada estratificada sobre el train."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    filas = []
    for nombre, modelo in modelos_candidatos().items():
        res = cross_validate(
            modelo,
            train[variables],
            train[config.TARGET],
            cv=cv,
            scoring=["roc_auc", "average_precision", "neg_brier_score"],
        )
        filas.append(
            {
                "modelo": nombre,
                "auc_media": float(res["test_roc_auc"].mean()),
                "auc_std": float(res["test_roc_auc"].std()),
                "pr_auc_media": float(res["test_average_precision"].mean()),
                "brier_media": float(-res["test_neg_brier_score"].mean()),
            }
        )
    return pd.DataFrame(filas).sort_values("auc_media", ascending=False).reset_index(drop=True)


def metricas_clasificacion(y_real, prob) -> dict:
    """Métricas de discriminación, calibración y de los umbrales de nivel de riesgo."""
    y_real = np.asarray(y_real)
    prob = np.asarray(prob)
    fpr, tpr, _ = roc_curve(y_real, prob)
    # Calibración por deciles de probabilidad predicha.
    deciles = pd.qcut(prob, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": prob, "real": y_real, "grupo": deciles}).groupby("grupo").mean()

    umbrales = {}
    for nombre, t in [("moderado", config.UMBRAL_MODERADO), ("alto", config.UMBRAL_ALTO)]:
        marcados = prob >= t
        umbrales[nombre] = {
            "umbral": t,
            "recall": float(recall_score(y_real, marcados)),
            "precision": float(precision_score(y_real, marcados, zero_division=0)),
            "pct_marcados": float(marcados.mean()),
        }
    return {
        "auc": float(roc_auc_score(y_real, prob)),
        "pr_auc": float(average_precision_score(y_real, prob)),
        "brier": float(brier_score_loss(y_real, prob)),
        "brier_baseline": float(brier_score_loss(y_real, np.full(len(y_real), y_real.mean()))),
        "prevalencia": float(y_real.mean()),
        "prob_media": float(prob.mean()),
        "umbrales": umbrales,
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "calibracion": {"pred": calib["pred"].round(4).tolist(), "real": calib["real"].round(4).tolist()},
    }


def coeficientes(modelo: Pipeline, variables: list[str]) -> pd.DataFrame:
    """Coeficientes de la logística: estandarizados, por unidad original y odds ratio."""
    escalador = modelo.named_steps["standardscaler"]
    logistica = modelo.named_steps["logisticregression"]
    coef_std = logistica.coef_[0]
    coef_unidad = coef_std / escalador.scale_
    tabla = pd.DataFrame(
        {
            "variable": variables,
            "nombre": [config.NOMBRES_VARIABLES[v] for v in variables],
            "coef_estandarizado": coef_std,
            "coef_por_unidad": coef_unidad,
            "odds_ratio_por_unidad": np.exp(coef_unidad),
        }
    )
    tabla["peso_pct"] = 100 * tabla["coef_estandarizado"].abs() / tabla["coef_estandarizado"].abs().sum()
    return tabla.sort_values("peso_pct", ascending=False).reset_index(drop=True)


# --- Entrenamiento completo -----------------------------------------------------------
def entrenar(df: pd.DataFrame, comparar: bool = True) -> tuple[dict, dict]:
    """Evalúa en train/test y entrena los modelos finales con todos los datos.

    Parameters:
        df: datos limpios (salida del ETL).
        comparar: si True, compara LR / RF / GB con validación cruzada (más lento).
    Output:
        (modelos finales por variante, diccionario de métricas serializable a JSON)
    """
    train, test = dividir(df)
    y = config.TARGET

    metricas: dict = {
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "prevalencia": float(df[y].mean()),
        "umbral_moderado": config.UMBRAL_MODERADO,
        "umbral_alto": config.UMBRAL_ALTO,
    }
    if comparar:
        metricas["comparacion_modelos"] = comparar_modelos(train, config.VARIANTES["completo"]).to_dict("records")

    # Evaluación honesta de cada variante: entrena en train, mide en test.
    # Con ~4.200 filas una sola partición varía ±0,02 de AUC, así que se añade
    # también el AUC por validación cruzada repetida (5 folds x 5 repeticiones).
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=config.RANDOM_STATE)
    metricas["variantes"] = {}
    for variante, variables in config.VARIANTES.items():
        modelo = pipeline_logistica().fit(train[variables], train[y])
        prob = modelo.predict_proba(test[variables])[:, 1]
        resultado = metricas_clasificacion(test[y], prob)
        aucs = cross_val_score(pipeline_logistica(), df[variables], df[y], cv=cv, scoring="roc_auc")
        resultado["auc_cv_media"] = float(aucs.mean())
        resultado["auc_cv_std"] = float(aucs.std())
        # Calibración con predicciones fuera de muestra de las 4.238 personas: con
        # solo el test (~850) cada decil tiene 85 casos y la curva es muy ruidosa.
        prob_oof = cross_val_predict(
            pipeline_logistica(), df[variables], df[y], method="predict_proba",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE),
        )[:, 1]
        resultado["calibracion"] = metricas_clasificacion(df[y], prob_oof)["calibracion"]
        metricas["variantes"][variante] = resultado

    # Modelos finales: misma receta, entrenada con todos los datos.
    modelos = {
        variante: pipeline_logistica().fit(df[variables], df[y])
        for variante, variables in config.VARIANTES.items()
    }
    metricas["coeficientes"] = {
        variante: coeficientes(modelos[variante], variables).to_dict("records")
        for variante, variables in config.VARIANTES.items()
    }
    return modelos, metricas


def guardar(modelos: dict, metricas: dict) -> None:
    """Guarda los modelos (joblib) y las métricas (JSON)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelos, config.MODELOS_PATH)
    with open(config.METRICAS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)


def cargar_modelos(path: Path = config.MODELOS_PATH) -> dict:
    """Carga los modelos entrenados."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No hay modelos entrenados en {path}. Ejecuta antes: python train.py")
    return joblib.load(path)


def cargar_metricas(path: Path = config.METRICAS_PATH) -> dict:
    """Carga las métricas generadas en el entrenamiento."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Inferencia ------------------------------------------------------------------------
def elegir_variante(tiene_tension: bool, tiene_colesterol: bool) -> str:
    """Elige el modelo según los campos opcionales que ha rellenado el usuario."""
    if tiene_tension and tiene_colesterol:
        return "completo"
    if tiene_tension:
        return "tension"
    if tiene_colesterol:
        return "colesterol"
    return "base"


def variante_para(datos: dict) -> str:
    """Variante adecuada para un diccionario de datos del usuario."""
    tiene_tension = all(datos.get(v) is not None for v in config.VARS_TENSION)
    tiene_colesterol = all(datos.get(v) is not None for v in config.VARS_COLESTEROL)
    return elegir_variante(tiene_tension, tiene_colesterol)


def _fila(datos: dict, variables: list[str]) -> pd.DataFrame:
    """DataFrame de una fila con las columnas en el orden que espera el modelo."""
    return pd.DataFrame([{v: datos[v] for v in variables}], dtype=float)


def predecir(modelos: dict, datos: dict) -> tuple[float, str]:
    """Probabilidad de evento coronario a 10 años y variante del modelo usada."""
    variante = variante_para(datos)
    variables = config.VARIANTES[variante]
    prob = modelos[variante].predict_proba(_fila(datos, variables))[0, 1]
    return float(prob), variante


def nivel_riesgo(prob: float) -> str:
    """Traduce una probabilidad a nivel bajo / moderado / alto."""
    if prob >= config.UMBRAL_ALTO:
        return "alto"
    if prob >= config.UMBRAL_MODERADO:
        return "moderado"
    return "bajo"


def contribuciones(modelos: dict, datos: dict) -> pd.Series:
    """Contribución de cada factor a la predicción, en log-odds, frente a la persona media del estudio.

    Para una logística: contribución = coeficiente estandarizado x valor estandarizado.
    Positivo = sube el riesgo respecto a la media; negativo = lo baja. Las variables
    relacionadas (p. ej. fumador + cigarrillos) se suman en un mismo factor.
    """
    variante = variante_para(datos)
    variables = config.VARIANTES[variante]
    modelo = modelos[variante]
    x = modelo.named_steps["simpleimputer"].transform(_fila(datos, variables))
    z = modelo.named_steps["standardscaler"].transform(x)[0]
    coef = modelo.named_steps["logisticregression"].coef_[0]
    por_variable = pd.Series(coef * z, index=variables)
    por_grupo = por_variable.groupby(lambda v: config.GRUPOS_FACTORES[v]).sum()
    return por_grupo.sort_values(key=np.abs, ascending=False)


def escenarios(modelos: dict, datos: dict) -> list[dict]:
    """Riesgo estimado si cambiaran factores modificables con efecto robusto (tabaco, tensión).

    No se simulan IMC ni colesterol: en estos datos su efecto directo es pequeño y
    el mensaje "bajar de peso no cambia tu riesgo" sería clínicamente engañoso.
    """
    actual, _ = predecir(modelos, datos)
    salida = []
    if datos.get("currentSmoker") == 1:
        nuevo = {**datos, "currentSmoker": 0, "cigsPerDay": 0}
        salida.append({"escenario": "Si dejaras de fumar", "prob": predecir(modelos, nuevo)[0]})
    sys_bp, dia_bp = datos.get("sysBP"), datos.get("diaBP")
    if sys_bp is not None and dia_bp is not None and (sys_bp > 120 or dia_bp > 80):
        nuevo = {**datos, "sysBP": min(sys_bp, 120.0), "diaBP": min(dia_bp, 80.0)}
        salida.append({"escenario": "Con la tensión en 120/80", "prob": predecir(modelos, nuevo)[0]})
    for item in salida:
        item["diferencia"] = item["prob"] - actual
    return salida


def riesgo_optimo(modelos: dict, datos: dict) -> float:
    """Riesgo de una persona de la misma edad y sexo con todos los factores modificables en valores óptimos.

    Sirve de referencia ("lo mínimo alcanzable a tu edad"): no fuma, sin hipertensión,
    diabetes ni ictus, IMC 22 y, si se dieron cifras, tensión 120/80 y colesterol 180.
    """
    optimo = {
        **datos,
        "currentSmoker": 0,
        "cigsPerDay": 0,
        "BPMeds": 0,
        "prevalentHyp": 0,
        "diabetes": 0,
        "prevalentStroke": 0,
        "BMI": 22.0,
    }
    if datos.get("sysBP") is not None:
        optimo.update(sysBP=120.0, diaBP=80.0)
    if datos.get("totChol") is not None:
        optimo["totChol"] = 180.0
    return predecir(modelos, optimo)[0]


def riesgo_similares(df: pd.DataFrame, edad: float, male: int, margen: int = 5) -> tuple[float, int]:
    """% observado de eventos en personas del estudio del mismo sexo y edad ± margen años.

    Output:
        (proporción de eventos, número de personas comparables)
    """
    grupo = df[(df["male"] == male) & (df["age"].between(edad - margen, edad + margen))]
    if grupo.empty:
        return float("nan"), 0
    return float(grupo[config.TARGET].mean()), int(len(grupo))
