import numpy as np
import pytest

from src import config, model
from src.formulario import procesar_formulario


def _datos(respuestas):
    datos, errores, _ = procesar_formulario(respuestas)
    assert not errores
    return datos


@pytest.mark.parametrize(
    "tension, colesterol, esperado",
    [(False, False, "base"), (True, False, "tension"), (False, True, "colesterol"), (True, True, "completo")],
)
def test_elegir_variante(tension, colesterol, esperado):
    assert model.elegir_variante(tension, colesterol) == esperado


@pytest.mark.parametrize("prob, nivel", [(0.0, "bajo"), (0.0999, "bajo"), (0.10, "moderado"),
                                         (0.1999, "moderado"), (0.20, "alto"), (0.9, "alto")])
def test_nivel_riesgo(prob, nivel):
    assert model.nivel_riesgo(prob) == nivel


def test_metricas_razonables(modelos_y_metricas):
    _, metricas = modelos_y_metricas
    for variante, m in metricas["variantes"].items():
        assert 0.65 < m["auc_cv_media"] < 0.80, variante  # literatura Framingham: ~0,70-0,75
        assert m["brier"] < m["brier_baseline"], variante  # mejor que predecir siempre la media
        assert abs(m["prob_media"] - m["prevalencia"]) < 0.03, variante  # bien calibrado en media


def test_prediccion_y_variante(modelos, respuestas_validas):
    datos = _datos(respuestas_validas)
    prob, variante = model.predecir(modelos, datos)
    assert 0 < prob < 1 and variante == "base"

    respuestas_validas.update(sistolica=140, diastolica=90, colesterol=230)
    prob, variante = model.predecir(modelos, _datos(respuestas_validas))
    assert 0 < prob < 1 and variante == "completo"


def test_fumar_y_edad_suben_el_riesgo(modelos, respuestas_validas):
    base = _datos(respuestas_validas)
    no_fumador = {**base, "currentSmoker": 0, "cigsPerDay": 0}
    mayor = {**base, "age": base["age"] + 10}
    assert model.predecir(modelos, no_fumador)[0] < model.predecir(modelos, base)[0]
    assert model.predecir(modelos, mayor)[0] > model.predecir(modelos, base)[0]


def test_contribuciones_cuadran_con_la_prediccion(modelos, respuestas_validas):
    """En una logística: logit(p) = intercepto + suma de contribuciones."""
    datos = _datos(respuestas_validas)
    prob, variante = model.predecir(modelos, datos)
    contrib = model.contribuciones(modelos, datos)
    intercepto = modelos[variante].named_steps["logisticregression"].intercept_[0]
    assert np.log(prob / (1 - prob)) == pytest.approx(intercepto + contrib.sum(), abs=1e-9)
    assert set(contrib.index) <= set(config.GRUPOS_FACTORES.values())


def test_escenarios(modelos, respuestas_validas):
    respuestas_validas.update(sistolica=150, diastolica=95)
    datos = _datos(respuestas_validas)
    escenarios = model.escenarios(modelos, datos)
    nombres = [e["escenario"] for e in escenarios]
    assert nombres == ["Si dejaras de fumar", "Con la tensión en 120/80"]
    assert all(e["diferencia"] < 0 for e in escenarios)


def test_sin_escenarios_para_no_fumador_sin_tension(modelos, respuestas_validas):
    respuestas_validas.update(fumador=0)
    assert model.escenarios(modelos, _datos(respuestas_validas)) == []


def test_riesgo_optimo_no_supera_el_actual(modelos, respuestas_validas):
    respuestas_validas.update(sistolica=150, diastolica=95, colesterol=250)
    datos = _datos(respuestas_validas)
    assert model.riesgo_optimo(modelos, datos) < model.predecir(modelos, datos)[0]


def test_riesgo_similares(datos):
    prop, n = model.riesgo_similares(datos, 55, 1)
    assert n > 100 and 0 < prop < 1
    prop_joven, _ = model.riesgo_similares(datos, 38, 0)
    assert prop_joven < prop  # mujeres jóvenes: menos eventos que hombres de 55
    assert model.riesgo_similares(datos, 20, 1)[1] == 0


def test_coeficientes_con_sentido_clinico(modelos):
    tabla = model.coeficientes(modelos["completo"], config.VARIANTES["completo"]).set_index("variable")
    for variable in ["age", "male", "cigsPerDay", "sysBP", "prevalentStroke"]:
        assert tabla.loc[variable, "coef_estandarizado"] > 0, variable
    assert tabla["peso_pct"].sum() == pytest.approx(100)
