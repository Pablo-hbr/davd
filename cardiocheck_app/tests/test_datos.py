"""Tests del ETL, la base de datos de encuestas y las recomendaciones."""

import pytest

from src import config, db, etl
from src.formulario import procesar_formulario
from src.recomendaciones import RECOMENDACION_NIVEL, generar_recomendaciones


def test_limpieza(datos):
    assert list(datos.columns) == config.VARIANTES["completo"] + [config.TARGET]
    assert len(datos) == 4238
    assert datos["cigsPerDay"].isna().sum() == 0
    assert (datos.loc[datos["currentSmoker"] == 0, "cigsPerDay"] == 0).all()
    assert (datos.loc[datos["currentSmoker"] == 1, "cigsPerDay"] > 0).all()
    assert datos.loc[datos["prevalentHyp"] == 0, "BPMeds"].isna().sum() == 0


def test_diagnostico(datos):
    tabla = etl.diagnostico(datos)
    assert {"dtype", "nulos", "pct_nulos"} <= set(tabla.columns)
    assert tabla.loc[config.TARGET, "nulos"] == 0


def test_etl_ida_y_vuelta_por_la_bd(tmp_path, datos):
    ruta = tmp_path / "prueba.db"
    leidos = etl.ejecutar_etl(db_path=ruta)
    assert leidos.shape == datos.shape


def test_encuestas(tmp_path, respuestas_validas):
    ruta = tmp_path / "encuestas.db"
    assert db.leer_encuestas(ruta).empty
    datos, _, _ = procesar_formulario(respuestas_validas)
    id_1 = db.guardar_encuesta(datos, "base", 0.123, "moderado", db_path=ruta)
    id_2 = db.guardar_encuesta(datos, "base", 0.05, "bajo", db_path=ruta)
    tabla = db.leer_encuestas(ruta)
    assert (id_1, id_2) == (1, 2)
    assert tabla["nivel"].tolist() == ["moderado", "bajo"]
    assert tabla.loc[0, "probabilidad"] == pytest.approx(0.123)
    assert tabla["sysBP"].isna().all()  # los opcionales vacíos se guardan como NULL


def test_recomendaciones(respuestas_validas):
    datos, _, _ = procesar_formulario(respuestas_validas)
    recs = generar_recomendaciones(datos, "alto")
    assert recs[0] == RECOMENDACION_NIVEL["alto"]
    assert all(set(r) == {"icono", "titulo", "texto"} for r in recs)
    textos = [f"{r['titulo']} {r['texto']}" for r in recs]
    assert any("fumar" in t for t in textos)
    assert any("sobrepeso" in t for t in textos)  # IMC 26,8


def test_recomendacion_tension_alta(respuestas_validas):
    respuestas_validas.update(sistolica=150, diastolica=95, colesterol=260)
    datos, _, _ = procesar_formulario(respuestas_validas)
    textos = [r["texto"] for r in generar_recomendaciones(datos, "moderado")]
    assert any("150/95" in t for t in textos)
    assert any("260" in t for t in textos)


def test_formato_espanol():
    from src.formato import num, pct

    assert num(12345.678) == "12.345,7"
    assert num(4238, 0) == "4.238"
    assert pct(0.363) == "36,3 %"
