"""Tests de humo de la app Dash: el layout se construye y los callbacks responden."""

import pytest
from dash import html

import app as cardio_app


@pytest.fixture
def encuestas_guardadas(monkeypatch):
    """Evita escribir en la base de datos real durante los tests."""
    guardadas = []
    monkeypatch.setattr(cardio_app.db, "guardar_encuesta", lambda *args, **kwargs: guardadas.append(args))
    return guardadas


def _texto(componente) -> str:
    """Concatena todo el texto de un árbol de componentes Dash."""
    if componente is None:
        return ""
    if isinstance(componente, (str, int, float)):
        return str(componente)
    if isinstance(componente, (list, tuple)):
        return " ".join(_texto(c) for c in componente)
    return _texto(getattr(componente, "children", None))


def test_layout_y_servidor():
    assert cardio_app.server is not None
    assert cardio_app.app.layout is not None


def test_calcular_riesgo_valido(encuestas_guardadas):
    resultado = cardio_app.calcular_riesgo(
        1, 55, 1, 175, 82, 1, 15, 1, 1, 0, 0, 150, 95, 250
    )
    texto = _texto(resultado)
    assert "Riesgo" in texto and "Recomendaciones" in texto
    assert len(encuestas_guardadas) == 1


def test_calcular_riesgo_con_errores(encuestas_guardadas):
    resultado = cardio_app.calcular_riesgo(1, None, None, None, None, 0, None, 0, 0, 0, 0, None, None, None)
    assert "Revisa el formulario" in _texto(resultado)
    assert encuestas_guardadas == []


def test_callbacks_simples():
    assert cardio_app.mostrar_cigarrillos(1) == {"display": "block"}
    assert cardio_app.mostrar_cigarrillos(0) == {"display": "none"}
    assert cardio_app.mostrar_medicacion(1) == {"display": "block"}
    for clave in cardio_app.graphics.VARIABLES_EXPLORAR:
        assert cardio_app.actualizar_exploracion(clave).data


def test_pestana_uso():
    assert cardio_app.actualizar_uso("calcular") is None
    contenido = cardio_app.actualizar_uso("uso")
    assert isinstance(contenido, list) and isinstance(contenido[0], html.Div)
