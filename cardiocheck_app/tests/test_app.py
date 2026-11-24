"""Tests de humo de la app Dash: el layout se construye y los callbacks responden."""

import pytest

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
    assert "De cada 100 personas" in texto and "Con factores óptimos" in texto
    assert len(encuestas_guardadas) == 1


def test_calcular_riesgo_con_errores(encuestas_guardadas):
    resultado = cardio_app.calcular_riesgo(1, None, None, None, None, 0, None, 0, 0, 0, 0, None, None, None)
    assert "Revisa el formulario" in _texto(resultado)
    assert encuestas_guardadas == []


def test_navegacion():
    estilos = cardio_app.navegar("datos")
    assert [e["display"] for e in estilos] == ["none", "block", "none"]
    assert [e["display"] for e in cardio_app.navegar("calcular")] == ["block", "none", "none"]


def test_portada():
    portada = cardio_app.portada()
    assert portada.id == "portada"
    assert getattr(portada, "data-objetivo") == "15"  # 15,2 % de eventos en el estudio
    assert "Calcula tu riesgo" in _texto(portada)


def test_imc_en_vivo():
    assert cardio_app.imc_en_vivo(None, 80) is None
    assert cardio_app.imc_en_vivo(180, 1000) is None  # fuera de rango: no se muestra
    assert "24,7" in _texto(cardio_app.imc_en_vivo(180, 80))
    assert "obesidad" in _texto(cardio_app.imc_en_vivo(170, 95))


def test_callbacks_simples():
    assert cardio_app.mostrar_cigarrillos(1) == {"display": "block"}
    assert cardio_app.mostrar_cigarrillos(0) == {"display": "none"}
    assert cardio_app.mostrar_medicacion(1) == {"display": "block"}
    for clave in cardio_app.graphics.VARIABLES_EXPLORAR:
        assert cardio_app.actualizar_exploracion(clave).data

