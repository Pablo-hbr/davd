"""Fixtures compartidas: datos limpios y modelos entrenados una sola vez por sesión."""

import pytest

from src import etl, model


@pytest.fixture(scope="session")
def datos():
    return etl.limpiar(etl.leer_csv())


@pytest.fixture(scope="session")
def modelos_y_metricas(datos):
    return model.entrenar(datos, comparar=False)


@pytest.fixture(scope="session")
def modelos(modelos_y_metricas):
    return modelos_y_metricas[0]


@pytest.fixture
def respuestas_validas():
    """Respuestas de formulario de un hombre de 55 años, fumador e hipertenso."""
    return dict(
        edad=55, sexo=1, altura=175, peso=82, fumador=1, cigarrillos=15, hipertension=1, medicacion=1,
        diabetes=0, ictus=0, sistolica=None, diastolica=None, colesterol=None,
    )
