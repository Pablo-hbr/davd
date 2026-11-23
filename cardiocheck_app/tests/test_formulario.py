import pytest

from src.formulario import calcular_imc, procesar_formulario


def test_imc():
    assert calcular_imc(180, 81) == pytest.approx(25.0)


def test_formulario_valido(respuestas_validas):
    datos, errores, avisos = procesar_formulario(respuestas_validas)
    assert errores == [] and avisos == []
    assert datos["age"] == 55 and datos["male"] == 1
    assert datos["currentSmoker"] == 1 and datos["cigsPerDay"] == 15
    assert datos["BPMeds"] == 1 and datos["prevalentHyp"] == 1
    assert datos["BMI"] == pytest.approx(82 / 1.75**2, abs=0.01)
    assert datos["sysBP"] is None and datos["totChol"] is None


@pytest.mark.parametrize("campo", ["edad", "altura", "peso"])
def test_obligatorios(respuestas_validas, campo):
    respuestas_validas[campo] = None
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and len(errores) == 1


def test_sexo_obligatorio(respuestas_validas):
    respuestas_validas["sexo"] = None
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and errores == ["Indica tu sexo."]


def test_fuera_de_rango(respuestas_validas):
    respuestas_validas["edad"] = 150
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and "edad" in errores[0]


def test_fumador_sin_cigarrillos(respuestas_validas):
    respuestas_validas["cigarrillos"] = None
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and "cigarrillos" in errores[0]


def test_no_fumador_ignora_cigarrillos(respuestas_validas):
    respuestas_validas.update(fumador=0, cigarrillos=20)
    datos, _, _ = procesar_formulario(respuestas_validas)
    assert datos["currentSmoker"] == 0 and datos["cigsPerDay"] == 0


def test_medicacion_sin_hipertension_se_ignora(respuestas_validas):
    respuestas_validas.update(hipertension=0, medicacion=1)
    datos, _, _ = procesar_formulario(respuestas_validas)
    assert datos["BPMeds"] == 0


def test_tension_incompleta(respuestas_validas):
    respuestas_validas["sistolica"] = 130
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and "dos cifras" in errores[0]


def test_tension_invertida(respuestas_validas):
    respuestas_validas.update(sistolica=80, diastolica=120)
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and any("menor" in e for e in errores)


def test_opcionales_validos(respuestas_validas):
    respuestas_validas.update(sistolica=135, diastolica=85, colesterol=220)
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert errores == []
    assert (datos["sysBP"], datos["diaBP"], datos["totChol"]) == (135, 85, 220)


def test_imc_imposible(respuestas_validas):
    respuestas_validas.update(altura=220, peso=30)
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and "altura y peso" in errores[0]


def test_aviso_edad_fuera_del_modelo(respuestas_validas):
    respuestas_validas["edad"] = 25
    datos, errores, avisos = procesar_formulario(respuestas_validas)
    assert datos is not None and errores == [] and len(avisos) == 1


def test_texto_no_numerico(respuestas_validas):
    respuestas_validas["peso"] = "abc"
    datos, errores, _ = procesar_formulario(respuestas_validas)
    assert datos is None and len(errores) == 1
