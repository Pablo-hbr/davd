"""Validación del formulario y traducción de las respuestas a la codificación del modelo.

Se mantiene separado de Dash para poder testearlo sin levantar la app.
"""

from __future__ import annotations

from src import config

# (mínimo, máximo) admitidos en el formulario.
RANGOS = {
    "edad": (20, 90),
    "altura": (120, 220),
    "peso": (30, 250),
    "cigarrillos": (1, 80),
    "sistolica": (80, 250),
    "diastolica": (40, 150),
    "colesterol": (100, 500),
}
ETIQUETAS = {
    "edad": "la edad",
    "altura": "la altura (cm)",
    "peso": "el peso (kg)",
    "cigarrillos": "los cigarrillos al día",
    "sistolica": "la tensión sistólica",
    "diastolica": "la tensión diastólica",
    "colesterol": "el colesterol total",
}


def calcular_imc(altura_cm: float, peso_kg: float) -> float:
    """Índice de masa corporal = peso (kg) / altura (m)^2."""
    return peso_kg / (altura_cm / 100) ** 2


def _numero(valor) -> float | None:
    """Convierte la entrada del formulario a float; None si está vacía."""
    if valor is None or valor == "":
        return None
    return float(valor)


def _en_rango(nombre: str, valor: float, errores: list[str]) -> None:
    minimo, maximo = RANGOS[nombre]
    if not minimo <= valor <= maximo:
        errores.append(f"Revisa {ETIQUETAS[nombre]}: debe estar entre {minimo} y {maximo}.")


def procesar_formulario(respuestas: dict) -> tuple[dict | None, list[str], list[str]]:
    """Valida las respuestas y las convierte a las variables del modelo.

    Parameters:
        respuestas: dict con claves edad, sexo, altura, peso, fumador, cigarrillos,
            hipertension, medicacion, diabetes, ictus, sistolica, diastolica, colesterol.
    Output:
        (datos para el modelo o None si hay errores, lista de errores, lista de avisos)
    """
    errores: list[str] = []
    avisos: list[str] = []

    obligatorios = {"edad": "Indica tu edad.", "altura": "Indica tu altura.", "peso": "Indica tu peso."}
    valores = {}
    for campo in ["edad", "altura", "peso", "cigarrillos", "sistolica", "diastolica", "colesterol"]:
        try:
            valores[campo] = _numero(respuestas.get(campo))
        except (TypeError, ValueError):
            errores.append(f"Revisa {ETIQUETAS[campo]}: no es un número válido.")
            valores[campo] = None

    for campo, mensaje in obligatorios.items():
        if valores[campo] is None and not any(ETIQUETAS[campo] in e for e in errores):
            errores.append(mensaje)
        elif valores[campo] is not None:
            _en_rango(campo, valores[campo], errores)

    if respuestas.get("sexo") not in (0, 1):
        errores.append("Indica tu sexo.")

    fumador = 1 if respuestas.get("fumador") == 1 else 0
    cigarrillos = 0.0
    if fumador:
        if valores["cigarrillos"] is None:
            errores.append("Indica cuántos cigarrillos fumas al día.")
        else:
            _en_rango("cigarrillos", valores["cigarrillos"], errores)
            cigarrillos = valores["cigarrillos"]

    hipertension = 1 if respuestas.get("hipertension") == 1 else 0
    # La medicación solo tiene sentido si hay hipertensión diagnosticada.
    medicacion = 1 if (hipertension and respuestas.get("medicacion") == 1) else 0

    sistolica, diastolica = valores["sistolica"], valores["diastolica"]
    if (sistolica is None) != (diastolica is None):
        errores.append("Si indicas tu tensión, rellena las dos cifras (máxima y mínima).")
    elif sistolica is not None:
        _en_rango("sistolica", sistolica, errores)
        _en_rango("diastolica", diastolica, errores)
        if diastolica >= sistolica:
            errores.append("La tensión mínima (diastólica) debe ser menor que la máxima (sistólica).")

    colesterol = valores["colesterol"]
    if colesterol is not None:
        _en_rango("colesterol", colesterol, errores)

    imc = None
    if valores["altura"] is not None and valores["peso"] is not None and not errores:
        imc = calcular_imc(valores["altura"], valores["peso"])
        if not 12 <= imc <= 70:
            errores.append("La combinación de altura y peso no parece correcta. Revísala.")

    if errores:
        return None, errores, avisos

    edad = valores["edad"]
    if not config.EDAD_MIN_MODELO <= edad <= config.EDAD_MAX_MODELO:
        avisos.append(
            f"El modelo se entrenó con personas de {config.EDAD_MIN_MODELO} a {config.EDAD_MAX_MODELO} años: "
            "con tu edad la estimación es menos fiable."
        )

    datos = {
        "age": edad,
        "male": int(respuestas["sexo"]),
        "currentSmoker": fumador,
        "cigsPerDay": cigarrillos,
        "BPMeds": medicacion,
        "prevalentStroke": 1 if respuestas.get("ictus") == 1 else 0,
        "prevalentHyp": hipertension,
        "diabetes": 1 if respuestas.get("diabetes") == 1 else 0,
        "BMI": round(imc, 2),
        "sysBP": sistolica,
        "diaBP": diastolica,
        "totChol": colesterol,
    }
    return datos, errores, avisos
