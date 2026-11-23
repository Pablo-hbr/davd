"""Recomendaciones según el nivel de riesgo y los factores del usuario.

Las recomendaciones se basan en reglas de prevención cardiovascular, NO en los
coeficientes del modelo: el modelo estima asociaciones, no indica qué cambio
causaría qué efecto.
"""

from __future__ import annotations

from src.formato import num

RECOMENDACION_NIVEL = {
    "bajo": "Tu riesgo estimado es bajo. Mantén tus hábitos saludables y haz revisiones médicas rutinarias.",
    "moderado": (
        "Tu riesgo estimado es moderado. Merece la pena mejorar los hábitos con más impacto "
        "y controlar periódicamente tu tensión y tu colesterol."
    ),
    "alto": (
        "Tu riesgo estimado es alto. Te recomendamos consultar con tu médico de cabecera o un "
        "cardiólogo para una evaluación completa."
    ),
}


def generar_recomendaciones(datos: dict, nivel: str) -> list[str]:
    """Lista de recomendaciones: primero la del nivel, después las específicas."""
    recs = [RECOMENDACION_NIVEL[nivel]]

    if datos["currentSmoker"] == 1:
        recs.append(
            "Dejar de fumar es el cambio con más impacto sobre tu corazón. Tu centro de salud "
            "ofrece programas gratuitos de deshabituación."
        )
    if datos["prevalentHyp"] == 1 and datos["BPMeds"] == 0:
        recs.append("Tienes hipertensión diagnosticada sin tratamiento: coméntalo con tu médico.")
    sys_bp, dia_bp = datos.get("sysBP"), datos.get("diaBP")
    if sys_bp is not None and (sys_bp >= 140 or dia_bp >= 90):
        recs.append(
            f"Tu tensión ({sys_bp:.0f}/{dia_bp:.0f}) está por encima de 140/90: conviene revisarla con tu médico."
        )
    elif sys_bp is None and datos["prevalentHyp"] == 0:
        recs.append("Si no conoces tu tensión arterial, mídetela: en muchas farmacias es gratis.")
    colesterol = datos.get("totChol")
    if colesterol is not None and colesterol >= 240:
        recs.append(f"Tu colesterol total ({colesterol:.0f} mg/dL) es alto: consulta un análisis completo (LDL/HDL).")
    if datos["diabetes"] == 1:
        recs.append("Con diabetes, mantener la glucosa bien controlada protege también tu corazón.")
    if datos["BMI"] >= 30:
        recs.append(
            f"Tu IMC ({num(datos['BMI'])}) indica obesidad: perder peso mejora la tensión, la glucosa y el colesterol."
        )
    elif datos["BMI"] >= 25:
        recs.append(f"Tu IMC ({num(datos['BMI'])}) indica sobrepeso: una dieta mediterránea ayuda a reducirlo.")
    recs.append("Haz al menos 150 minutos de actividad física moderada a la semana (por ejemplo, caminar a paso rápido).")
    return recs
