"""Figuras de Plotly que usa la app (resultado, exploración de datos, modelo y uso)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.formato import num

COLOR_PRINCIPAL = "#b71c1c"
COLOR_SECUNDARIO = "#455a64"
COLOR_SUBE = "#c62828"
COLOR_BAJA = "#2e7d32"
COLORES_VARIANTES = {"base": "#90a4ae", "tension": "#ef8f00", "colesterol": "#1e88e5", "completo": "#b71c1c"}


def _estilo(fig: go.Figure, titulo: str = "", altura: int = 380) -> go.Figure:
    """Estilo común: fondo blanco, márgenes compactos (pantallas de móvil) y título."""
    fig.update_layout(
        title={"text": titulo, "x": 0.02, "xanchor": "left", "font": {"size": 16}},
        template="plotly_white",
        height=altura,
        margin={"l": 10, "r": 10, "t": 50 if titulo else 20, "b": 10},
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 13},
        hoverlabel={"font_size": 13},
        separators=",.",  # coma decimal en todos los números de Plotly
    )
    return fig


def figura_vacia(mensaje: str) -> go.Figure:
    """Figura sin datos con un mensaje centrado."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font={"size": 15, "color": COLOR_SECUNDARIO})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _estilo(fig, altura=250)


# --- Resultado individual -------------------------------------------------------------
def figura_gauge(prob: float, nivel: str) -> go.Figure:
    """Indicador tipo velocímetro con las bandas de riesgo bajo / moderado / alto."""
    valor = 100 * prob
    maximo = max(50, int(np.ceil(valor / 10) * 10))
    fig = go.Figure(
        go.Indicator(
            mode="gauge",  # el porcentaje ya aparece en grande encima del gráfico
            value=valor,
            gauge={
                "axis": {"range": [0, maximo], "ticksuffix": "%"},
                "bar": {"color": "#263238", "thickness": 0.25},
                "steps": [
                    {"range": [0, 100 * config.UMBRAL_MODERADO], "color": "#c8e6c9"},
                    {"range": [100 * config.UMBRAL_MODERADO, 100 * config.UMBRAL_ALTO], "color": "#ffe0b2"},
                    {"range": [100 * config.UMBRAL_ALTO, maximo], "color": "#ffcdd2"},
                ],
            },
        )
    )
    return _estilo(fig, altura=240)


def figura_factores(contrib: pd.Series) -> go.Figure:
    """Barras horizontales: qué factores suben (rojo) o bajan (verde) tu riesgo frente a la media."""
    datos = contrib[contrib.abs() > 0.01].sort_values()
    if datos.empty:
        return figura_vacia("Tus factores están en la media del estudio")
    colores = [COLOR_SUBE if v > 0 else COLOR_BAJA for v in datos]
    textos = ["Sube tu riesgo" if v > 0 else "Baja tu riesgo" for v in datos]
    fig = go.Figure(
        go.Bar(
            x=datos.values,
            y=datos.index,
            orientation="h",
            marker_color=colores,
            customdata=textos,
            hovertemplate="%{y}: %{customdata}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="← baja el riesgo · sube el riesgo →", showticklabels=False, zeroline=True)
    return _estilo(fig, "Qué factores han influido", altura=90 + 40 * len(datos))


def figura_escenarios(prob_actual: float, escenarios: list[dict]) -> go.Figure:
    """Comparación del riesgo actual con escenarios de cambio de hábitos."""
    etiquetas = ["Tu riesgo actual"] + [e["escenario"] for e in escenarios]
    valores = [100 * prob_actual] + [100 * e["prob"] for e in escenarios]
    colores = [COLOR_PRINCIPAL] + [COLOR_BAJA] * len(escenarios)
    fig = go.Figure(
        go.Bar(
            x=etiquetas,
            y=valores,
            marker_color=colores,
            text=[f"{num(v)} %" for v in valores],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f} %<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Riesgo a 10 años (%)", range=[0, max(valores) * 1.25])
    return _estilo(fig, "¿Y si cambiaras…?", altura=320)


# --- Exploración de los datos de Framingham ----------------------------------------------
def _grupos(df: pd.DataFrame, clave: str) -> pd.Series:
    """Agrupa una variable en categorías legibles para la exploración."""
    if clave == "edad":
        return pd.cut(df["age"], [31, 39, 49, 59, 70], labels=["32-39", "40-49", "50-59", "60-70"])
    if clave == "sexo":
        return df["male"].map({0: "Mujer", 1: "Hombre"})
    if clave == "tabaco":
        return pd.cut(
            df["cigsPerDay"], [-1, 0, 9, 19, 100], labels=["No fuma", "1-9 al día", "10-19 al día", "20 o más"]
        )
    if clave == "hipertension":
        return df["prevalentHyp"].map({0: "No", 1: "Sí"})
    if clave == "diabetes":
        return df["diabetes"].map({0: "No", 1: "Sí"})
    if clave == "imc":
        return pd.cut(df["BMI"], [0, 18.5, 25, 30, 100], right=False, labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"])
    if clave == "sistolica":
        return pd.cut(df["sysBP"], [0, 120, 140, 160, 400], right=False, labels=["<120", "120-139", "140-159", "≥160"])
    if clave == "colesterol":
        return pd.cut(df["totChol"], [0, 200, 240, 1000], right=False, labels=["<200", "200-239", "≥240"])
    raise ValueError(f"Variable de exploración desconocida: {clave}")


VARIABLES_EXPLORAR = {
    "edad": "Edad",
    "sexo": "Sexo",
    "tabaco": "Tabaco",
    "hipertension": "Hipertensión diagnosticada",
    "diabetes": "Diabetes",
    "imc": "Índice de masa corporal",
    "sistolica": "Tensión sistólica (mmHg)",
    "colesterol": "Colesterol total (mg/dL)",
}


def tabla_prevalencia(df: pd.DataFrame, clave: str) -> pd.DataFrame:
    """% de personas con evento coronario a 10 años por categoría (sin nulos)."""
    grupos = _grupos(df, clave)
    tabla = (
        pd.DataFrame({"grupo": grupos, "evento": df[config.TARGET]})
        .dropna()
        .groupby("grupo", observed=True)["evento"]
        .agg(["mean", "size"])
        .reset_index()
    )
    tabla["pct"] = 100 * tabla["mean"]
    return tabla


def figura_prevalencia(df: pd.DataFrame, clave: str) -> go.Figure:
    """Barras con el % de eventos coronarios a 10 años por categoría y la media global."""
    tabla = tabla_prevalencia(df, clave)
    media = 100 * df[config.TARGET].mean()
    fig = go.Figure(
        go.Bar(
            x=tabla["grupo"].astype(str),
            y=tabla["pct"],
            marker_color=COLOR_PRINCIPAL,
            text=[f"{num(p)} %" for p in tabla["pct"]],
            textposition="outside",
            customdata=tabla["size"],
            hovertemplate="%{x}: %{y:.1f} % de %{customdata} personas<extra></extra>",
        )
    )
    fig.add_hline(y=media, line_dash="dash", line_color=COLOR_SECUNDARIO)
    fig.update_yaxes(title_text="% con evento coronario en 10 años", range=[0, tabla["pct"].max() * 1.3])
    titulo = (f"Riesgo observado según {VARIABLES_EXPLORAR[clave].lower()}"
              f"<br><sup>Línea discontinua: media del estudio ({num(media)} %)</sup>")
    return _estilo(fig, titulo, altura=400)


def figura_edad_sexo_tabaco(df: pd.DataFrame) -> go.Figure:
    """% de eventos por grupo de edad, separando por sexo y tabaco."""
    aux = pd.DataFrame(
        {
            "edad": _grupos(df, "edad"),
            "perfil": df["male"].map({0: "Mujer", 1: "Hombre"})
            + df["currentSmoker"].map({0: " no fumador/a", 1: " fumador/a"}),
            "evento": df[config.TARGET],
        }
    )
    tabla = aux.groupby(["perfil", "edad"], observed=True)["evento"].mean().mul(100).reset_index()
    colores = {
        "Hombre fumador/a": "#b71c1c", "Hombre no fumador/a": "#ef9a9a",
        "Mujer fumador/a": "#283593", "Mujer no fumador/a": "#9fa8da",
    }
    fig = go.Figure()
    for perfil, color in colores.items():
        sub = tabla[tabla["perfil"] == perfil]
        fig.add_trace(
            go.Scatter(
                x=sub["edad"].astype(str), y=sub["evento"], mode="lines+markers", name=perfil,
                line={"color": color, "width": 3}, hovertemplate=f"{perfil}<br>%{{x}}: %{{y:.1f}} %<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="% con evento coronario en 10 años")
    fig.update_xaxes(title_text="Edad")
    fig.update_layout(legend={"orientation": "h", "y": -0.25})
    return _estilo(fig, "Edad, sexo y tabaco combinados", altura=420)


# --- Sobre el modelo -------------------------------------------------------------------------
def figura_comparacion_modelos(comparacion: list[dict]) -> go.Figure:
    """AUC medio ± desviación en validación cruzada de cada modelo candidato."""
    tabla = pd.DataFrame(comparacion).sort_values("auc_media")
    fig = go.Figure(
        go.Bar(
            x=tabla["auc_media"],
            # El AUC va en la etiqueta del eje: dentro de la barra se taparía con la barra de error.
            y=[f"{m}<br><b>AUC {num(a, 3)}</b>" for m, a in zip(tabla["modelo"], tabla["auc_media"])],
            orientation="h",
            error_x={"type": "data", "array": tabla["auc_std"]},
            marker_color=[COLOR_PRINCIPAL if m == "Regresión logística" else "#b0bec5" for m in tabla["modelo"]],
            customdata=tabla["auc_std"],
            hovertemplate="%{y}<br>± %{customdata:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="AUC (validación cruzada)", range=[0.5, 0.8])
    return _estilo(fig, "Comparación de modelos", altura=280)


def figura_roc(metricas: dict) -> go.Figure:
    """Curvas ROC en test de las cuatro variantes del modelo."""
    fig = go.Figure()
    for variante, m in metricas["variantes"].items():
        fig.add_trace(
            go.Scatter(
                x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
                name=f"{config.DESCRIPCION_VARIANTES[variante]} (AUC {num(m['auc'], 3)})",
                line={"color": COLORES_VARIANTES[variante], "width": 2},
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC 0,5)", line={"dash": "dash", "color": "#9e9e9e"})
    )
    fig.update_xaxes(title_text="Falsos positivos (1 - especificidad)")
    fig.update_yaxes(title_text="Verdaderos positivos (sensibilidad)")
    fig.update_layout(legend={"orientation": "h", "y": -0.3, "font": {"size": 11}})
    return _estilo(fig, "Curva ROC en test", altura=460)


def figura_calibracion(metricas: dict, variante: str = "completo") -> go.Figure:
    """Probabilidad predicha frente a frecuencia real, por deciles (test)."""
    calib = metricas["variantes"][variante]["calibracion"]
    pred = 100 * np.array(calib["pred"])
    real = 100 * np.array(calib["real"])
    limite = float(max(pred.max(), real.max()) * 1.1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, limite], y=[0, limite], mode="lines", name="Calibración perfecta",
                   line={"dash": "dash", "color": "#9e9e9e"})
    )
    fig.add_trace(
        go.Scatter(x=pred, y=real, mode="lines+markers", name="Modelo",
                   line={"color": COLOR_PRINCIPAL, "width": 3},
                   hovertemplate="Predicho %{x:.1f} % → real %{y:.1f} %<extra></extra>")
    )
    fig.update_xaxes(title_text="Riesgo predicho (%)", range=[0, limite])
    fig.update_yaxes(title_text="Eventos observados (%)", range=[0, limite])
    fig.update_layout(legend={"orientation": "h", "y": -0.25})
    return _estilo(fig, "Calibración: ¿el % que mostramos es real?<br><sup>Predicciones de validación cruzada, agrupadas en deciles</sup>", altura=420)


def figura_coeficientes(metricas: dict, variante: str = "completo") -> go.Figure:
    """Coeficientes estandarizados de la regresión logística (peso de cada variable)."""
    tabla = pd.DataFrame(metricas["coeficientes"][variante]).sort_values("coef_estandarizado")
    fig = go.Figure(
        go.Bar(
            x=tabla["coef_estandarizado"],
            y=tabla["nombre"],
            orientation="h",
            marker_color=[COLOR_SUBE if c > 0 else COLOR_BAJA for c in tabla["coef_estandarizado"]],
            customdata=np.stack([tabla["odds_ratio_por_unidad"], tabla["peso_pct"]], axis=1),
            hovertemplate="%{y}<br>Coef. estandarizado: %{x:.3f}<br>Odds ratio por unidad: %{customdata[0]:.3f}"
            "<br>Peso: %{customdata[1]:.1f} %<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Coeficiente estandarizado (log-odds por desviación típica)")
    return _estilo(fig, "Peso de cada variable en el modelo", altura=460)


# --- Uso de la app -------------------------------------------------------------------------
def figura_niveles_encuestas(encuestas: pd.DataFrame) -> go.Figure:
    """Número de encuestas por nivel de riesgo."""
    orden = ["bajo", "moderado", "alto"]
    conteo = encuestas["nivel"].value_counts().reindex(orden, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[n.capitalize() for n in orden],
            y=conteo.values,
            marker_color=[config.COLORES_NIVEL[n] for n in orden],
            text=conteo.values,
            textposition="outside",
        )
    )
    fig.update_yaxes(title_text="Encuestas", range=[0, max(conteo.max(), 1) * 1.25], tickformat="d")
    return _estilo(fig, "Encuestas por nivel de riesgo", altura=320)


def figura_distribucion_encuestas(encuestas: pd.DataFrame) -> go.Figure:
    """Histograma del riesgo estimado en las encuestas recibidas."""
    fig = go.Figure(
        go.Histogram(x=100 * encuestas["probabilidad"], xbins={"start": 0, "size": 2.5},
                     marker_color=COLOR_PRINCIPAL, hovertemplate="%{x} %: %{y} encuestas<extra></extra>")
    )
    for umbral in (config.UMBRAL_MODERADO, config.UMBRAL_ALTO):
        fig.add_vline(x=100 * umbral, line_dash="dash", line_color=COLOR_SECUNDARIO)
    fig.update_xaxes(title_text="Riesgo estimado (%)")
    fig.update_yaxes(title_text="Encuestas", tickformat="d")
    fig.update_layout(bargap=0.05)
    return _estilo(fig, "Distribución del riesgo estimado", altura=320)
