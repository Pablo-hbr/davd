"""Figuras de Plotly de la app (exploración de datos y modelo), con un estilo común."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.formato import num

# Paleta (coherente con assets/style.css)
ROJO = "#e11d48"
ROJO_OSCURO = "#9f1239"
TINTA = "#0f172a"
GRIS = "#64748b"
REJILLA = "#eef2f6"
GRIS_CLARO = "#cbd5e1"
COLORES_VARIANTES = {"base": "#94a3b8", "tension": "#f59e0b", "colesterol": "#3b82f6", "completo": ROJO}


def _estilo(fig: go.Figure, titulo: str = "", subtitulo: str = "", altura: int = 380) -> go.Figure:
    """Estilo común: fondo transparente, tipografía Inter, rejilla suave y márgenes compactos."""
    texto = f"<b>{titulo}</b>" if titulo else ""
    if subtitulo:
        texto += f"<br><span style='font-size:12.5px;color:{GRIS}'>{subtitulo}</span>"
    fig.update_layout(
        title={"text": texto, "x": 0, "xanchor": "left", "y": 1, "yref": "container", "yanchor": "top",
               "pad": {"t": 12}, "font": {"size": 16, "color": TINTA}},
        height=altura,
        margin={"l": 8, "r": 8, "t": 78 if subtitulo else (52 if titulo else 16), "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "size": 13, "color": TINTA},
        hoverlabel={"font_size": 13, "font_family": "Inter, system-ui, sans-serif", "bgcolor": "white"},
        separators=",.",  # coma decimal en todos los números de Plotly
        legend={"font": {"size": 12}},
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA, title_font={"size": 12, "color": GRIS})
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA, title_font={"size": 12, "color": GRIS})
    return fig


# --- Exploración de los datos de Framingham ----------------------------------------------
VARIABLES_EXPLORAR = {
    "edad": "Edad",
    "sexo": "Sexo",
    "tabaco": "Tabaco",
    "hipertension": "Hipertensión",
    "diabetes": "Diabetes",
    "imc": "IMC",
    "sistolica": "Tensión sistólica",
    "colesterol": "Colesterol",
}


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
    colores = [ROJO if p > media else GRIS_CLARO for p in tabla["pct"]]
    fig = go.Figure(
        go.Bar(
            x=tabla["grupo"].astype(str),
            y=tabla["pct"],
            marker={"color": colores, "cornerradius": 8},
            text=[f"{num(p)} %" for p in tabla["pct"]],
            textposition="outside",
            textfont={"size": 13, "color": TINTA},
            customdata=tabla["size"],
            hovertemplate="%{x}: %{y:.1f} % de %{customdata} personas<extra></extra>",
        )
    )
    fig.add_hline(y=media, line_dash="dot", line_color=GRIS, line_width=1.5)
    fig.update_yaxes(title_text="% con evento en 10 años", range=[0, tabla["pct"].max() * 1.3], ticksuffix=" %")
    fig.update_layout(bargap=0.35)
    return _estilo(
        fig,
        f"Riesgo observado según {VARIABLES_EXPLORAR[clave].lower()}",
        f"Línea punteada: media del estudio ({num(media)} %). En rojo, los grupos por encima de la media.",
        altura=400,
    )


def figura_edad_sexo_tabaco(df: pd.DataFrame) -> go.Figure:
    """% de eventos por grupo de edad, separando por sexo y tabaco."""
    aux = pd.DataFrame(
        {
            "edad": _grupos(df, "edad"),
            "perfil": df["male"].map({0: "Mujer", 1: "Hombre"})
            + df["currentSmoker"].map({0: " · no fuma", 1: " · fuma"}),
            "evento": df[config.TARGET],
        }
    )
    tabla = aux.groupby(["perfil", "edad"], observed=True)["evento"].mean().mul(100).reset_index()
    estilos = {
        "Hombre · fuma": (ROJO, "solid"), "Hombre · no fuma": (ROJO, "dot"),
        "Mujer · fuma": ("#6366f1", "solid"), "Mujer · no fuma": ("#6366f1", "dot"),
    }
    fig = go.Figure()
    for perfil, (color, guion) in estilos.items():
        sub = tabla[tabla["perfil"] == perfil]
        fig.add_trace(
            go.Scatter(
                x=sub["edad"].astype(str), y=sub["evento"], mode="lines+markers", name=perfil,
                line={"color": color, "width": 3, "dash": guion}, marker={"size": 8},
                hovertemplate=f"{perfil}<br>%{{x}} años: %{{y:.1f}} %<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="% con evento en 10 años", ticksuffix=" %", rangemode="tozero")
    fig.update_xaxes(title_text="Edad")
    fig.update_layout(legend={"orientation": "h", "y": -0.22, "x": 0})
    return _estilo(fig, "Edad, sexo y tabaco a la vez",
                   "El riesgo crece con la edad y es mayor en hombres. Ojo: solo hay 84 mujeres fumadoras de 60-70 años",
                   altura=440)


# --- Sobre el modelo -------------------------------------------------------------------------
def figura_comparacion_modelos(comparacion: list[dict]) -> go.Figure:
    """AUC medio ± desviación en validación cruzada de cada modelo candidato."""
    tabla = pd.DataFrame(comparacion).sort_values("auc_media")
    fig = go.Figure(
        go.Bar(
            x=tabla["auc_media"],
            # El AUC va en la etiqueta del eje: dentro de la barra se taparía con la barra de error.
            y=[f"{m}  <b>{num(a, 3)}</b>" for m, a in zip(tabla["modelo"], tabla["auc_media"])],
            orientation="h",
            error_x={"type": "data", "array": tabla["auc_std"], "color": GRIS, "thickness": 1.5},
            marker={"color": [ROJO if m == "Regresión logística" else GRIS_CLARO for m in tabla["modelo"]],
                    "cornerradius": 6},
            customdata=tabla["auc_std"],
            hovertemplate="%{y}<br>± %{customdata:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="AUC en validación cruzada (0,5 = azar)", range=[0.5, 0.8])
    fig.update_layout(bargap=0.4)
    return _estilo(fig, "Comparación de modelos", "Mismas variables, misma validación cruzada", altura=290)


def figura_roc(metricas: dict) -> go.Figure:
    """Curvas ROC en test de las cuatro variantes del modelo."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"dash": "dot", "color": GRIS_CLARO})
    )
    for variante, m in metricas["variantes"].items():
        fig.add_trace(
            go.Scatter(
                x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
                name=f"{config.DESCRIPCION_VARIANTES[variante].capitalize()} ({num(m['auc'], 3)})",
                line={"color": COLORES_VARIANTES[variante], "width": 2.5 if variante == "completo" else 1.8},
            )
        )
    fig.update_xaxes(title_text="Falsos positivos", range=[0, 1])
    fig.update_yaxes(title_text="Verdaderos positivos", range=[0, 1.02])
    fig.update_layout(legend={"orientation": "h", "y": -0.28, "x": 0, "font": {"size": 11}})
    return _estilo(fig, "Curva ROC", "En test, para las cuatro versiones del modelo", altura=470)


def figura_calibracion(metricas: dict, variante: str = "completo") -> go.Figure:
    """Probabilidad predicha frente a frecuencia real, por deciles."""
    calib = metricas["variantes"][variante]["calibracion"]
    pred = 100 * np.array(calib["pred"])
    real = 100 * np.array(calib["real"])
    limite = float(max(pred.max(), real.max()) * 1.1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, limite], y=[0, limite], mode="lines", name="Calibración perfecta",
                   line={"dash": "dot", "color": GRIS_CLARO})
    )
    fig.add_trace(
        go.Scatter(x=pred, y=real, mode="lines+markers", name="Modelo",
                   line={"color": ROJO, "width": 3}, marker={"size": 8},
                   hovertemplate="Predicho %{x:.1f} % → real %{y:.1f} %<extra></extra>")
    )
    fig.update_xaxes(title_text="Riesgo predicho", range=[0, limite], ticksuffix=" %")
    fig.update_yaxes(title_text="Eventos observados", range=[0, limite], ticksuffix=" %")
    fig.update_layout(legend={"orientation": "h", "y": -0.28, "x": 0})
    return _estilo(fig, "Calibración", "¿El % que mostramos se cumple? Validación cruzada, por deciles", altura=470)


def figura_coeficientes(metricas: dict, variante: str = "completo") -> go.Figure:
    """Coeficientes estandarizados de la regresión logística (peso de cada variable)."""
    tabla = pd.DataFrame(metricas["coeficientes"][variante]).sort_values("coef_estandarizado")
    fig = go.Figure(
        go.Bar(
            x=tabla["coef_estandarizado"],
            y=tabla["nombre"],
            orientation="h",
            marker={"color": [ROJO if c > 0 else "#10b981" for c in tabla["coef_estandarizado"]],
                    "cornerradius": 5},
            customdata=np.stack([tabla["odds_ratio_por_unidad"], tabla["peso_pct"]], axis=1),
            hovertemplate="%{y}<br>Coef. estandarizado: %{x:.3f}<br>Odds ratio por unidad: %{customdata[0]:.3f}"
            "<br>Peso: %{customdata[1]:.1f} %<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Coeficiente estandarizado (log-odds por desviación típica)")
    fig.update_layout(bargap=0.3)
    return _estilo(fig, "Peso de cada variable", "Modelo completo. Rojo: sube el riesgo · verde: lo baja", altura=480)
