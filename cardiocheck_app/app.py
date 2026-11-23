"""CardioCheck: aplicación Dash para estimar el riesgo cardiovascular a 10 años.

Ejecución en local:   python app.py   ->  http://127.0.0.1:8050/
Producción (Render):  gunicorn --timeout 600 app:server
"""

from __future__ import annotations

import logging

from dash import Dash, Input, Output, State, dcc, html

from src import config, db, etl, graphics, model
from src.formato import num, pct
from src.formulario import procesar_formulario
from src.recomendaciones import generar_recomendaciones

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cardiocheck")


# --- Carga de modelos y datos (una sola vez, nunca dentro de un callback) ---------------
def cargar_recursos() -> tuple[dict, dict]:
    """Carga los modelos entrenados; si no existen, ejecuta el entrenamiento rápido."""
    try:
        return model.cargar_modelos(), model.cargar_metricas()
    except FileNotFoundError:
        log.warning("No hay modelos entrenados: ejecutando entrenamiento rápido (python train.py --rapido)")
        modelos, metricas = model.entrenar(etl.ejecutar_etl(), comparar=False)
        model.guardar(modelos, metricas)
        return modelos, metricas


MODELOS, METRICAS = cargar_recursos()
DATOS = etl.leer_de_bd()
db.crear_tabla_encuestas()

app = Dash(
    __name__,
    title="CardioCheck",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # necesario para gunicorn (Procfile / render.yaml)

AVISO_MEDICO = (
    "CardioCheck ofrece una estimación orientativa basada en un modelo estadístico. "
    "No sustituye un diagnóstico ni el consejo de un profesional sanitario."
)
OPCIONES_SI_NO = [{"label": "No", "value": 0}, {"label": "Sí", "value": 1}]


# --- Componentes reutilizables del formulario ----------------------------------------------
def campo_numero(id_: str, etiqueta: str, placeholder: str, ayuda: str | None = None) -> html.Div:
    """Campo numérico con etiqueta y texto de ayuda opcional.

    Sin min/max en el navegador: si el valor queda fuera de rango, dcc.Input lo
    convierte en None y el usuario no sabría por qué. Los rangos se validan en
    src/formulario.py con mensajes claros.
    """
    return html.Div(
        className="campo",
        children=[
            html.Label(etiqueta, htmlFor=id_),
            dcc.Input(id=id_, type="number", placeholder=placeholder, step="any", className="entrada"),
            html.Small(ayuda, className="ayuda") if ayuda else None,
        ],
    )


def campo_si_no(id_: str, etiqueta: str, ayuda: str | None = None) -> html.Div:
    """Pregunta de sí / no con botones grandes (cómodos en el móvil)."""
    return html.Div(
        className="campo",
        children=[
            html.Label(etiqueta),
            dcc.RadioItems(id=id_, options=OPCIONES_SI_NO, value=0, inline=True, className="radio",
                           inputClassName="radio-input", labelClassName="radio-opcion"),
            html.Small(ayuda, className="ayuda") if ayuda else None,
        ],
    )


def tarjeta(titulo: str | None, children, clase: str = "") -> html.Div:
    """Bloque con fondo blanco y sombra."""
    return html.Div(className=f"tarjeta {clase}".strip(),
                    children=([html.H3(titulo)] if titulo else []) + list(children))


def grafico(figura, id_: str | None = None) -> dcc.Graph:
    """dcc.Graph responsive y sin barra de herramientas."""
    kwargs = {"id": id_} if id_ else {}
    return dcc.Graph(figure=figura, config={"displayModeBar": False, "responsive": True}, **kwargs)


# --- Pestaña 1: calcular el riesgo -----------------------------------------------------------
formulario = tarjeta(
    None,
    [
        html.H2("Calcula tu riesgo"),
        html.P("Responde a unas preguntas sencillas (2 minutos). No necesitas análisis médicos.", className="intro"),
        html.H4("Sobre ti"),
        html.Div(className="fila", children=[
            campo_numero("in-edad", "Edad", "p. ej. 52"),
            html.Div(className="campo", children=[
                html.Label("Sexo"),
                dcc.RadioItems(id="in-sexo", options=[{"label": "Mujer", "value": 0}, {"label": "Hombre", "value": 1}],
                               value=None, inline=True, className="radio",
                               inputClassName="radio-input", labelClassName="radio-opcion"),
            ]),
        ]),
        html.Div(className="fila", children=[
            campo_numero("in-altura", "Altura (cm)", "p. ej. 172"),
            campo_numero("in-peso", "Peso (kg)", "p. ej. 75"),
        ]),
        html.H4("Hábitos"),
        campo_si_no("in-fumador", "¿Fumas actualmente?"),
        html.Div(id="bloque-cigarrillos", style={"display": "none"}, children=[
            campo_numero("in-cigarrillos", "¿Cuántos cigarrillos al día?", "p. ej. 10"),
        ]),
        html.H4("Antecedentes"),
        campo_si_no("in-hipertension", "¿Te han diagnosticado hipertensión (tensión alta)?"),
        html.Div(id="bloque-medicacion", style={"display": "none"}, children=[
            campo_si_no("in-medicacion", "¿Tomas medicación para la tensión?"),
        ]),
        campo_si_no("in-diabetes", "¿Tienes diabetes?"),
        campo_si_no("in-ictus", "¿Has tenido un ictus?"),
        html.Details(className="opcionales", children=[
            html.Summary("Opcional: añade tu tensión y tu colesterol para afinar la estimación"),
            html.P("Si conoces tus cifras (de un análisis o una toma reciente), la estimación será más precisa. "
                   "Si no, déjalo en blanco: usaremos un modelo que no las necesita.", className="ayuda"),
            html.Div(className="fila", children=[
                campo_numero("in-sistolica", "Tensión máxima (sistólica)", "p. ej. 130", ayuda="mmHg"),
                campo_numero("in-diastolica", "Tensión mínima (diastólica)", "p. ej. 85", ayuda="mmHg"),
            ]),
            campo_numero("in-colesterol", "Colesterol total", "p. ej. 210", ayuda="mg/dL"),
        ]),
        html.Button("Calcular mi riesgo", id="boton-calcular", n_clicks=0, className="boton"),
    ],
    clase="formulario",
)

pestana_calcular = html.Div(className="contenido", children=[
    formulario,
    dcc.Loading(html.Div(id="resultado"), type="circle", color=graphics.COLOR_PRINCIPAL),
    html.Div(AVISO_MEDICO, className="aviso"),
])


# --- Pestaña 2: explorar los datos ---------------------------------------------------------
def kpi(valor: str, etiqueta: str) -> html.Div:
    return html.Div(className="kpi", children=[html.Span(valor, className="kpi-valor"),
                                               html.Span(etiqueta, className="kpi-etiqueta")])


pestana_explorar = html.Div(className="contenido", children=[
    tarjeta(None, [
        html.H2("Explora los datos"),
        html.P(
            "El modelo aprende del Framingham Heart Study: personas de Framingham (EE. UU.) a las que se "
            "siguió durante 10 años para ver quién sufría un evento coronario (infarto, angina o muerte "
            "coronaria). Estos gráficos muestran el riesgo observado en ellas.",
            className="intro",
        ),
        html.Div(className="kpis", children=[
            kpi(num(len(DATOS), 0), "personas seguidas"),
            kpi(pct(DATOS[config.TARGET].mean()), "tuvo un evento en 10 años"),
            kpi(f"{DATOS['age'].mean():.0f} años", "edad media al inicio"),
            kpi(f"{100 * DATOS['currentSmoker'].mean():.0f} %", "fumaba"),
        ]),
    ]),
    tarjeta(None, [
        html.Label("Elige un factor", htmlFor="explorar-variable"),
        dcc.Dropdown(id="explorar-variable", value="edad", clearable=False, searchable=False,
                     options=[{"label": v, "value": k} for k, v in graphics.VARIABLES_EXPLORAR.items()]),
        grafico(graphics.figura_prevalencia(DATOS, "edad"), "explorar-figura"),
        html.Small(
            "Ojo: son datos brutos y algunas relaciones están mezcladas con la edad. Por ejemplo, los fumadores "
            "del estudio eran más jóvenes, por eso el tabaco parece pesar poco aquí. El gráfico de abajo separa "
            "por edad, y el modelo tiene en cuenta todos los factores a la vez.",
            className="ayuda",
        ),
    ]),
    tarjeta(None, [grafico(graphics.figura_edad_sexo_tabaco(DATOS))]),
])


# --- Pestaña 3: sobre el modelo -----------------------------------------------------------------
def tabla_variantes(metricas: dict) -> html.Table:
    """Tabla con el rendimiento en test de las cuatro variantes del modelo."""
    cabecera = html.Tr([html.Th(c) for c in
                        ["Modelo", "AUC (CV)", "AUC (test)", "Brier", "Recall ≥10 %", "Precisión ≥10 %"]])
    filas = []
    for variante, m in metricas["variantes"].items():
        mod = m["umbrales"]["moderado"]
        auc_cv = f"{num(m['auc_cv_media'], 3)} ± {num(m['auc_cv_std'], 3)}" if "auc_cv_media" in m else "–"
        filas.append(html.Tr([
            html.Td(config.DESCRIPCION_VARIANTES[variante].capitalize()),
            html.Td(auc_cv),
            html.Td(num(m["auc"], 3)),
            html.Td(num(m["brier"], 3)),
            html.Td(f"{100 * mod['recall']:.0f} %"),
            html.Td(f"{100 * mod['precision']:.0f} %"),
        ]))
    return html.Table(className="tabla", children=[html.Thead(cabecera), html.Tbody(filas)])


bloques_modelo = [
    tarjeta(None, [
        html.H2("Sobre el modelo"),
        html.P(
            f"Regresión logística entrenada con {num(METRICAS['n_total'], 0)} personas del Framingham Heart Study "
            f"({pct(METRICAS['prevalencia'])} tuvo un evento coronario en 10 años). Evaluación con un "
            f"{int(100 * config.TEST_SIZE)} % de los datos reservado como test y validación cruzada.",
            className="intro",
        ),
        html.Ul([
            html.Li("Se eligió la regresión logística porque iguala o supera a Random Forest y Gradient Boosting "
                    "en estos datos, da probabilidades bien calibradas y permite explicar cada resultado."),
            html.Li("Hay cuatro versiones del modelo según conozcas o no tu tensión y tu colesterol: nunca "
                    "inventamos un valor que no nos has dado."),
            html.Li(f"Niveles de riesgo a 10 años: bajo (< {100 * config.UMBRAL_MODERADO:.0f} %), moderado "
                    f"({100 * config.UMBRAL_MODERADO:.0f}-{100 * config.UMBRAL_ALTO:.0f} %) y alto "
                    f"(≥ {100 * config.UMBRAL_ALTO:.0f} %)."),
        ]),
    ]),
]
if "comparacion_modelos" in METRICAS:
    bloques_modelo.append(tarjeta(None, [grafico(graphics.figura_comparacion_modelos(METRICAS["comparacion_modelos"]))]))
bloques_modelo += [
    tarjeta("Rendimiento en test", [
        tabla_variantes(METRICAS),
        html.Small("AUC: capacidad de ordenar a las personas por riesgo (0,5 = azar, 1 = perfecto). "
                   "AUC (CV): media de validación cruzada 5x5, más estable que una sola partición de test. "
                   "Brier: error de las probabilidades (menor es mejor; predecir siempre la media da "
                   f"{num(METRICAS['variantes']['base']['brier_baseline'], 3)}). Recall: % de personas que tuvieron "
                   "un evento y la app habría marcado como riesgo moderado o alto.", className="ayuda"),
    ]),
    tarjeta(None, [grafico(graphics.figura_roc(METRICAS))]),
    tarjeta(None, [grafico(graphics.figura_calibracion(METRICAS))]),
    tarjeta(None, [
        grafico(graphics.figura_coeficientes(METRICAS)),
        html.Small(
            "Cada barra es el efecto de la variable manteniendo las demás fijas. La tensión diastólica sale "
            "ligeramente negativa porque va muy unida a la sistólica (el modelo reparte el efecto entre las dos), "
            "y «Fumador» pesa poco porque el efecto del tabaco ya lo recoge «Cigarrillos al día».",
            className="ayuda",
        ),
    ]),
    tarjeta("Limitaciones", [html.Ul([
        html.Li("Datos de una sola ciudad de EE. UU. y de una cohorte antigua: el riesgo cardiovascular en la "
                "España actual es menor, así que el modelo tiende a sobrestimarlo."),
        html.Li(f"Solo incluye personas de {config.EDAD_MIN_MODELO} a {config.EDAD_MAX_MODELO} años."),
        html.Li("Con unas 4.200 personas, las métricas tienen un margen de error de unos ±0,02 de AUC."),
        html.Li("El modelo estima asociaciones estadísticas; las recomendaciones se basan en guías de prevención, "
                "no en los coeficientes."),
    ])]),
]
pestana_modelo = html.Div(className="contenido", children=bloques_modelo)

pestana_uso = html.Div(className="contenido", children=[
    dcc.Loading(html.Div(id="contenido-uso"), type="circle", color=graphics.COLOR_PRINCIPAL),
])


# --- Layout ------------------------------------------------------------------------------------
app.layout = html.Div(className="pagina", children=[
    html.Header(className="cabecera", children=[
        html.Div("❤", className="logo"),
        html.Div([html.H1("CardioCheck"),
                  html.P("Estima tu riesgo cardiovascular a 10 años en 2 minutos")]),
    ]),
    dcc.Tabs(id="pestanas", value="calcular", className="pestanas", mobile_breakpoint=0, children=[
        dcc.Tab(label="Calcular", value="calcular", children=pestana_calcular,
                className="pestana", selected_className="pestana-activa"),
        dcc.Tab(label="Datos", value="explorar", children=pestana_explorar,
                className="pestana", selected_className="pestana-activa"),
        dcc.Tab(label="Modelo", value="modelo", children=pestana_modelo,
                className="pestana", selected_className="pestana-activa"),
        dcc.Tab(label="Uso", value="uso", children=pestana_uso,
                className="pestana", selected_className="pestana-activa"),
    ]),
    html.Footer("CardioCheck · Desarrollo de Aplicaciones para la Visualización de Datos · Pablo Hebrero",
                className="pie"),
])


# --- Callbacks ---------------------------------------------------------------------------------
@app.callback(Output("bloque-cigarrillos", "style"), Input("in-fumador", "value"))
def mostrar_cigarrillos(fumador):
    """Solo se pregunta por los cigarrillos si la persona fuma."""
    return {"display": "block"} if fumador == 1 else {"display": "none"}


@app.callback(Output("bloque-medicacion", "style"), Input("in-hipertension", "value"))
def mostrar_medicacion(hipertension):
    """Solo se pregunta por la medicación si hay hipertensión diagnosticada."""
    return {"display": "block"} if hipertension == 1 else {"display": "none"}


def construir_resultado(datos: dict, prob: float, variante: str, nivel: str, avisos: list[str]) -> html.Div:
    """Bloque de resultado: porcentaje, nivel, factores, escenarios y recomendaciones."""
    color = config.COLORES_NIVEL[nivel]
    bloques = [
        html.Div(className="resultado-cabecera", children=[
            html.Span(pct(prob), className="resultado-porcentaje", style={"color": color}),
            html.Span(f"Riesgo {nivel}", className="etiqueta-nivel", style={"backgroundColor": color}),
        ]),
        html.P("Probabilidad estimada de sufrir un evento coronario (infarto, angina o muerte coronaria) "
               "en los próximos 10 años.", className="intro"),
        grafico(graphics.figura_gauge(prob, nivel)),
        html.P([f"Estimación con {config.DESCRIPCION_VARIANTES[variante]}. "] +
               ([html.Strong("Si conoces tu tensión o tu colesterol, añádelos para afinarla.")]
                if variante != "completo" else []),
               className="ayuda"),
    ]
    bloques += [html.Div(a, className="aviso-edad") for a in avisos]
    bloques.append(grafico(graphics.figura_factores(model.contribuciones(MODELOS, datos))))
    bloques.append(html.Small("Comparado con la persona media del estudio. Muestra asociaciones estadísticas, "
                              "no causas.", className="ayuda"))
    escenarios = model.escenarios(MODELOS, datos)
    if escenarios:
        bloques.append(grafico(graphics.figura_escenarios(prob, escenarios)))
    bloques += [
        html.H3("Recomendaciones"),
        html.Ul([html.Li(r) for r in generar_recomendaciones(datos, nivel)], className="recomendaciones"),
    ]
    return tarjeta(None, bloques, clase=f"resultado nivel-{nivel}")


@app.callback(
    Output("resultado", "children"),
    Input("boton-calcular", "n_clicks"),
    State("in-edad", "value"), State("in-sexo", "value"), State("in-altura", "value"),
    State("in-peso", "value"), State("in-fumador", "value"), State("in-cigarrillos", "value"),
    State("in-hipertension", "value"), State("in-medicacion", "value"), State("in-diabetes", "value"),
    State("in-ictus", "value"), State("in-sistolica", "value"), State("in-diastolica", "value"),
    State("in-colesterol", "value"),
    prevent_initial_call=True,
)
def calcular_riesgo(n_clicks, edad, sexo, altura, peso, fumador, cigarrillos, hipertension, medicacion,
                    diabetes, ictus, sistolica, diastolica, colesterol):
    """Valida el formulario, predice, guarda la encuesta y muestra el resultado."""
    respuestas = dict(edad=edad, sexo=sexo, altura=altura, peso=peso, fumador=fumador, cigarrillos=cigarrillos,
                      hipertension=hipertension, medicacion=medicacion, diabetes=diabetes, ictus=ictus,
                      sistolica=sistolica, diastolica=diastolica, colesterol=colesterol)
    datos, errores, avisos = procesar_formulario(respuestas)
    if errores:
        return tarjeta("Revisa el formulario", [html.Ul([html.Li(e) for e in errores])], clase="errores")

    prob, variante = model.predecir(MODELOS, datos)
    nivel = model.nivel_riesgo(prob)
    try:
        db.guardar_encuesta(datos, variante, prob, nivel)
    except Exception:  # guardar nunca debe impedir mostrar el resultado
        log.exception("No se pudo guardar la encuesta")
    return construir_resultado(datos, prob, variante, nivel, avisos)


@app.callback(Output("explorar-figura", "figure"), Input("explorar-variable", "value"))
def actualizar_exploracion(variable):
    """Cambia el gráfico de exploración según el factor elegido."""
    return graphics.figura_prevalencia(DATOS, variable)


@app.callback(Output("contenido-uso", "children"), Input("pestanas", "value"))
def actualizar_uso(pestana):
    """Estadísticas de las encuestas guardadas (se recalculan al abrir la pestaña)."""
    if pestana != "uso":
        return None
    encuestas = db.leer_encuestas()
    cabecera = tarjeta(None, [
        html.H2("Uso de la app"),
        html.P("Cada cálculo se guarda de forma anónima (sin nombre ni datos de contacto) en la base de datos "
               "de la aplicación.", className="intro"),
        html.Div(className="kpis", children=[
            kpi(str(len(encuestas)), "encuestas"),
            kpi(pct(encuestas["probabilidad"].mean()) if len(encuestas) else "–", "riesgo medio"),
            kpi(f"{encuestas['age'].mean():.0f} años" if len(encuestas) else "–", "edad media"),
        ]),
    ])
    if encuestas.empty:
        return [cabecera, tarjeta(None, [html.P("Todavía no hay encuestas. ¡Calcula tu riesgo para estrenar la base de datos!")])]
    return [
        cabecera,
        tarjeta(None, [grafico(graphics.figura_niveles_encuestas(encuestas))]),
        tarjeta(None, [grafico(graphics.figura_distribucion_encuestas(encuestas))]),
    ]


if __name__ == "__main__":
    app.run(debug=True)
