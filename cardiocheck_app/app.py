"""CardioCheck: aplicación Dash para estimar el riesgo cardiovascular a 10 años.

Ejecución en local:   python app.py   ->  http://127.0.0.1:8050/
Producción (Render):  gunicorn --timeout 600 app:server
"""

from __future__ import annotations

import logging

from dash import Dash, Input, Output, State, dcc, html

from src import config, db, etl, graphics, model
from src.formato import num, pct
from src.formulario import RANGOS, calcular_imc, procesar_formulario
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
    title="CardioCheck · Tu riesgo cardiovascular",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Estima tu riesgo cardiovascular a 10 años en 2 minutos."},
        {"name": "theme-color", "content": "#e11d48"},
    ],
)
server = app.server  # necesario para gunicorn (Procfile / render.yaml)

AVISO_MEDICO = (
    "CardioCheck ofrece una estimación orientativa basada en un modelo estadístico. "
    "No sustituye un diagnóstico ni el consejo de un profesional sanitario."
)
SI_NO = [{"label": "No", "value": 0}, {"label": "Sí", "value": 1}]
LOGO = app.get_asset_url("logo.svg")


# --- Componentes reutilizables --------------------------------------------------------------
def tarjeta(children, clase: str = "") -> html.Div:
    """Bloque blanco con borde suave y sombra."""
    return html.Div(className=f"tarjeta {clase}".strip(), children=children)


def grafico(figura, id_: str | None = None) -> dcc.Graph:
    """dcc.Graph responsive y sin barra de herramientas."""
    kwargs = {"id": id_} if id_ else {}
    return dcc.Graph(figure=figura, className="grafico",
                     config={"displayModeBar": False, "responsive": True}, **kwargs)


def kpi(valor: str, etiqueta: str, icono: str) -> html.Div:
    """Indicador numérico grande con icono."""
    return html.Div(className="kpi", children=[
        html.Span(icono, className="kpi-icono"),
        html.Span(valor, className="kpi-valor"),
        html.Span(etiqueta, className="kpi-etiqueta"),
    ])


def campo_numero(id_: str, etiqueta: str, placeholder: str, unidad: str | None = None) -> html.Div:
    """Campo numérico con unidad dentro de la caja.

    Sin min/max en el navegador: si el valor queda fuera de rango, dcc.Input lo
    convierte en None y el usuario no sabría por qué. Los rangos se validan en
    src/formulario.py con mensajes claros.
    """
    return html.Div(className="campo", children=[
        html.Label(etiqueta, htmlFor=id_, className="campo-etiqueta"),
        html.Div(className="entrada-grupo", children=[
            dcc.Input(id=id_, type="number", placeholder=placeholder, step="any", className="entrada"),
            html.Span(unidad, className="unidad") if unidad else None,
        ]),
    ])


def campo_opciones(id_: str, etiqueta: str, opciones: list[dict] = SI_NO, valor=0) -> html.Div:
    """Pregunta de opción única con botones segmentados (cómodos en el móvil)."""
    return html.Div(className="campo", children=[
        html.Span(etiqueta, className="campo-etiqueta"),
        dcc.RadioItems(id=id_, options=opciones, value=valor, inline=True, className="segmentado"),
    ])


def paso(numero: int, titulo: str, children: list, opcional: bool = False) -> html.Section:
    """Bloque numerado del formulario."""
    return html.Section(className="paso", children=[
        html.Div(className="paso-cabecera", children=[
            html.Span(str(numero), className="paso-numero"),
            html.H3(titulo),
            html.Span("Opcional", className="insignia-opcional") if opcional else None,
        ]),
        *children,
    ])


# --- Cabecera y navegación -------------------------------------------------------------------
cabecera = html.Header(className="cabecera", children=html.Div(className="contenedor cabecera-interior", children=[
    html.Div(className="marca", children=[
        html.Img(src=LOGO, alt="", className="marca-logo"),
        html.Span("CardioCheck", className="marca-nombre"),
    ]),
    dcc.RadioItems(
        id="nav", value="calcular", inline=True, className="nav",
        options=[
            {"label": "Calcular", "value": "calcular"},
            {"label": "Explorar datos", "value": "datos"},
            {"label": "El modelo", "value": "modelo"},
        ],
    ),
]))


# --- Sección 1: calculadora ------------------------------------------------------------------
def portada() -> html.Div:
    """Portada con efecto parallax: 100 personas se «encienden» al hacer scroll hasta llegar a las que
    tuvieron un evento coronario en el estudio. assets/parallax.js mueve las capas y enciende las figuras;
    el número objetivo sale de los datos (data-objetivo)."""
    objetivo = round(100 * DATOS[config.TARGET].mean())
    return html.Div(id="portada", className="portada", **{"data-objetivo": str(objetivo)}, children=html.Div(
        className="portada-escena", children=[
            html.Div(className="portada-fondo", children=[
                html.Span(className="burbuja burbuja-1"), html.Span(className="burbuja burbuja-2"),
            ]),
            html.Div(className="contenedor portada-contenido", children=[
                html.Div(className="portada-texto", children=[
                    html.Span("Basado en el Framingham Heart Study", className="hero-insignia"),
                    html.H1(["Tu corazón, ", html.Span("en cifras", className="resaltado")]),
                    html.Div(className="portada-cifra", children=[
                        html.Span("0", className="portada-contador"),
                        html.Span(["de cada", html.Br(), "100 personas"], className="portada-unidad"),
                    ]),
                    html.P("tendrán un infarto, una angina o una muerte coronaria en los próximos 10 años.",
                           className="portada-frase"),
                    html.Div(className="portada-final", children=[
                        html.H2("¿Serás una de ellas?"),
                        html.A(["Calcula tu riesgo en 2 minutos", html.Span("↓", className="boton-flecha")],
                               href="#calculadora", className="boton-principal portada-boton"),
                    ]),
                ]),
                html.Div(className="portada-personas", **{"aria-hidden": "true"},
                         children=[html.Span(className="persona-figura") for _ in range(100)]),
            ]),
            html.Div(className="portada-desliza", children=["Desliza", html.Span("↓")]),
        ],
    ))


cabecera_calculadora = html.Div(id="calculadora", className="cabecera-seccion cabecera-calculadora", children=[
    html.H2(["Calcula tu ", html.Span("riesgo cardiovascular", className="resaltado")]),
    html.P("Responde unas preguntas sencillas, sin análisis médicos, y descubre tu probabilidad de tener un problema "
           "de corazón en los próximos 10 años y qué puedes hacer para reducirla."),
    html.Div(className="hero-puntos", children=[
        html.Span("⏱️ 2 minutos"), html.Span("🔒 Anónimo"), html.Span("📊 Resultado explicado"),
    ]),
])

formulario = tarjeta([
    paso(1, "Sobre ti", [
        html.Div(className="rejilla-2", children=[
            campo_numero("in-edad", "Edad", "52", "años"),
            campo_opciones("in-sexo", "Sexo", [{"label": "Mujer", "value": 0}, {"label": "Hombre", "value": 1}],
                           valor=None),
        ]),
        html.Div(className="rejilla-2", children=[
            campo_numero("in-altura", "Altura", "172", "cm"),
            campo_numero("in-peso", "Peso", "75", "kg"),
        ]),
        html.Div(id="imc-vivo"),
    ]),
    paso(2, "Hábitos", [
        campo_opciones("in-fumador", "¿Fumas actualmente?"),
        html.Div(id="bloque-cigarrillos", style={"display": "none"}, children=[
            campo_numero("in-cigarrillos", "¿Cuántos cigarrillos al día?", "10", "al día"),
        ]),
    ]),
    paso(3, "Tu salud", [
        campo_opciones("in-hipertension", "¿Te han diagnosticado hipertensión (tensión alta)?"),
        html.Div(id="bloque-medicacion", style={"display": "none"}, children=[
            campo_opciones("in-medicacion", "¿Tomas medicación para la tensión?"),
        ]),
        campo_opciones("in-diabetes", "¿Tienes diabetes?"),
        campo_opciones("in-ictus", "¿Has tenido un ictus?"),
    ]),
    paso(4, "Si conoces tus cifras", [
        html.P("De un análisis o una toma reciente. Si no las sabes, déjalas en blanco: usaremos un modelo que "
               "no las necesita.", className="ayuda"),
        html.Div(className="rejilla-2", children=[
            campo_numero("in-sistolica", "Tensión máxima", "130", "mmHg"),
            campo_numero("in-diastolica", "Tensión mínima", "85", "mmHg"),
        ]),
        campo_numero("in-colesterol", "Colesterol total", "210", "mg/dL"),
    ], opcional=True),
    html.Button(["Calcular mi riesgo", html.Span("→", className="boton-flecha")],
                id="boton-calcular", n_clicks=0, className="boton-principal"),
], clase="formulario")


def resultado_vacio() -> html.Div:
    """Lo que se ve en la columna de resultado antes de calcular."""
    return tarjeta([
        html.Img(src=LOGO, alt="", className="vacio-logo"),
        html.H3("Tu resultado aparecerá aquí"),
        html.P("Completa el formulario y pulsa «Calcular mi riesgo». Obtendrás:", className="ayuda"),
        html.Ul(className="lista-check", children=[
            html.Li("Tu probabilidad de evento coronario a 10 años"),
            html.Li("Cómo te comparas con personas de tu edad"),
            html.Li("Qué factores pesan más en tu caso"),
            html.Li("Cuánto podrías reducirlo y recomendaciones"),
        ]),
    ], clase="resultado-vacio")


pagina_calcular = html.Section(id="sec-calcular", children=[
    portada(),
    cabecera_calculadora,
    html.Div(className="calculadora", children=[
        formulario,
        html.Div(className="columna-resultado", children=[
            dcc.Loading(html.Div(id="resultado", children=resultado_vacio()), type="dot",
                        color=graphics.ROJO),
            html.P(AVISO_MEDICO, className="aviso"),
        ]),
    ]),
])


# --- Bloques del resultado -------------------------------------------------------------------
def escala_riesgo(prob: float) -> html.Div:
    """Barra de colores bajo / moderado / alto con un marcador en la posición del usuario (escala 0-40 %)."""
    posicion = min(prob, 0.4) / 0.4 * 100
    return html.Div(className="escala", children=[
        html.Div(className="escala-barra", children=[
            html.Div(className="tramo tramo-bajo"),
            html.Div(className="tramo tramo-moderado"),
            html.Div(className="tramo tramo-alto"),
            html.Div(className="escala-marcador", style={"left": f"{posicion:.1f}%"}),
        ]),
        html.Div(className="escala-etiquetas", children=[
            html.Span("Bajo · <10 %", className="tramo-bajo-texto"),
            html.Span("Moderado", className="tramo-moderado-texto"),
            html.Span("Alto · ≥20 %", className="tramo-alto-texto"),
        ]),
    ])


def matriz_personas(prob: float, nivel: str) -> html.Div:
    """Matriz de 100 personas: forma recomendada de comunicar un riesgo en salud."""
    afectadas = min(max(round(100 * prob), 0), 100)
    texto_n = html.Strong(str(afectadas)) if afectadas > 0 else html.Strong("menos de 1")
    return html.Div(className="matriz", children=[
        html.Div(className=f"matriz-rejilla nivel-{nivel}",
                 children=[html.Span(className="persona activa" if i < afectadas else "persona") for i in range(100)]),
        html.P(["De cada 100 personas con tu perfil, ", texto_n,
                " tendrían un infarto, una angina o una muerte coronaria en los próximos 10 años."],
               className="matriz-texto"),
    ])


def comparativa(datos: dict) -> html.Div:
    """Riesgo con factores óptimos y riesgo observado en personas parecidas del estudio."""
    optimo = model.riesgo_optimo(MODELOS, datos)
    similar, n_similar = model.riesgo_similares(DATOS, datos["age"], datos["male"])
    edad = int(round(datos["age"]))
    sexo = "hombres" if datos["male"] == 1 else "mujeres"
    bloques = [html.Div(className="comparativa-dato", children=[
        html.Span(pct(optimo), className="comparativa-valor"),
        html.Span("Con factores óptimos", className="comparativa-titulo"),
        html.Span("Tu edad y sexo, sin tabaco, hipertensión ni diabetes", className="comparativa-detalle"),
    ])]
    if n_similar >= 30:
        bloques.append(html.Div(className="comparativa-dato", children=[
            html.Span(pct(similar), className="comparativa-valor"),
            html.Span("Personas como tú", className="comparativa-titulo"),
            html.Span(f"tuvo un evento entre los {num(n_similar, 0)} {sexo} de {edad - 5}-{edad + 5} años del estudio",
                      className="comparativa-detalle"),
        ]))
    return html.Div(className="comparativa", children=bloques)


def lista_factores(contrib) -> html.Div:
    """Barras propias: cuánto sube o baja cada factor el riesgo frente a la persona media del estudio."""
    relevantes = contrib[contrib.abs() > 0.02]
    if relevantes.empty:
        return html.P("Tus factores están muy cerca de la media del estudio.", className="ayuda")
    maximo = relevantes.abs().max()
    filas = []
    for nombre, valor in relevantes.items():
        sube = valor > 0
        filas.append(html.Div(className="factor", children=[
            html.Span(nombre, className="factor-nombre"),
            html.Div(className="factor-pista", children=html.Div(
                className=f"factor-barra {'sube' if sube else 'baja'}",
                style={"width": f"{max(6, 100 * abs(valor) / maximo):.0f}%"},
            )),
            html.Span("↑ sube" if sube else "↓ baja", className=f"factor-efecto {'sube' if sube else 'baja'}"),
        ]))
    return html.Div(className="factores", children=filas)


def tarjetas_escenarios(prob: float, escenarios: list[dict]) -> html.Div:
    """Tarjetas «¿y si…?» con el riesgo actual, el nuevo y la diferencia."""
    iconos = {"Si dejaras de fumar": "🚭", "Con la tensión en 120/80": "📉"}
    return html.Div(className="escenarios", children=[
        html.Div(className="escenario", children=[
            html.Span(iconos.get(e["escenario"], "💡"), className="escenario-icono"),
            html.Div(className="escenario-cuerpo", children=[
                html.Span(e["escenario"], className="escenario-titulo"),
                html.Span([pct(prob), " → ", html.Strong(pct(e["prob"]))], className="escenario-valores"),
            ]),
            html.Span(f"−{num(-100 * e['diferencia'])} pts", className="escenario-delta"),
        ])
        for e in escenarios
    ])


def construir_resultado(datos: dict, prob: float, variante: str, nivel: str, avisos: list[str]) -> html.Div:
    """Resultado completo: cifra, escala, matriz, comparativa, factores, escenarios y recomendaciones."""
    nota_variante = [f"Estimación con {config.DESCRIPCION_VARIANTES[variante]}."]
    faltan = {"base": "tu tensión y tu colesterol", "tension": "tu colesterol", "colesterol": "tu tensión"}
    if variante in faltan:
        nota_variante += [" ", html.Strong(f"Si conoces {faltan[variante]}, añádelo para afinarla.")]

    principal = tarjeta([
        html.Span("Tu riesgo cardiovascular a 10 años", className="sobretitulo"),
        html.Div(className="resultado-cifra", children=[
            html.Span(pct(prob), className=f"cifra nivel-{nivel}"),
            html.Span(f"Riesgo {nivel}", className=f"insignia-nivel nivel-{nivel}"),
        ]),
        escala_riesgo(prob),
        matriz_personas(prob, nivel),
        comparativa(datos),
        *[html.Div(["⚠️ ", aviso], className="aviso-edad") for aviso in avisos],
        html.P(nota_variante, className="nota-modelo"),
    ], clase=f"resultado nivel-{nivel}")

    bloques = [
        principal,
        tarjeta([
            html.H3("Qué pesa más en tu caso"),
            html.P("Comparado con la persona media del estudio. Son asociaciones estadísticas, no causas.",
                   className="ayuda"),
            lista_factores(model.contribuciones(MODELOS, datos)),
        ]),
    ]
    escenarios = model.escenarios(MODELOS, datos)
    if escenarios:
        bloques.append(tarjeta([html.H3("¿Y si cambiaras…?"), tarjetas_escenarios(prob, escenarios)]))
    bloques.append(tarjeta([
        html.H3("Recomendaciones"),
        html.Div(className="recomendaciones", children=[
            html.Div(className="recomendacion", children=[
                html.Span(r["icono"], className="rec-icono"),
                html.Div([html.Strong(r["titulo"]), html.P(r["texto"])]),
            ])
            for r in generar_recomendaciones(datos, nivel)
        ]),
    ]))
    return html.Div(className="resultado-pila", children=bloques)


# --- Sección 2: explorar los datos ---------------------------------------------------------------
pagina_datos = html.Section(id="sec-datos", style={"display": "none"}, children=[
    html.Div(className="cabecera-seccion", children=[
        html.H2("Explora los datos"),
        html.P("El modelo aprende del Framingham Heart Study: vecinos de Framingham (EE. UU.) a los que se siguió "
               "durante 10 años para ver quién sufría un evento coronario (infarto, angina o muerte coronaria)."),
    ]),
    html.Div(className="kpis", children=[
        kpi(num(len(DATOS), 0), "personas seguidas 10 años", "👥"),
        kpi(pct(DATOS[config.TARGET].mean()), "tuvo un evento coronario", "🫀"),
        kpi(f"{DATOS['age'].mean():.0f} años", "edad media al empezar", "🎂"),
        kpi(pct(DATOS["currentSmoker"].mean(), 0), "fumaba", "🚬"),
    ]),
    tarjeta([
        html.Span("Elige un factor", className="campo-etiqueta"),
        dcc.RadioItems(
            id="explorar-variable", value="edad", inline=True, className="chips",
            options=[{"label": v, "value": k} for k, v in graphics.VARIABLES_EXPLORAR.items()],
        ),
        grafico(graphics.figura_prevalencia(DATOS, "edad"), "explorar-figura"),
        html.P(
            "Ojo: son datos brutos y algunas relaciones se mezclan con la edad. Por ejemplo, los fumadores del "
            "estudio eran más jóvenes, por eso el tabaco parece pesar poco aquí. El gráfico de abajo separa por "
            "edad, y el modelo tiene en cuenta todos los factores a la vez.",
            className="nota",
        ),
    ]),
    tarjeta([grafico(graphics.figura_edad_sexo_tabaco(DATOS))]),
])


# --- Sección 3: el modelo ------------------------------------------------------------------------
def tabla_variantes(metricas: dict) -> html.Div:
    """Tabla con el rendimiento de las cuatro variantes del modelo."""
    cabecera = html.Tr([html.Th(c) for c in
                        ["Datos usados", "AUC (CV)", "AUC (test)", "Brier", "Recall ≥10 %", "Precisión ≥10 %"]])
    filas = []
    for variante, m in metricas["variantes"].items():
        mod = m["umbrales"]["moderado"]
        auc_cv = f"{num(m['auc_cv_media'], 3)} ± {num(m['auc_cv_std'], 3)}" if "auc_cv_media" in m else "–"
        filas.append(html.Tr([
            html.Td(config.DESCRIPCION_VARIANTES[variante].capitalize()),
            html.Td(auc_cv),
            html.Td(num(m["auc"], 3)),
            html.Td(num(m["brier"], 3)),
            html.Td(pct(mod["recall"], 0)),
            html.Td(pct(mod["precision"], 0)),
        ]))
    return html.Div(className="tabla-scroll", children=html.Table(
        className="tabla", children=[html.Thead(cabecera), html.Tbody(filas)]))


METRICAS_BASE = METRICAS["variantes"]["base"]
bloques_modelo = [
    html.Div(className="cabecera-seccion", children=[
        html.H2("Cómo funciona el modelo"),
        html.P(f"Una regresión logística entrenada con {num(METRICAS['n_total'], 0)} personas, evaluada con "
               f"validación cruzada y con un {int(100 * config.TEST_SIZE)} % de los datos reservado como test."),
    ]),
    html.Div(className="kpis", children=[
        kpi(num(METRICAS_BASE.get("auc_cv_media", METRICAS_BASE["auc"]), 2), "AUC con los datos básicos", "🎯"),
        kpi(pct(METRICAS_BASE["umbrales"]["moderado"]["recall"], 0), "de los eventos, marcados como riesgo ≥10 %",
            "🔎"),
        kpi("4", "versiones según los datos que conozcas", "🧩"),
        kpi(num(METRICAS["n_total"], 0), "personas para entrenar", "👥"),
    ]),
    tarjeta([
        html.H3("En tres pasos"),
        html.Div(className="pasos-modelo", children=[
            html.Div(className="paso-modelo", children=[
                html.Span("1", className="paso-numero"),
                html.Strong("Tus respuestas"),
                html.P("Se convierten a las variables del estudio: el IMC sale de tu altura y peso, por ejemplo."),
            ]),
            html.Div(className="paso-modelo", children=[
                html.Span("2", className="paso-numero"),
                html.Strong("El modelo adecuado"),
                html.P("Hay 4 versiones según conozcas o no tu tensión y tu colesterol: nunca inventamos un dato "
                       "que no nos has dado."),
            ]),
            html.Div(className="paso-modelo", children=[
                html.Span("3", className="paso-numero"),
                html.Strong("Probabilidad y nivel"),
                html.P(f"Riesgo bajo (<{pct(config.UMBRAL_MODERADO, 0)}), moderado o alto "
                       f"(≥{pct(config.UMBRAL_ALTO, 0)}), los cortes clásicos de las escalas de Framingham."),
            ]),
        ]),
    ]),
]
if "comparacion_modelos" in METRICAS:
    bloques_modelo.append(tarjeta([
        grafico(graphics.figura_comparacion_modelos(METRICAS["comparacion_modelos"])),
        html.P("La regresión logística iguala o supera a modelos más complejos, da probabilidades bien calibradas "
               "y permite explicar cada resultado.", className="nota"),
    ]))
bloques_modelo += [
    tarjeta([
        html.H3("Rendimiento"),
        tabla_variantes(METRICAS),
        html.P("AUC: capacidad de ordenar a las personas por riesgo (0,5 = azar, 1 = perfecto); en CV, media de "
               "validación cruzada 5x5. Brier: error de las probabilidades (menor es mejor; predecir siempre la media "
               f"da {num(METRICAS_BASE['brier_baseline'], 3)}). Recall: % de las personas que tuvieron un evento a "
               "las que la app habría dado riesgo moderado o alto.", className="nota"),
    ]),
    html.Div(className="rejilla-graficos", children=[
        tarjeta([grafico(graphics.figura_roc(METRICAS))]),
        tarjeta([grafico(graphics.figura_calibracion(METRICAS))]),
    ]),
    tarjeta([
        grafico(graphics.figura_coeficientes(METRICAS)),
        html.P("Cada barra es el efecto de la variable manteniendo las demás fijas. La tensión diastólica sale "
               "ligeramente negativa porque va muy unida a la sistólica (el modelo reparte el efecto entre las dos), "
               "y «Fumador» pesa poco porque el efecto del tabaco ya lo recoge «Cigarrillos al día».", className="nota"),
    ]),
    tarjeta([
        html.H3("Limitaciones"),
        html.Ul(className="lista-limitaciones", children=[
            html.Li("Datos de una sola ciudad de EE. UU. y de una cohorte antigua: el riesgo cardiovascular en la "
                    "España actual es menor, así que el modelo tiende a sobrestimarlo."),
            html.Li(f"Solo incluye personas de {config.EDAD_MIN_MODELO} a {config.EDAD_MAX_MODELO} años."),
            html.Li("Con unas 4.200 personas, las métricas tienen un margen de error de unos ±0,02 de AUC."),
            html.Li("El modelo estima asociaciones estadísticas; las recomendaciones se basan en guías de "
                    "prevención, no en los coeficientes."),
        ]),
    ], clase="limitaciones"),
]
pagina_modelo = html.Section(id="sec-modelo", style={"display": "none"}, children=bloques_modelo)


# --- Layout --------------------------------------------------------------------------------------
app.layout = html.Div(className="app", children=[
    cabecera,
    html.Main(className="contenedor", children=[pagina_calcular, pagina_datos, pagina_modelo]),
    html.Footer(className="pie", children=html.Div(className="contenedor", children=[
        html.Div(className="marca", children=[
            html.Img(src=LOGO, alt="", className="marca-logo pequeno"),
            html.Span("CardioCheck", className="marca-nombre"),
        ]),
        html.P("Proyecto de Desarrollo de Aplicaciones para la Visualización de Datos · Pablo Hebrero"),
        html.P(AVISO_MEDICO, className="pie-aviso"),
    ])),
    dcc.Store(id="scroll-resultado"),
    dcc.Store(id="scroll-nav"),
])


# --- Callbacks -----------------------------------------------------------------------------------
SECCIONES = ("calcular", "datos", "modelo")


@app.callback(
    [Output(f"sec-{s}", "style") for s in SECCIONES],
    Input("nav", "value"),
)
def navegar(seccion):
    """Muestra solo la sección elegida en la navegación."""
    return [{"display": "block"} if s == seccion else {"display": "none"} for s in SECCIONES]


@app.callback(Output("bloque-cigarrillos", "style"), Input("in-fumador", "value"))
def mostrar_cigarrillos(fumador):
    """Solo se pregunta por los cigarrillos si la persona fuma."""
    return {"display": "block"} if fumador == 1 else {"display": "none"}


@app.callback(Output("bloque-medicacion", "style"), Input("in-hipertension", "value"))
def mostrar_medicacion(hipertension):
    """Solo se pregunta por la medicación si hay hipertensión diagnosticada."""
    return {"display": "block"} if hipertension == 1 else {"display": "none"}


@app.callback(Output("imc-vivo", "children"), Input("in-altura", "value"), Input("in-peso", "value"))
def imc_en_vivo(altura, peso):
    """Muestra el IMC mientras se escriben la altura y el peso."""
    try:
        altura, peso = float(altura), float(peso)
    except (TypeError, ValueError):
        return None
    if not (RANGOS["altura"][0] <= altura <= RANGOS["altura"][1] and RANGOS["peso"][0] <= peso <= RANGOS["peso"][1]):
        return None
    imc = calcular_imc(altura, peso)
    if imc < 18.5:
        categoria, clase = "bajo peso", "aviso"
    elif imc < 25:
        categoria, clase = "peso normal", "ok"
    elif imc < 30:
        categoria, clase = "sobrepeso", "aviso"
    else:
        categoria, clase = "obesidad", "alerta"
    return html.Div(className=f"imc-vivo {clase}", children=[
        html.Span("Tu IMC"), html.Strong(num(imc)), html.Span(f"· {categoria}"),
    ])


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
    """Valida el formulario, predice, guarda la encuesta de forma anónima y muestra el resultado."""
    respuestas = dict(edad=edad, sexo=sexo, altura=altura, peso=peso, fumador=fumador, cigarrillos=cigarrillos,
                      hipertension=hipertension, medicacion=medicacion, diabetes=diabetes, ictus=ictus,
                      sistolica=sistolica, diastolica=diastolica, colesterol=colesterol)
    datos, errores, avisos = procesar_formulario(respuestas)
    if errores:
        return tarjeta([
            html.Div(className="error-cabecera", children=[html.Span("✋"), html.H3("Revisa el formulario")]),
            html.Ul([html.Li(e) for e in errores]),
        ], clase="errores")

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


# Al calcular, la página se desplaza hasta el principio del resultado (en el móvil queda debajo del
# formulario y en escritorio el botón está al final). Se descuenta la altura de la cabecera fija.
app.clientside_callback(
    """
    function(hijos) {
        setTimeout(function () {
            var r = document.getElementById('resultado');
            if (r) {
                var cab = document.querySelector('.cabecera');
                var y = r.getBoundingClientRect().top + window.scrollY - (cab ? cab.offsetHeight : 0) - 12;
                window.scrollTo({top: y, behavior: 'smooth'});
            }
        }, 150);
        return window.dash_clientside.no_update;
    }
    """,
    Output("scroll-resultado", "data"),
    Input("resultado", "children"),
    prevent_initial_call=True,
)

# Al cambiar de sección: arriba del todo (o a la calculadora) y reajuste de los gráficos.
app.clientside_callback(
    """
    function(seccion) {
        // Volver a «Calcular» desde el menú lleva directamente a la calculadora: la animación de la
        // portada es solo la entrada (y durante ella el menú está oculto).
        var destino = seccion === 'calcular' ? document.getElementById('calculadora') : null;
        if (destino) {
            // La sección la hace visible un callback del servidor: se espera a que lo esté (máx. 3 s).
            var intentos = 0;
            var esperar = setInterval(function () {
                intentos += 1;
                if (destino.offsetParent !== null || intentos > 60) {
                    clearInterval(esperar);
                    var cab = document.querySelector('.cabecera');
                    var y = destino.getBoundingClientRect().top + window.scrollY - (cab ? cab.offsetHeight : 0) + 1;
                    window.scrollTo({top: y, behavior: 'instant'});
                }
            }, 50);
        } else {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        // Los gráficos de una sección oculta se dibujan con un ancho por defecto: al mostrarla,
        // se fuerza un 'resize' para que Plotly los ajuste al ancho real.
        setTimeout(function () { window.dispatchEvent(new Event('resize')); }, 60);
        return window.dash_clientside.no_update;
    }
    """,
    Output("scroll-nav", "data"),
    Input("nav", "value"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=True)
