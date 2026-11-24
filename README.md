# Desarrollo de Aplicaciones para la Visualización de Datos — Pablo Hebrero

Repositorio de la asignatura DAVD (curso 2026-2027).

## Contenido

| Carpeta | Descripción |
| --- | --- |
| [`cardiocheck_app/`](cardiocheck_app/) | **CardioCheck**: proyecto de la asignatura. App Dash que estima el riesgo cardiovascular a 10 años a partir de una encuesta de salud |
| [`Inter/`](Inter/) | Prueba intertrimestral: análisis de alquileres horarios de bicis y patinetes compartidos (`city_micromobility_hourly.csv`) |

## CardioCheck

El usuario responde a unas preguntas desde el móvil (edad, sexo, peso, tabaco, hipertensión, diabetes…) y la app
devuelve su probabilidad de sufrir un evento coronario en 10 años, un nivel de riesgo (bajo / moderado / alto), los
factores que más han influido y recomendaciones.

- **Datos:** Framingham Heart Study (4.238 personas seguidas durante 10 años).
- **Modelo:** regresión logística (scikit-learn), comparada con Random Forest y Gradient Boosting.
- **App:** Dash + Plotly, base de datos SQLite y despliegue en Render.

Instalación, ejecución, decisiones de modelado y limitaciones: [`cardiocheck_app/README.md`](cardiocheck_app/README.md).

## Prueba intertrimestral

Notebook [`Inter/Prueba_intertrimestral_14102027.ipynb`](Inter/Prueba_intertrimestral_14102027.ipynb) con datos de
micromovilidad urbana (enero 2025 – diciembre 2026). Descripción de las columnas en
[`Inter/dataset_columns.txt`](Inter/dataset_columns.txt).

1. Programación: funciones de tarifa y de nivel de demanda, integradas en una clase.
2. Exploración y visualización: diagnóstico de nulos, correlaciones y demanda por hora.
3. Regresión: predicción del total de alquileres por hora.
4. Clasificación: predicción de las horas de alta demanda (`high_demand`).
5. Conclusiones.
