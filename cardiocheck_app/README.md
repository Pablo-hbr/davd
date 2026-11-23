# CardioCheck ❤

**Estimación del riesgo cardiovascular a 10 años a partir de una encuesta de salud sencilla.**

Proyecto de la asignatura *Desarrollo de Aplicaciones para la Visualización de Datos* (DAVD, curso 2026-2027) · Pablo Hebrero

La persona usuaria responde a unas preguntas desde el móvil (unos 2 minutos, sin análisis médicos). Un modelo de
*machine learning* entrenado con el **Framingham Heart Study** estima su probabilidad de sufrir un evento coronario
(infarto, angina o muerte coronaria) en los próximos 10 años. La app la traduce a un nivel de riesgo, muestra
qué factores han influido y da recomendaciones.

> ⚠️ La estimación es orientativa y no sustituye un diagnóstico médico. La app lo indica junto al resultado.

---

## Funcionalidades

| Pestaña | Contenido |
| --- | --- |
| **Calcular** | Formulario adaptado al móvil, validación con mensajes claros, % de riesgo + nivel (bajo / moderado / alto), velocímetro, factores que han influido, escenarios «¿y si dejaras de fumar?» y recomendaciones |
| **Datos** | Exploración interactiva del dataset: riesgo observado por edad, sexo, tabaco, tensión, colesterol, IMC… |
| **Modelo** | Comparación de modelos, métricas, curva ROC, calibración, coeficientes y limitaciones |
| **Uso** | Estadísticas de las encuestas guardadas en la base de datos |

## Datos

- **Framingham Heart Study** (versión pública de Kaggle): 4.238 personas de 32 a 70 años, seguidas 10 años.
- Objetivo: `TenYearCHD` (1 = evento coronario en los 10 años siguientes). Prevalencia: 15,2 %.
- Fichero: `data/raw/framingham.csv`.

**¿Por qué Framingham y no BRFSS 2015?** Se compararon ambos. BRFSS tiene más filas y un AUC mayor (≈ 0,84), pero su
objetivo es *haber tenido alguna vez* un infarto (transversal): los hábitos se miden después del diagnóstico y el
modelo aprende relaciones al revés (el alcohol sale protector, el IMC no influye…). Framingham es **prospectivo**
(mide riesgo futuro, lo que promete la app) y sus coeficientes tienen sentido clínico. Su AUC (≈ 0,73) es el
esperable al predecir el futuro.

## Arquitectura

```text
data/raw/framingham.csv ──► src/etl.py ──► SQLite (tabla framingham) ──► train.py ──► models/*.joblib + metricas.json
                                                                                         │
     móvil / navegador ◄── app.py (Dash) ◄── src/model.py, graphics.py, recomendaciones ◄─┘
                                │
                                └──► SQLite (tabla encuestas): cada cálculo se guarda de forma anónima
```

```text
cardiocheck_app/
├── app.py                 # App Dash (layout + callbacks). Expone `server` para gunicorn
├── train.py               # ETL + evaluación + entrenamiento -> models/
├── src/
│   ├── config.py          # Rutas, variables, umbrales (sin números mágicos repartidos)
│   ├── etl.py             # CSV -> limpieza -> SQLite
│   ├── model.py           # Entrenamiento, evaluación, predicción y explicación
│   ├── formulario.py      # Validación del formulario y conversión a variables del modelo
│   ├── recomendaciones.py # Recomendaciones por reglas de prevención
│   ├── graphics.py        # Figuras Plotly
│   ├── db.py              # Guardado / lectura de encuestas
│   └── formato.py         # Números en formato español
├── assets/style.css       # Estilos (mobile first)
├── tests/                 # pytest
├── requirements.txt · requirements-dev.txt · Procfile · render.yaml · .python-version
└── data/raw/framingham.csv
```

## Instalación y ejecución en local

Requiere Python 3.11+ (en Render se usa 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python train.py                    # ~20 s: crea la BD, evalúa y guarda los modelos
python app.py                      # http://127.0.0.1:8050/
pytest                             # tests
```

`python train.py --rapido` omite la comparación de modelos por validación cruzada. Si se arranca la app sin modelos
entrenados, la propia app ejecuta el entrenamiento rápido.

Variable de entorno opcional: `CARDIOCHECK_DB` (ruta de la base de datos SQLite; por defecto `data/cardiocheck.db`).

## Despliegue en Render

1. Sube el proyecto a GitHub (con `.gitignore`: no se suben `.venv`, la BD ni los modelos).
2. En Render: **New → Blueprint** (detecta `render.yaml`) o **New → Web Service** con:
   - Build command: `pip install -r requirements.txt && python train.py`
   - Start command: `gunicorn --timeout 600 app:server`
   - Si la app está en una subcarpeta del repositorio, indica esa carpeta en **Root Directory**.
3. Los modelos se entrenan durante el build, así siempre coinciden con la versión de scikit-learn del servidor.

⚠️ En el plan gratuito de Render el disco no es persistente: las encuestas guardadas en SQLite se pierden en cada
redespliegue o reinicio. Para conservarlas habría que usar una base de datos gestionada (p. ej. PostgreSQL).

## Modelo

**Variables que pregunta la app**: edad, sexo, altura y peso (→ IMC), si fuma y cuántos cigarrillos, hipertensión
diagnosticada, medicación para la tensión, diabetes e ictus previo. **Opcionales**: tensión arterial
(sistólica/diastólica) y colesterol total.

**Decisiones:**

1. **Regresión logística** (`StandardScaler` + `LogisticRegression` en un `Pipeline`, imputación con la mediana del train).
   En validación cruzada iguala o supera a Random Forest y Gradient Boosting (AUC 0,733 frente a 0,720-0,721), da
   probabilidades bien calibradas y cada predicción se explica con sus coeficientes.
2. **Sin `class_weight="balanced"`**: la app muestra una probabilidad, y re-ponderar las clases la inflaría
   (comprobado en BRFSS: predecía un 36 % de media para una prevalencia real del 10 %). El desbalanceo se gestiona con
   los umbrales de nivel.
3. **Cuatro modelos** (base, +tensión, +colesterol, completo) en lugar de imputar los campos opcionales: rellenar con la
   media subestimaba el riesgo de los hipertensos (33 % en vez de 42 % en un caso de ejemplo).
4. **Niveles de riesgo a 10 años**: bajo < 10 %, moderado 10-20 %, alto ≥ 20 % (cortes clásicos de las escalas
   Framingham). Con el umbral del 10 % se detecta en torno al 80 % de las personas que tendrán un evento.
5. **Recomendaciones por reglas** de prevención (tabaco, tensión, colesterol, IMC, actividad física), no derivadas de
   los coeficientes: el modelo estima asociaciones, no efectos causales.

**Resultados** (`python train.py`):

| Modelo | AUC (validación cruzada 5x5) | Recall con umbral 10 % |
| --- | --- | --- |
| Datos básicos | 0,722 ± 0,017 | 78 % |
| + tensión arterial | 0,726 ± 0,019 | 81 % |
| + colesterol | 0,722 ± 0,018 | 80 % |
| + tensión + colesterol | 0,726 ± 0,020 | 83 % |

Saber las cifras de tensión y colesterol apenas mejora el AUC global, pero sí ajusta la estimación individual
(p. ej. con 120 → 180 mmHg de sistólica, el riesgo de un mismo perfil pasa del 30 % al 50 %).

## Limitaciones

- Cohorte de una sola ciudad de EE. UU. y de hace décadas: en la población española actual el modelo tenderá a
  **sobrestimar** el riesgo.
- Solo incluye personas de 32 a 70 años (la app avisa fuera de ese rango).
- ~4.200 personas: las métricas tienen un margen de error de unos ±0,02 de AUC.
- Asociaciones estadísticas, no causalidad.

## Plan de trabajo

- [x] Propuesta (entrega 1)
- [x] Selección del dataset (BRFSS 2015 frente a Framingham) y análisis exploratorio
- [x] Modelo, evaluación y explicabilidad
- [x] App Dash: formulario, resultado, exploración de datos, modelo y uso
- [x] Base de datos SQLite (datos + encuestas) y tests
- [ ] Despliegue en Render
- [ ] Base de datos persistente para las encuestas
- [ ] Presentación

## Uso de IA

Parte del código y del análisis se desarrolló con ayuda de un asistente de IA (Claude, de Anthropic), revisando y
validando cada propuesta: comparación de datasets, decisiones de modelado y estructura de la app.
