
<div align="center">

# Predicción de popularidad de canciones

Repositorio base del taller en dos sprints: regresión y clasificación binaria.

[![Python](https://img.shields.io/badge/Python-3.11%20a%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

</div>

---

Una disquera debe decidir, antes del lanzamiento, en qué canciones invertir promoción. Este proyecto desarrolla un servicio de predicción mediante Machine Learning, Python y FastAPI, desplegado en Render.

El servicio utiliza un único artefacto, `models/modelo.pkl`, que contiene el modelo entrenado y su ficha de información. En cada sprint se reemplaza este archivo por el modelo correspondiente, sin modificar el código de la API.

## Los dos sprints

| Característica | Sprint 1: regresión | Sprint 2: clasificación |
|---|---|---|
| Objetivo | Estimar la popularidad de una canción | Predecir si será un hit |
| Variable objetivo | `popularidad` (0–100) | `es_hit` (1 si popularidad ≥ 70) |
| Variables excluidas | `reproducciones_sem1`, `es_hit` | `reproducciones_sem1`, `popularidad` |
| Modelo de referencia | Regresión lineal con duración² | Regresión logística |
| Métrica principal | MAE | Costo de clasificación |
| Línea base de referencia | MAE 10,44 | Costo 318 |
| Resultado de referencia | MAE 5,75 | Costo 99 |
| Decisión | Promocionar si popularidad esperada > 65 | Promocionar si probabilidad de hit > 0,27 |

Ambos modelos se entrenan con canciones de 1995 a 2019 y se evalúan con datos de 2020 a 2025. Las variables excluidas corresponden a información que no estaría disponible al momento de decidir la promoción o que revelaría directamente el objetivo.

## Resultados obtenidos

### Sprint 1: regresión

Se desarrolló y desplegó un modelo propio basado en `HistGradientBoostingRegressor`, incorporando características derivadas para mejorar la predicción de popularidad.

| Métrica | Resultado |
|---|---:|
| Modelo | HistGradientBoosting + características derivadas |
| Periodo de entrenamiento | 1995–2019 |
| Periodo de evaluación | 2020–2025 |
| Registros de prueba | 415 |
| MAE del modelo | 6,61 |
| MAE de la línea base | 10,44 |
| RMSE | 8,35 |
| R² | 0,616 |
| Sesgo medio | 1,76 |

El modelo obtuvo un MAE de 6,61, inferior al 10,44 de la línea base, lo que representa una reducción aproximada del 36,7 % en el error absoluto medio. El coeficiente R² de 0,616 indica que el modelo explica una parte importante de la variabilidad de la popularidad en el conjunto evaluado.

La regla de decisión establece recomendar la promoción cuando la popularidad esperada supera los 65 puntos.

**Servicio desplegado:** https://demo-reg-clas-canciones-ctmb.onrender.com

- Documentación interactiva: [/docs](https://demo-reg-clas-canciones-ctmb.onrender.com/docs)
- Estado del servicio: [/salud](https://demo-reg-clas-canciones-ctmb.onrender.com/salud)
- Ficha y métricas del modelo: [/modelo](https://demo-reg-clas-canciones-ctmb.onrender.com/modelo)

La prueba del servicio confirmó que el modelo se carga correctamente, produce predicciones, aplica el umbral de promoción y rechaza variables posteriores al lanzamiento. El resultado de la prueba de humo fue `servicio en orden`.

### Sprint 2: clasificación binaria

El segundo sprint reemplaza el modelo de regresión por un clasificador que estima la probabilidad de que una canción sea un hit. La selección del umbral considera el costo de los falsos negativos y falsos positivos.

Los resultados de referencia del taller son un costo de 99 frente a 318 de la línea base. El desempeño del modelo propio debe documentarse después de entrenarlo y desplegarlo, consultando las métricas de `GET /modelo`.

> Nota: los resultados de referencia del taller y los obtenidos con el modelo propio son diferentes y deben reportarse por separado. Los datos utilizados son sintéticos y sus resultados no constituyen evidencia causal sobre el éxito musical real.

## Cómo ejecutar el proyecto

Se requiere Python 3.11, 3.12 o 3.13.

```bash
git clone https://github.com/ebuitrago/demo-reg-clas-canciones.git
cd demo-reg-clas-canciones

python -m venv .venv
# Windows: py -3.13 -m venv .venv
# Activación en Windows: .venv\Scripts\activate
# Activación en Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pytest -q
uvicorn src.app.main:app --reload
```

Para entrenar cada modelo:

```bash
python -m src.entrenar_regresion
python -m src.entrenar_clasificacion
```

Cada script genera `models/modelo.pkl`. Entrene el modelo correspondiente al sprint que vaya a entregar.

## API disponible

| Método | Ruta | Función |
|---|---|---|
| GET | `/docs` | Documentación interactiva |
| GET | `/salud` | Estado del servicio y del modelo |
| GET | `/modelo` | Ficha técnica y métricas |
| POST | `/predecir` | Predicción individual |
| POST | `/predecir_lote` | Predicción de hasta 1.000 canciones |

Ejemplo de entrada:

```json
{
  "bailabilidad": 0.72,
  "energia": 0.65,
  "valencia": 0.55,
  "acustica": 0.12,
  "tempo": 118,
  "duracion_min": 3.4,
  "volumen_db": -6.5,
  "anio": 2026,
  "colaboracion": 1,
  "genero": "urbano"
}
```

El servicio responde con la predicción y la decisión correspondiente al tipo de modelo cargado. Si se envían variables que no están disponibles antes del lanzamiento, como `reproducciones_sem1`, la API rechaza la entrada.

## Despliegue en Render

1. Sube el proyecto y `models/modelo.pkl` a un repositorio propio de GitHub.
2. En Render, selecciona **New + → Blueprint** y conecta el repositorio que contiene `render.yaml`.
3. Espera a que finalice el despliegue y copia la URL pública asignada.
4. Comprueba el servicio:

```bash
python -m src.probar_servicio https://su-servicio.onrender.com
```

La entrega se considera terminada cuando `pytest -q` pasa, la ficha presenta las métricas del modelo propio y este supera la línea base, el servicio está desplegado y la prueba de humo termina con `servicio en orden`.

En el plan gratuito de Render, el servicio puede suspenderse tras un periodo de inactividad; la primera petición posterior puede tardar más de lo habitual.

## Estructura del proyecto

```text
demo-reg-clas-canciones/
├── data/canciones.csv
├── models/modelo.pkl
├── src/
│   ├── variables.py
│   ├── artefacto.py
│   ├── entrenar_regresion.py
│   ├── entrenar_clasificacion.py
│   ├── probar_servicio.py
│   └── app/
│       ├── esquemas.py
│       └── main.py
├── tests/test_servicio.py
├── requirements.txt
└── render.yaml
```

## Tecnologías utilizadas

- **Python:** procesamiento de datos y entrenamiento de modelos.
- **scikit-learn:** construcción, entrenamiento y evaluación de los modelos.
- **FastAPI:** exposición de predicciones mediante una API REST.
- **Render:** despliegue del servicio en la nube.
- **pytest:** pruebas automatizadas del servicio.

---

<div align="center">
<sub>Universidad Central, Ingeniería de Sistemas, 2026-2. Prof. Elias Buitrago Bolivar</sub>
</div>
