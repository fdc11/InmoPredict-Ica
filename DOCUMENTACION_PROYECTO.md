# Documentación del proyecto InmoPredict Ica

## 1. Resumen

**InmoPredict Ica** es un prototipo educativo de una empresa inmobiliaria ficticia. Su propósito es demostrar un flujo de aprendizaje automático de regresión y cómo conectar un modelo entrenado con una aplicación web.

La aplicación tiene dos partes:

1. Una página web en la que se ingresan características de una vivienda.
2. Una API en Python que escala las entradas, ejecuta el modelo de Keras y devuelve un precio estimado.

> **Estado importante:** la versión conectada a la web fue entrenada con 500 observaciones sintéticas generadas por el proyecto. No se debe presentar la predicción ni las métricas actuales como tasaciones reales de Ica.

## 2. Datos y alcance

### Dataset que usa actualmente el modelo

El pipeline principal (`run_pipeline.py`) genera un dataset de 500 registros educativos con estas variables:

| Variable | Descripción |
|---|---|
| `area_m2` | Área construida en metros cuadrados |
| `habitaciones` | Número de habitaciones |
| `distancia_centro_km` | Distancia simulada al centro de Ica |
| `antiguedad_anos` | Antigüedad simulada de la vivienda |
| `precio_usd` | Precio generado artificialmente en dólares |

La relación entre las características y el precio se define mediante una fórmula simulada que incluye ruido, un efecto no lineal de la distancia y una interacción entre área y habitaciones. Esto sirve para practicar el entrenamiento y la comparación de modelos, pero **no representa transacciones ni anuncios reales**.

### Archivo `urbania_data.csv`

El archivo añadido al proyecto contiene 7,663 registros de distintas zonas del país. En una revisión preliminar, 59 filas tienen `Provincia` igual a `Ica`; aparecen distritos como Ica, Paracas y Chincha Alta. Las fechas de los registros de muestra revisados son de 2019; hace falta comprobar el rango completo. Por tanto, el archivo no es exclusivamente de Ica y su cobertura geográfica debe confirmarse.

Antes de usarlo para entrenar se debe verificar:

- La fuente original, sus condiciones de uso y si está permitido reutilizar estos datos.
- Si `Precio` es un precio publicado en un anuncio o el monto de una venta cerrada.
- La moneda de cada precio. No se debe asumir que está en USD o PEN: la moneda no está confirmada en el esquema revisado.
- La correspondencia entre `Provincia`, `Distrito` y la región geográfica que se quiere modelar.
- La vigencia de los registros y la calidad de las áreas y precios.

Este archivo **todavía no está conectado al pipeline ni a la web**. Hasta que se verifiquen esos puntos, el sistema sigue utilizando el dataset sintético. No se usan como características las columnas que puedan identificar a personas o domicilios, como anunciante, dirección o descripción.

## 3. Flujo de trabajo

```text
Datos sintéticos educativos
          │
          ▼
Exploración y separación entrenamiento/prueba (80/20)
          │
          ├── Regresión lineal simple: área → precio
          ├── Regresión lineal multivariable: 4 características → precio
          └── Red neuronal Keras: 4 características → precio
                         │
                         ▼
             Evaluación en el conjunto de prueba
                         │
                         ▼
             Guardado del modelo y escaladores
                         │
                         ▼
       FastAPI recibe datos de la página web
                         │
                         ▼
          Predicción orientativa en USD
```

La separación de entrenamiento y prueba permite evaluar el modelo con registros que no se usaron para ajustar sus parámetros. En la red, los escaladores se ajustan solo con el conjunto de entrenamiento; luego se aplican sin reajustar al conjunto de prueba.

## 4. Modelos y evaluación

### Regresión lineal simple

Utiliza únicamente `area_m2` para predecir `precio_usd`. Produce una línea de tendencia y sirve como modelo de referencia sencillo.

### Regresión lineal multivariable

Usa área, habitaciones, distancia al centro y antigüedad. Es un punto de comparación útil para distinguir la mejora por añadir variables de la mejora por cambiar la arquitectura del modelo.

### Red neuronal

La red se construye con Keras Sequential:

```text
4 entradas → Dense(64, ReLU) → Dense(128, ReLU)
           → Dense(64, ReLU) → Dense(32, ReLU)
           → Dense(1, lineal)
```

Se compila con Adam, pérdida MSE y métrica MAE. Usa `StandardScaler` para las características y el precio, una validación interna durante el entrenamiento y parada temprana para detener el ajuste cuando la pérdida de validación deja de mejorar.

### Métricas

- **MSE:** promedio del error al cuadrado; penaliza más los errores grandes y se expresa en unidades de precio al cuadrado.
- **RMSE:** raíz del MSE; está en las mismas unidades del precio y aproxima el tamaño típico del error.
- **R²:** proporción de variación del precio explicada por el modelo; puede ser menor que cero si el modelo predice peor que usar la media del conjunto.

El pipeline calcula las métricas sobre el conjunto de prueba. Los resultados del informe generado son demostrativos mientras se entrene con datos sintéticos; una red con más variables no necesariamente será mejor en datos reales.

## 5. Aplicación web y API

La página de [web/index.html](./web/index.html) envía cuatro campos a la API:

| Campo | Rango permitido |
|---|---:|
| Área construida | 40–300 m² |
| Habitaciones | 1–6 |
| Distancia al centro de Ica | 0.5–25 km |
| Antigüedad | 0–40 años |

La API carga `models/neural_model.keras`, `models/scaler_X.pkl` y `models/scaler_y.pkl` al iniciar. Escala los datos de entrada, calcula la predicción, revierte el escalado del precio y devuelve un resultado en USD. Los rangos de entrada están definidos para el dataset sintético actual; no se deben interpretar como rangos válidos de todo el mercado inmobiliario.

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/` | Página web de tasación |
| `GET` | `/health` | Comprueba que el modelo y los escaladores estén cargados |
| `POST` | `/predict` | Estima una propiedad |
| `POST` | `/predict/batch` | Estima una lista de propiedades |
| `GET` | `/docs` | Documentación interactiva de FastAPI |

Ejemplo de cuerpo JSON para `/predict`:

```json
{
  "area_m2": 120,
  "habitaciones": 3,
  "distancia_centro_km": 5,
  "antiguedad_anos": 10
}
```

`/health` solo indica que los artefactos se cargaron correctamente; no certifica la precisión del modelo. La respuesta incluye una advertencia de que la estimación es orientativa y no está validada con tasaciones reales de Ica.

## 6. Instalación y ejecución en Windows

Se recomienda Python 3.11 para instalar las dependencias declaradas en `requirements.txt`.

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Entrenar los modelos y generar el informe y gráficos:

```powershell
.\.venv\Scripts\python run_pipeline.py
```

Iniciar la página y la API:

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Abrir [http://127.0.0.1:8000](http://127.0.0.1:8000) para usar la web. También se puede abrir [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) para probar los endpoints. Si el servidor indica que faltan artefactos, primero hay que ejecutar el pipeline.

Ejecutar las pruebas:

```powershell
.\.venv\Scripts\python -m pytest tests -v
```

## 7. Archivos principales

| Archivo | Responsabilidad |
|---|---|
| `run_pipeline.py` | Ejecuta el flujo completo de generación, entrenamiento y comparación |
| `src/data_generator.py` | Genera y guarda los datos sintéticos |
| `src/preprocessing.py` | Carga datos, presenta estadísticas y prepara divisiones/escalado |
| `src/linear_model.py` | Entrena y evalúa las regresiones lineales |
| `src/neural_model.py` | Construye, entrena, evalúa y guarda la red neuronal |
| `src/comparison.py` | Genera gráficos y `reports/informe_tecnico.md` |
| `api/main.py` | Expone la predicción mediante FastAPI |
| `web/index.html` | Interfaz web para asesores o clientes |
| `tests/test_pipeline.py` | Pruebas del dataset generado y de la API |
| `urbania_data.csv` | Archivo externo añadido; pendiente de validar y no usado por el entrenamiento |

Al finalizar, el modelo y sus escaladores quedan en `models/`; los gráficos y el informe se guardan bajo `reports/`; el CSV sintético se guarda en `data/raw/ica_houses.csv`.

## 8. Interpretación y siguientes pasos

El prototipo demuestra el recorrido desde los datos hasta una predicción servida por una web. No es todavía un sistema de tasación comercial. La empresa inmobiliaria del ejemplo es ficticia y la salida no debe usarse para fijar precios, otorgar créditos ni tomar decisiones financieras.

Para avanzar hacia una evaluación con datos reales:

1. Documentar la fuente y el permiso de uso del CSV.
2. Confirmar la moneda y si el objetivo es precio de publicación o precio de venta.
3. Definir qué provincias y distritos pertenecen al ámbito de Ica que se quiere cubrir.
4. Depurar registros repetidos, rangos imposibles y columnas personales; conservar solo variables disponibles también al momento de predecir.
5. Obtener más registros recientes y representativos. Con solo 59 filas preliminares en `Provincia = Ica`, una red neuronal profunda puede sobreajustarse y la evaluación sería muy inestable.
6. Volver a entrenar y evaluar con una partición apropiada, comparar con una regresión base y validar los errores por zona y rango de precio.
7. Actualizar la API y la interfaz para reflejar moneda, unidades, variables y límites que correspondan a los datos validados.

Hasta completar estos pasos, las cifras del modelo se deben presentar únicamente como una demostración técnica con datos sintéticos.
