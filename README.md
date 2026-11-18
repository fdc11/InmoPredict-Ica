# InmoPredict Ica 🏠

> **Sistema Inteligente de Tasación Inmobiliaria – Región Ica, Perú**

La documentación detallada del sistema, su entrenamiento, la API y las limitaciones de los datos está en [DOCUMENTACION_PROYECTO.md](./DOCUMENTACION_PROYECTO.md).

---

## 📋 Descripción del Proyecto

**InmoPredict Ica** es un sistema de machine learning end-to-end para la predicción de precios de propiedades inmobiliarias en la región de Ica, Perú. El proyecto compara tres modelos:

1. **Regresión Lineal Simple** – Modelo clásico usando solo el área (m²) como predictor.
2. **Regresión Lineal Multivariable** – Control con las mismas cuatro características que la red.
3. **Red Neuronal Profunda (MLP)** – Modelo Keras con múltiples características.

Adicionalmente, expone una **API REST** (FastAPI) para consumir las predicciones en tiempo real.

---

## 📁 Estructura del Proyecto

```
InmoPredict Ica/
├── data/
│   └── raw/                  # Dataset sintético generado (ica_houses.csv)
├── notebooks/                # Jupyter notebooks de exploración
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Fase 0: Generación del dataset sintético
│   ├── preprocessing.py      # Fase 1: EDA y preprocesamiento
│   ├── linear_model.py       # Fase 2: Regresión Lineal Simple
│   ├── neural_model.py       # Fase 3: Red Neuronal (MLP)
│   └── comparison.py         # Fase 4: Comparación de modelos
├── models/                   # Modelos entrenados serializados (.pkl, .keras)
├── reports/
│   └── figures/              # Gráficos generados (PNG)
├── api/
│   ├── __init__.py
│   └── main.py               # API FastAPI para predicciones
├── web/
│   └── index.html            # Interfaz web que consume la API
├── tests/
│   └── __init__.py
├── requirements.txt
├── README.md
└── run_pipeline.py           # Script maestro que ejecuta todo el pipeline
```

---

## 🐍 Versión de Python

> [!WARNING]
> Este proyecto requiere **Python 3.11** (recomendado) para garantizar compatibilidad con TensorFlow 2.19.0.
>
> TensorFlow 2.19 soporta oficialmente **Python 3.9 – 3.12**.
>
> ⚠️ **Su sistema tiene Python 3.14**, que **NO es compatible** con TensorFlow 2.19.0.
> Se recomienda crear un entorno virtual con Python 3.11:
>
> ```bash
> # Con conda
> conda create -n inmopred python=3.11
> conda activate inmopred
>
> # O con pyenv + venv
> pyenv install 3.11.9
> pyenv local 3.11.9
> python -m venv .venv
> .venv\Scripts\activate
> ```

---

## ⚙️ Instalación

### 1. Clonar / ubicarse en el proyecto

```bash
cd "InmoPredict Ica"
```

### 2. Crear entorno virtual (Python 3.11)

```bash
py -3.11 -m venv .venv
```

### 3. Instalar dependencias

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

---

## 🚀 Cómo Ejecutar

### Pipeline completo (recomendado)

Ejecuta todas las fases en secuencia:

```bash
.venv\Scripts\python run_pipeline.py
```

El pipeline guarda el dataset sintético en `data/raw/ica_houses.csv`, además de los
modelos, métricas e imágenes. Los datos son **sintéticos y educativos**; no son anuncios
ni transacciones reales de Ica.

### Ejecutar las pruebas

```bash
.venv\Scripts\python -m pytest tests -v
```

### Fases individuales

```bash
.venv\Scripts\python src\data_generator.py   # Generar dataset
.venv\Scripts\python src\preprocessing.py    # EDA y preprocesamiento
.venv\Scripts\python src\linear_model.py     # Entrenar modelo lineal
.venv\Scripts\python src\neural_model.py     # Entrenar red neuronal
```

Para comparar los tres modelos y generar el informe, use el pipeline completo: los scripts
individuales no guardan un archivo de métricas compartido.

---

## 🌐 API de Predicción

### Iniciar el servidor

```bash
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Después de iniciar el servidor, abre **http://127.0.0.1:8000**. La página web consulta
`/health`, permite ingresar las cuatro características de una vivienda y envía la tasación
a `/predict`. Si el modelo no cargó, la página deshabilita el cálculo y muestra el estado
de la API.

### Endpoints disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Web de tasación para asesores/clientes |
| GET | `/health` | Estado del modelo |
| POST | `/predict` | Predecir precio de una propiedad |
| POST | `/predict/batch` | Predecir precios en lote |

`/health` devuelve `503` si el modelo y los escaladores no están disponibles. Si falla
su carga al iniciar, el servidor reporta el error en lugar de anunciar un estado saludable.
Los rangos aceptados corresponden al dominio sintético de entrenamiento: área 40–300 m²,
1–6 habitaciones, distancia 0.5–25 km y antigüedad 0–40 años. Una petición fuera de rango
responde `422`; no se debe extrapolar el modelo fuera de esos valores.

### Ejemplo con curl

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"area_m2":120,"habitaciones":3,"distancia_centro_km":5,"antiguedad_anos":10}'
```

La respuesta contiene `precio_estimado_usd` (número), `mensaje` y `confianza`. El último
campo advierte que la estimación es orientativa y no ha sido validada con tasaciones
reales. La respuesta concreta depende de los artefactos generados; no se publica aquí un
precio ficticio como si fuera resultado de una ejecución.

### Verificación manual de la API

Con el servidor ejecutándose en otra terminal:

```powershell
curl.exe -i http://127.0.0.1:8000/health
curl.exe -i -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"area_m2":120,"habitaciones":3,"distancia_centro_km":5,"antiguedad_anos":10}'
```

La primera petición debe responder `200` con `"status":"healthy"` y la segunda `200` con
un precio positivo. Un área inválida (por ejemplo, `area_m2: 0`) debe responder `422`.

### Documentación interactiva (Swagger)

Abre en tu navegador: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Fases del Pipeline

| Fase | Módulo | Descripción |
|------|--------|-------------|
| 0 | `data_generator.py` | Genera dataset sintético de 500 propiedades en Ica |
| 1 | `preprocessing.py` | EDA, limpieza y normalización con StandardScaler |
| 2 | `linear_model.py` | Regresión Lineal Simple (area_m2 → precio_usd) |
| 3 | `neural_model.py` | Red Neuronal MLP (4 features → precio_usd) |
| 4 | `comparison.py` | Comparación de los tres modelos y visualizaciones finales |

---

## 📈 Métricas

Las métricas dependen de la versión del código, semilla y entorno. El pipeline las calcula
sobre un conjunto de prueba separado y actualiza `reports/informe_tecnico.md`. Al ser datos
sintéticos, no representan el desempeño en el mercado inmobiliario real de Ica.

---

## 👥 Autores

Proyecto desarrollado como sistema demostrativo de comparación de modelos de ML aplicado al mercado inmobiliario de Ica, Perú.

---

## 📄 Licencia

MIT License – Libre uso educativo y comercial.
