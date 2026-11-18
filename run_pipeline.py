"""
run_pipeline.py
===============
Script maestro del proyecto InmoPredict Ica.
Ejecuta secuencialmente todas las fases del pipeline de machine learning.

Uso:
    python run_pipeline.py

Fases:
    0 - Generación de datos sintéticos
    1 - Preprocesamiento y EDA
    2 - Modelo de Regresión Lineal Simple
    3 - Red Neuronal (MLP)
    4 - Comparación de modelos
"""

import logging
import sys
import time
from pathlib import Path

# Avoid Windows legacy console code pages aborting the pipeline on emoji output.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ── Configuración de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def banner(title: str) -> None:
    """Imprime un banner visual para separar fases."""
    sep = "═" * 60
    logger.info(sep)
    logger.info(f"  {title}")
    logger.info(sep)


def run_phase(phase_num: int, description: str, func, *args, **kwargs):
    """
    Ejecuta una fase del pipeline con manejo de errores y temporización.

    Args:
        phase_num:   Número de fase (0-4).
        description: Descripción de la fase.
        func:        Función a ejecutar.
        *args:       Argumentos posicionales para func.
        **kwargs:    Argumentos de palabra clave para func.

    Returns:
        El valor de retorno de func, o None si falla.
    """
    banner(f"FASE {phase_num}: {description}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"✅ Fase {phase_num} completada en {elapsed:.2f}s")
        return result
    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"❌ Fase {phase_num} falló después de {elapsed:.2f}s: {exc}")
        raise


# =============================================================================
# FASE 0 – Generación de datos
# =============================================================================
def phase0_generate_data():
    """Genera el dataset sintético de propiedades en Ica."""
    print("\n🏗️  Generando dataset sintético de propiedades en Ica, Perú...")
    from src.data_generator import generate_dataset, save_dataset

    df = generate_dataset()
    dataset_path = save_dataset(df)
    logger.info(f"Dataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
    logger.info(f"Dataset guardado en: {dataset_path}")
    logger.info(f"Columnas: {list(df.columns)}")
    logger.info(f"Precio promedio: ${df['precio_usd'].mean():,.0f} USD")
    return df


# =============================================================================
# FASE 1 – Preprocesamiento y EDA
# =============================================================================
def phase1_preprocess(df=None):
    """Carga, explora y preprocesa el dataset."""
    print("\n🔍 Iniciando EDA y preprocesamiento...")
    from src.preprocessing import load_data, explore_data, preprocess_data

    if df is None:
        df = load_data()

    logger.info("Ejecutando análisis exploratorio de datos (EDA)...")
    explore_data(df)

    logger.info("Preprocesando datos (split 80/20, StandardScaler)...")
    result = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = result[:6]

    logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test  set: X={X_test.shape}, y={y_test.shape}")
    return result


# =============================================================================
# FASE 2 – Regresión Lineal Simple
# =============================================================================
def phase2_linear_model(df=None):
    """Entrena y evalúa el modelo de regresión lineal simple."""
    print("\n📈 Entrenando modelo de Regresión Lineal Simple...")
    from src.linear_model import run_linear_phase, run_multivariable_linear_phase
    if df is None:
        from src.data_generator import generate_dataset
        df = generate_dataset()
    simple_metrics = run_linear_phase(df)
    multivariable_metrics = run_multivariable_linear_phase(df)
    return None, simple_metrics, multivariable_metrics


# =============================================================================
# FASE 3 – Red Neuronal (MLP)
# =============================================================================
def phase3_neural_model(df=None):
    """Entrena y evalúa la red neuronal profunda."""
    print("\n🧠 Entrenando Red Neuronal (MLP)...")
    from src.neural_model import run_neural_phase
    if df is None:
        from src.data_generator import generate_dataset
        df = generate_dataset()
    metrics = run_neural_phase(df)
    return None, metrics, None


# =============================================================================
# FASE 4 – Comparación de modelos
# =============================================================================
def phase4_comparison(linear_metrics=None, neural_metrics=None, multivariable_metrics=None):
    """Compara ambos modelos y genera reporte final."""
    print("\n📊 Comparando modelos y generando reporte final...")
    from src.comparison import compare_models, generate_technical_report

    compare_models(linear_metrics, neural_metrics, multivariable_metrics)
    report_path = BASE_DIR / "reports" / "informe_tecnico.md"
    generate_technical_report(linear_metrics, neural_metrics, report_path, multivariable_metrics)
    logger.info(f"Reporte de comparación guardado en: {report_path}")
    return {
        "linear_simple": linear_metrics,
        "linear_multivariable": multivariable_metrics,
        "neural": neural_metrics,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  InmoPredict Ica – Pipeline de Machine Learning")
    logger.info("  Sistema de Tasación Inmobiliaria – Ica, Perú")
    logger.info("=" * 60)

    # Fase 0: Generar datos
    df = run_phase(0, "GENERACIÓN DE DATOS SINTÉTICOS", phase0_generate_data)

    # Fase 1: Preprocesamiento
    preprocess_result = run_phase(1, "PREPROCESAMIENTO Y EDA", phase1_preprocess, df)

    # Fase 2: Modelo lineal
    _, linear_metrics, multivariable_metrics = run_phase(
        2, "REGRESIÓN LINEAL SIMPLE", phase2_linear_model, df
    )

    # Fase 3: Red neuronal
    _, neural_metrics, _ = run_phase(
        3, "RED NEURONAL (MLP)", phase3_neural_model, df
    )

    # Fase 4: Comparación
    run_phase(
        4,
        "COMPARACIÓN DE MODELOS",
        phase4_comparison,
        linear_metrics,
        neural_metrics,
        multivariable_metrics,
    )

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"🎉 Pipeline completado en {total_elapsed:.1f}s")
    logger.info("   Revisa reports/ para gráficos y el reporte de comparación.")
    logger.info("   Para la API: uvicorn api.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
