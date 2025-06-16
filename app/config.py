# Configuración de la API de Predicción de Riesgo Cardiovascular
import os
from pathlib import Path
from typing import List

# Rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Archivos de modelos
MODEL_PATH = MODELS_DIR / "cardiovascular_risk_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
ENCODING_PATH = MODELS_DIR / "encoding_info.pkl"

# Configuración de la API
API_TITLE = "Cardiovascular Risk Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## API para Predicción de Riesgo Cardiovascular 🫀

Esta API utiliza un modelo de Machine Learning (Random Forest) para predecir
el riesgo de enfermedad cardiovascular basándose en 9 factores de riesgo.

### Características:
* Predicción en tiempo real
* Validación automática de datos
* Recomendaciones personalizadas
* Documentación interactiva

### Uso:
1. Envíe los datos del paciente al endpoint `/predict`
2. Reciba la predicción con nivel de riesgo y recomendaciones
"""

# Configuración de entorno
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# URLs del frontend desde variables de entorno
FRONTEND_URL_PROD = os.getenv("FRONTEND_URL", "")  # URL de producción del frontend
FRONTEND_URL_DEV = os.getenv("FRONTEND_DEV_URL", "http://localhost:3000")  # URL de desarrollo

def get_allowed_origins() -> List[str]:
    """
    Determina automáticamente los orígenes permitidos basado en el entorno
    """
    # Orígenes base (siempre permitidos para desarrollo)
    base_origins = [
        "http://localhost:3000",    # React/Next.js dev
        "http://localhost:5173",    # Vite dev
        "http://localhost:8080",    # Vue dev
        "http://localhost:4200",    # Angular dev
        "http://localhost:8000",    # API local
        "http://127.0.0.1:3000",   # Variante localhost
        "http://127.0.0.1:5173",   # Variante localhost
        "http://127.0.0.1:8000",   # API local variante
    ]
    
    # Si estamos en producción y tenemos URL del frontend
    if ENVIRONMENT == "production" and FRONTEND_URL_PROD:
        production_origins = [
            FRONTEND_URL_PROD,
            # Asegurar que tanto HTTP como HTTPS estén permitidos
            FRONTEND_URL_PROD.replace("http://", "https://") if FRONTEND_URL_PROD.startswith("http://") else FRONTEND_URL_PROD,
            FRONTEND_URL_PROD.replace("https://", "http://") if FRONTEND_URL_PROD.startswith("https://") else FRONTEND_URL_PROD,
        ]
        # En producción, incluir tanto los orígenes de producción como localhost para testing
        return list(set(production_origins + ["http://localhost:3000", "http://localhost:8000"]))
    
    # Si tenemos URL de desarrollo personalizada
    if FRONTEND_URL_DEV and FRONTEND_URL_DEV not in base_origins:
        base_origins.append(FRONTEND_URL_DEV)
    
    # En desarrollo, permitir todos los orígenes base
    return base_origins

# Configuración de CORS
ALLOWED_ORIGINS = get_allowed_origins()

# Para desarrollo local, también permitir cualquier origen
ALLOW_ALL_ORIGINS = ENVIRONMENT == "development"

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Imprimir configuración al cargar
print(f"🌐 Entorno: {ENVIRONMENT}")
print(f"🔗 Orígenes permitidos: {ALLOWED_ORIGINS}")
print(f"🔓 Permitir todos los orígenes: {ALLOW_ALL_ORIGINS}")
