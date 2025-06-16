from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

from .config import (
    API_TITLE, 
    API_VERSION, 
    API_DESCRIPTION, 
    ALLOWED_ORIGINS, 
    ALLOW_ALL_ORIGINS, 
    ENVIRONMENT
)
from .models import PatientData, PredictionResponse, HealthResponse
from .predictor import predictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Configuración de CORS más flexible
if ALLOW_ALL_ORIGINS:
    # En desarrollo, permitir todos los orígenes
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    logger.info("🔓 CORS: Permitiendo todos los orígenes (modo desarrollo)")
else:
    # En producción, usar orígenes específicos
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    logger.info(f"🔒 CORS: Usando orígenes específicos: {ALLOWED_ORIGINS}")

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Riesgo Cardiovascular",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "cors_origins": ALLOWED_ORIGINS if not ALLOW_ALL_ORIGINS else ["*"],
        "cors_mode": "development (all origins)" if ALLOW_ALL_ORIGINS else "production (specific origins)",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud de la API", 
            "/predict": "Realizar predicción de riesgo",
            "/docs": "Documentación interactiva"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verificar el estado de la API"""
    return HealthResponse(
        status="healthy",
        models_loaded=predictor.model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_risk(patient_data: PatientData):
    """
    Predecir el riesgo cardiovascular de un paciente.
    
    Recibe los datos del paciente y retorna:
    - Nivel de riesgo (Bajo/Moderado/Alto)
    - Probabilidad numérica
    - Factores de riesgo identificados
    - Recomendaciones personalizadas
    """
    try:
        # Realizar predicción
        prediction, probability, confidence_interval = predictor.predict(patient_data)
        
        # Obtener nivel de riesgo
        risk_level = predictor.get_risk_level(probability)
        
        # Obtener factores de riesgo
        risk_factors = predictor.get_risk_factors(patient_data)
        
        # Obtener recomendaciones
        recommendations = predictor.get_recommendations(probability, risk_factors)
        
        # Log de la predicción
        logger.info(f"Predicción realizada - Riesgo: {risk_level} ({probability:.2%})")
        
        return PredictionResponse(
            risk_prediction=risk_level,
            risk_probability=round(probability, 3),
            confidence_interval=[round(ci, 3) for ci in confidence_interval],
            risk_factors=risk_factors,
            recommendations=recommendations,
            disclaimer="Esta predicción es solo orientativa y no reemplaza la consulta médica profesional."
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
