
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional

class PatientData(BaseModel):
    """Modelo para los datos del paciente"""
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Edad del paciente (18-100 años)"
    )
    gender: str = Field(
        ..., 
        description="Género: Male o Female"
    )
    smoking: str = Field(
        ..., 
        description="Hábito de fumar: Never, Former, Current"
    )
    alcohol_intake: str = Field(
        ..., 
        description="Consumo de alcohol: None, Light, Moderate, Heavy"
    )
    exercise_hours: float = Field(
        ..., 
        ge=0, 
        le=24, 
        description="Horas de ejercicio por semana (0-24)"
    )
    diabetes: str = Field(
        ..., 
        description="Tiene diabetes: Yes o No"
    )
    family_history: str = Field(
        ..., 
        description="Historia familiar de enfermedad cardíaca: Yes o No"
    )
    obesity: str = Field(
        ..., 
        description="Tiene obesidad: Yes o No"
    )
    stress_level: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Nivel de estrés (1-10)"    )
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender debe ser Male o Female')
        return v
    
    @field_validator('smoking')
    @classmethod
    def validate_smoking(cls, v):
        if v not in ['Never', 'Former', 'Current']:
            raise ValueError('Smoking debe ser Never, Former o Current')
        return v
    
    @field_validator('alcohol_intake')
    @classmethod
    def validate_alcohol(cls, v):
        if v not in ['None', 'Light', 'Moderate', 'Heavy']:
            raise ValueError('Alcohol intake debe ser None, Light, Moderate o Heavy')
        return v
    
    @field_validator('diabetes', 'family_history', 'obesity')
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Debe ser Yes o No')
        return v    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 55,
                "gender": "Male",
                "smoking": "Never",
                "alcohol_intake": "Moderate",
                "exercise_hours": 3.5,
                "diabetes": "No",
                "family_history": "Yes",
                "obesity": "No",
                "stress_level": 6
            }
        }
    )

class PredictionResponse(BaseModel):
    """Modelo para la respuesta de predicción"""
    risk_prediction: str = Field(..., description="Nivel de riesgo predicho")
    risk_probability: float = Field(..., description="Probabilidad de riesgo (0-1)")
    confidence_interval: List[float] = Field(..., description="Intervalo de confianza 95%")
    risk_factors: Dict[str, str] = Field(..., description="Factores de riesgo identificados")
    recommendations: List[str] = Field(..., description="Recomendaciones personalizadas")
    disclaimer: str = Field(..., description="Aviso legal")

class HealthResponse(BaseModel):
    """Modelo para la respuesta de health check"""
    status: str
    models_loaded: bool
    timestamp: str