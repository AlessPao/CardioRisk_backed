 # tests/test_api.py
"""Tests para la API"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test del endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test del health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid():
    """Test de predicción con datos válidos"""
    patient_data = {
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
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "risk_prediction" in result
    assert "risk_probability" in result
    assert "recommendations" in result
    assert 0 <= result["risk_probability"] <= 1

def test_predict_invalid_age():
    """Test con edad inválida"""
    patient_data = {
        "age": 150,  # Edad inválida
        "gender": "Male",
        "smoking": "Never",
        "alcohol_intake": "Moderate",
        "exercise_hours": 3.5,
        "diabetes": "No",
        "family_history": "Yes",
        "obesity": "No",
        "stress_level": 6
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_gender():
    """Test con género inválido"""
    patient_data = {
        "age": 55,
        "gender": "Other",  # Género inválido
        "smoking": "Never",
        "alcohol_intake": "Moderate",
        "exercise_hours": 3.5,
        "diabetes": "No",
        "family_history": "Yes",
        "obesity": "No",
        "stress_level": 6
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422
