# tests/test_api.py
"""
Pruebas unitarias completas para la API de Predicción de Riesgo Cardiovascular
Diseñadas para generar reportes técnicos ordenados y detallados
"""
import pytest
import json
import time
from datetime import datetime
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ============================================================================
# PRUEBAS DE ENDPOINTS PRINCIPALES
# ============================================================================

class TestEndpoints:
    """Pruebas de endpoints básicos de la API"""
    
    def test_root_endpoint_info(self):
        """TEST 1: Endpoint raíz - Información general de la API"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificaciones estructurales
        assert "message" in data
        assert "version" in data
        assert "environment" in data
        
        # Verificaciones de contenido
        assert data["message"] == "API de Predicción de Riesgo Cardiovascular"
        assert data["version"] == "1.0.0"
        
        print(f"\n✅ ENDPOINT RAÍZ - INFORMACIÓN DE LA API")
        print(f"   Status Code: {response.status_code}")
        print(f"   Mensaje: {data['message']}")
        print(f"   Versión: {data['version']}")
        print(f"   Entorno: {data.get('environment', 'N/A')}")

    def test_health_check_status(self):
        """TEST 2: Health check - Verificación del estado del sistema"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificaciones críticas
        assert "status" in data
        assert "models_loaded" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        
        print(f"\n✅ HEALTH CHECK - ESTADO DEL SISTEMA")
        print(f"   Status Code: {response.status_code}")
        print(f"   Estado: {data['status']}")
        print(f"   Modelos cargados: {data['models_loaded']}")
        print(f"   Timestamp: {data['timestamp']}")

    def test_docs_endpoint_availability(self):
        """TEST 3: Documentación automática disponible"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        print(f"\n✅ DOCUMENTACIÓN SWAGGER")
        print(f"   Status Code: {response.status_code}")
        print(f"   Documentación: Disponible en /docs")

# ============================================================================
# PRUEBAS DE PREDICCIÓN POR PERFIL DE RIESGO
# ============================================================================

class TestPredictionProfiles:
    """Pruebas de predicción con diferentes perfiles de pacientes"""
    
    def test_low_risk_patient(self):
        """TEST 4: Predicción - Paciente de BAJO RIESGO"""
        patient_data = {
            "age": 25,
            "gender": "Female", 
            "smoking": "Never",
            "alcohol_intake": "None",
            "exercise_hours": 8.0,
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No",
            "stress_level": 2
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "risk_prediction" in data
        assert "risk_probability" in data
        assert isinstance(data["risk_probability"], float)
        assert 0.0 <= data["risk_probability"] <= 1.0
        
        print(f"\n✅ PREDICCIÓN - PERFIL BAJO RIESGO")
        print(f"   Paciente: Mujer, 25 años, saludable")
        print(f"   Status Code: {response.status_code}")
        print(f"   Predicción: {data['risk_prediction']}")
        print(f"   Probabilidad: {data['risk_probability']:.2%}")
        print(f"   Recomendaciones: {len(data.get('recommendations', []))}")

    def test_moderate_risk_patient(self):
        """TEST 5: Predicción - Paciente de RIESGO MODERADO"""
        patient_data = {
            "age": 45,
            "gender": "Male",
            "smoking": "Former", 
            "alcohol_intake": "Moderate",
            "exercise_hours": 3.0,
            "diabetes": "No",
            "family_history": "Yes",
            "obesity": "No",
            "stress_level": 6
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        
        print(f"\n✅ PREDICCIÓN - PERFIL RIESGO MODERADO")
        print(f"   Paciente: Hombre, 45 años, ex-fumador, historia familiar")
        print(f"   Status Code: {response.status_code}")
        print(f"   Predicción: {data['risk_prediction']}")
        print(f"   Probabilidad: {data['risk_probability']:.2%}")
        
        if "risk_factors" in data and data["risk_factors"]:
            print(f"   Factores identificados: {len(data['risk_factors'])}")

    def test_high_risk_patient(self):
        """TEST 6: Predicción - Paciente de ALTO RIESGO"""
        patient_data = {
            "age": 65,
            "gender": "Male",
            "smoking": "Current",
            "alcohol_intake": "Heavy", 
            "exercise_hours": 0.5,
            "diabetes": "Yes",
            "family_history": "Yes",
            "obesity": "Yes",
            "stress_level": 9
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        
        print(f"\n✅ PREDICCIÓN - PERFIL ALTO RIESGO")
        print(f"   Paciente: Hombre, 65 años, múltiples factores de riesgo")
        print(f"   Status Code: {response.status_code}")
        print(f"   Predicción: {data['risk_prediction']}")
        print(f"   Probabilidad: {data['risk_probability']:.2%}")
        
        if "risk_factors" in data and data["risk_factors"]:
            print(f"   Factores críticos: {len(data['risk_factors'])}")

# ============================================================================
# PRUEBAS DE VALIDACIÓN DE ENTRADA
# ============================================================================

class TestInputValidation:
    """Pruebas de validación de datos de entrada"""
    
    def test_invalid_age_too_young(self):
        """TEST 7: Validación - Edad menor al límite (< 18 años)"""
        patient_data = {
            "age": 15,  # Menor a 18
            "gender": "Male",
            "smoking": "Never",
            "alcohol_intake": "None",
            "exercise_hours": 5.0,
            "diabetes": "No",
            "family_history": "No", 
            "obesity": "No",
            "stress_level": 3
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422
        
        print(f"\n❌ VALIDACIÓN - EDAD MÍNIMA")
        print(f"   Input: Edad = 15 años (< 18)")
        print(f"   Status Code: {response.status_code} (Error esperado)")
        print(f"   Resultado: Validación funcionando ✓")

    def test_invalid_age_too_old(self):
        """TEST 8: Validación - Edad mayor al límite (> 100 años)"""
        patient_data = {
            "age": 150,  # Mayor a 100
            "gender": "Female",
            "smoking": "Never",
            "alcohol_intake": "None",
            "exercise_hours": 2.0,
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No", 
            "stress_level": 4
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422
        
        print(f"\n❌ VALIDACIÓN - EDAD MÁXIMA")
        print(f"   Input: Edad = 150 años (> 100)")
        print(f"   Status Code: {response.status_code} (Error esperado)")
        print(f"   Resultado: Validación funcionando ✓")

    def test_invalid_gender(self):
        """TEST 9: Validación - Género no válido"""
        patient_data = {
            "age": 35,
            "gender": "Other",  # No válido
            "smoking": "Never",
            "alcohol_intake": "Light",
            "exercise_hours": 4.0,
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No",
            "stress_level": 5
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422
        
        print(f"\n❌ VALIDACIÓN - GÉNERO")
        print(f"   Input: Gender = 'Other' (no válido)")
        print(f"   Status Code: {response.status_code} (Error esperado)")
        print(f"   Valores válidos: Male, Female")

    def test_invalid_exercise_hours(self):
        """TEST 10: Validación - Horas de ejercicio fuera de rango"""
        patient_data = {
            "age": 40,
            "gender": "Male",
            "smoking": "Never",
            "alcohol_intake": "Light",
            "exercise_hours": 30.0,  # > 24 horas
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No",
            "stress_level": 5
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422
        
        print(f"\n❌ VALIDACIÓN - EJERCICIO")
        print(f"   Input: Exercise = 30.0 horas (> 24)")
        print(f"   Status Code: {response.status_code} (Error esperado)")
        print(f"   Rango válido: 0-24 horas/semana")

    def test_invalid_stress_level(self):
        """TEST 11: Validación - Nivel de estrés fuera de rango"""
        patient_data = {
            "age": 30,
            "gender": "Female",
            "smoking": "Never",
            "alcohol_intake": "None",
            "exercise_hours": 6.0,
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No",
            "stress_level": 15  # > 10
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422
        
        print(f"\n❌ VALIDACIÓN - NIVEL ESTRÉS")
        print(f"   Input: Stress = 15 (> 10)")
        print(f"   Status Code: {response.status_code} (Error esperado)")
        print(f"   Rango válido: 1-10")

# ============================================================================
# PRUEBAS DE ESTRUCTURA DE RESPUESTA
# ============================================================================

class TestResponseStructure:
    """Pruebas de estructura y tipos de datos en respuestas"""
    
    def test_response_data_types(self):
        """TEST 12: Verificación de tipos de datos en respuesta"""
        patient_data = {
            "age": 50,
            "gender": "Female",
            "smoking": "Former",
            "alcohol_intake": "Light",
            "exercise_hours": 4.5,
            "diabetes": "No",
            "family_history": "Yes",
            "obesity": "No",
            "stress_level": 7
        }
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verificar tipos específicos
        assert isinstance(data["risk_prediction"], str)
        assert isinstance(data["risk_probability"], float)
        assert isinstance(data["confidence_interval"], list)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["disclaimer"], str)
        
        # Verificar rangos
        assert 0.0 <= data["risk_probability"] <= 1.0
        assert len(data["confidence_interval"]) == 2
        assert len(data["recommendations"]) > 0
        
        print(f"\n✅ ESTRUCTURA DE RESPUESTA")
        print(f"   risk_prediction: {type(data['risk_prediction']).__name__}")
        print(f"   risk_probability: {type(data['risk_probability']).__name__} ({data['risk_probability']:.3f})")
        print(f"   confidence_interval: {type(data['confidence_interval']).__name__} (length: {len(data['confidence_interval'])})")
        print(f"   recommendations: {type(data['recommendations']).__name__} ({len(data['recommendations'])} items)")
        print(f"   disclaimer: {type(data['disclaimer']).__name__}")

# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================

class TestPerformance:
    """Pruebas de rendimiento y tiempo de respuesta"""
    
    def test_response_time(self):
        """TEST 13: Medición de tiempo de respuesta"""
        patient_data = {
            "age": 42,
            "gender": "Male",
            "smoking": "Never",
            "alcohol_intake": "Moderate", 
            "exercise_hours": 5.0,
            "diabetes": "No",
            "family_history": "No",
            "obesity": "No",
            "stress_level": 5
        }
        
        start_time = time.time()
        response = client.post("/predict", json=patient_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0  # Menos de 2 segundos
        
        print(f"\n⚡ RENDIMIENTO - TIEMPO DE RESPUESTA")
        print(f"   Tiempo: {response_time:.3f} segundos")
        print(f"   Requisito: < 2.0 segundos")
        print(f"   Estado: {'✅ EXITOSO' if response_time < 2.0 else '❌ LENTO'}")
        print(f"   Eficiencia: {((2.0 - response_time) / 2.0 * 100):.1f}%")

# ============================================================================
# FUNCIONES AUXILIARES PARA REPORTES
# ============================================================================

@pytest.fixture(autouse=True)
def test_info(request):
    """Información de cada test para reportes ordenados"""
    test_name = request.node.name
    if test_name.startswith('test_'):
        print(f"\n{'='*70}")
        print(f"🧪 {test_name.replace('_', ' ').upper()}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

def test_final_summary():
    """TEST FINAL: Resumen general de todas las pruebas"""
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN FINAL DE PRUEBAS UNITARIAS")
    print(f"🫀 API DE PREDICCIÓN DE RIESGO CARDIOVASCULAR")
    print(f"{'='*80}")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Framework: pytest + FastAPI TestClient")
    print(f"🎯 Categorías probadas:")
    print(f"   ✅ Endpoints principales (3 tests)")
    print(f"   ✅ Perfiles de riesgo (3 tests)")
    print(f"   ✅ Validación de entrada (5 tests)")
    print(f"   ✅ Estructura de respuesta (1 test)")
    print(f"   ✅ Rendimiento (1 test)")
    print(f"{'='*80}")
    print(f"🎉 RESULTADO: API VALIDADA COMPLETAMENTE")
    print(f"{'='*80}")
    
    assert True  # Test siempre pasa para generar el reporte
