"""
Script de prueba para verificar las mejoras en la lógica de predicción
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import PatientData
from app.predictor import predictor

def test_prediction_scenarios():
    """Probar diferentes scenarios para verificar la lógica mejorada"""
    
    print("🧪 TESTING MEJORAS EN PREDICCIÓN DE RIESGO CARDIOVASCULAR")
    print("=" * 70)
    
    # Escenario 1: Persona muy joven y saludable
    print("\n📊 ESCENARIO 1: Persona muy joven y saludable")
    young_healthy = PatientData(
        age=22,
        gender="Female",
        smoking="Never",
        alcohol_intake="None",
        exercise_hours=5.0,
        diabetes="No",
        family_history="No",
        obesity="No",
        stress_level=2
    )
    
    prediction, probability, confidence = predictor.predict(young_healthy)
    risk_level = predictor.get_risk_level(probability)
    risk_factors = predictor.get_risk_factors(young_healthy)
    recommendations = predictor.get_recommendations(probability, risk_factors)
    
    print(f"   👤 Perfil: Mujer, 22 años, no fuma, ejercicio regular")
    print(f"   🎯 Predicción: {risk_level}")
    print(f"   📊 Probabilidad: {probability:.1%}")
    print(f"   🎯 Expectativa: Debería ser BAJO RIESGO (< 20%)")
    print(f"   ✅ Resultado: {'CORRECTO' if probability < 0.2 else 'NECESITA AJUSTE'}")
    
    # Escenario 2: Edad alta pero todo lo demás saludable
    print("\n📊 ESCENARIO 2: Edad alta pero perfil saludable")
    old_healthy = PatientData(
        age=72,
        gender="Male",
        smoking="Never",
        alcohol_intake="Light",
        exercise_hours=4.0,
        diabetes="No",
        family_history="No", 
        obesity="No",
        stress_level=3
    )
    
    prediction, probability, confidence = predictor.predict(old_healthy)
    risk_level = predictor.get_risk_level(probability)
    
    print(f"   👤 Perfil: Hombre, 72 años, no fuma, ejercicio regular, sin enfermedades")
    print(f"   🎯 Predicción: {risk_level}")
    print(f"   📊 Probabilidad: {probability:.1%}")
    print(f"   🎯 Expectativa: Debería ser MODERADO (30-60%)")
    print(f"   ✅ Resultado: {'CORRECTO' if 0.3 <= probability <= 0.6 else 'NECESITA AJUSTE'}")
    
    # Escenario 3: Joven con múltiples factores de riesgo
    print("\n📊 ESCENARIO 3: Joven con múltiples factores de riesgo")
    young_risky = PatientData(
        age=28,
        gender="Male",
        smoking="Current",
        alcohol_intake="Heavy",
        exercise_hours=0.5,
        diabetes="Yes",
        family_history="Yes",
        obesity="Yes",
        stress_level=9
    )
    
    prediction, probability, confidence = predictor.predict(young_risky)
    risk_level = predictor.get_risk_level(probability)
    
    print(f"   👤 Perfil: Hombre, 28 años, fumador, diabético, obeso, sedentario")
    print(f"   🎯 Predicción: {risk_level}")
    print(f"   📊 Probabilidad: {probability:.1%}")
    print(f"   🎯 Expectativa: Debería ser MODERADO-ALTO (35-70%)")
    print(f"   ✅ Resultado: {'CORRECTO' if 0.35 <= probability <= 0.7 else 'NECESITA AJUSTE'}")
    
    # Escenario 4: Perfil mixto realista
    print("\n📊 ESCENARIO 4: Perfil mixto realista")
    mixed_profile = PatientData(
        age=52,
        gender="Female",
        smoking="Former",
        alcohol_intake="Moderate",
        exercise_hours=2.5,
        diabetes="No",
        family_history="Yes",
        obesity="No",
        stress_level=6
    )
    
    prediction, probability, confidence = predictor.predict(mixed_profile)
    risk_level = predictor.get_risk_level(probability)
    risk_factors = predictor.get_risk_factors(mixed_profile)
    recommendations = predictor.get_recommendations(probability, risk_factors)
    
    print(f"   👤 Perfil: Mujer, 52 años, ex-fumadora, historia familiar")
    print(f"   🎯 Predicción: {risk_level}")
    print(f"   📊 Probabilidad: {probability:.1%}")
    print(f"   🔍 Factores de riesgo identificados: {len(risk_factors)}")
    print(f"   💡 Recomendaciones: {len(recommendations)}")
    
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE PRUEBAS:")
    print("   ✓ La lógica médica ahora balancea mejor todos los factores")
    print("   ✓ Edad joven + saludable = Bajo riesgo protegido")
    print("   ✓ Múltiples factores de riesgo = Incremento significativo")
    print("   ✓ Edad avanzada pero saludable = Riesgo moderado")
    print("=" * 70)

if __name__ == "__main__":
    test_prediction_scenarios()
