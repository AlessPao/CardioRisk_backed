"""
Script de prueba avanzado para verificar las mejoras realistas en la lógica de predicción
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import PatientData
from app.predictor import predictor

def test_realistic_scenarios():
    """Probar escenarios más realistas y complejos"""
    
    print("🧬 TESTING LÓGICA MÉDICA AVANZADA - PREDICCIONES REALISTAS")
    print("=" * 80)
    
    scenarios = [
        {
            "name": "👩 Mujer joven muy saludable",
            "data": PatientData(
                age=24, gender="Female", smoking="Never", alcohol_intake="Light",
                exercise_hours=4.0, diabetes="No", family_history="No", 
                obesity="No", stress_level=3
            ),
            "expected": "Muy bajo riesgo (< 10%)"
        },
        {
            "name": "👨 Hombre joven con múltiples factores",
            "data": PatientData(
                age=32, gender="Male", smoking="Current", alcohol_intake="Heavy",
                exercise_hours=0.5, diabetes="Yes", family_history="Yes", 
                obesity="Yes", stress_level=9
            ),
            "expected": "Alto riesgo por factores múltiples (40-60%)"
        },
        {
            "name": "👩 Mujer post-menopáusica saludable",
            "data": PatientData(
                age=58, gender="Female", smoking="Never", alcohol_intake="Light",
                exercise_hours=5.0, diabetes="No", family_history="No", 
                obesity="No", stress_level=4
            ),
            "expected": "Riesgo moderado por edad, mitigado por hábitos (25-40%)"
        },
        {
            "name": "👨 Hombre mayor con diabetes controlada",
            "data": PatientData(
                age=67, gender="Male", smoking="Former", alcohol_intake="Moderate",
                exercise_hours=3.0, diabetes="Yes", family_history="No", 
                obesity="No", stress_level=5
            ),
            "expected": "Alto riesgo por edad+diabetes, mejorado por ejercicio (50-70%)"
        },
        {
            "name": "👩 Mujer joven con historia familiar fuerte",
            "data": PatientData(
                age=28, gender="Female", smoking="Never", alcohol_intake="None",
                exercise_hours=6.0, diabetes="No", family_history="Yes", 
                obesity="No", stress_level=2
            ),
            "expected": "Bajo riesgo pese a historia familiar (10-20%)"
        },
        {
            "name": "👨 Atleta masculino con estrés extremo",
            "data": PatientData(
                age=35, gender="Male", smoking="Never", alcohol_intake="None",
                exercise_hours=8.0, diabetes="No", family_history="No", 
                obesity="No", stress_level=10
            ),
            "expected": "Bajo-moderado (ejercicio vs estrés) (15-30%)"
        },
        {
            "name": "👩 Mujer con síndrome metabólico",
            "data": PatientData(
                age=45, gender="Female", smoking="Current", alcohol_intake="Heavy",
                exercise_hours=0.0, diabetes="Yes", family_history="Yes", 
                obesity="Yes", stress_level=8
            ),
            "expected": "Muy alto riesgo - todos los factores (70-85%)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 ESCENARIO {i}: {scenario['name']}")
        print("-" * 60)
        
        # Realizar predicción
        prediction, probability, confidence = predictor.predict(scenario['data'])
        risk_level = predictor.get_risk_level(probability)
        risk_factors = predictor.get_risk_factors(scenario['data'])
        recommendations = predictor.get_recommendations(probability, risk_factors)
        
        # Mostrar resultados
        print(f"   👤 Perfil: {scenario['data'].gender}, {scenario['data'].age} años")
        print(f"   🎯 Predicción: {risk_level}")
        print(f"   📊 Probabilidad: {probability:.1%}")
        print(f"   📋 Esperado: {scenario['expected']}")
        
        # Factores de riesgo identificados
        if risk_factors:
            print(f"   🔴 Factores de riesgo ({len(risk_factors)}):")
            for factor, description in list(risk_factors.items())[:3]:  # Mostrar solo los primeros 3
                print(f"      • {factor}: {description[:50]}{'...' if len(description) > 50 else ''}")
        
        # Algunas recomendaciones
        if recommendations:
            print(f"   💡 Recomendaciones principales ({len(recommendations)} total):")
            for rec in recommendations[:2]:  # Mostrar solo las primeras 2
                print(f"      • {rec}")
        
        print()
    
    print("=" * 80)
    print("🎯 RESUMEN DE MEJORAS IMPLEMENTADAS:")
    print("   ✓ Diferencias de género (protección hormonal femenina)")
    print("   ✓ Interacciones entre factores (efectos sinérgicos)")
    print("   ✓ Pesos dinámicos modelo vs lógica médica")
    print("   ✓ Efectos no lineales del ejercicio")
    print("   ✓ Curva en J del alcohol")
    print("   ✓ Relevancia de factores según edad")
    print("   ✓ Combinaciones críticas (diabetes + tabaquismo)")
    print("   ✓ Protección especial para jóvenes muy saludables")
    print("   ✓ Penalización por síndrome metabólico")
    print("=" * 80)

def test_edge_cases():
    """Probar casos extremos para verificar límites"""
    
    print("\n🧪 CASOS EXTREMOS:")
    print("=" * 50)
    
    # Caso 1: Perfección absoluta
    perfect_case = PatientData(
        age=22, gender="Female", smoking="Never", alcohol_intake="Light",
        exercise_hours=7.0, diabetes="No", family_history="No", 
        obesity="No", stress_level=1
    )
    _, prob, _ = predictor.predict(perfect_case)
    print(f"🌟 Perfección absoluta: {prob:.1%}")
    
    # Caso 2: Desastre absoluto
    disaster_case = PatientData(
        age=68, gender="Male", smoking="Current", alcohol_intake="Heavy",
        exercise_hours=0.0, diabetes="Yes", family_history="Yes", 
        obesity="Yes", stress_level=10
    )
    _, prob, _ = predictor.predict(disaster_case)
    print(f"💀 Desastre absoluto: {prob:.1%}")
    
    # Caso 3: Contradicción interesante
    contradiction_case = PatientData(
        age=25, gender="Male", smoking="Current", alcohol_intake="Heavy",
        exercise_hours=8.0, diabetes="No", family_history="No", 
        obesity="No", stress_level=2
    )
    _, prob, _ = predictor.predict(contradiction_case)
    print(f"🤔 Atleta fumador joven: {prob:.1%}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_realistic_scenarios()
    test_edge_cases()
