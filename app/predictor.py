"""Lógica de predicción y procesamiento de datos"""
import joblib
import numpy as np
import logging
from typing import Dict, List, Tuple
from .models import PatientData
from .config import MODEL_PATH, SCALER_PATH, ENCODING_PATH

logger = logging.getLogger(__name__)

class CardiovascularRiskPredictor:
    """Clase para manejar las predicciones del modelo"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoding_info = None
        self._load_models()
    
    def _load_models(self):
        """Cargar los modelos desde los archivos pickle"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.encoding_info = joblib.load(ENCODING_PATH)
            logger.info("Modelos cargados exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            raise
    
    def preprocess_data(self, patient_data: PatientData) -> np.ndarray:
        """Preprocesar los datos del paciente para el modelo"""        # Convertir a diccionario
        data_dict = patient_data.model_dump()
        
        # Obtener encodings
        encodings = self.encoding_info['encodings']
        
        # Codificar variables categóricas
        data_dict['gender'] = encodings['Gender'][patient_data.gender]
        data_dict['smoking'] = encodings['Smoking'][patient_data.smoking]
        data_dict['diabetes'] = encodings['Diabetes'][patient_data.diabetes]
        data_dict['family_history'] = encodings['Family History'][patient_data.family_history]
        data_dict['obesity'] = encodings['Obesity'][patient_data.obesity]
        
        # Codificar alcohol_intake
        alcohol_mapping = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3}
        data_dict['alcohol_intake'] = alcohol_mapping.get(patient_data.alcohol_intake, 2)
        
        # Crear array en el orden correcto
        feature_order = self.encoding_info['features']
        processed_data = []
        
        # Mapeo de nombres
        feature_mapping = {
            'Age': 'age',
            'Gender': 'gender',
            'Smoking': 'smoking',
            'Alcohol Intake': 'alcohol_intake',
            'Exercise Hours': 'exercise_hours',
            'Diabetes': 'diabetes',
            'Family History': 'family_history',
            'Obesity': 'obesity',
            'Stress Level': 'stress_level'
        }
        
        for feature in feature_order:
            key = feature_mapping.get(feature, feature.lower().replace(' ', '_'))
            processed_data.append(data_dict[key])
        
        # Convertir a numpy array
        data_array = np.array(processed_data).reshape(1, -1)
        
        # Escalar características numéricas
        scaler_features = self.encoding_info['scaler_features']
        feature_indices = [feature_order.index(feat) for feat in scaler_features]
        
        scaled_data = data_array.copy()
        scaled_data[:, feature_indices] = self.scaler.transform(data_array[:, feature_indices])
        
        return scaled_data
    
    def predict(self, patient_data: PatientData) -> Tuple[int, float, List[float]]:
        """Realizar predicción de riesgo con lógica médica mejorada"""
        # Preprocesar datos
        processed_data = self.preprocess_data(patient_data)
        
        # Predicción base del modelo
        model_prediction = self.model.predict(processed_data)[0]
        model_probability = self.model.predict_proba(processed_data)[0, 1]
        
        # Aplicar lógica médica para ajustar la predicción
        adjusted_probability = self._apply_medical_logic(patient_data, model_probability)
        
        # Determinar predicción final basada en probabilidad ajustada
        final_prediction = 1 if adjusted_probability >= 0.5 else 0
        
        # Calcular intervalo de confianza
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict_proba(processed_data)[0, 1] 
                for tree in self.model.estimators_
            ])
            std_dev = np.std(tree_predictions)
            confidence_interval = [
                max(0, adjusted_probability - 1.96 * std_dev),
                min(1, adjusted_probability + 1.96 * std_dev)
            ]
        else:
            confidence_interval = [
                max(0, adjusted_probability - 0.1),
                min(1, adjusted_probability + 0.1)
            ]
        
        return final_prediction, adjusted_probability, confidence_interval
    
    def _apply_medical_logic(self, patient_data: PatientData, base_probability: float) -> float:
        """Aplicar lógica médica para ajustar la predicción del modelo"""
        # Comenzar con la probabilidad base del modelo
        adjusted_prob = base_probability
        
        # Calcular puntajes de factores de riesgo y protección
        risk_score = 0
        protective_score = 0
        
        # FACTORES DE RIESGO PRINCIPALES (aumentan el riesgo)
        
        # Edad - Factor progresivo más realista con diferencias de género
        if patient_data.gender == "Male":
            # Hombres tienen riesgo más temprano
            if patient_data.age >= 65:
                risk_score += 0.28  # Riesgo muy alto
            elif patient_data.age >= 55:
                risk_score += 0.18  # Riesgo alto
            elif patient_data.age >= 45:
                risk_score += 0.12  # Riesgo moderado
            elif patient_data.age >= 35:
                risk_score += 0.05  # Riesgo leve
            elif patient_data.age <= 25:
                protective_score += 0.18  # Protección por edad muy joven
        else:  # Female
            # Mujeres tienen protección hormonal hasta menopausia
            if patient_data.age >= 70:
                risk_score += 0.25  # Riesgo muy alto post-menopausia
            elif patient_data.age >= 60:
                risk_score += 0.15  # Riesgo alto post-menopausia
            elif patient_data.age >= 50:
                risk_score += 0.08  # Inicio de menopausia
            elif patient_data.age >= 40:
                risk_score += 0.03  # Riesgo mínimo pre-menopáusico
            elif patient_data.age <= 30:
                protective_score += 0.20  # Mayor protección hormonal en mujeres jóvenes
        
        # Tabaquismo - Factor muy importante con efectos acumulativos
        if patient_data.smoking == "Current":
            # El riesgo del tabaquismo aumenta con la edad
            base_smoking_risk = 0.22
            if patient_data.age >= 50:
                risk_score += base_smoking_risk + 0.08  # Efecto acumulativo
            elif patient_data.age >= 35:
                risk_score += base_smoking_risk + 0.04
            else:
                risk_score += base_smoking_risk
        elif patient_data.smoking == "Former":
            # Riesgo residual que disminuye con el tiempo (asumimos reciente)
            if patient_data.age >= 50:
                risk_score += 0.08  # Más riesgo residual en mayores
            else:
                risk_score += 0.04  # Menor riesgo residual en jóvenes
        
        # Diabetes - Factor crítico que empeora con la edad
        if patient_data.diabetes == "Yes":
            base_diabetes_risk = 0.20
            if patient_data.age >= 50:
                risk_score += base_diabetes_risk + 0.05  # Diabetes + edad = muy grave
            else:
                risk_score += base_diabetes_risk
        
        # Obesidad - Factor importante, más grave en jóvenes relativamente
        if patient_data.obesity == "Yes":
            if patient_data.age <= 40:
                risk_score += 0.15  # Obesidad temprana es más preocupante
            else:
                risk_score += 0.12
        
        # Historia familiar - Más relevante en jóvenes
        if patient_data.family_history == "Yes":
            if patient_data.age <= 45:
                risk_score += 0.12  # Historia familiar importante en jóvenes
            else:
                risk_score += 0.08  # Menos relevante en mayores
        
        # Alcohol - Efecto complejo dependiendo de cantidad y edad
        if patient_data.alcohol_intake == "Heavy":
            if patient_data.age >= 50:
                risk_score += 0.12  # Más dañino en mayores
            else:
                risk_score += 0.08
        
        # Estrés - Impacto variable según edad y otros factores
        if patient_data.stress_level >= 8:
            base_stress_risk = 0.12
            # El estrés es más dañino si ya hay otros factores
            stress_multiplier = 1 + (risk_score * 0.5)  # Efecto sinérgico
            risk_score += base_stress_risk * min(stress_multiplier, 1.5)
        elif patient_data.stress_level >= 6:
            risk_score += 0.06
        
        # FACTORES PROTECTORES (reducen el riesgo)
        
        # Ejercicio regular - Factor protector con efectos exponenciales
        if patient_data.exercise_hours >= 7:
            protective_score += 0.20  # Ejercicio muy intenso
        elif patient_data.exercise_hours >= 5:
            protective_score += 0.16  # Excelente nivel de ejercicio
        elif patient_data.exercise_hours >= 3:
            protective_score += 0.12   # Buen nivel de ejercicio
        elif patient_data.exercise_hours >= 1.5:
            protective_score += 0.07  # Ejercicio moderado
        elif patient_data.exercise_hours >= 1:
            protective_score += 0.03  # Ejercicio mínimo
        elif patient_data.exercise_hours < 0.5:
            risk_score += 0.12  # Sedentarismo extremo
        elif patient_data.exercise_hours < 1:
            risk_score += 0.08  # Sedentarismo como factor de riesgo
        
        # No fumar - Factor protector que aumenta con la edad
        if patient_data.smoking == "Never":
            if patient_data.age >= 50:
                protective_score += 0.15  # Muy protector en mayores
            else:
                protective_score += 0.10
        
        # Consumo de alcohol - Curva en J (moderado puede ser protector)
        if patient_data.alcohol_intake == "None":
            protective_score += 0.06  # Abstinencia total es protectora
        elif patient_data.alcohol_intake == "Light":
            protective_score += 0.08  # Consumo ligero óptimo
        elif patient_data.alcohol_intake == "Moderate":
            if patient_data.age >= 40:
                protective_score += 0.04  # Beneficio moderado en adultos
            else:
                protective_score += 0.01  # Menor beneficio en jóvenes
        
        # Manejo del estrés - Más importante de lo que se piensa
        if patient_data.stress_level <= 2:
            protective_score += 0.12  # Estrés muy bajo es muy protector
        elif patient_data.stress_level <= 4:
            protective_score += 0.08
        elif patient_data.stress_level <= 6:
            protective_score += 0.04
        
        # Ausencia de condiciones médicas - Beneficio compuesto
        medical_conditions_absent = 0
        if patient_data.diabetes == "No":
            protective_score += 0.06
            medical_conditions_absent += 1
        
        if patient_data.obesity == "No":
            protective_score += 0.06
            medical_conditions_absent += 1
        
        # Bonus por múltiples factores saludables
        if medical_conditions_absent == 2 and patient_data.smoking == "Never":
            protective_score += 0.05  # Bonus por perfil muy limpio
        
        # AJUSTE FINAL CON LÓGICA MÉDICA AVANZADA
        # Aplicar factores de riesgo y protección
        net_adjustment = risk_score - protective_score
        
        # Peso dinámico del modelo vs lógica médica según confiabilidad
        # Si tenemos muchos factores claros, damos más peso a la lógica médica
        total_factors = risk_score + protective_score
        if total_factors > 0.4:  # Muchos factores identificados
            model_weight = 0.45
            logic_weight = 0.55
        elif total_factors > 0.2:  # Factores moderados
            model_weight = 0.55
            logic_weight = 0.45
        else:  # Pocos factores, confiamos más en el modelo
            model_weight = 0.65
            logic_weight = 0.35
        
        # Calcular probabilidad ajustada con límites más inteligentes
        logic_probability = max(0.02, min(0.98, base_probability + net_adjustment))
        adjusted_prob = (model_weight * base_probability) + (logic_weight * logic_probability)
        
        # CASOS ESPECIALES - Combinaciones extremas con lógica médica real
        
        # Perfil muy joven y completamente saludable
        if (patient_data.age <= 25 and 
            patient_data.smoking == "Never" and 
            patient_data.diabetes == "No" and 
            patient_data.obesity == "No" and 
            patient_data.exercise_hours >= 3 and
            patient_data.stress_level <= 5):
            adjusted_prob = min(adjusted_prob, 0.12)  # Máximo 12% de riesgo
        
        # Perfil de alto riesgo indiscutible
        if (patient_data.age >= 60 and 
            patient_data.smoking == "Current" and 
            patient_data.diabetes == "Yes"):
            adjusted_prob = max(adjusted_prob, 0.65)  # Mínimo 65% de riesgo
        
        # Mujer joven con factores protectores
        if (patient_data.gender == "Female" and 
            patient_data.age <= 35 and 
            patient_data.smoking == "Never" and 
            patient_data.exercise_hours >= 2):
            adjusted_prob = min(adjusted_prob, 0.08)  # Máximo 8% por protección hormonal
        
        # Hombre joven con múltiples factores de riesgo (más preocupante)
        if (patient_data.gender == "Male" and 
            patient_data.age <= 35 and 
            patient_data.smoking == "Current" and 
            (patient_data.diabetes == "Yes" or patient_data.obesity == "Yes")):
            adjusted_prob = max(adjusted_prob, 0.25)  # Preocupante en hombres jóvenes
        
        # Síndrome metabólico (obesidad + diabetes + sedentarismo)
        if (patient_data.obesity == "Yes" and 
            patient_data.diabetes == "Yes" and 
            patient_data.exercise_hours < 1):
            adjusted_prob = max(adjusted_prob, 0.55)  # Muy alto riesgo
        
        # Perfil de ejercicio extremo (protección adicional)
        if (patient_data.exercise_hours >= 6 and 
            patient_data.smoking == "Never" and 
            patient_data.stress_level <= 4):
            adjusted_prob = adjusted_prob * 0.8  # Reducción del 20% por ejercicio intenso
        
        # Múltiples factores de riesgo con consideración de edad
        major_risk_factors = sum([
            patient_data.smoking == "Current",
            patient_data.diabetes == "Yes", 
            patient_data.obesity == "Yes",
            patient_data.family_history == "Yes",
            patient_data.exercise_hours < 1,
            patient_data.stress_level >= 8,
            patient_data.alcohol_intake == "Heavy"
        ])
        
        # Ajustes por múltiples factores más precisos
        if major_risk_factors >= 4:
            if patient_data.age >= 50:
                adjusted_prob = max(adjusted_prob, 0.60)  # Muy grave en mayores
            elif patient_data.age >= 30:
                adjusted_prob = max(adjusted_prob, 0.45)  # Preocupante en adultos
            else:
                adjusted_prob = max(adjusted_prob, 0.35)  # Alarmante en jóvenes
        elif major_risk_factors >= 3:
            if patient_data.age >= 45:
                adjusted_prob = max(adjusted_prob, 0.50)
            elif patient_data.age >= 25:
                adjusted_prob = max(adjusted_prob, 0.40)
        elif major_risk_factors >= 2:
            if patient_data.age >= 55:
                adjusted_prob = max(adjusted_prob, 0.40)
        
        # Efecto sinérgico: algunos factores se potencian entre sí
        if (patient_data.smoking == "Current" and patient_data.diabetes == "Yes"):
            adjusted_prob = adjusted_prob * 1.15  # Combinación especialmente dañina
        
        if (patient_data.stress_level >= 8 and patient_data.exercise_hours < 1):
            adjusted_prob = adjusted_prob * 1.10  # Estrés + sedentarismo
        
        # Asegurar que esté en rango válido
        return max(0.01, min(0.99, adjusted_prob))
    
    def get_risk_level(self, probability: float) -> str:
        """Determinar el nivel de riesgo basado en la probabilidad ajustada"""
        if probability >= 0.6:
            return "Alto Riesgo"
        elif probability >= 0.35:
            return "Riesgo Moderado"
        else:
            return "Bajo Riesgo"
    
    def get_risk_factors(self, patient_data: PatientData) -> Dict[str, str]:
        """Identificar factores de riesgo del paciente con análisis más detallado"""
        risk_factors = {}
        
        # Edad con contexto de género
        if patient_data.gender == "Male":
            if patient_data.age >= 60:
                risk_factors["Edad"] = f"{patient_data.age} años - Hombre (riesgo alto por edad avanzada)"
            elif patient_data.age >= 45:
                risk_factors["Edad"] = f"{patient_data.age} años - Hombre (riesgo moderado por edad)"
            elif patient_data.age >= 35:
                risk_factors["Edad"] = f"{patient_data.age} años - Hombre (inicio de riesgo por edad)"
        else:  # Female
            if patient_data.age >= 65:
                risk_factors["Edad"] = f"{patient_data.age} años - Mujer (riesgo alto post-menopáusico)"
            elif patient_data.age >= 55:
                risk_factors["Edad"] = f"{patient_data.age} años - Mujer (riesgo moderado post-menopáusico)"
            elif patient_data.age >= 50:
                risk_factors["Edad"] = f"{patient_data.age} años - Mujer (transición menopáusica)"
        
        # Tabaquismo con historia
        if patient_data.smoking == "Current":
            if patient_data.age >= 50:
                risk_factors["Tabaquismo"] = "Fumador actual - CRÍTICO en su edad (daño acumulativo)"
            else:
                risk_factors["Tabaquismo"] = "Fumador actual - Factor de riesgo mayor modificable"
        elif patient_data.smoking == "Former":
            risk_factors["Tabaquismo"] = "Ex-fumador (riesgo residual, pero ¡felicitaciones por dejarlo!)"
        
        # Diabetes con severidad por edad
        if patient_data.diabetes == "Yes":
            if patient_data.age >= 50:
                risk_factors["Diabetes"] = "Diabetes presente - MUY GRAVE en su edad (riesgo cardiovascular duplicado)"
            else:
                risk_factors["Diabetes"] = "Diabetes presente - Requiere control estricto para prevenir complicaciones"
        
        # Obesidad con contexto
        if patient_data.obesity == "Yes":
            if patient_data.age <= 40:
                risk_factors["Obesidad"] = "Obesidad presente - Especialmente preocupante a su edad"
            else:
                risk_factors["Obesidad"] = "Obesidad presente - Factor de riesgo modificable importante"
        
        # Historia familiar con relevancia por edad
        if patient_data.family_history == "Yes":
            if patient_data.age <= 45:
                risk_factors["Historia Familiar"] = "Historia familiar positiva - MUY RELEVANTE a su edad (predisposición genética)"
            else:
                risk_factors["Historia Familiar"] = "Historia familiar positiva - Aumenta el riesgo base"
        
        # Sedentarismo con gradientes
        if patient_data.exercise_hours < 0.5:
            risk_factors["Sedentarismo"] = f"Sedentarismo extremo - Solo {patient_data.exercise_hours}h/semana (¡URGENTE cambiar!)"
        elif patient_data.exercise_hours < 1.5:
            risk_factors["Sedentarismo"] = f"Actividad insuficiente - {patient_data.exercise_hours}h/semana (mínimo recomendado: 2.5h)"
        elif patient_data.exercise_hours < 2.5:
            risk_factors["Actividad Baja"] = f"Ejercicio por debajo del óptimo - {patient_data.exercise_hours}h/semana"
        
        # Estrés con impacto
        if patient_data.stress_level >= 9:
            risk_factors["Estrés"] = f"Estrés extremo - Nivel {patient_data.stress_level}/10 (¡CRISIS! Afecta directamente al corazón)"
        elif patient_data.stress_level >= 8:
            risk_factors["Estrés"] = f"Estrés muy alto - Nivel {patient_data.stress_level}/10 (daño cardiovascular comprobado)"
        elif patient_data.stress_level >= 6:
            risk_factors["Estrés"] = f"Estrés elevado - Nivel {patient_data.stress_level}/10 (puede aumentar presión arterial)"
        
        # Alcohol con contexto
        if patient_data.alcohol_intake == "Heavy":
            if patient_data.age >= 50:
                risk_factors["Alcohol"] = "Consumo alto de alcohol - Especialmente dañino a su edad"
            else:
                risk_factors["Alcohol"] = "Consumo alto de alcohol - Factor de riesgo cardiovascular"
        
        # Factores sinérgicos (combinaciones peligrosas)
        if patient_data.smoking == "Current" and patient_data.diabetes == "Yes":
            risk_factors["⚠️ COMBINACIÓN CRÍTICA"] = "Diabetes + Tabaquismo = Riesgo exponencial"
        
        if patient_data.obesity == "Yes" and patient_data.diabetes == "Yes" and patient_data.exercise_hours < 1:
            risk_factors["⚠️ SÍNDROME METABÓLICO"] = "Obesidad + Diabetes + Sedentarismo = Muy alto riesgo"
        
        if patient_data.stress_level >= 8 and patient_data.exercise_hours < 1:
            risk_factors["⚠️ CÍRCULO VICIOSO"] = "Estrés alto + Sedentarismo = Se refuerzan mutuamente"
        
        return risk_factors
    
    def get_recommendations(self, probability: float, risk_factors: Dict[str, str]) -> List[str]:
        """Generar recomendaciones personalizadas basadas en la probabilidad ajustada"""
        recommendations = []
        
        # Recomendaciones según nivel de riesgo ajustado
        if probability >= 0.6:
            recommendations.append("🚨 ALTA PRIORIDAD: Consulte a un cardiólogo INMEDIATAMENTE")
            recommendations.append("📊 Exámenes urgentes: ECG, perfil lipídico completo, ecocardiograma")
            recommendations.append("💊 Evalúe medicación preventiva con su médico (estatinas, aspirina)")
            recommendations.append("📈 Monitoreo semanal de presión arterial")
        elif probability >= 0.35:
            recommendations.append("⚠️ Programe consulta médica en las próximas 2-4 semanas")
            recommendations.append("🔍 Realice exámenes cardiovasculares: ECG, perfil lipídico")
            recommendations.append("� Chequeos médicos cada 3-6 meses")
            recommendations.append("📈 Monitoree presión arterial mensualmente")
        else:
            recommendations.append("✅ Excelente perfil cardiovascular - Continue así")
            recommendations.append("📅 Mantenga chequeos médicos anuales preventivos")
            recommendations.append("🎯 Enfoque en mantener hábitos saludables actuales")
            recommendations.append("💚 Su estilo de vida está protegiendo su corazón")
        
        # Recomendaciones específicas por factor de riesgo
        if "Tabaquismo" in risk_factors:
            if "actual" in risk_factors["Tabaquismo"]:
                recommendations.append("🚭 CRÍTICO: Deje de fumar HOY - es su prioridad #1")
                recommendations.append("📞 Línea de ayuda para dejar de fumar: consiga apoyo profesional")
            else:
                recommendations.append("👏 Excelente por haber dejado de fumar - manténgase así")
        
        if "Sedentarismo" in risk_factors:
            recommendations.append("🏃‍♂️ URGENTE: Inicie programa de ejercicio gradual")
            recommendations.append("🚶‍♂️ Comience con 15 min diarios de caminata, aumente gradualmente")
            recommendations.append("🎯 Meta: 150 minutos de ejercicio moderado por semana")
        elif probability < 0.35:  # Si tiene bajo riesgo, reconocer su buen nivel de actividad
            recommendations.append("🏆 Su nivel de ejercicio está protegiendo su corazón")
        
        if "Obesidad" in risk_factors:
            recommendations.append("⚖️ IMPORTANTE: Plan de pérdida de peso con nutricionista")
            recommendations.append("🥗 Dieta mediterránea recomendada para la salud cardiovascular")
            recommendations.append("🎯 Objetivo inicial: reducir 5-10% del peso actual")
        
        if "Estrés" in risk_factors:
            if "muy alto" in risk_factors["Estrés"]:
                recommendations.append("🧘‍♂️ URGENTE: Técnicas de manejo del estrés - meditación, yoga")
                recommendations.append("😴 Priorice 7-8 horas de sueño reparador")
                recommendations.append("🗣️ Considere apoyo psicológico para manejo del estrés")
            else:
                recommendations.append("🧘‍♂️ Practique relajación diaria: 10-15 minutos")
        
        if "Diabetes" in risk_factors:
            recommendations.append("🩺 CRÍTICO: Control estricto de diabetes con endocrinólogo")
            recommendations.append("📊 Hemoglobina glicada (HbA1c) objetivo: < 7%")
            recommendations.append("🍎 Nutricionista especializada en diabetes")
        
        if "Alcohol" in risk_factors and "alto" in risk_factors["Alcohol"]:
            recommendations.append("🍷 Reduzca consumo de alcohol: máximo 1-2 bebidas/día")
            recommendations.append("🚫 Considere días sin alcohol durante la semana")
        
        # Recomendaciones adicionales para bajo riesgo (refuerzo positivo)
        if probability < 0.2:
            recommendations.append("🌟 ¡Felicitaciones! Su perfil cardiovascular es excelente")
            recommendations.append("🛡️ Sus hábitos saludables son su mejor medicina preventiva")
        
        return recommendations

# Instancia global del predictor
predictor = CardiovascularRiskPredictor()
