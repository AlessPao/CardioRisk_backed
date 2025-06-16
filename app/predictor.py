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
        """Realizar predicción de riesgo"""
        # Preprocesar datos
        processed_data = self.preprocess_data(patient_data)
        
        # Predicción
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0, 1]
        
        # Calcular intervalo de confianza
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict_proba(processed_data)[0, 1] 
                for tree in self.model.estimators_
            ])
            std_dev = np.std(tree_predictions)
            confidence_interval = [
                max(0, probability - 1.96 * std_dev),
                min(1, probability + 1.96 * std_dev)
            ]
        else:
            confidence_interval = [
                max(0, probability - 0.1),
                min(1, probability + 0.1)
            ]
        
        return prediction, probability, confidence_interval
    
    def get_risk_level(self, probability: float) -> str:
        """Determinar el nivel de riesgo basado en la probabilidad"""
        if probability >= 0.7:
            return "Alto Riesgo"
        elif probability >= 0.4:
            return "Riesgo Moderado"
        else:
            return "Bajo Riesgo"
    
    def get_risk_factors(self, patient_data: PatientData) -> Dict[str, str]:
        """Identificar factores de riesgo del paciente"""
        risk_factors = {}
        
        # Edad
        if patient_data.age >= 60:
            risk_factors["Edad"] = f"{patient_data.age} años (riesgo alto por edad avanzada)"
        elif patient_data.age >= 45:
            risk_factors["Edad"] = f"{patient_data.age} años (riesgo moderado por edad)"
        
        # Tabaquismo
        if patient_data.smoking == "Current":
            risk_factors["Tabaquismo"] = "Fumador actual (factor de riesgo importante)"
        elif patient_data.smoking == "Former":
            risk_factors["Tabaquismo"] = "Ex-fumador (riesgo residual)"
        
        # Diabetes
        if patient_data.diabetes == "Yes":
            risk_factors["Diabetes"] = "Presente (aumenta significativamente el riesgo)"
        
        # Obesidad
        if patient_data.obesity == "Yes":
            risk_factors["Obesidad"] = "Presente (factor de riesgo modificable)"
        
        # Historia familiar
        if patient_data.family_history == "Yes":
            risk_factors["Historia Familiar"] = "Positiva (predisposición genética)"
        
        # Sedentarismo
        if patient_data.exercise_hours < 2:
            risk_factors["Sedentarismo"] = f"Solo {patient_data.exercise_hours}h/semana de ejercicio"
        
        # Estrés
        if patient_data.stress_level >= 8:
            risk_factors["Estrés"] = f"Nivel {patient_data.stress_level}/10 (muy alto)"
        elif patient_data.stress_level >= 6:
            risk_factors["Estrés"] = f"Nivel {patient_data.stress_level}/10 (moderado-alto)"
        
        # Alcohol
        if patient_data.alcohol_intake == "Heavy":
            risk_factors["Alcohol"] = "Consumo alto (factor de riesgo)"
        
        return risk_factors
    
    def get_recommendations(self, probability: float, risk_factors: Dict[str, str]) -> List[str]:
        """Generar recomendaciones personalizadas"""
        recommendations = []
        
        # Recomendaciones según nivel de riesgo
        if probability >= 0.7:
            recommendations.append("⚠️ CONSULTE A UN CARDIÓLOGO lo antes posible")
            recommendations.append("📊 Solicite exámenes completos: electrocardiograma, perfil lipídico")
            recommendations.append("💊 Evalúe con su médico la necesidad de medicación preventiva")
        elif probability >= 0.4:
            recommendations.append("👨‍⚕️ Programe una consulta médica para evaluación cardiovascular")
            recommendations.append("🔍 Realice chequeos médicos semestrales")
            recommendations.append("📈 Monitoree su presión arterial regularmente")
        else:
            recommendations.append("✅ Mantenga sus hábitos saludables actuales")
            recommendations.append("📅 Continúe con chequeos médicos anuales")
            recommendations.append("🎯 Siga enfocándose en la prevención")
        
        # Recomendaciones específicas
        if "Tabaquismo" in risk_factors and "actual" in risk_factors["Tabaquismo"]:
            recommendations.append("🚭 DEJE DE FUMAR: es la medida más importante que puede tomar")
        
        if "Sedentarismo" in risk_factors:
            recommendations.append("🏃‍♂️ Aumente gradualmente el ejercicio hasta 150 min/semana")
            recommendations.append("🚶‍♂️ Comience con caminatas de 30 minutos diarios")
        
        if "Obesidad" in risk_factors:
            recommendations.append("🥗 Consulte un nutricionista para un plan alimenticio")
            recommendations.append("⚖️ Objetivo: reducir 5-10% del peso corporal")
        
        if "Estrés" in risk_factors and ("alto" in risk_factors["Estrés"] or "moderado" in risk_factors["Estrés"]):
            recommendations.append("🧘‍♂️ Practique técnicas de relajación: meditación, yoga")
            recommendations.append("😴 Priorice dormir 7-8 horas diarias")
        
        if "Diabetes" in risk_factors:
            recommendations.append("🩺 Control estricto de glucosa con endocrinólogo")
            recommendations.append("🍎 Dieta específica para diabéticos")
        
        if "Alcohol" in risk_factors and "alto" in risk_factors["Alcohol"]:
            recommendations.append("🍷 Reduzca el consumo de alcohol gradualmente")
        
        return recommendations

# Instancia global del predictor
predictor = CardiovascularRiskPredictor()
