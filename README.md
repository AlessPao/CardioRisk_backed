 Cardiovascular Risk Prediction API
API REST para predicción de riesgo cardiovascular usando Machine Learning (Random Forest).
🚀 Instalación

Clonar el repositorio:

bashgit clone <tu-repositorio>
cd cardiovascular-risk-api

Crear entorno virtual:

bashpython -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

Instalar dependencias:

bashpip install -r requirements.txt

Asegurarse de que los archivos .pkl estén en la carpeta models/:

cardiovascular_risk_model.pkl
feature_scaler.pkl
encoding_info.pkl



🏃‍♂️ Ejecutar localmente
bash# Desarrollo con recarga automática
uvicorn app.main:app --reload

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000
La API estará disponible en: http://localhost:8000
📚 Documentación

Documentación interactiva: http://localhost:8000/docs
Documentación alternativa: http://localhost:8000/redoc

🧪 Tests
Ejecutar tests:
bashpytest tests/
📋 Endpoints
GET /
Información general de la API
GET /health
Estado de salud de la API
POST /predict
Predicción de riesgo cardiovascular
Body de ejemplo:
json{
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
🚀 Despliegue en Render

Crear cuenta en Render
Conectar repositorio GitHub
Crear nuevo Web Service
Configurar:

Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT



📊 Modelo

Algoritmo: Random Forest (200 árboles)
Accuracy: 88%
AUC-ROC: 0.908
Features: 9 variables de entrada

🔒 Seguridad

Validación estricta de datos de entrada
CORS configurado
Manejo seguro de errores
