from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import os

# Definir el modelo de entrada de datos
class InputData(BaseModel):
    rsi: float
    sma: float
    bb_high: float
    bb_low: float
    volume: float

# Cargar el modelo y escaladores
SYMBOL = "TSLA"
model_path = f"/Users/sebabustos/Documents/trading_bot/models/{SYMBOL}_model.keras"
scaler_X_path = f"/Users/sebabustos/Documents/trading_bot/models/{SYMBOL}_1m_scaler_X.pkl"
scaler_y_path = f"/Users/sebabustos/Documents/trading_bot/models/{SYMBOL}_1m_scaler_y.pkl"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    model_loaded = True
else:
    model = None
    model_loaded = False

if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    scaling_enabled = True
else:
    scaler_X = None
    scaler_y = None
    scaling_enabled = False

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 API del modelo de trading está en ejecución"}

@app.post("/predict/")
def predict(data: InputData):
    """Recibe un JSON con los valores y devuelve la predicción."""
    if not model_loaded:
        return {"error": "El modelo no está disponible. Verifica la ruta o entrena el modelo."}

    try:
        # Convertir los datos a un array de numpy
        input_data = np.array([[data.rsi, data.sma, data.bb_high, data.bb_low, data.volume]])

        print(f"🔍 Datos originales antes de escalar: {input_data}")  # ✅ Verificación inicial

        # Aplicar escalado si está disponible
        if scaling_enabled:
            input_data_scaled = scaler_X.transform(input_data)
        else:
            input_data_scaled = input_data  # No aplicar escalado si no está habilitado

        print(f"🔍 Datos escalados antes de LSTM: {input_data_scaled}")  # ✅ Verificación post-escalado

        # Ajustar forma para el modelo LSTM
        WINDOW_SIZE = 12
        FEATURES = 5  # Aseguramos que coincida con el entrenamiento
        input_data_lstm = np.tile(input_data_scaled, (WINDOW_SIZE, 1)).reshape(1, WINDOW_SIZE, FEATURES)

        # Hacer la predicción
        prediction_scaled = model.predict(input_data_lstm)

        print(f"🔍 Predicción escalada antes de desescalar: {prediction_scaled}")  # ✅ Verificación antes de desescalar

        # Desescalar el valor de salida
        if scaling_enabled:
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        else:
            prediction = prediction_scaled[0][0]

        print(f"🔍 Predicción desescalada final: {prediction}")  # ✅ Verificación después de desescalar

        return {
            "rsi": round(float(data.rsi), 2),
            "sma": round(float(data.sma), 2),
            "bb_high": round(float(data.bb_high), 2),
            "bb_low": round(float(data.bb_low), 2),
            "volume": round(float(data.volume), 2),
            "prediction": round(float(prediction), 2),  # 🔥 Convertir `numpy.float32` a `float`
            "scaling_used": scaling_enabled
        }

    except Exception as e:
        return {"error": str(e)}
