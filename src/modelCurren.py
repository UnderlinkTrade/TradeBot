import os
import gc
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from configCurren import SYMBOLS  # Importar lista de símbolos

# 🔹 Configuración de TensorFlow para optimizar uso de CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# 🔹 Parámetros del modelo
WINDOW_SIZE = 12  # Se amplía la ventana para capturar más contexto
FEATURES = 15  # Se agregan Volumen_Relativo, RSI_15m, RSI_1h, Distancia_SMA20

def cargar_modelo(symbol):
    """Carga el modelo LSTM entrenado para una acción específica."""
    symbol_formatted = symbol.replace(":", "_")
    model_filename = f"models/{symbol_formatted}_model.keras"
    
    if not os.path.exists(model_filename):
        print(f"❌ Error: No se encontró el modelo LSTM para {symbol} en {model_filename}.")
        return None

    try:
        model = tf.keras.models.load_model(model_filename)
        print(f"✅ Modelo LSTM para {symbol} cargado correctamente.")
        return model
    except Exception as e:
        print(f"⚠️ Error al cargar el modelo LSTM para {symbol}: {e}")
        return None

def entrenar_y_guardar_modelos():
    """Entrena un modelo LSTM para cada acción en SYMBOLS."""
    os.makedirs("models", exist_ok=True)

    for symbol in SYMBOLS:
        gc.collect()
        symbol_formatted = symbol.replace(":", "_")
        data_file = f"data/{symbol_formatted}_5m_data.csv"

        if not os.path.exists(data_file):
            print(f"⚠️ Error: El archivo '{data_file}' no existe para {symbol}.")
            continue

        data = pd.read_csv(data_file, parse_dates=["date"])
        if len(data) < WINDOW_SIZE:
            print(f"⚠️ No hay suficientes datos para entrenar el modelo de {symbol}.")
            continue

        # 🔹 Definir las columnas de features antes de la verificación
        feature_columns = ["RSI", "MACD", "EMA_20", "ATR", "SMA_20", "BB_High", "BB_Low", "v", 
                           "ADX", "+DI", "-DI", "Volumen_Relativo", "RSI_15m", "RSI_1h", "Distancia_SMA20"]

        # 🔹 Verificar si las columnas existen en el dataset antes de continuar
        if data.empty or data.columns is None:
            print(f"❌ Error: Dataset vacío o con formato incorrecto para {symbol}.")
            continue  # Saltar este activo

        missing_features = [feature for feature in feature_columns if feature not in data.columns]
        if missing_features:
            print(f"❌ Error: Las siguientes columnas faltan en el dataset de {symbol}: {missing_features}")
            continue  # Saltar entrenamiento si faltan columnas

        feature_columns = ["RSI", "MACD", "EMA_20", "ATR", "SMA_20", "BB_High", "BB_Low", "v", "ADX", "+DI", "-DI", "Volumen_Relativo", "RSI_15m", "RSI_1h", "Distancia_SMA20"]
        target_column = "c"  # Precio de cierre como objetivo

        X, y = [], []
        for i in range(len(data) - WINDOW_SIZE):
            X.append(data[feature_columns].iloc[i: i + WINDOW_SIZE].values)
            y.append(data[target_column].iloc[i + WINDOW_SIZE])

        X, y = np.array(X), np.array(y)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # 🔹 Guardar escaladores
        joblib.dump(scaler_X, f"models/{symbol_formatted}_5m_scaler_X.pkl")
        joblib.dump(scaler_y, f"models/{symbol_formatted}_5m_scaler_y.pkl")

        # 🔹 Definir la arquitectura optimizada del modelo
        model = Sequential([
            Input(shape=(WINDOW_SIZE, FEATURES)),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.1),

            Dense(10, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=tf.keras.losses.Huber(),
            metrics=['mae']
        )

        checkpoint_callback = ModelCheckpoint(
            filepath=f"models/{symbol_formatted}_best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            checkpoint_callback
        ]

        print(f"🚀 Iniciando entrenamiento para {symbol}...")
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=callbacks)

        # 🔹 Guardar modelo entrenado
        model.save(f"models/{symbol_formatted}_model.keras")
        print(f"✅ Modelo para {symbol} guardado en models/{symbol_formatted}_model.keras")
        gc.collect()

if __name__ == "__main__":
    print("🚀 Iniciando el entrenamiento de modelos para múltiples acciones...")
    entrenar_y_guardar_modelos()
    print("✅ Entrenamiento completado y modelos guardados.")