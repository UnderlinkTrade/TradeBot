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
# Longitud de la ventana de observaciones.  Se mantiene en 12 para capturar
# una hora de datos de velas de 5 min.
WINDOW_SIZE = 12  # se amplía la ventana para capturar más contexto

# Lista de columnas utilizadas como variables explicativas.  Se incluyen
# indicadores técnicos clásicos, estadísticas de volumen y nuevas columnas
# provenientes del archivo 5m (MACD_signal, vw, o, h, l, n y pips_5m).  Esta
# lista se utiliza tanto para comprobar que el dataset contiene todas las
# variables necesarias como para generar las secuencias de entrada.  Si se
# añaden o eliminan columnas, actualiza esta lista y el valor de FEATURES.
FEATURE_COLUMNS = [
    "RSI", "MACD", "MACD_signal", "EMA_20", "ATR", "SMA_20",
    "BB_High", "BB_Low", "v", "vw", "o", "h", "l", "n",
    "ADX", "+DI", "-DI", "Volumen_Relativo", "RSI_15m", "RSI_1h",
    "Distancia_SMA20", "pips_5m"
]

# Número total de características derivado de la longitud de FEATURE_COLUMNS.
FEATURES = len(FEATURE_COLUMNS)

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

        # 🔹 Verificar si el dataset contiene todas las columnas necesarias
        # Usamos la lista global FEATURE_COLUMNS definida al inicio.
        if data.empty or data.columns is None:
            print(f"❌ Error: Dataset vacío o con formato incorrecto para {symbol}.")
            continue  # Saltar este activo

        missing_features = [feature for feature in FEATURE_COLUMNS if feature not in data.columns]
        if missing_features:
            print(f"❌ Error: Las siguientes columnas faltan en el dataset de {symbol}: {missing_features}")
            continue  # Saltar entrenamiento si faltan columnas

        # 🔹 Ordenar cronológicamente por fecha y definir la columna objetivo (precio de cierre)
        data.sort_values("date", inplace=True)
        data.reset_index(drop=True, inplace=True)
        target_column = "c"  # Precio de cierre como objetivo

        # 🔹 Construir secuencias de entrada y objetivos separando entrenamiento y validación cronológicamente.
        X_train, y_train, X_val, y_val = [], [], [], []
        # Índice de corte temporal (80% de datos) para evitar mezclar futuro y pasado.
        train_limit = int(len(data) * 0.8)
        for i in range(len(data) - WINDOW_SIZE):
            # features de la ventana
            seq_features = data[FEATURE_COLUMNS].iloc[i: i + WINDOW_SIZE].values
            # objetivo: precio de cierre inmediatamente después de la ventana
            target_value = data[target_column].iloc[i + WINDOW_SIZE]
            # Si la última posición de la secuencia está dentro del conjunto de entrenamiento, asignamos a train
            if i + WINDOW_SIZE <= train_limit:
                X_train.append(seq_features)
                y_train.append(target_value)
            else:
                X_val.append(seq_features)
                y_val.append(target_value)

        # Convertir a arrays y gestionar valores NaN/infinito
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        X_val = np.array(X_val, dtype=float) if X_val else np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)))
        y_val = np.array(y_val, dtype=float) if y_val else np.empty((0,))
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

        # 🔹 Inicializar y ajustar escaladores únicamente con datos de entrenamiento
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        if len(X_train) == 0:
            print(f"⚠️ No hay suficientes secuencias de entrenamiento para {symbol} tras la división temporal.")
            continue
        # Aplanar las dimensiones de secuencia para ajustar el escalador en cada característica
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler_X.fit(X_train_flat)
        # Ajustar escalador de la variable objetivo
        scaler_y.fit(y_train.reshape(-1, 1))
        # Transformar conjuntos de entrenamiento y validación
        X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
        if len(X_val) > 0:
            X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
        else:
            X_val_scaled = None
            y_val_scaled = None

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
        # Entrenar con los datos escalados, usando la partición de validación explícita.  No se barajan
        if X_val_scaled is not None and y_val_scaled is not None and len(X_val_scaled) > 0:
            val_data = (X_val_scaled, y_val_scaled)
        else:
            val_data = None
        model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_data=val_data,
            verbose=1,
            shuffle=False,
            callbacks=callbacks
        )

        # 🔹 Guardar modelo entrenado
        model.save(f"models/{symbol_formatted}_model.keras")
        print(f"✅ Modelo para {symbol} guardado en models/{symbol_formatted}_model.keras")
        gc.collect()

if __name__ == "__main__":
    print("🚀 Iniciando el entrenamiento de modelos para múltiples acciones...")
    entrenar_y_guardar_modelos()
    print("✅ Entrenamiento completado y modelos guardados.")
