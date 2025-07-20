import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📂 Ruta del archivo de historial de operaciones
historial_file = "data/historial_operaciones.csv"

# Cargar los datos del historial
df_historial = pd.read_csv(historial_file)

# Convertir la columna 'tipo_señal' a valores numéricos (1 = Compra, 0 = Descartada)
df_historial["tipo_señal"] = df_historial["tipo_señal"].apply(lambda x: 1 if x == "Compra" else 0)

# Seleccionar las características (X) y la variable objetivo (y)
features = ["RSI", "MACD", "MACD_signal", "ATR", "BB_High", "BB_Low", "volumen", "volumen_ma20", "diferencia"]
X = df_historial[features]
y = df_historial["tipo_señal"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
y_pred = modelo_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Guardar el modelo entrenado
modelo_path = "models/modelo_rf.pkl"
joblib.dump(modelo_rf, modelo_path)

print(f"✅ Modelo entrenado y guardado en {modelo_path}")
print(f"✅ Precisión del modelo: {accuracy:.2%}")
print("\n📊 Reporte de Clasificación:")
print(report)
