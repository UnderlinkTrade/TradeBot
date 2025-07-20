import os
import pandas as pd
from datetime import datetime
import json
import re

SEÑALES_PATH = "data/señales_historicas.csv"
VELAS_DIR = "data"
LOGS_GPT_PATH = "logs/logs_gpt.jsonl"

# Cargar señales
if not os.path.exists(SEÑALES_PATH):
    raise FileNotFoundError(f"No se encontró {SEÑALES_PATH}")

senales = pd.read_csv(SEÑALES_PATH)
senales['fecha_prediccion'] = pd.to_datetime(senales['fecha_prediccion'], format='mixed', utc=True, errors='coerce')
senales['fecha_prediccion'] = senales['fecha_prediccion'].dt.floor('min')

# Normalizar nombres de símbolos
senales["symbol"] = senales["symbol"].str.replace("C_", "C:").str.replace("_", "").str.replace("/", "").str.upper().str.strip()

# Evaluar cada señal
for symbol in senales['symbol'].unique():
    df_simbolo = senales[senales['symbol'] == symbol]
    simbolo_archivo = symbol.replace(":", "_")
    velas_path = os.path.join(VELAS_DIR, f"{simbolo_archivo}_5m_data.csv")

    if not os.path.exists(velas_path):
        print(f"⚠️ No se encontró archivo de velas para {symbol}")
        continue

    velas = pd.read_csv(velas_path, parse_dates=['date'])
    velas = velas.rename(columns={'h': 'high', 'l': 'low'})

    for idx, row in df_simbolo.iterrows():
        fecha_pred = row['fecha_prediccion']
        precio_actual = row['precio_actual']
        tp = precio_actual + row['take_profit']
        sl = precio_actual - row['stop_loss']

        futuras = velas[velas['date'] >= fecha_pred].reset_index(drop=True)
        resultado = 'NINGUNO'
        tardadas = -1
        motivo = ''

        if futuras.empty:
            motivo = 'Sin velas posteriores'
        else:
            for i, vela in futuras.iterrows():
                hit_tp = vela['high'] >= tp
                hit_sl = vela['low'] <= sl

                if hit_tp and hit_sl:
                    if abs(tp - precio_actual) < abs(precio_actual - sl):
                        resultado = 'TP alcanzado'
                    else:
                        resultado = 'SL alcanzado'
                    tardadas = i
                    break
                elif hit_tp:
                    resultado = 'TP alcanzado'
                    tardadas = i
                    break
                elif hit_sl:
                    resultado = 'SL alcanzado'
                    tardadas = i
                    break

            if resultado == 'NINGUNO':
                motivo = 'Niveles no alcanzados'

        senales.at[idx, 'cumplida'] = resultado
        senales.at[idx, 'velas_tardadas'] = tardadas
        senales.at[idx, 'motivo'] = motivo

# Agregar resultado GPT si el archivo existe
if os.path.exists(LOGS_GPT_PATH):
    logs = []
    with open(LOGS_GPT_PATH, 'r') as f:
        for line in f:
            logs.append(json.loads(line))

    gpt_data = []
    for entry in logs:
        fecha = pd.to_datetime(entry.get("fecha_senal"), utc=True, errors='coerce')
        symbol = entry.get("symbol")
        output = entry.get("output", "")

        # Buscar "Recomendación: <valor>" en todo el output, incluso con salto de línea
        if "COMPRA" in output.upper():
            resultado = "COMPRA"
        elif "VENTA" in output.upper():
            resultado = "VENTA"
        elif "ESPERAR" in output.upper():
            resultado = "ESPERAR"
        else:
            resultado = "INDEFINIDO"

        gpt_data.append({
            "symbol": symbol,
            "fecha_prediccion": fecha.floor('min'),
            "resultado_gpt": resultado
        })

    df_gpt = pd.DataFrame(gpt_data)
    df_gpt = df_gpt.dropna(subset=["fecha_prediccion"])
    df_gpt = df_gpt.drop_duplicates(subset=["symbol", "fecha_prediccion"], keep="last")
    df_gpt["symbol"] = df_gpt["symbol"].str.replace("C_", "C:").str.replace("_", "").str.replace("/", "").str.upper().str.strip()

    print("🔍 Tipos de resultado_gpt encontrados:", set(df_gpt["resultado_gpt"]))

    if 'resultado_gpt' in senales.columns:
        senales = senales.drop(columns=['resultado_gpt'])

    senales = pd.merge(senales, df_gpt, on=["symbol", "fecha_prediccion"], how="left")
    print(f"📊 Coincidencias de señales con análisis GPT: {senales['resultado_gpt'].notna().sum()} de {len(senales)}")

    # Evaluar la recomendación GPT contra el resultado alcanzado
    def evaluar_eficacia(row):
        if pd.isna(row.get('resultado_gpt')):
            return 'SIN RESULTADO GPT'
        if row['resultado_gpt'] == 'ESPERAR':
            return 'GPT ESPERAR'
        if row['resultado_gpt'] in ['COMPRA', 'VENTA']:
            if row['cumplida'] == 'TP alcanzado':
                return f'GPT ACIERTO {row["resultado_gpt"]}'
            elif row['cumplida'] == 'SL alcanzado':
                return 'GPT FALLO'
            elif row['cumplida'] in ['NINGUNO', 'pendiente']:
                return 'GPT SIN RESULTADO'
        return 'SIN EVALUAR'

    senales['evaluacion_gpt'] = senales.apply(evaluar_eficacia, axis=1)

    # Resumen de evaluación
    resumen = senales['evaluacion_gpt'].value_counts()
    print("\n📈 Resumen de desempeño de GPT:")
    for key, val in resumen.items():
        print(f"  {key}: {val}")
else:
    print("⚠️ No se encontró el archivo de logs GPT. Se omite la columna 'resultado_gpt'.")

# Guardar los resultados sobreescribiendo el archivo
senales.to_csv(SEÑALES_PATH, index=False)
print("✅ Se actualizaron las señales con los resultados de cumplimiento y GPT.")