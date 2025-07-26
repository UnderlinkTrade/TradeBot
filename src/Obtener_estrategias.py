import pandas as pd
import numpy as np
from itertools import product
import os
import re
from datetime import datetime  
from tqdm import tqdm  # Asegúrate de que esté importado

API_KEY = "MItpr9kHZmufmbqbGvxu_S7FsWF8Sljb"

# === Configuración general ===
fecha_inicio = "2022-01-01"
fecha_fin = "2025-07-25"
fecha_inicio_dt = pd.to_datetime(fecha_inicio)
fecha_fin_dt = pd.to_datetime(fecha_fin)
fecha_formateada = f"{fecha_inicio.replace('-', '')}_hasta_{fecha_fin.replace('-', '')}"

activos_ejecutar = {
    "USDJPY": False,
    "EURGBP": False,
    "EURUSD": False,
    "GBPJPY": True,
    "GBPUSD": False
}

modo_operacion = "AMBAS"

configuraciones = {
    "USDJPY": {
        "ruta": "data/usdjpy_5min_2025.csv",
        "pip_size": 0.01,
        "atr_min_vals": [0.03, 0.05, 0.07, 0.1],
    },
    "EURGBP": {
        "ruta": "data/eurgbp_5min_2025.csv",
        "pip_size": 0.0001,
        "atr_min_vals": [0.0003, 0.00045, 0.0006, 0.0009],
    },
    "EURUSD": {
        "ruta": "data/eurusd_5min_2025.csv",
        "pip_size": 0.0001,
        "atr_min_vals": [0.0004, 0.00055, 0.0007, 0.001],
    },
    "GBPJPY": {
        "ruta": "data/gbpjpy_5min_2025.csv",
        "pip_size": 0.01,
        "atr_min_vals": [0.07, 0.1, 0.14, 0.2],
    },
    "GBPUSD": {
        "ruta": "data/gbpusd_5min_2025.csv",
        "pip_size": 0.0001,
        "atr_min_vals": [0.00035, 0.0005, 0.0007, 0.001],
    }
}

# Asegura carpeta `data`
os.makedirs("data", exist_ok=True)

def descargar_datos_polygon(simbolo: str, inicio: datetime, fin: datetime, ruta: str):
    print(f"📥 Descargando datos reales de {simbolo} desde Polygon...")
    client = RESTClient(API_KEY)
    try:
        aggs = client.get_aggregate_bars(
            symbol=f"C:{simbolo}",
            multiplier=5,
            timespan="minute",
            from_=inicio.strftime("%Y-%m-%d"),
            to=fin.strftime("%Y-%m-%d"),
            adjusted=True,
            full_range=True
        )
        records = [a._asdict() for a in aggs]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'Open', 'h': 'High', 'l': 'Low',
            'c': 'Close', 'v': 'Volume'
        })[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(ruta, index=False)
        print(f"✅ Datos guardados en: {ruta}")
    except Exception as e:
        print(f"❌ Error al descargar datos de {simbolo}: {e}")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        abs(df['h'] - df['c'].shift()),
        abs(df['l'] - df['c'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calcular_adx(df, period=14):
    plus_dm = df['h'].diff()
    minus_dm = df['l'].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[df['l'].diff() > 0] = 0
    tr = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period).mean()

def simular(df, tp_pips, sl_pips, pip_size, tipo):
    df = df.copy()
    spread_pips = 2  # Spread en pips usado solo para evaluación, no para PnL

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['entrada'] = df['c']
    df['tp_valor'] = df['entrada'] + tp_pips * pip_size if tipo == "COM" else df['entrada'] - tp_pips * pip_size
    df['sl_valor'] = df['entrada'] - sl_pips * pip_size if tipo == "COM" else df['entrada'] + sl_pips * pip_size

    # Para evaluación (TP y SL ajustados por spread)
    tp_eval = (tp_pips - spread_pips) * pip_size if tipo == "COM" else (tp_pips + spread_pips) * pip_size
    sl_eval = sl_pips * pip_size if tipo == "COM" else (sl_pips - spread_pips) * pip_size

    df['take_eval'] = df['entrada'] + tp_eval if tipo == "COM" else df['entrada'] - tp_eval
    df['stop_eval'] = df['entrada'] - sl_eval if tipo == "COM" else df['entrada'] + sl_eval

    df['resultado'] = None
    df['pnl_usd'] = 0.0

    for i in range(len(df) - 1):
        hora_i = df.loc[i, 'timestamp'].hour
        if 21 <= hora_i < 23:
            continue  # 🚫 No abrir operaciones entre 21:00 y 22:59

        take_eval = df.loc[i, 'take_eval']
        stop_eval = df.loc[i, 'stop_eval']
        tp_valor = df.loc[i, 'tp_valor']
        sl_valor = df.loc[i, 'sl_valor']
        entrada = df.loc[i, 'entrada']

        pnl = 0  # comisión fija
        resultado = None

        for j in range(i + 1, len(df)):
            hora_j = df.loc[j, 'timestamp'].hour
            if 21 <= hora_j < 23:
                continue  # 🚫 No cerrar operaciones en horario de spread alto

            h, l = df.loc[j, 'h'], df.loc[j, 'l']

            if tipo == "COM":
                if h >= take_eval:
                    resultado = 'TP'
                    pnl += tp_pips * (exposicion / 10000)
                    break
                elif l <= stop_eval:
                    resultado = 'SL'
                    pnl -= sl_pips * (exposicion / 10000)
                    break
            else:  # "VEN"
                if l <= take_eval:
                    resultado = 'TP'
                    pnl += tp_pips * (exposicion / 10000)
                    break
                elif h >= stop_eval:
                    resultado = 'SL'
                    pnl -= sl_pips * (exposicion / 10000)
                    break

        df.at[i, 'resultado'] = resultado
        df.at[i, 'pnl_usd'] = pnl if resultado else 0

    return df

def procesar_tipo_operacion(df, simbolo, tipo, tp_vals, sl_vals, rsi_min_vals, rsi_max_vals,
                            atr_min_vals, adx_min_vals, ema_filter, ema_slope_filter,
                            candle_filter, pip_size):

    prefijo = f"{tipo}_{simbolo}"
    resultados_por_tp = {}

    for tp in tp_vals:
        resultados = []
        trazas = {}

        for sl in sl_vals:
            print(f"{simbolo} - {tipo} - TP={tp} / SL={sl}")
            df_sim = simular(df, tp, sl, pip_size, tipo)


            combinaciones = list(product(
                rsi_min_vals, rsi_max_vals, atr_min_vals, adx_min_vals,
                ema_filter, ema_slope_filter, candle_filter
            ))
            
            for idx, (rsi_min, rsi_max, atr_min, adx_min, ema_c, ema_slope, candle_ok) in enumerate(tqdm(combinaciones, desc=f"{simbolo}-{tipo}-TP{tp}-SL{sl}")):
                print(f"⏳ [{idx+1}/{len(combinaciones)}] RSI {rsi_min}-{rsi_max}, ATR>{atr_min}, ADX>{adx_min}, EMA={ema_c}, Slope={ema_slope}, Candle={candle_ok}")

            for rsi_min, rsi_max, atr_min, adx_min, ema_c, ema_slope, candle_ok in product(
                rsi_min_vals, rsi_max_vals, atr_min_vals, adx_min_vals,
                ema_filter, ema_slope_filter, candle_filter):

                filtro = (
                    (df_sim['rsi'] >= rsi_min) & (df_sim['rsi'] <= rsi_max) &
                    (df_sim['atr'] >= atr_min) &
                    (df_sim['adx'] >= adx_min) &
                    ((df_sim['macd'] > df_sim['macd_signal']) if tipo == "COM" else (df_sim['macd'] < df_sim['macd_signal'])) &
                    ((df_sim['ema_confluencia'] if tipo == "COM" else df_sim['ema_confluencia_ven']) if ema_c else True) &
                    ((df_sim['ema_slope_pos'] if tipo == "COM" else df_sim['ema_slope_neg']) if ema_slope else True) &
                    ((df_sim['candle_bullish'] if tipo == "COM" else df_sim['candle_bearish']) if candle_ok else True) &
                    df_sim['resultado'].notnull()
                )

                subset = df_sim[filtro].copy()
                if len(subset) <= 20:
                    continue

                subset["mes"] = pd.to_datetime(subset["timestamp"]).dt.to_period("M").astype(str)
                subset["anio"] = pd.to_datetime(subset["timestamp"]).dt.year
                subset["pnl"] = subset["pnl_usd"]

                # === Nuevas métricas agregadas ===
                pnl_por_mes = subset.groupby("mes")["pnl"].sum().to_dict()
                pnl_por_anio = subset.groupby("anio")["pnl"].sum().to_dict()
                ops_por_mes = subset.groupby("mes").size().to_dict()
                ops_por_anio = subset.groupby("anio").size().to_dict()

                winrate_general = (subset['resultado'] == 'TP').mean()
                winrate_por_anio = subset.groupby("anio")['resultado'].apply(lambda x: (x == 'TP').mean()).to_dict()
                winrate_por_mes = subset.groupby("mes")['resultado'].apply(lambda x: (x == 'TP').mean()).to_dict()

                resultado = {
                    'tp': tp, 'sl': sl, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
                    'atr_min': atr_min, 'adx_min': adx_min,
                    'ema_confluencia': ema_c, 'ema_slope': ema_slope,
                    'candle_filter': candle_ok, 'ops': len(subset),
                    'pnl': round(subset['pnl'].sum(), 2),
                    'winrate': round(winrate_general * 100, 2)
                }

                for anio, val in pnl_por_anio.items():
                    resultado[f"PNL_ANIO_{anio}"] = round(val, 2)
                for anio, val in winrate_por_anio.items():
                    resultado[f"WINRATE_ANIO_{anio}"] = round(val * 100, 2)
                    resultado[f"OPS_ANIO_{anio}"] = ops_por_anio.get(anio, 0)
                for mes, val in pnl_por_mes.items():
                    resultado[f"PNL_{mes[-2:]}_{mes[:4]}"] = round(val, 2)
                for mes, val in winrate_por_mes.items():
                    resultado[f"WINRATE_{mes[-2:]}_{mes[:4]}"] = round(val * 100, 2)
                    resultado[f"OPS_{mes[-2:]}_{mes[:4]}"] = ops_por_mes.get(mes, 0)

                resultados.append(resultado)
                trazas[(tp, sl, rsi_min, rsi_max, atr_min, adx_min, ema_c, ema_slope, candle_ok)] = subset

        if resultados:
            resultados_por_tp[tp] = (pd.DataFrame(resultados), trazas)

    if resultados_por_tp:
        output_path = f"data/RES_{prefijo}_{fecha_formateada}_resultados.xlsx"
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for tp, (df_resultado, trazas) in resultados_por_tp.items():
                columnas = list(df_resultado.columns)
                base = [c for c in columnas if not c.startswith(("PNL_", "WINRATE_", "OPS_"))]
                pnl_anio = sorted([c for c in columnas if c.startswith("PNL_ANIO_")])
                win_anio = sorted([c for c in columnas if c.startswith("WINRATE_ANIO_")])
                ops_anio = sorted([c for c in columnas if c.startswith("OPS_ANIO_")])
                pnl_mes = sorted([c for c in columnas if re.match(r'^PNL_\d{2}_\d{4}$', c)], key=lambda x: datetime.strptime(x[4:], "%m_%Y"))
                win_mes = sorted([c for c in columnas if re.match(r'^WINRATE_\d{2}_\d{4}$', c)], key=lambda x: datetime.strptime(x[8:], "%m_%Y"))
                ops_mes = sorted([c for c in columnas if re.match(r'^OPS_\d{2}_\d{4}$', c)], key=lambda x: datetime.strptime(x[4:], "%m_%Y"))

                orden_final = base + pnl_anio + win_anio + ops_anio + pnl_mes + win_mes + ops_mes
                df_resultado = df_resultado[orden_final]
                df_resultado = df_resultado.sort_values(by='pnl', ascending=False).reset_index(drop=True)

                nombre_pestana = f"Resumen_TP{tp}"
                df_resultado.to_excel(writer, sheet_name=nombre_pestana, index=False)

                if not df_resultado.empty:
                    best = df_resultado.iloc[0]
                    clave = (
                        best['tp'], best['sl'], best['rsi_min'], best['rsi_max'],
                        best['atr_min'], best['adx_min'],
                        best['ema_confluencia'], best['ema_slope'], best['candle_filter']
                    )
                    if clave in trazas:
                        df_mejor = trazas[clave].drop(columns=["pnl_usd"])
                        df_mejor.to_excel(writer, sheet_name=f"Mejor_COMBINATORIA_TP{tp}", index=False)

        print(f"✅ Resultados guardados en: {output_path}")



def ejecutar():
    tipos_operacion = []
    if modo_operacion in ["COM", "AMBAS"]:
        tipos_operacion.append("COM")
    if modo_operacion in ["VEN", "AMBAS"]:
        tipos_operacion.append("VEN")

    for simbolo, config in configuraciones.items():
        if not activos_ejecutar.get(simbolo, False):
            print(f"⏭️ {simbolo} está marcado como OFF. Se omite su ejecución.")
            continue

        ruta = config['ruta']
        pip_size = config['pip_size']
        salida_dir = os.path.dirname(ruta)
        
        if not os.path.exists(ruta):
            print(f"⚠️ Archivo no encontrado: {ruta}. Generando archivo simulado...")
            
            filas = 2000
            base_price = 180.0 if "gbpjpy" in ruta else 1.1
            spread = 0.03 if "gbpjpy" in ruta else 0.001

            fechas = pd.date_range(start="2025-01-01", periods=filas, freq="5T")
            precios_base = base_price + np.cumsum(np.random.normal(0, spread, size=(filas,)))
            df_fake = pd.DataFrame({
                "timestamp": fechas,
                "Open": precios_base,
                "High": precios_base + abs(np.random.normal(0, spread / 2, size=filas)),
                "Low": precios_base - abs(np.random.normal(0, spread / 2, size=filas)),
                "Close": precios_base + np.random.normal(0, spread / 4, size=filas),
                "Volume": np.random.randint(100, 500, size=filas)
            })
            df_fake.to_csv(ruta, index=False)
            print(f"✅ Archivo generado en: {ruta}")
        df = pd.read_csv(ruta, names=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'], skiprows=1)
        # ⛔ Si el archivo no existe, lo crea con datos de muestra
        # ⛔ Si el archivo no existe, lo crea con datos de muestra


        df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c'})
        df = df.sort_values('timestamp').reset_index(drop=True)
        for col in ['o', 'h', 'l', 'c', 'Volume']:
            df[col] = df[col].astype(float)

        df['rsi'] = rsi(df['c'])
        df['atr'] = atr(df)
        df['macd'], df['macd_signal'] = macd(df['c'])
        df['adx'] = calcular_adx(df)
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
        df['ema_confluencia'] = df['c'] > df['ema20']
        df['ema_confluencia_ven'] = df['c'] < df['ema20']
        df['ema_slope_pos'] = df['ema20'].diff() > 0
        df['ema_slope_neg'] = df['ema20'].diff() < 0
        df['candle_bullish'] = (df['c'] > df['o']) & ((df['c'] - df['o']) > (df['h'] - df['l']) * 0.5)
        df['candle_bearish'] = (df['c'] < df['o']) & ((df['o'] - df['c']) > (df['h'] - df['l']) * 0.5)
        df = df.bfill().ffill()

        for tipo in tipos_operacion:
            print(f"🔄 Ejecutando {tipo} para {simbolo}...")
            procesar_tipo_operacion(
                df, simbolo, tipo, tp_vals, sl_vals, rsi_min_vals, rsi_max_vals,
                config['atr_min_vals'], adx_min_vals,
                ema_filter, ema_slope_filter,
                candle_bullish_filter if tipo == "COM" else candle_bearish_filter,
                pip_size
            )

# === PUNTO DE ENTRADA ===
if __name__ == '__main__':
    ejecutar()
