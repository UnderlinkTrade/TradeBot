import time
import pandas as pd
import requests
import os
import joblib
import json
import re
import numpy as np
import random
import tensorflow as tf
from datetime import datetime, timezone, timedelta, time as dt_time
import time as time_module
from configCurren import SYMBOLS, FILTROS_POR_SYMBOL, PIP_DECIMALS, MIN_TP_PIPS, MIN_DISTANCIA_IG_POR_SYMBOL, OPENAI_API_KEY
from modelCurren import cargar_modelo
from data_loaderCurren import descargar_datos, recalcular_indicadores, calcular_indicadores_sin_volumen, evaluar_ema_flags, obtener_ultima_y_actualizar_csv
from openai_validator import validar_senal_con_gpt
from ig_api import IGClient
from ig_accounts import IG_ACCOUNTS
from backtest import backtest 
import threading


DATA_DIR = "data"
# 🔹 Configuración de Telegram
TELEGRAM_BOT_TOKEN = "7646940729:AAEXmjYTWUCIlqL723nmSRBRo8tZtPYJ8Vg"
TELEGRAM_CHAT_ID = "-4730295727"
FEATURES = 15  
WINDOW_SIZE = 12
INTERVALO_SEGUNDOS = 60
GOLD_MAX_VELAS = 4


GOLD_PIP_THRESHOLDS = {
    "pips_muy_bajo": 5,
    "pips_baja": 10,
    "pips_media": 20,
    "pips_alta": 70,
    "pips_muy_alta": 70  # umbral inferior
}



EPIC_MAP = {
    "C:USDJPY": "CS.D.USDJPY.CFD.IP",
    "C:EURUSD": "CS.D.EURUSD.CFD.IP",
    "C:GBPUSD": "CS.D.GBPUSD.CFD.IP",
    "C:EURGBP": "CS.D.EURGBP.CFD.IP",
    "C:GBPJPY": "CS.D.GBPJPY.CFD.IP",
    "C:XAUUSD": "CS.D.CFDGOLD.CFM.IP"
}






def es_horario_spread_alto(now=None):
    """
    Retorna True si el horario actual cae dentro de las ventanas típicas de spread alto
    de lunes a viernes. Usa hora de Chile (CLT = UTC−4).
    """
    now = now or datetime.now(timezone(timedelta(hours=-4)))  # CLT
    hora_actual = now.time()
    dia_semana = now.weekday()  # 0=lunes, ..., 6=domingo

    if dia_semana > 4:
        return False

    return dt_time(16, 0) <= hora_actual <= dt_time(19, 0)

# 📌 Cargar modelos y escaladores
modelos = {}
scalers_X = {}
scalers_y = {}


print("🔍 Cargando modelos y escaladores...")

for symbol in SYMBOLS:
    symbol_formatted = symbol.replace(":", "_")

    modelos[symbol] = cargar_modelo(symbol_formatted)
    if modelos[symbol] is None:
        print(f"❌ Error al cargar el modelo para {symbol}. Se omitirá en el monitoreo.")
        continue

    try:
        scalers_X[symbol] = joblib.load(f"models/{symbol_formatted}_5m_scaler_X.pkl")
        scalers_y[symbol] = joblib.load(f"models/{symbol_formatted}_5m_scaler_y.pkl")
        print(f"✅ Escaladores cargados para {symbol}")
    except Exception as e:
        print(f"⚠️ Error al cargar escaladores para {symbol}: {e}")



print("✅ Todos los modelos y escaladores han sido cargados.")

# 📌 Almacenar últimas señales enviadas
ultimas_senales_enviadas = {}
ultimas_fechas_procesadas = {}  # ⏱ Evitar reprocesar velas ya usadas

def convertir_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def guardar_log(prompt, respuesta, symbol, fecha):
    entrada = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "fecha_senal": fecha,
        "input": json.loads(json.dumps(prompt, default=convertir_numpy)),
        "output": respuesta
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/logs_gpt.jsonl", "a") as f:
        f.write(json.dumps(entrada) + "\n")



def enviar_alerta_telegram(mensaje):
    """Envía una alerta a Telegram con los detalles de la señal."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("✅ Alerta enviada a Telegram")
        else:
            print(f"⚠️ Error en la respuesta de Telegram: {response.text}")
    except Exception as e:
        print(f"⚠️ Error al enviar alerta a Telegram: {e}")



def calcular_objetivos(precio_actual, atr, direction):
    """
    Calcula Stop-Loss y Take-Profit basado en ATR para IBKR.
    Retorna precios absolutos, sin lógica de pips.

    :param precio_actual: Precio actual del instrumento
    :param atr: Volatilidad actual (ATR)
    :param direction: 'BUY' o 'SELL'
    :return: (take_profit, stop_loss) como precios absolutos
    """
    factor_sl = 2
    factor_tp = 2
    min_distance = 0.0000001  # Protección para evitar límites demasiado cerca

    distancia_sl = max(atr * factor_sl, min_distance)
    distancia_tp = max(atr * factor_tp, min_distance)

    if direction == "BUY":
        stop_loss_price = precio_actual - distancia_sl
        take_profit_price = precio_actual + distancia_tp
    else:  # SELL
        stop_loss_price = precio_actual + distancia_sl
        take_profit_price = precio_actual - distancia_tp

    return take_profit_price, stop_loss_price



def predecir_precio(symbol, historial):
    """Realiza la predicción del precio futuro con detección de NaN."""
    if symbol not in modelos or symbol not in scalers_X or symbol not in scalers_y:
        return None

    if len(historial) < WINDOW_SIZE:
        return None

    try:
        feature_columns = ["RSI", "MACD", "EMA_20", "ATR", "SMA_20", "BB_High", "BB_Low", "v", 
                           "ADX", "+DI", "-DI", "Volumen_Relativo", "RSI_15m", "RSI_1h", "Distancia_SMA20"]
        X_nuevo = historial.tail(WINDOW_SIZE)[feature_columns]

        if X_nuevo.isnull().values.any():
            return None
        #print(f"📊 Datos antes de normalización:\n{X_nuevo}")  # 🔍 Debug
        
        X_nuevo = scalers_X[symbol].transform(X_nuevo.values)
        X_nuevo = X_nuevo.reshape(1, WINDOW_SIZE, FEATURES)
        #print(f"📊 Datos después de normalización:\n{X_nuevo}")  # 🔍 Debug
        
        prediccion = modelos[symbol].predict(X_nuevo)
        precio_futuro = scalers_y[symbol].inverse_transform(prediccion.reshape(-1, 1))[0][0]
        print(f"🔮 {symbol} - Predicción cruda: {precio_futuro}")


        return (prediccion, precio_futuro) if not np.isnan(precio_futuro) else (prediccion, None)

    except Exception as e:
        print(f"❌ {symbol}: Error en la predicción -> {e}")
        return None


#def calcular_sl_tp_dinamico(atr, riesgo_factor=1.5):
    """
    Calcula el Stop-Loss y Take-Profit dinámicos basados en el ATR.
    - atr: valor del Average True Range (volatilidad)
    - riesgo_factor: factor para ajustar la distancia de SL y TP
    """
#    stop_loss = atr * riesgo_factor
#    take_profit = atr * (riesgo_factor * 2)  # TP más amplio que SL
#    return stop_loss, take_profit     

def agregar_contexto_a_data_gpt(data_gpt, historial):
    velas = historial.tail(30)[["date", "o", "h", "l", "c"]]
    ultimas_velas = []
    alcistas = bajistas = neutrales = 0
    rangos = []

    for _, row in velas.iterrows():
        o, h, l, c = float(row["o"]), float(row["h"]), float(row["l"]), float(row["c"])
        if c > o:
            alcistas += 1
        elif c < o:
            bajistas += 1
        else:
            neutrales += 1
        rangos.append(h - l)
        ultimas_velas.append({
            "timestamp": str(row["date"]),
            "o": round(o, 5),
            "h": round(h, 5),
            "l": round(l, 5),
            "c": round(c, 5)
        })

    resumen = {
        "alcistas": alcistas,
        "bajistas": bajistas,
        "neutrales": neutrales,
        "rango_promedio": sum(rangos) / len(rangos),
        "ultimo_cierre": ultimas_velas[-1]["c"],
        "comportamiento": "Mayoría alcista" if alcistas > bajistas else "Mayoría bajista" if bajistas > alcistas else "Balanceado"
    }

    data_gpt["ultimas_velas"] = ultimas_velas
    data_gpt["resumen_velas"] = resumen
    return data_gpt

def extraer_calidad_senal(respuesta_gpt):
    """
    Extrae el valor de calidad de la señal desde el análisis GPT.
    Busca líneas como: 'Calidad de la señal: 8'
    """
    match = re.search(r"Calidad de la señal:\s*([0-9]+(?:\.\d+)?)", respuesta_gpt)
    return float(match.group(1)) if match else None


def redondear_precio(symbol, price):
    if price is None:
        return None
    pair = symbol.replace("C:", "")  # Ejemplo: "C:USDJPY" → "USDJPY"
    decimales = PIP_DECIMALS.get(pair, 5)  # Usa el par completo como clave
    return round(price, decimales)



def evaluar_senal(historial, precio_actual, precio_futuro, symbol, prediccion, direction):

    global ultimas_senales_enviadas
    auto_validado = False  # ← Se define aquí para evitar errores de referencia
    # Obtener configuración específica para el símbolo
    filtros = FILTROS_POR_SYMBOL.get(symbol, {
        "min_cambio_pct": 0.1,
        "rsi_rebote_min": 40,
        "rsi_rebote_max": 65,
        "min_volumen_relativo": 0.2,
        "min_tp_pips": 3,
        "min_atr": 0.0001,  # valor genérico por defecto
    })
            

        # Seleccionar configuración en base al tipo_operacion
    tipo_operacion = "Compra" if direction == "BUY" else "Venta"
    configuraciones = FILTROS_POR_SYMBOL.get(symbol, [])
    filtros = next((f for f in configuraciones if f.get("tipo_operacion") == tipo_operacion), None)

        
    fecha_ultima_vela = historial.iloc[-1]["date"]
    print(f"💰 {symbol} → Precio actual utilizado para evaluación: {precio_actual}")

    # ─── Nueva lógica ORO ───────────────────────────────────────────
    if symbol == "C:XAUUSD":
        pips_last5 = historial["pips_5m"].tail(5)
        variacion5 = pips_last5.sum()           # ∑ de las 5 velas
        ultima_pips = pips_last5.iloc[-1]       # pips de la vela más reciente
        pips_alta   = GOLD_PIP_THRESHOLDS["pips_alta"]  # 70 por defecto

        criterios_ok = (variacion5 <= -100) and (abs(ultima_pips) <= pips_alta)

        if not criterios_ok:
            return False, (
                f"❌ {symbol}: variación 5 velas = {variacion5:.1f} pips, "
                f"última = {ultima_pips:.1f} pips (límite {pips_alta})"
            )

        print(
            f"📉 ORO | variación 5 velas {variacion5:.1f} pips ≤ -100 y "
            f"última vela {ultima_pips:.1f} pips ≤ {pips_alta} → señal ACEPTADA"
        )
    # ────────────────────────────────────────────────────────────────




    macd, macd_signal, rsi, atr = historial.iloc[-1][["MACD", "MACD_signal", "RSI", "ATR"]]
    bb_high, bb_low = historial.iloc[-1][["BB_High", "BB_Low"]]
    adx, di_plus, di_minus = historial.iloc[-1][["ADX", "+DI", "-DI"]]

    min_tp_distance = filtros["min_tp_pips"]  # ya están expresados en precio
    min_sl_distance = filtros["min_sl_pips"]





    # ✅ Validación MACD direccional según tipo de operación
    if not auto_validado and direction == "BUY" and macd <= macd_signal:
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        print("   - ⚠️ MACD no apoya compra (no está por encima de la señal).")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}: MACD no apoya compra (MACD {macd:.5f} <= Signal {macd_signal:.5f})",
            direction=direction
        )
        return False, f"❌ {symbol}: MACD no apoya compra (MACD {macd:.5f} <= Signal {macd_signal:.5f})"

    if not auto_validado and direction == "SELL" and macd >= macd_signal:
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        print("   - ⚠️ MACD no apoya venta (no está por debajo de la señal).")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}: MACD no apoya venta (MACD {macd:.5f} >= Signal {macd_signal:.5f})",
            direction=direction
        )
        return False, f"❌ {symbol}: MACD no apoya venta (MACD {macd:.5f} >= Signal {macd_signal:.5f})"


    # ✅ Validación de ATR mínimo
    if not auto_validado and atr < filtros["min_atr"]:
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}:ATR demasiado bajo ({atr:.5f} < {filtros['min_atr']})",
            direction=direction
        )  
        return False, f"❌ {symbol}: ATR demasiado bajo ({atr:.5f} < {filtros['min_atr']})"

    # ✅ Validación de ADX mínimo
    if not auto_validado and adx < filtros.get("adx_min", 0):  # fallback a 0 si no se define
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}: ADX demasiado bajo ({adx:.2f} < {filtros['adx_min']})",
            direction=direction
        )  
        return False, f"❌ {symbol}: ADX demasiado bajo ({adx:.2f} < {filtros['adx_min']})"

    # ✅ Validación de RSI mínimo
    if not auto_validado and rsi < filtros["rsi_rebote_min"]:
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}: RSI demasiado bajo ({rsi:.2f} < {filtros['rsi_rebote_min']})",
            direction=direction
        )
        return False, f"❌ {symbol}: RSI demasiado bajo ({rsi:.2f} < {filtros['rsi_rebote_min']})"

    # ✅ Validación de RSI máximo
    if not auto_validado and rsi > filtros["rsi_rebote_max"]:
        print(f"\n🔎 Evaluación de señal para {symbol}:")
        print(f"   - Última vela utilizada:")
        print(historial.iloc[-1][["date", "o", "h", "l", "c", "v"]].to_string())
        print(f"   - ATR: {atr:.5f}")
        print(f"   - ADX: {adx:.5f}")  # ✅ Uso correcto
        print(f"   - RSI: {rsi:.2f}")  # ✅ Uso correcto
        print(f"   - MACD: {macd:.5f}")
        print(f"   - MACD Signal: {macd_signal:.5f}")
        print(f"   - Dirección: {direction}")
        print(f"   - Precio actual: {precio_actual}")
        print(f"   - Precio predicho: {precio_futuro}")
        print(f"   - Umbral ATR mínimo requerido: {filtros['min_atr']:.5f}\n")
        registrar_rechazo_completo(
            symbol=symbol,
            historial=historial,
            precio_actual=precio_actual,
            precio_predicho=precio_futuro,
            motivo_rechazo=f"{symbol}: RSI demasiado alto ({rsi:.2f} > {filtros['rsi_rebote_max']})",
            direction=direction
        )
        return False, f"❌ {symbol}: RSI demasiado alto ({rsi:.2f} > {filtros['rsi_rebote_max']})"

    
    if not auto_validado and "ema_conf" in filtros and filtros["ema_conf"]:
        ema_flags = evaluar_ema_flags(historial)
        ema_conf_valor = ema_flags["ema_conf"]

        # Debug exhaustivo
        print(f"\n🔍 DEBUG EMA_CONF para {symbol}")
        print(f"   - Cierre actual: {historial['c'].iloc[-1]:.5f}")
        print(f"   - EMA_20 actual: {historial['EMA_20'].iloc[-1]:.5f}")
        print(f"   - EMA_20 anterior: {historial['EMA_20'].iloc[-2]:.5f}")
        print(f"   - Resultado ema_conf: {ema_conf_valor}")

        if not ema_conf_valor:
            registrar_rechazo_completo(
                symbol=symbol,
                historial=historial,
                precio_actual=precio_actual,
                precio_predicho=precio_futuro,
                motivo_rechazo=f"{symbol}: ❌ ema_conf esperado True, pero fue False",
                direction=direction
            )
            return False, f"❌ {symbol}: ema_conf esperado True, pero fue False"


    if not auto_validado and "ema_slope" in filtros and filtros["ema_slope"]:
        ema_flags = evaluar_ema_flags(historial)
        ema_slope_valor = ema_flags["ema_slope"]
        print(f"\n🔍 DEBUG EMA_SLOPE para {symbol}")
        print(f"   - EMA_20 actual: {historial['EMA_20'].iloc[-1]:.5f}")
        print(f"   - EMA_20 anterior: {historial['EMA_20'].iloc[-2]:.5f}")
        print(f"   - Resultado ema_slope: {ema_slope_valor}")

        if not ema_slope_valor:
            registrar_rechazo_completo(
                symbol=symbol,
                historial=historial,
                precio_actual=precio_actual,
                precio_predicho=precio_futuro,
                motivo_rechazo=f"{symbol}: ❌ ema_slope esperado True, pero fue False",
                direction=direction
            )
            return False, f"❌ {symbol}: ema_slope esperado True, pero fue False"


            
 #if len(historial) >= 3:
    #    macd_tendencia = historial["MACD"].iloc[-3:].diff().fillna(0).sum()
    #    if macd < macd_signal and abs(macd - macd_signal) > 0.01 and macd_tendencia <= 0:
    #        return False, "❌ Señal descartada: MACD no confirma aún reversión."

        # MA Confluencia
    #if not (
    #    historial.iloc[-1]["EMA_20"] > historial.iloc[-1]["SMA_20"] > historial.iloc[-1]["BB_Low"]
    #):
    #    return False, "❌ MA_CONFLUENCIA: Falso (no se cumple)"

    # Pendiente positiva de EMA
    #if historial["EMA_20"].iloc[-1] <= historial["EMA_20"].iloc[-2]:
    #    return False, "❌ EMA_SLOPE_POS: falso (no se cumple)"


    # SE elimina porque los indicadores nuevos reemplazan la fucnion 
    #if not (historial["c"].iloc[-3] < historial["c"].iloc[-2] < historial["c"].iloc[-1]):
    #    return False, "❌ Señal descartada: No hay mínimos ascendentes."

    #volumen = historial.iloc[-1]["v"]
    #volumen_medio = historial["v"].tail(50).mean()
    #if volumen < volumen_medio * filtros["min_volumen_relativo"]:
    #    return False, "❌ Señal descartada: Volumen demasiado bajo."



    if not auto_validado and direction == "BUY":
        take_profit = precio_actual + min_tp_distance
        stop_loss = precio_actual - min_sl_distance
    else:  # SELL
        take_profit = precio_actual - min_tp_distance
        stop_loss = precio_actual + min_sl_distance
        
    print(f"🧾 {symbol} ({direction}) - SL/TP calculados: SL={stop_loss}, TP={take_profit}")


    ultimas_senales_enviadas[symbol] = fecha_ultima_vela
    print(f"\n✅ Evaluación positiva para {symbol}:")
    print(f"   - Fecha vela: {fecha_ultima_vela}")
    print(f"   - Dirección: {direction}")
    print(f"   - Precio actual: {precio_actual}")
    print(f"   - Precio predicho: {precio_futuro}")
    print(f"   - ATR: {atr:.5f}")
    print(f"   - ADX: {adx:.2f}")
    print(f"   - RSI: {rsi:.2f}")
    print(f"   - MACD: {macd:.5f}")
    print(f"   - MACD Signal: {macd_signal:.5f}")


    pair = symbol.replace("C:", "")  # Agrega esta línea justo antes de usar 'pair
    base = pair[:3]
    quote = pair[3:]
    capital = 10
    apalancamiento = 10

    stop_loss = redondear_precio(symbol, stop_loss)
    take_profit = redondear_precio(symbol, take_profit)

    quantity = int((capital * apalancamiento) // 100) * 100


        # ❌ Rechazo por horario con spreads altos
    try:
        if es_horario_spread_alto():
            hora_clt = datetime.now(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d %H:%M:%S')
            hora_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

            mensaje = (
                f"⛔ Señal rechazada por horario con spread potencialmente alto\n"
                f"📈 {symbol} ({direction})\n"
                f"🕒 Hora CLT: {hora_clt}\n"
                f"🌐 Hora UTC: {hora_utc}\n"
                f"⚠️ Fuera de horario seguro (alta probabilidad de spread elevado)"
            )
            enviar_alerta_telegram(mensaje)

            registrar_rechazo_completo(
                symbol=symbol,
                historial=historial,
                precio_actual=precio_actual,
                precio_predicho=precio_futuro,
                motivo_rechazo=f"{symbol}: horario con spread alto",
                direction=direction
            )

            return False, f"❌ {symbol}: horario con spread alto"
    except Exception as e:
        print(f"⚠️ Error en la validación de horario con spread alto: {e}")
        

    # ✅ Verificar límite de posiciones abiertas antes de ejecutar la orden
    for cuenta in IG_ACCOUNTS:
        cliente_ig = IGClient(
            username=cuenta["username"],
            password=cuenta["password"],
            api_key=cuenta["api_key"],
            account_type=cuenta.get("account_type", "REAL")
        )

        try:
            cliente_ig.login()
            open_positions = cliente_ig.get_open_positions()

            if open_positions is not None and len(open_positions) >= 120:
                print(f"⛔ [{cuenta['username']}] No se enviará la orden para {symbol}: ya hay {len(open_positions)} posiciones abiertas.")
                registrar_rechazo_completo(
                    symbol=symbol,
                    historial=historial,
                    precio_actual=precio_actual,
                    precio_predicho=precio_futuro,
                    motivo_rechazo=f"{symbol} [{cuenta['username']}]: Límite de posiciones abiertas alcanzado ({len(open_positions)} >= 10)",
                    direction=direction
                )
                continue  # pasa a la siguiente cuenta
            # 🔹 Obtener lotaje personalizado por símbolo, default=1
            lot_sizes = cuenta.get("lot_sizes", {})
            size = lot_sizes.get(symbol, 1)
            print(f"🚀 Ejecutando orden para {cuenta['username']} → {symbol} → lotaje: {size}")

            cliente_ig.ejecutar_orden(
                symbol=symbol,
                entry_price=precio_actual,
                sl_price=stop_loss,
                tp_price=take_profit,
                direction=direction,
                size=size
            )

        except Exception as e:
            print(f"❌ Error con la cuenta {cuenta['username']}: {e}")


    hora_envio = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    registrar_prediccion(
        symbol=symbol,
        prediccion=prediccion,
        fecha=fecha_ultima_vela,
        precio_actual=precio_actual,
        precio_predicho=precio_futuro,
        precio_objetivo=precio_actual,
        stop_loss=stop_loss,
        take_profit=take_profit,
        historial=historial,
        atr=atr,
        direction=direction,
        hora_envio=hora_envio
    )


    return True, precio_actual, stop_loss, take_profit, direction



SIGNALS_FILE = "data/señales_historicas.csv"

def registrar_rechazo_completo(symbol, historial, precio_actual, precio_predicho, motivo_rechazo, direction=None):
    file_path = SIGNALS_FILE
    os.makedirs("data", exist_ok=True)

    fecha = historial.iloc[-1]["date"]
    hora_envio = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "symbol": symbol,
        "fecha_prediccion": fecha,
        "hora_envio": hora_envio,
        "precio_actual": precio_actual,
        "precio_predicho": precio_predicho,
        "precio_objetivo": precio_actual,
        "stop_loss": None,
        "take_profit": None,
        "rsi_5m": historial.iloc[-1]["RSI"],
        "rsi_15m": historial.iloc[-1]["RSI_15m"] if "RSI_15m" in historial.columns else None,
        "rsi_1h": historial.iloc[-1]["RSI_1h"] if "RSI_1h" in historial.columns else None,
        "macd": historial.iloc[-1]["MACD"],
        "macd_signal": historial.iloc[-1]["MACD_signal"],
        "adx_5m": historial.iloc[-1]["ADX"],
        "adx_15m": historial.iloc[-1]["ADX_15m"] if "ADX_15m" in historial.columns else None,
        "adx_1h": historial.iloc[-1]["ADX_1h"] if "ADX_1h" in historial.columns else None,
        "volumen_relativo": historial.iloc[-1]["Volumen_Relativo"] if "Volumen_Relativo" in historial.columns else None,
        "distancia_sma20": historial.iloc[-1]["Distancia_SMA20"] if "Distancia_SMA20" in historial.columns else None,
        "bb_high": historial.iloc[-1]["BB_High"],
        "bb_low": historial.iloc[-1]["BB_Low"],
        "atr": historial.iloc[-1]["ATR"],
        "direction": direction,
        "cumplida": "rechazada",
        "velas_tardadas": "rechazada",
        "motivo_rechazo": motivo_rechazo
    }

    df_rechazo = pd.DataFrame([row])

    if os.path.exists(file_path):
        df_ant = pd.read_csv(file_path)
        for col in df_rechazo.columns:
            if col not in df_ant.columns:
                df_ant[col] = np.nan
            if col not in df_rechazo.columns:
                df_rechazo[col] = np.nan
        df = pd.concat([df_ant, df_rechazo], ignore_index=True)
    else:
        df = df_rechazo

    df.to_csv(file_path, index=False)
    print(f"❌ Rechazo registrado para {symbol}: {motivo_rechazo}")





def registrar_prediccion(symbol, fecha, precio_actual, precio_predicho, precio_objetivo, stop_loss, take_profit, historial, atr, direction, prediccion, hora_envio):
    os.makedirs("data", exist_ok=True)
    file_path = SIGNALS_FILE

    rsi_5m = historial.iloc[-1]["RSI"]
    rsi_15m = historial.iloc[-1]["RSI_15m"] if "RSI_15m" in historial.columns else None
    rsi_1h = historial.iloc[-1]["RSI_1h"] if "RSI_1h" in historial.columns else None
    macd = historial.iloc[-1]["MACD"]
    macd_signal = historial.iloc[-1]["MACD_signal"]
    adx_5m = historial.iloc[-1]["ADX"]
    adx_15m = historial.iloc[-1]["ADX_15m"] if "ADX_15m" in historial.columns else None
    adx_1h = historial.iloc[-1]["ADX_1h"] if "ADX_1h" in historial.columns else None
    volumen_relativo = historial.iloc[-1]["Volumen_Relativo"] if "Volumen_Relativo" in historial.columns else None
    distancia_sma20 = historial.iloc[-1]["Distancia_SMA20"] if "Distancia_SMA20" in historial.columns else None
    bb_high = historial.iloc[-1]["BB_High"]
    bb_low = historial.iloc[-1]["BB_Low"]

    nueva_prediccion = pd.DataFrame([{
        "symbol": symbol,
        "fecha_prediccion": fecha,
        "hora_envio": hora_envio,
        "precio_actual": precio_actual,
        "precio_predicho": precio_predicho,
        "precio_objetivo": precio_objetivo,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rsi_5m": rsi_5m,
        "rsi_15m": rsi_15m,
        "rsi_1h": rsi_1h,
        "macd": macd,
        "macd_signal": macd_signal,
        "adx_5m": adx_5m,
        "adx_15m": adx_15m,
        "adx_1h": adx_1h,
        "volumen_relativo": volumen_relativo,
        "distancia_sma20": distancia_sma20,
        "bb_high": bb_high,
        "bb_low": bb_low,
        "atr": atr,
        "direction": direction,
        "cumplida": "pendiente",
        "velas_tardadas": "pendiente",
        "motivo_rechazo": None  # ← clave para mantener consistencia
    }])

    columnas_esperadas = nueva_prediccion.columns.tolist()

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        for col in columnas_esperadas:
            if col not in df.columns:
                df[col] = np.nan
            if col not in nueva_prediccion.columns:
                nueva_prediccion[col] = np.nan

        df = pd.concat([df, nueva_prediccion], ignore_index=True)
    else:
        df = nueva_prediccion

    df = df[columnas_esperadas]
    df.to_csv(file_path, index=False)
    print(f"✅ Señal registrada en {file_path}")

    

def verificar_cumplimiento(symbol, historial):
    """Verifica si las predicciones previas se cumplieron."""
    file_path = SIGNALS_FILE
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        if row["cumplida"] != "pendiente" or row["symbol"] != symbol:
            continue  # Saltar verificadas o de otro símbolo

        fecha_pred = pd.to_datetime(row["fecha_prediccion"], utc=True)
        pred_idx = historial[historial["date"] >= fecha_pred].index

        if len(pred_idx) == 0:
            continue

        pred_idx = pred_idx[0]
        precio_entrada = row["precio_actual"]
        take_profit_delta = row["take_profit"]
        stop_loss_delta = row["stop_loss"]

        # Calcular niveles reales
        tp_real = row["take_profit"]
        sl_real = row["stop_loss"]


        es_compra = row["precio_predicho"] > precio_entrada

        for i in range(pred_idx, len(historial)):
            precio_cierre = historial.iloc[i]["c"]

            if es_compra:
                if precio_cierre >= tp_real:
                    df.at[index, "cumplida"] = "TP alcanzado"
                    df.at[index, "velas_tardadas"] = i - pred_idx
                    break

                

sl_backup_por_epic = {}  # ← nuevo diccionario para guardar SL originales
epics_sl_modificados = set()  # ← faltaba definir este conjunto global

def ejecutar_control_spread():
    now = datetime.now(timezone(timedelta(hours=-4)))  # CLT
    hora_actual = now.time()
    minuto = now.minute

    en_pre_spread = dt_time(16, 0) <= hora_actual < dt_time(19, 0)  # de 16:00 a 19:00
    post_spread = hora_actual >= dt_time(19, 0)
    es_19_01 = hora_actual >= dt_time(19, 1) and hora_actual < dt_time(19, 2)

    if not en_pre_spread and not es_19_01:
        return

    for cuenta in IG_ACCOUNTS:
        try:
            cliente_ig = IGClient(
                username=cuenta["username"],
                password=cuenta["password"],
                api_key=cuenta["api_key"],
                account_type=cuenta.get("account_type", "REAL")
            )
            cliente_ig.login()

            if en_pre_spread and minuto % 5 != 0:
                print(f"🕒 {now.strftime('%H:%M')} [{cuenta['username']}] → Pre-spread, evaluando cierres...")
                posiciones = cliente_ig.get_open_positions()
                cerradas = False

                for pos in posiciones:
                    profit, error = calcular_profit_bruto(pos['position'], pos['market'])
                    if error:
                        print(error)
                        continue

                    epic = pos["market"]["epic"]
                    direction = pos["position"]["direction"]
                    size = pos["position"]["size"]

                    try:
                        profit = float(profit)
                    except Exception as e:
                        print(f"⚠️ [{cuenta['username']}] Error convirtiendo profit: {e}")
                        continue

                    if profit >= 0.1:
                        try:
                            deal_id = pos['position']['dealId']
                            opposite = "SELL" if direction == "BUY" else "BUY"
                            cliente_ig.close_position(
                                deal_id=deal_id,
                                direction=opposite,
                                size=size
                            )
                            print(f"✅ [{cuenta['username']}] Posición cerrada: {epic}")
                            cerradas = True
                        except Exception as e:
                            print(f"❌ Error cerrando posición {deal_id}: {e}")

                if not cerradas and hora_actual >= dt_time(15, 55):
                    print(f"⚠️ [{cuenta['username']}] No se cerraron posiciones. Aumentando SL x5 para proteger en spread...")
                    for pos in posiciones:
                        sl = pos['position'].get('stopLevel')
                        if sl:
                            level = float(pos['position']['level'])
                            direction = pos['position']['direction']
                            epic = pos['market']['epic']
                            if epic not in sl_backup_por_epic:
                                sl_backup_por_epic[epic] = sl  # Guardamos SL original
                            nuevo_sl = calcular_sl_ampliado(level, sl, direction, factor=5)
                            print(f"🔧 [{cuenta['username']}] Modificando SL de {epic} a {nuevo_sl}")
                            deal_id = pos['position']['dealId']
                            cliente_ig.modificar_stop_loss(deal_id=deal_id, stop_level=nuevo_sl)
                            epics_sl_modificados.add(epic)

            elif post_spread and epics_sl_modificados:
                print(f"🌅 [{cuenta['username']}] Fin de spread. Restaurando SL originales...")
                posiciones = cliente_ig.get_open_positions()
                for pos in posiciones:
                    epic = pos['market']['epic']
                    if epic in epics_sl_modificados and epic in sl_backup_por_epic:
                        original_sl = sl_backup_por_epic[epic]
                        print(f"♻️ [{cuenta['username']}] Restaurando SL de {epic} a {original_sl}")
                        deal_id = pos['position']['dealId']
                        cliente_ig.modificar_stop_loss(deal_id=deal_id, stop_level=original_sl)
                epics_sl_modificados.clear()
                sl_backup_por_epic.clear()

        except Exception as e:
            print(f"❌ Error en control de spread para {cuenta['username']}: {e}")






def calcular_sl_ampliado(entry, sl, direction, factor=5):
    if direction == "BUY":
        return entry - abs(entry - sl) * factor
    else:  # SELL
        return entry + abs(entry - sl) * factor

def loop_control_spread():
    """Ejecuta control de spread cada minuto en segundo plano."""
    while True:
        ejecutar_control_spread()
        time_module.sleep(60)



def calcular_profit_bruto(pos, market):
    try:
        entrada = float(pos['level'])
        size = float(pos['size'])
        direction = pos['direction'].upper()
        scaling = market.get('scalingFactor', 1)

        bid = market.get('bid')
        offer = market.get('offer')

        if bid is None or offer is None:
            return None, f"⚠️ Precios de mercado no disponibles para {market.get('instrumentName')}."

        if direction == 'BUY':
            precio_actual = bid  # precio al que puedes vender
            profit = (precio_actual - entrada) * size * scaling
        elif direction == 'SELL':
            precio_actual = offer  # precio al que puedes recomprar
            profit = (entrada - precio_actual) * size * scaling
        else:
            return None, f"❌ Dirección desconocida: {direction}"

        return round(profit, 2), None

    except Exception as e:
        return None, f"❌ Error al calcular profit: {str(e)}"




def esperar_hasta_instante_final(multiplo=5, delay_segundos=20):
    """
    Espera hasta justo antes de que cierre una vela de 5 minutos (por ejemplo, 5 segundos antes).
    """
    while True:
        ahora = datetime.now(timezone.utc)
        minuto_actual = ahora.minute
        segundo_actual = ahora.second

        if minuto_actual % multiplo == (multiplo - 1) and segundo_actual >= (60 - delay_segundos):
            break
        time_module.sleep(1)

def esperar_hasta_cierre_vela(intervalo_min=5, delay_extra=10):
    ahora = datetime.utcnow()
    minuto_actual = ahora.minute
    segundos_actuales = ahora.second

    minutos_faltantes = intervalo_min - (minuto_actual % intervalo_min)
    segundos_espera = (minutos_faltantes * 60) - segundos_actuales

    print(f"⏳ Esperando {segundos_espera} segundos hasta el cierre de la vela de {intervalo_min}m...")
    time_module.sleep(max(segundos_espera, 1))

        
ultimas_fechas_procesadas = {}
backtest_ejecutado_hoy = False 

def monitorear_mercado():
    global backtest_ejecutado_hoy
    while True:
        ejecutar_control_spread()  # ✅ Agregado: control dinámico de SL según spread
        esperar_hasta_cierre_vela()  # ✅ Esperar cierre de vela 5m
                # ⏰ ------------- gatillo a las 04:00 UTC -------------
        ahora = datetime.now(timezone.utc)
        if (ahora.hour, ahora.minute) == (4, 0) and not backtest_ejecutado_hoy:
            print("🛠 04:00 UTC – lanzando back-test…")
            try:
                backtest()               # llamada directa
                # subprocess.run(["python", "src/backtest.py"], check=True)
            except Exception as e:
                print(f"❌ Back-test falló: {e}")
            backtest_ejecutado_hoy = True

        # 🔄 reinicia la bandera a medianoche UTC
        if (ahora.hour, ahora.minute) == (0, 5):
            backtest_ejecutado_hoy = False

        for symbol in SYMBOLS:
            try:
                symbol_fmt = symbol.replace(":", "_")
                filepath = os.path.join(DATA_DIR, f"{symbol_fmt}_5m_data.csv")

                # 🔄 Obtener y actualizar la última vela desde Polygon
                df = obtener_ultima_y_actualizar_csv(symbol)
                # Descarta la última vela (la más reciente) y usa solo velas confirmadas
                #df = df.iloc[:-1]  # <- Esto evita usar una vela que pudo no haberse cerrado correctamente
                if df is None or len(df) < WINDOW_SIZE:
                    print(f"⚠️ {symbol}: Datos insuficientes tras actualización")
                    continue

                # ⏱ Prevenir reprocesamiento de la misma vela
                fecha_ultima_vela = df.iloc[-1]["date"]
                if symbol in ultimas_fechas_procesadas and ultimas_fechas_procesadas[symbol] == fecha_ultima_vela:
                    continue
                ultimas_fechas_procesadas[symbol] = fecha_ultima_vela

                print(f"📥 {symbol} - Última vela descargada: {fecha_ultima_vela} - Open: {df.iloc[-1]['o']} - Close: {df.iloc[-1]['c']}")

                # 📈 Ejecutar predicción
                precio_actual = df.iloc[-1]["c"]
                prediccion, precio_futuro = predecir_precio(symbol, df)
                if precio_futuro is None:
                    print(f"❌ {symbol}: Predicción no válida")
                    continue

                # 🧠 Evaluar señal
                for direction in ["BUY", "SELL"]:
                    resultado = evaluar_senal(df, precio_actual, precio_futuro, symbol, prediccion, direction)
                    print(f"📈 {symbol} → Dirección {direction} → Resultado evaluación: {resultado}")


                try:
                    descargar_datos(symbol, interval="5m")
                except Exception as e:
                    print(f"⚠️ Error al refrescar datos para {symbol}: {e}")
                # 📝 Luego retorna el resultado             

            except Exception as e:
                print(f"❌ Error en monitoreo para {symbol}: {e}")


if __name__ == "__main__":
    # 🔄 Ejecutar control de spread cada minuto en un thread paralelo
    threading.Thread(target=loop_control_spread, daemon=True).start()

    # 🧠 Ejecutar el ciclo principal
    monitorear_mercado()
