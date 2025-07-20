import requests
import pandas as pd
import os
import importlib
import configCurren
import sys
import ta
import time
import random
from datetime import datetime, timezone, timedelta
from ib_insync import *
from configCurren import SYMBOLS, API_KEY, PIP_DECIMALS
from polygon import RESTClient

# 🔥 Recargar configuración de símbolos y API Key
importlib.reload(configCurren)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

client = RESTClient(API_KEY)

# 🔹 Mapeo de intervalos correctos para Polygon.io
INTERVAL_MAP = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
}

# 📂 Crear carpeta 'data' si no existe
os.makedirs("data", exist_ok=True)


# ✅ NUEVO: Función ATR personalizada
def atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        abs(df['h'] - df['c'].shift()),
        abs(df['l'] - df['c'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ✅ NUEVO: RSI personalizado
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ✅ NUEVO: ADX personalizado
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

def evaluar_ema_flags(df):
    """
    Devuelve True/False para:
    - ema_conf: cierre por encima de la EMA
    - ema_slope: EMA actual mayor que EMA anterior
    """
    ema_conf_valor = df["c"].iloc[-1] > df["EMA_20"].iloc[-1]
    ema_slope_valor = df["EMA_20"].iloc[-1] > df["EMA_20"].iloc[-2]
    return {"ema_conf": ema_conf_valor, "ema_slope": ema_slope_valor}


def recalcular_indicadores(df):
    """Recalcula los indicadores técnicos con los datos más recientes."""
    
    df["RSI"] = rsi(df["c"])
    df["MACD"] = ta.trend.MACD(df["c"], window_slow=26, window_fast=12, window_sign=9).macd()
    df["MACD_signal"] = ta.trend.MACD(df["c"], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    df["EMA_20"] = ta.trend.EMAIndicator(df["c"], window=20).ema_indicator()
    df["ATR"] = atr(df)  # ✅ Usamos función personalizada
    df["SMA_20"] = ta.trend.SMAIndicator(df["c"], window=20).sma_indicator()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["c"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    
    # Cálculo de ADX, +DI y -DI
    df["ADX"] = calcular_adx(df)
    period = 14
    plus_dm = df["h"].diff()
    minus_dm = -df["l"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = atr(df, period)
    df["+DI"] = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    df["-DI"] = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    
    # 🔹 Nuevos indicadores adicionales
    df["Volumen_Relativo"] = df["v"] / df["v"].rolling(50).mean()
    df["RSI_15m"] = df["RSI"].rolling(3).mean()  # RSI en 15m (promedio de los últimos 3 valores de RSI 5m)
    df["RSI_1h"] = df["RSI"].rolling(12).mean()  # RSI en 1h (promedio de los últimos 12 valores de RSI 5m)
    df["Distancia_SMA20"] = (df["c"] - df["SMA_20"]) / df["SMA_20"] * 100  # Distancia en %

    # 🛠️ Llenar valores NaN que puedan generarse en los cálculos
    df.bfill(inplace=True)  # ✅ CORRECTO
    df.ffill(inplace=True)

    return df





def calcular_indicadores_sin_volumen(df):
    """
    Calcula indicadores técnicos sin requerir volumen (Forex compatible).
    """
    df["RSI"] = rsi(df["c"])
    macd = ta.trend.MACD(df["c"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["EMA_20"] = ta.trend.EMAIndicator(df["c"], window=20).ema_indicator()
    df["ATR"] = atr(df)  # ✅ Usamos función personalizada
    df["SMA_20"] = ta.trend.SMAIndicator(df["c"], window=20).sma_indicator()

    bb = ta.volatility.BollingerBands(df["c"], window=20)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    adx = ta.trend.ADXIndicator(df["h"], df["l"], df["c"], window=14)
    df["ADX"] = calcular_adx(df)
    period = 14
    plus_dm = df["h"].diff()
    minus_dm = -df["l"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = atr(df, period)
    df["+DI"] = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    df["-DI"] = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)

    df["RSI_15m"] = df["RSI"].rolling(3).mean()
    df["RSI_1h"] = df["RSI"].rolling(12).mean()
    df["Distancia_SMA20"] = (df["c"] - df["SMA_20"]) / df["SMA_20"] * 100

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df



def obtener_order_flow_polygon_fx(symbol):
    """
    Obtiene bid/ask real y datos relevantes desde Polygon Snapshot v2.
    Compatible con símbolos como 'C:USDJPY'.
    """
    if not symbol.startswith("C:") or len(symbol) < 7:
        print(f"❌ Formato de símbolo inválido: {symbol}")
        return None

    url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/forex/tickers/{symbol}?apiKey={API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        ticker = data.get("ticker", {})
        quote = ticker.get("lastQuote", {})

        bid = quote.get("b")
        ask = quote.get("a")
        spread = ask - bid if ask and bid else None

        return {
            "symbol": symbol,
            "ultimo_bid": bid,
            "ultimo_ask": ask,
            "spread": spread,
            "ordenes_bid": bid,
            "ordenes_ask": ask,
            "desequilibrio_ordenes": round((bid - ask) / (bid + ask), 4) if bid and ask else None
        }

    except Exception as e:
        print(f"❌ Error al obtener Snapshot de Polygon para {symbol}: {e}")
        return {
            "symbol": symbol,
            "ultimo_bid": None,
            "ultimo_ask": None,
            "spread": None,
            "ordenes_bid": None,
            "ordenes_ask": None,
            "desequilibrio_ordenes": None
        }

def obtener_ultima_y_actualizar_csv(symbol):

    client = RESTClient(API_KEY)

    try:
        symbol_fmt = symbol.replace(":", "_")
        file_path = os.path.join("data", f"{symbol_fmt}_5m_data.csv")

        if os.path.exists(file_path):
            df_hist = pd.read_csv(file_path, parse_dates=["date"])
            ultima_fecha = df_hist["date"].max()
            desde_ts = int(ultima_fecha.timestamp() * 1000)
        else:
            df_hist = pd.DataFrame()
            desde_ts = int((datetime.utcnow() - timedelta(days=2)).timestamp() * 1000)

        hasta_ts = int(datetime.utcnow().timestamp() * 1000)

        resp = client.get_aggs(
            ticker=symbol,
            multiplier=5,
            timespan="minute",
            from_=desde_ts,
            to=hasta_ts,
            limit=20
        )

        if not resp:
            print(f"⚠️ No se encontraron velas para {symbol}")
            return None

        velas = [{
            "date": datetime.utcfromtimestamp(v.timestamp / 1000).replace(tzinfo=timezone.utc),
            "o": v.open,
            "h": v.high,
            "l": v.low,
            "c": v.close,
            "v": v.volume
        } for v in resp]

        df_nuevo = pd.DataFrame(velas).sort_values("date")

        if os.path.exists(file_path):
            df_hist = pd.read_csv(file_path, parse_dates=["date"])
        else:
            df_hist = pd.DataFrame()

        df_comb = pd.concat([df_hist, df_nuevo], ignore_index=True)
        df_comb.drop_duplicates(subset=["date"], keep="last", inplace=True)
        df_comb.sort_values("date", inplace=True)
        df_comb.reset_index(drop=True, inplace=True)

        from data_loaderCurren import calcular_indicadores_sin_volumen
        df_comb = calcular_indicadores_sin_volumen(df_comb)
        pair = symbol.replace("C:", "")
        pip_size = 10 ** -PIP_DECIMALS.get(pair, 2)
        df_comb["pips_5m"] = abs(df_comb["c"] - df_comb["c"].shift(1)) / pip_size
        df_comb.ffill(inplace=True)
        df_comb.bfill(inplace=True)

        df_comb.to_csv(file_path, index=False)
        print(f"✅ {symbol}: CSV actualizado con {len(df_comb)} velas. Última: {df_comb['date'].max()}")
        return df_comb

    except Exception as e:
        print(f"❌ Error al actualizar {symbol}: {e}")
        return None



def descargar_datos(symbol, interval="5m"):
    """Descarga y acumula datos para evitar perder velas antiguas."""
    if not API_KEY:
        print("❌ ERROR: API Key no definida.")
        return

    print(f"🔑 API Key usada: {API_KEY}")

    symbol_formatted = symbol.replace(":", "_")  
    file_path = os.path.join("data", f"{symbol_formatted}_{interval}_data.csv")

    # Verificar si ya tenemos datos recientes
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, parse_dates=["date"])
        existing_data["date"] = pd.to_datetime(existing_data["date"], errors="coerce", utc=True)
        
        # Asegurar que last_date tenga un valor correcto
        last_date = existing_data["date"].max()
        if pd.isnull(last_date):  
            last_date = pd.Timestamp.utcnow() - pd.DateOffset(days=180)
    else:
        last_date = pd.Timestamp.utcnow() - pd.DateOffset(days=180)

    start_date = last_date.strftime('%Y-%m-%d')
    end_date = pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{start_date}/{end_date}?limit=50000&apiKey={API_KEY}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

            # Si ya tenemos datos, concatenar con los nuevos
            if os.path.exists(file_path):
                full_data = pd.concat([existing_data, df]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
            else:
                full_data = df
            
            # Recalcular indicadores antes de guardar
            full_data = recalcular_indicadores(full_data)
            pair = symbol.replace("C:", "")
            pip_size = 10 ** -PIP_DECIMALS.get(pair, 2)
            full_data["pips_5m"] = abs(full_data["c"] - full_data["c"].shift(1)) / pip_size

            full_data.to_csv(file_path, index=False)
            print(f"✅ Datos guardados en {file_path}. Última fecha: {full_data['date'].max()}")


if __name__ == "__main__":
    for symbol in SYMBOLS:
        descargar_datos(symbol, interval="5m")
