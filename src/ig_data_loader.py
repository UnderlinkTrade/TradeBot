"""
Versión adaptada para Bloomberg (velas de 5 min) — descarga inicial de hasta 2 meses
y control de CSV vacío.

Novedad en esta revisión (19‑jul‑2025):
• Cambiado proveedor de IG Markets → Bloomberg.
• Verificación automática de conexión y permisos antes de descargar.
• Paginación defensiva (1 día por llamada) para cumplir límite de ~1400 barras.
• Mantiene cálculo de indicadores y estructura de pipeline.
"""

from __future__ import annotations

import os, sys, importlib, time
from datetime import datetime, timezone, timedelta

import pandas as pd
import ta

# ──────────────────────────────────────────────────────────────────────────────
# Bloomberg API
# ──────────────────────────────────────────────────────────────────────────────
try:
    from xbbg import blp  # envuelve blpapi
except ImportError as e:
    raise ImportError(
        "Falta la librería `xbbg`/`blpapi`. Instala con `pip install xbbg` y \
        asegúrate de tener Bloomberg Terminal abierto y la API habilitada."
    ) from e

# ──────────────────────────────────────────────────────────────────────────────
# Configuración del proyecto
# ──────────────────────────────────────────────────────────────────────────────
import configCurren  # tu módulo con claves, símbolos, etc.
importlib.reload(configCurren)

from configCurren import (
    SYMBOLS,
    PIP_DECIMALS,
    BLOOMBERG_TICKERS,  # nuevo: mapeo símbolo interno → ticker BBG
)

VERBOSE               = True   # impresión de progreso
SAFE_DELAY_MINUTES    = 1      # evita pedir la vela que aún se está formando
CHUNK_DAYS            = 1      # ≤1400 barras por request; 1 día ≈ 288 barras 5m
INITIAL_DAYS_DEFAULT  = 5      # descarga inicial (~2 meses de velas 5 m)
INTERVAL_BBG          = {"5m": 5}

os.makedirs("data", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Verificación de acceso a Bloomberg
# ──────────────────────────────────────────────────────────────────────────────

def _verify_bloomberg_connection(test_ticker: str = "EURUSD Curncy"):
    """Lanza RuntimeError si no hay sesión Bloomberg operativa."""
    if VERBOSE:
        print("🔑 Verificando conexión Bloomberg …")
    try:
        test = blp.bdp(tickers=test_ticker, flds="PX_LAST")
        if test.empty or pd.isna(test.iloc[0, 0]):
            raise RuntimeError("Respuesta vacía desde Bloomberg")
    except Exception as e:
        raise RuntimeError(
            "No se pudo establecer conexión con Bloomberg. Abre Bloomberg Terminal \
            y comprueba que tu licencia permite peticiones API." 
        ) from e
    if VERBOSE:
        print("✅ Bloomberg disponible")

_verify_bloomberg_connection()

# ──────────────────────────────────────────────────────────────────────────────
# Indicadores auxiliares (idénticos a la versión IG)
# ──────────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14):
    tr = pd.concat(
        [
            df["h"] - df["l"],
            (df["h"] - df["c"].shift()).abs(),
            (df["l"] - df["c"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calcular_adx(df: pd.DataFrame, period: int = 14):
    plus_dm = df["h"].diff()
    minus_dm = df["l"].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[df["l"].diff() > 0] = 0
    tr = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / tr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1 / period).mean()

# ──────────────────────────────────────────────────────────────────────────────
# Descarga de precios desde Bloomberg
# ──────────────────────────────────────────────────────────────────────────────

def fetch_bbg_prices(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "5m",
    *,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Descarga velas desde Bloomberg y devuelve DataFrame con columnas estándar."""
    ticker = BLOOMBERG_TICKERS.get(symbol, symbol)  # fallback símbolo mismo
    freq   = INTERVAL_BBG[interval]

    if verbose:
        print(f"📅 {ticker} {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}", end=" … ")

    try:
        df = blp.bdib(
            ticker,
            start=start_dt,
            end=end_dt,
            interval=freq,
            tz="UTC",
            session="all",
        )
    except Exception as e:
        print(f"⚠️ Error Bloomberg: {e}")
        return pd.DataFrame()

    if df.empty:
        print("0 filas")
        return df

    df = df.rename(
        columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    ).reset_index().rename(columns={"date_time": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if verbose:
        print(f"{len(df)} filas")
    return df[["date", "o", "h", "l", "c", "v"]]

# ──────────────────────────────────────────────────────────────────────────────
# Indicadores técnicos (sin cambios)
# ──────────────────────────────────────────────────────────────────────────────

def recalcular_indicadores(df: pd.DataFrame):
    df["RSI"] = rsi(df["c"])
    macd = ta.trend.MACD(df["c"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["EMA_20"]      = ta.trend.EMAIndicator(df["c"], window=20).ema_indicator()
    df["ATR"]         = atr(df)
    df["SMA_20"]      = ta.trend.SMAIndicator(df["c"], window=20).sma_indicator()

    bb = ta.volatility.BollingerBands(df["c"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()

    df["ADX"] = calcular_adx(df)

    period = 14
    plus_dm  = df["h"].diff().clip(lower=0)
    minus_dm = -df["l"].diff().clip(lower=0)
    tr = atr(df, period)
    df["+DI"] = 100 * (plus_dm.ewm(alpha=1 / period).mean() / tr)
    df["-DI"] = 100 * (minus_dm.ewm(alpha=1 / period).mean() / tr)

    vol_avg = df["v"].rolling(50).mean().replace(0, pd.NA)
    df["Volumen_Relativo"] = df["v"] / vol_avg

    df["RSI_15m"]       = df["RSI"].rolling(3).mean()
    df["RSI_1h"]        = df["RSI"].rolling(12).mean()
    df["Distancia_SMA20"] = (df["c"] - df["SMA_20"]) / df["SMA_20"] * 100

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def obtener_ultima_y_actualizar_csv(
    symbol: str,
    interval: str = "5m",
    *,
    initial_days: int = INITIAL_DAYS_DEFAULT,
):
    symbol_fmt = symbol.replace(":", "_")
    file_path  = os.path.join("data", f"{symbol_fmt}_{interval}_data_BBG.csv")

    now       = datetime.now(timezone.utc)
    safe_now  = now - timedelta(minutes=SAFE_DELAY_MINUTES)

    # ── Cargar histórico existente (si lo hay)
    if os.path.exists(file_path):
        df_hist = pd.read_csv(file_path, parse_dates=["date"], keep_default_na=False)
        dates   = pd.to_datetime(df_hist["date"], errors="coerce").dropna()

        if not dates.empty:
            last_dt = dates.max()
            if last_dt.tzinfo is None:
                last_dt = last_dt.tz_localize("UTC")
            else:
                last_dt = last_dt.tz_convert("UTC")
        else:
            last_dt = safe_now - timedelta(days=initial_days)
    else:
        df_hist = pd.DataFrame()
        last_dt = safe_now - timedelta(days=initial_days)

    # ── Descargar nuevos bloques
    df_all      = []
    current_dt  = last_dt

    while current_dt < safe_now:
        chunk_end = min(current_dt + timedelta(days=CHUNK_DAYS), safe_now)
        df_chunk  = fetch_bbg_prices(symbol, current_dt, chunk_end, interval)
        if not df_chunk.empty:
            df_all.append(df_chunk)
        current_dt = chunk_end

    # ── Control si no hay datos nuevos
    if not df_all:
        if df_hist.empty:
            pd.DataFrame(columns=["date", "o", "h", "l", "c", "v"]).to_csv(
                file_path, index=False
            )
            print(f"⚠️ {symbol}: CSV creado vacío (sin datos).")
        else:
            print(f"ℹ️ {symbol}: sin nuevas velas.")
        return None

    # ── Unir histórico + nuevo
    df_new  = pd.concat(df_all).drop_duplicates(subset=["date"])
    df_comb = (
        pd.concat([df_hist, df_new])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ── Recalcular indicadores
    df_comb = recalcular_indicadores(df_comb)

    # ── Pips 5 m
    pair      = symbol.replace("C:", "")
    pip_size  = 10 ** -PIP_DECIMALS.get(pair, 2)
    df_comb["pips_5m"] = df_comb["c"].diff().abs() / pip_size

    df_comb.to_csv(file_path, index=False)
    print(f"✅ {symbol}: CSV actualizado → {file_path} ({len(df_new)} velas nuevas)")
    return df_comb


def descargar_datos(
    symbol: str,
    interval: str = "5m",
    *,
    initial_days: int = INITIAL_DAYS_DEFAULT,
):
    obtener_ultima_y_actualizar_csv(symbol, interval, initial_days=initial_days)


if __name__ == "__main__":
    # ejemplo: 10 días iniciales
    descargar_datos("C:EURGBP", initial_days=10)
