#!/usr/bin/env python3
"""
backtest.py – back‑test + cruce de señales + Telegram
====================================================

* Lee los CSV de velas 5 m ubicados en `./data/C_{symbol}_5m_data.csv` (uno por
  símbolo).  Los símbolos a simular se controlan desde el dict `EJECUTAR`.
* Calcula indicadores técnicos (RSI, ATR, MACD, ADX, EMA20).
* Genera entradas modelo (compra/venta) según un set de filtros; luego recorre
  velas futuras para ver si toca TP o SL.
* Por cada símbolo+dirección guarda un Excel con 3 hojas
  (Resumen, PnL_mensual, Operaciones_filtradas).
* Después de crear el Excel, llama a `evaluar_vs_senales()` para cruzar las
  operaciones de las últimas 24 h con `data/señales_historicas.csv` y envía un
  resumen al canal Telegram configurado en `configCurren.py`.

Requisitos:
    pip install pandas numpy requests xlsxwriter openpyxl
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from configCurren import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------

FECHA_INICIO_SIMULACION = "2025-07-01"  # AAAA-MM-DD
FECHA_FORMATEADA = FECHA_INICIO_SIMULACION.replace("-", "")
TELEGRAM_BOT_TOKEN = "7646940729:AAEXmjYTWUCIlqL723nmSRBRo8tZtPYJ8Vg"
TELEGRAM_CHAT_ID = "-4730295727"
HOURS_WINDOW = 24

EJECUTAR: Dict[str, bool] = {
    "USDJPY": False,
    "EURGBP": True,
    "EURUSD": False,
    "GBPJPY": False,
    "GBPUSD": False,
}

BASE_PATH = "./data"

CONFIGURACIONES: Dict[str, Dict] = {}
for symbol in EJECUTAR:
    CONFIGURACIONES[symbol] = {
        "ruta": f"{BASE_PATH}/C_{symbol}_5m_data.csv",
        "pip_size": 0.0001 if symbol in {"EURGBP", "EURUSD", "GBPUSD"} else 0.01,
        "compra": {
            "tp": 12,
            "sl": 18,
            "rsi_min": 45,
            "rsi_max": 60,
            "atr_min": 0.0006 if symbol == "EURGBP" else 0.0008,
            "adx_min": 20,
            "ema_confluencia": False,
            "ema_slope_pos": False,
        },
        "venta": {
            "tp": 12,
            "sl": 16,
            "rsi_min": 45,
            "rsi_max": 60,
            "atr_min": 0.0006 if symbol == "EURGBP" else 0.0008,
            "adx_min": 15,
            "ema_confluencia": False,
            "ema_slope_pos": False,
        },
    }

# ----------------------------------------------------------------------------
# UTILIDADES GENÉRICAS
# ----------------------------------------------------------------------------

def _build_key(dt) -> str | None:
    """Normaliza un datetime o str → 'YYYY-MM-DD HH:MM' (sin zona)."""
    if pd.isna(dt):
        return None
    if isinstance(dt, str):
        dt = pd.to_datetime(dt, dayfirst=True, errors="coerce")
    if dt is pd.NaT:
        return None
    # elimina zona si existe
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None)
    return dt.strftime("%Y-%m-%d %H:%M")

MAX_TG = 4096  # límite de caracteres por mensaje en Telegram

def send_telegram(text: str) -> None:
    """Imprime el texto a consola y, si cabe, lo envía a Telegram en bloques."""
    print("\n===== MENSAJE A TELEGRAM =====")
    print(text)
    print("===== FIN MENSAJE =====\n")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. No se envía.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for start in range(0, len(text), MAX_TG):
        chunk = text[start : start + MAX_TG]
        try:
            r = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"⚠️  Telegram error {r.status_code}: {r.text}")
                break
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Error enviando Telegram → {exc}")
            break


# ----------------------------------------------------------------------------
# INDICADORES
# ----------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["h"] - df["l"],
            (df["h"] - df["c"].shift()).abs(),
            (df["l"] - df["c"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["h"].diff()
    minus_dm = df["l"].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[df["l"].diff() > 0] = 0
    t_range = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / t_range)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / t_range)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1 / period).mean()

# ----------------------------------------------------------------------------
# CARGA Y PREPROCESADO
# ----------------------------------------------------------------------------

def load_csv(path: str, start_date: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"❌ Archivo no encontrado → {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(
        columns={"open": "o", "high": "h", "low": "l", "close": "c", "date": "timestamp"}
    )
    if "timestamp" not in df.columns:
        print(f"⚠️ 'timestamp' no existe en {path}")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] >= start_date].sort_values("timestamp").reset_index(drop=True)
    for col in ["o", "h", "l", "c", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df

# ----------------------------------------------------------------------------
# CRUCE CON SEÑALES HISTÓRICAS
# ----------------------------------------------------------------------------


def evaluar_vs_senales(excel_path: str) -> None:
    """Cruza Operaciones_filtradas con señales_historicas y envía resumen a Telegram.

    Cambios solicitados:
    — Incluir en el mensaje si la operación terminó en TP, SL o PEND, usando la columna
      "resultado".
    — No enviar (silenciar) los resultados correspondientes a compras (BUY).
    """
    # --- Leer operaciones ---
    try:
        ops = pd.read_excel(excel_path, sheet_name="Operaciones_filtradas")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  No se pudo leer Operaciones_filtradas en {excel_path} → {exc}")
        return
    if ops.empty:
        return

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(hours=HOURS_WINDOW)

    ops["timestamp"] = pd.to_datetime(ops["timestamp"], errors="coerce").dt.tz_localize(None)
    ops = ops[(ops["timestamp"] >= window_start.replace(tzinfo=None)) & (ops["timestamp"] <= now_utc.replace(tzinfo=None))].copy()
    if ops.empty:
        return
    ops["key"] = ops["timestamp"].apply(_build_key)

    # --- Extraer metadatos del nombre de archivo ---
    try:
        par, sufijo, *_ = os.path.basename(excel_path).split("_")
    except ValueError:
        print(f"⚠️  Nombre inesperado de archivo: {excel_path}")
        return
    direction = "SELL" if sufijo == "VEN" else "BUY"

    # ⛔️  SALIR SIN ENVIAR TELEGRAM SI ES COMPRA (BUY)
    if direction == "BUY":
        print("🔕 Resultado BUY omitido en reporte Telegram según solicitud de usuario")
        return

    symbol_key = f"C:{par}"

    # --- Leer CSV de señales ---
    sig_path = next((p for p in (
        os.path.join(os.path.dirname(excel_path), "señales_historicas.csv"),
        os.path.join(os.path.dirname(excel_path), "señales_historicas.csv"),
    ) if os.path.exists(p)), "")
    if not sig_path:
        print("⚠️  señales_historicas.csv no hallado en carpeta")
        return

    sig = pd.read_csv(sig_path)
    sig["fecha_prediccion"] = pd.to_datetime(sig["fecha_prediccion"], errors="coerce").dt.tz_localize(None)
    sig = sig[(sig["fecha_prediccion"] >= window_start.replace(tzinfo=None)) & (sig["fecha_prediccion"] <= now_utc.replace(tzinfo=None))].copy()
    sig["key"] = sig["fecha_prediccion"].apply(_build_key)

    ok_set = set(sig[
        (sig["symbol"] == symbol_key) &
        (sig["direction"] == direction) &
        (sig["motivo_rechazo"].isna() | (sig["motivo_rechazo"] == ""))
    ]["key"])

    # --- Clasificación ---
    ops_keys = set(ops["key"])
    union_keys = ops_keys | ok_set

    lines: List[str] = []
    hits = ops_only = sig_only = 0

    for k in sorted(union_keys):
        in_ops = k in ops_keys
        in_sig = k in ok_set

        if in_ops and in_sig:
            status = "✅ Acierto"; hits += 1
        elif in_ops and not in_sig:
            status = "❌ Op‑sin‑señal"; ops_only += 1
        else:
            status = "⚠️ Señal‑sin‑op"; sig_only += 1

        # Hora y resultado para imprimir
        if in_ops:
            fila_op = ops.loc[ops["key"] == k].iloc[0]
            hora = fila_op["timestamp"]
            resultado_op = fila_op.get("resultado", "-")
        else:
            hora = pd.to_datetime(k)
            resultado_op = "-"

        lines.append(
            f"{par} {direction} {hora:%Y-%m-%d %H:%M}  {status}  [{resultado_op}]"
        )

    msg = "\n".join(lines) + (
        f"\nTotales → {hits}✅ / {ops_only}❌ / {sig_only}⚠️  | Ventana: {HOURS_WINDOW}h"
    )
    send_telegram(msg)


# ----------------------------------------------------------------------------
# BACKTEST PRINCIPAL
# ----------------------------------------------------------------------------

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = rsi(df["c"])
    df["atr"] = atr(df)
    df["macd"], df["macd_signal"] = macd(df["c"])
    df["adx"] = adx(df)
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema_confluencia"] = df["c"] > df["ema20"]
    df["ema_slope_pos"] = df["ema20"].diff() > 0
    df["ema20_previa"] = df["ema20"].shift(1)
    df["ema_diferencia"] = df["c"] - df["ema20"]
    return df.bfill()


def simular(df: pd.DataFrame, params: Dict, direccion: str, pip_size: float) -> pd.DataFrame:
    """Devuelve df con columnas de resultado TP/SL."""
    df = df.copy()
    tp, sl = params["tp"], params["sl"]
    spread_pips = 2

    # valores de entrada / objetivos segun compra/venta
    df["entrada"] = df["c"]
    if direccion == "compra":
        df["take_eval"] = df["entrada"] + ((tp - spread_pips) * pip_size)
        df["stop_eval"] = df["entrada"] - (sl * pip_size)
        df["tp_valor"] = df["entrada"] + tp * pip_size
        df["sl_valor"] = df["entrada"] - sl * pip_size
    else:
        df["take_eval"] = df["entrada"] - ((tp + spread_pips) * pip_size)
        df["stop_eval"] = df["entrada"] + ((sl - spread_pips) * pip_size)
        df["tp_valor"] = df["entrada"] - tp * pip_size
        df["sl_valor"] = df["entrada"] + sl * pip_size

    df[["resultado", "nro_vela_salida", "hora_salida"]] = None
    df["pnl_usd"] = 0.0

    for i in range(len(df) - 1):
        hora_i = df.at[i, "timestamp"].hour
        if 21 <= hora_i < 23:  # sesión illíquida
            df.at[i, "resultado"] = "SKIPPED"
            continue

        take, stop = df.at[i, "take_eval"], df.at[i, "stop_eval"]
        resultado = None
        hora_salida = None
        velo_salida = None
        pnl = 0.0

        for j in range(i + 1, len(df)):
            hora_j = df.at[j, "timestamp"].hour
            if 21 <= hora_j < 23:
                continue

            h, l = df.at[j, "h"], df.at[j, "l"]
            if direccion == "compra":
                if h >= take:
                    resultado = "TP"; pnl = tp * (30000 / 10000)
                elif l <= stop:
                    resultado = "SL"; pnl = -sl * (30000 / 10000)
            else:  # venta
                if l <= take:
                    resultado = "TP"; pnl = tp * (30000 / 10000)
                elif h >= stop:
                    resultado = "SL"; pnl = -sl * (30000 / 10000)
            if resultado:
                hora_salida = df.at[j, "timestamp"]
                velo_salida = j - i
                break

        df.at[i, "resultado"] = resultado or "PEND"
        df.at[i, "pnl_usd"] = pnl
        df.at[i, "nro_vela_salida"] = velo_salida
        df.at[i, "hora_salida"] = hora_salida

    return df


def filtrar_entradas(df: pd.DataFrame, params: Dict, direccion: str) -> pd.Series:
    cond = (
        (df["rsi"] >= params["rsi_min"]) & (df["rsi"] <= params["rsi_max"]) &
        (df["atr"] >= params["atr_min"]) & (df["adx"] >= params["adx_min"])
    )
    macd_cond = df["macd"] > df["macd_signal"] if direccion == "compra" else df["macd"] < df["macd_signal"]
    cond &= macd_cond
    if params["ema_confluencia"]:
        cond &= df["ema_confluencia"] if direccion == "compra" else ~df["ema_confluencia"]
    if params["ema_slope_pos"]:
        cond &= df["ema_slope_pos"] if direccion == "compra" else ~df["ema_slope_pos"]
    return cond


def guardar_excel(df_full: pd.DataFrame, symbol: str, direccion: str):
    sufijo = "COM" if direccion == "compra" else "VEN"
    filename = f"{BASE_PATH}/{symbol}_{sufijo}_{FECHA_FORMATEADA}.xlsx"
    os.makedirs(BASE_PATH, exist_ok=True)

    # timezones fuera
    df_aux = df_full.copy()
    for col in ("timestamp", "hora_salida"):
        df_aux[col] = pd.to_datetime(df_aux[col], errors="coerce").dt.tz_localize(None)

    resumen = pd.DataFrame({
        "total_ops": [len(df_aux[df_aux["entrada_modelo"]])],
        "TP": [(df_aux["resultado"] == "TP").sum()],
        "SL": [(df_aux["resultado"] == "SL").sum()],
        "Pendientes": [(df_aux["resultado"] == "PEND").sum()],
        "winrate": [round(100 * (df_aux["resultado"] == "TP").sum() / max(1, (df_aux["resultado"].isin(["TP", "SL"]).sum())), 2)],
        "pnl_usd": [round(df_aux["pnl_usd"].sum(), 2)],
    })
    pnl_mensual = (
        df_aux[df_aux["entrada_modelo"]]
        .groupby(df_aux["timestamp"].dt.strftime("%m/%Y"))
        ["pnl_usd"].sum().reset_index(name="pnl_mensual")
    )

    cols_traz = [
        "timestamp", "c", "h", "l", "rsi", "atr", "adx", "macd", "macd_signal", "ema20_previa",
        "ema20", "ema_diferencia", "ema_confluencia", "ema_slope_pos", "entrada_modelo", "resultado",
        "pnl_usd", "tp_valor", "sl_valor", "take_eval", "stop_eval", "nro_vela_salida", "hora_salida"
    ]

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        pnl_mensual.to_excel(writer, sheet_name="PnL_mensual", index=False)
        df_aux[cols_traz].to_excel(writer, sheet_name="Trazabilidad", index=False)
        df_aux[df_aux["entrada_modelo"]][cols_traz].to_excel(writer, sheet_name="Operaciones_filtradas", index=False)

    print(f"📝 Generado {filename}")
    evaluar_vs_senales(filename)


def backtest() -> None:
    for symbol, cfg in CONFIGURACIONES.items():
        if not EJECUTAR.get(symbol, False):
            continue
        df = load_csv(cfg["ruta"], FECHA_INICIO_SIMULACION)
        if df is None or df.empty:
            continue

        df = calcular_indicadores(df)

        for direccion in ("compra", "venta"):
            params = cfg[direccion]
            df_sim = simular(df, params, direccion, cfg["pip_size"])
            df_sim["entrada_modelo"] = filtrar_entradas(df_sim, params, direccion)
            guardar_excel(df_sim, symbol, direccion)

if __name__ == "__main__":
    backtest()
