"""
Script para descargar datos de velas de 5 minutos desde Polygon.io y evaluar
distintas combinaciones de parámetros técnicos con el fin de encontrar
estrategias intradía rentables. El objetivo es generar un archivo CSV
con las mejores estrategias por símbolo basado en el rendimiento pasado.

Este módulo asume que `data_loaderCurren.py` y `configCurren.py` están
disponibles en el mismo entorno de ejecución y que en `configCurren.py`
se define la lista de símbolos (`SYMBOLS`) y el API key para Polygon.

Para cada símbolo, se descargan los datos de 5 minutos (histórico de los
últimos 6 meses, según implemente `descargar_datos` en `data_loaderCurren`),
se recalculan indicadores técnicos (RSI, EMA, ATR, ADX, etc.) y se
evalúan múltiples combinaciones de límites RSI y umbrales ADX para
operaciones de compra y venta. Los stops y objetivos se calculan como
múltiplos del ATR (1.5× para el stop‑loss y 2.0× para el take‑profit por
defecto), lo que permite adaptar la estrategia a la volatilidad
subyacente.

El resultado de la búsqueda se guarda en un archivo CSV con las
principales métricas para cada combinación evaluada: tasa de acierto
(win_rate), beneficio medio por operación (avg_profit) y número total
de operaciones (trades). Las estrategias se ordenan por beneficio
medio descendente en el CSV final.

Nota: Este código no ejecuta operaciones reales. Es una herramienta
analítica para identificar parámetros prometedores que posteriormente
pueden integrarse al bot de trading. Se recomienda validar los
resultados mediante backtesting adicional y evaluación en tiempo real
antes de utilizar cualquier estrategia en entornos de producción.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# Importamos funciones de descarga y cálculo de indicadores
try:
    from data_loaderCurren import descargar_datos
except ImportError:
    raise ImportError(
        "No se puede importar 'descargar_datos' de data_loaderCurren.py."
        " Asegúrate de que data_loaderCurren.py esté disponible en el path."
    )

try:
    # SYMBOLS contiene la lista de instrumentos con prefijo 'C:' para Forex
    from configCurren import SYMBOLS
except ImportError:
    raise ImportError(
        "No se puede importar SYMBOLS de configCurren.py."
        " Asegúrate de que configCurren.py esté disponible y correctamente configurado."
    )


# --- Configuración de multiplicadores ATR para SL/TP ---
# Estos valores se inspiran en la práctica de usar un porcentaje del ATR para
# determinar stops dinámicos, que se adapta a la volatilidad de cada activo. Un
# stop de 1.5× ATR permite cierta holgura en velas de 5 minutos, mientras que
# un objetivo de 2.0× ATR ofrece una relación riesgo‑beneficio razonable.
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: float = 2.0


def generate_combinations() -> List[Tuple[int, int, int, str]]:
    """
    Genera una lista de tuplas con combinaciones de parámetros (RSI mínimo,
    RSI máximo, ADX mínimo, dirección) para evaluar. Se cubren tanto
    operaciones de compra (long) como de venta (short) con intervalos
    razonables de RSI y ADX usados habitualmente en intradía.

    Returns:
        List of tuples: Cada tupla tiene la forma
        (rsi_min, rsi_max, adx_min, direction).
    """
    combinations: List[Tuple[int, int, int, str]] = []
    # Rango de RSI para compras: sobreventa moderada a neutral
    buy_rsi_ranges = [(10, 30), (15, 35), (20, 40)]
    # Rango de RSI para ventas: neutral a sobrecompra
    sell_rsi_ranges = [(60, 80), (65, 85), (70, 90)]
    # Umbrales ADX: valores por encima de 20 suelen indicar tendencia, valores
    # mayores a 25 reflejan tendencia más fuerte【542196532393249†L369-L389】.
    adx_thresholds = [20, 25, 30]

    # Combinaciones para compras (long)
    for rsi_min, rsi_max in buy_rsi_ranges:
        for adx_min in adx_thresholds:
            combinations.append((rsi_min, rsi_max, adx_min, "buy"))

    # Combinaciones para ventas (short)
    for rsi_min, rsi_max in sell_rsi_ranges:
        for adx_min in adx_thresholds:
            combinations.append((rsi_min, rsi_max, adx_min, "sell"))

    return combinations


def evaluate_strategy(
    df: pd.DataFrame,
    rsi_min: int,
    rsi_max: int,
    adx_min: int,
    direction: str,
    atr_sl_mult: float = ATR_SL_MULTIPLIER,
    atr_tp_mult: float = ATR_TP_MULTIPLIER,
    max_lookahead: int = 12,
) -> dict:
    """
    Evalúa una estrategia simple basada en límites de RSI y ADX, así como
    confirmación con la EMA de 20. Se simulan operaciones y se calcula
    rendimiento en función de multiplicadores de ATR para el stop‑loss y
    take‑profit. Para cada señal se busca dentro de las `max_lookahead`
    velas siguientes si se alcanza primero el TP o el SL.

    Args:
        df: DataFrame con columnas necesarias: 'c' (close), 'h' (high),
            'l' (low), 'RSI', 'ADX', 'EMA_20', 'ATR'.
        rsi_min, rsi_max: Rangos de RSI para disparar la entrada.
        adx_min: Umbral mínimo de ADX para confirmar tendencia.
        direction: 'buy' o 'sell'.
        atr_sl_mult: Multiplicador del ATR para el stop‑loss.
        atr_tp_mult: Multiplicador del ATR para el take‑profit.
        max_lookahead: Número máximo de velas 5m para vigilar el TP/SL.

    Returns:
        Diccionario con métricas: número de operaciones, tasa de aciertos,
        beneficio medio y parámetros usados.
    """
    wins = 0
    losses = 0
    profits: List[float] = []

    # Pre‑obtener series para mejorar rendimiento
    closes = df["c"].values
    highs = df["h"].values
    lows = df["l"].values
    rsis = df["RSI"].values
    adxs = df["ADX"].values
    emas = df["EMA_20"].values
    atrs = df["ATR"].values

    # Iterar a partir de la segunda fila porque comparamos pendiente de EMA
    for i in range(1, len(df) - max_lookahead):
        rsi_val = rsis[i]
        adx_val = adxs[i]
        ema_current = emas[i]
        ema_prev = emas[i - 1]
        price_close = closes[i]

        # Condiciones de entrada comunes: RSI dentro del rango y ADX ≥ umbral
        if not (rsi_min <= rsi_val <= rsi_max and adx_val >= adx_min):
            continue

        if direction == "buy":
            # Confirmar que el cierre esté por encima de la EMA y que la EMA
            # tenga pendiente ascendente【542196532393249†L369-L389】.
            if not (price_close > ema_current and ema_current > ema_prev):
                continue
        else:  # sell
            # Confirmar que el cierre esté por debajo de la EMA y que la EMA
            # tenga pendiente descendente.
            if not (price_close < ema_current and ema_current < ema_prev):
                continue

        # Nivel de entrada y niveles de SL/TP
        entry_price = price_close
        sl_distance = atrs[i] * atr_sl_mult
        tp_distance = atrs[i] * atr_tp_mult

        # Variables para determinar si se ha cerrado la operación
        exit_found = False
        trade_profit = 0.0

        for j in range(1, max_lookahead + 1):
            next_high = highs[i + j]
            next_low = lows[i + j]
            
            if direction == "buy":
                stop_price = entry_price - sl_distance
                target_price = entry_price + tp_distance
                # Comprobar si se toca primero el SL
                if next_low <= stop_price:
                    trade_profit = -sl_distance
                    losses += 1
                    exit_found = True
                    break
                # Comprobar si se toca el TP
                if next_high >= target_price:
                    trade_profit = tp_distance
                    wins += 1
                    exit_found = True
                    break
            else:
                stop_price = entry_price + sl_distance
                target_price = entry_price - tp_distance
                if next_high >= stop_price:
                    trade_profit = -sl_distance
                    losses += 1
                    exit_found = True
                    break
                if next_low <= target_price:
                    trade_profit = tp_distance
                    wins += 1
                    exit_found = True
                    break

        # Si no se alcanzó ni TP ni SL, cerrar al final de la ventana
        if not exit_found:
            final_price = closes[i + max_lookahead]
            if direction == "buy":
                trade_profit = final_price - entry_price
            else:
                trade_profit = entry_price - final_price
            if trade_profit > 0:
                wins += 1
            else:
                losses += 1

        profits.append(trade_profit)

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0

    return {
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "adx_min": adx_min,
        "direction": direction,
        "trades": total_trades,
        "win_rate": round(win_rate, 4),
        "avg_profit": round(avg_profit, 5),
    }


def search_best_strategies(symbol: str) -> pd.DataFrame:
    """
    Descarga datos de 5 minutos para un símbolo, calcula indicadores y
    evalúa todas las combinaciones de parámetros definidas por
    `generate_combinations()`. Devuelve un DataFrame ordenado por
    beneficio medio descendente y lo guarda en disco.

    Args:
        symbol: Identificador del instrumento con prefijo 'C:' (ej. 'C:EURGBP').

    Returns:
        DataFrame con resultados de cada estrategia.
    """
    print(f"⬇️ Descargando datos y calculando indicadores para {symbol}...")
    df = descargar_datos(symbol, interval="5m")
    if df is None or df.empty:
        raise ValueError(f"No se pudieron obtener datos para {symbol}. Verifica el API Key y la disponibilidad de datos.")

    # Asegurarse de que los datos estén ordenados y sin valores nulos
    df = df.dropna(subset=["c", "h", "l", "RSI", "ADX", "EMA_20", "ATR"]).reset_index(drop=True)

    results: List[dict] = []
    combinations = generate_combinations()
    for rsi_min, rsi_max, adx_min, direction in combinations:
        metrics = evaluate_strategy(df, rsi_min, rsi_max, adx_min, direction)
        results.append(metrics)
        print(
            f"Evaluada estrategia {direction.upper()} | RSI {rsi_min}-{rsi_max} | ADX≥{adx_min} -> "
            f"Trades: {metrics['trades']}, WinRate: {metrics['win_rate']}, AvgProfit: {metrics['avg_profit']}"
        )

    results_df = pd.DataFrame(results)
    # Ordenar por beneficio medio descendente y luego por tasa de aciertos
    results_df = results_df.sort_values(by=["avg_profit", "win_rate"], ascending=False).reset_index(drop=True)
    # Guardar en CSV
    out_path = f"best_strategies_{symbol.replace(':', '_')}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"📁 Resultados guardados en {out_path}")
    return results_df


def main():
    """Función principal. Itera sobre los símbolos configurados y busca estrategias para cada uno."""
    for symbol in SYMBOLS:
        try:
            search_best_strategies(symbol)
        except Exception as e:
            print(f"⚠️ Error al procesar {symbol}: {e}")


if __name__ == "__main__":
    main()
