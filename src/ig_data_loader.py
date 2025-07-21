from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import yfinance as yf

VERBOSE = True
SAFE_DELAY_MINUTES = 1
CHUNK_DAYS = 1
INITIAL_DAYS_DEFAULT = 5
os.makedirs("data", exist_ok=True)


def fetch_yahoo_prices(symbol: str, start_dt: datetime, end_dt: datetime, interval="5m"):
    ticker = symbol if symbol.endswith("=X") else symbol + "=X"
    if VERBOSE:
        print(f"Fetching {ticker}: {start_dt} to {end_dt}...")
    df = yf.download(ticker, start=start_dt, end=end_dt, interval=interval, progress=False)
    if df.empty:
        print("No data from Yahoo")
        return pd.DataFrame()
    df = df.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})
    df = df.reset_index().rename(columns={"Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "o", "h", "l", "c", "v"]]


def obtener_ultima_y_actualizar_csv(symbol: str, interval: str = "5m", initial_days: int = INITIAL_DAYS_DEFAULT):
    symbol_fmt = symbol.replace(":", "_").replace("=X", "")
    file_path = os.path.join("data", f"{symbol_fmt}_{interval}_data_YF.csv")

    now = datetime.now(timezone.utc)
    safe_now = now - timedelta(minutes=SAFE_DELAY_MINUTES)

    if os.path.exists(file_path):
        df_hist = pd.read_csv(file_path, parse_dates=["date"])
        last_dt = df_hist["date"].max()
        if last_dt.tzinfo is None:
            last_dt = last_dt.tz_localize("UTC")
        else:
            last_dt = last_dt.tz_convert("UTC")
    else:
        df_hist = pd.DataFrame()
        last_dt = safe_now - timedelta(days=initial_days)

    df_all = []
    current_dt = last_dt

    while current_dt < safe_now:
        chunk_end = min(current_dt + timedelta(days=CHUNK_DAYS), safe_now)
        df_chunk = fetch_yahoo_prices(symbol, current_dt, chunk_end, interval)
        if not df_chunk.empty:
            df_all.append(df_chunk)
        current_dt = chunk_end

    if not df_all:
        if df_hist.empty:
            pd.DataFrame(columns=["date", "o", "h", "l", "c", "v"]).to_csv(file_path, index=False)
            print(f"{symbol}: CSV creado vacío (sin datos).")
        else:
            print(f"{symbol}: sin nuevas velas.")
        return None

    df_new = pd.concat(df_all).drop_duplicates(subset=["date"])
    df_comb = (
        pd.concat([df_hist, df_new])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_comb.to_csv(file_path, index=False)
    print(f"{symbol}: CSV actualizado → {file_path} ({len(df_new)} nuevas velas)")
    return df_comb


if __name__ == "__main__":
    obtener_ultima_y_actualizar_csv("EURGBP", interval="5m", initial_days=5)
