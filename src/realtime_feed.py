import threading
import pandas as pd
import time
import websocket
import json
from datetime import datetime, timezone
from data_loaderCurren import recalcular_indicadores
from configCurren import SYMBOLS, API_KEY


class CandleAggregator:
    def __init__(self):
        self.candles = {}  # Un agregador por símbolo

    def add_tick(self, symbol, tick):
        timestamp = pd.to_datetime(tick['timestamp'], utc=True)
        price = tick['price']

        print(f"[tick] {symbol} | {timestamp} | {price}")

        minute = timestamp.minute - (timestamp.minute % 5)
        rounded = timestamp.replace(minute=minute, second=0, microsecond=0)

        if symbol not in self.candles:
            self.candles[symbol] = {'timestamp': rounded, 'buffer': [], 'last_seen': timestamp}

        current = self.candles[symbol]

        # 🚨 Si ya pasó el bloque de 5m, cerrar vela
        if timestamp >= current['timestamp'] + pd.Timedelta(minutes=5):
            closed_candle = self.build_candle(symbol)
            self.candles[symbol] = {'timestamp': rounded, 'buffer': [(timestamp, price)], 'last_seen': timestamp}
            return closed_candle

        # 👇 Normal append
        current['buffer'].append((timestamp, price))
        current['last_seen'] = timestamp
        return None


    def build_candle(self, symbol):
        current = self.candles[symbol]
        if not current['buffer']:
            return None

        df = pd.DataFrame(current['buffer'], columns=["timestamp", "price"])
        candle = {
            "timestamp": current['timestamp'],
            "o": df["price"].iloc[0],
            "h": df["price"].max(),
            "l": df["price"].min(),
            "c": df["price"].iloc[-1]
        }
        return candle

def escuchar_ticks_polygon():
    aggregator = CandleAggregator()

    def on_message(ws, message):
        print(f"[raw] {message}")  # 👀 Para ver qué tipo de eventos realmente están llegando
        data = json.loads(message)
        for item in data:
            if item.get("ev") == "C":
                symbol = item.get("pair", "C.USDJPY")  # fallback
                tick = {
                    'timestamp': datetime.fromtimestamp(item['t'] / 1000, tz=timezone.utc),
                    'price': item['p']
                }
                result = aggregator.add_tick(symbol, tick)

                if result:
                    print(f"\n[+] Nueva vela cerrada para {symbol}: {result}")

                    symbol_file = symbol.replace(".", "_")  # reemplazar punto por guión bajo para archivos
                    file_path = f"data/{symbol_file}_5m_data.csv"
                    try:
                        historial = pd.read_csv(file_path)
                        nueva = pd.DataFrame([result])
                        df_candle = pd.concat([historial, nueva], ignore_index=True)
                        df_indicadores = recalcular_indicadores(df_candle)

                        precio_actual = result['c']
                        precio_futuro = predecir_precio(symbol.replace(".", ":"), df_indicadores)
                        if precio_futuro is None:
                            print(f"⚠️ {symbol} - Predicción inválida o insuficiente para esta vela.")
                            return

                        print(f"📈 Evaluando señal para {symbol} con precio actual {precio_actual} y predicho {precio_futuro}")
                        resultado = evaluar_senal(df_indicadores, precio_actual, precio_futuro, symbol.replace(".", ":"))

                        if len(resultado) == 2:
                            valido, mensaje = resultado
                            print(f"🔍 {symbol} - {mensaje}")
                            return

                        valido, mensaje, precio_objetivo, stop_loss, take_profit = resultado
                        registrar_prediccion(symbol_file, result['timestamp'], precio_actual, precio_futuro, precio_objetivo, stop_loss, take_profit, df_indicadores)
                        enviar_alerta_telegram(mensaje)
                    except Exception as e:
                        print(f"❌ Error procesando vela para {symbol}: {e}")

    def on_open(ws):
        print("🔗 Conectado a WebSocket de Polygon")
        ws.send(json.dumps({"action": "auth", "params": API_KEY}))
        for symbol in SYMBOLS:
            ws_symbol = symbol.replace(":", ".")
            print(f"📡 Subscrito a: {ws_symbol}")
            ws.send(json.dumps({"action": "subscribe", "params": ws_symbol}))

    ws = websocket.WebSocketApp(
        "wss://socket.polygon.io/forex",
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

def iniciar_feed_en_hilo():
    hilo = threading.Thread(target=escuchar_ticks_polygon, daemon=True)
    hilo.start()
    print("[i] Feed de datos en tiempo real iniciado en segundo plano.")

if __name__ == "__main__":
    iniciar_feed_en_hilo()
    while True:
        time.sleep(1)
