import websocket
import json
import threading
import time
import pandas as pd
from datetime import datetime, UTC
from config import API_KEY, POLYGON_WS_URL, SYMBOL  # ✅ Importamos configuración
import ssl
import os

class PolygonWebSocket:
    def __init__(self, tickers=None):
        """Inicia la conexión WebSocket para los tickers especificados."""
        self.ws = None
        self.tickers = tickers if tickers else [SYMBOL]  
        self.reconnect_attempts = 5  # Intentos de reconexión antes de abortar
        self.latest_data = {}  
        self.vela_actual = None  
        self.datos_vela = []  

    def on_message(self, ws, message):
        """Callback cuando se recibe un mensaje del WebSocket."""
        data = json.loads(message)
    
        for trade in data:
            if trade.get("ev") == "T":  
                timestamp_unix = trade["t"]

                # 🔹 Detectar si el timestamp ya es un string (caso especial)
                if isinstance(timestamp_unix, str):
                    timestamp_legible = timestamp_unix  # Si ya es legible, no lo convertimos
                else:
                    # 🔹 Si el timestamp es un número, verificar si está en milisegundos o segundos
                    if len(str(timestamp_unix)) == 13:  # Si tiene 13 dígitos, está en milisegundos
                        timestamp_unix /= 1000  # Convertir a segundos

                    timestamp_legible = datetime.utcfromtimestamp(timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')

                precio = trade["p"]
                volumen = trade["s"]

                self.latest_data = {
                    "symbol": trade["sym"],
                    "price": precio,
                    "size": volumen,
                    "timestamp": timestamp_legible
                }

                # 🔹 AGRUPAR TRADES EN VELAS DE 5 MINUTOS
                minuto_actual = timestamp_legible[:-3]  # Tomar solo hasta los minutos (YYYY-MM-DD HH:MM)

                if self.vela_actual is None:
                    self.vela_actual = minuto_actual

                if minuto_actual == self.vela_actual:
                    self.datos_vela.append({"timestamp": timestamp_legible, "price": precio, "volume": volumen})
                else:
                    self.guardar_vela()
                    self.vela_actual = minuto_actual
                    self.datos_vela = [{"timestamp": timestamp_legible, "price": precio, "volume": volumen}]


    def guardar_vela(self):
        """Guarda la vela actual en un archivo CSV sin sobrescribir datos anteriores."""
        if not self.datos_vela:
            return

        df = pd.DataFrame(self.datos_vela)

        if df.empty:
            print("⚠️ No hay datos en la vela actual. No se guardará.")
            return

        vela = {
            "symbol": SYMBOL,
            "date": self.vela_actual + ":00",
            "Open": df.iloc[0]["price"],
            "Close": df.iloc[-1]["price"],
            "High": df["price"].max(),
            "Low": df["price"].min(),
            "Volume": df["volume"].sum()
        }

        archivo_velas = "data/live_tsla_5m.csv"

        try:
            df_velas = pd.read_csv(archivo_velas, parse_dates=["date"])

            # 🔹 Si el archivo existe pero está vacío, inicializar con columnas correctas
            if df_velas.empty:
                df_velas = pd.DataFrame(columns=["symbol", "date", "Open", "Close", "High", "Low", "Volume"])

        except (FileNotFoundError, pd.errors.EmptyDataError):
            df_velas = pd.DataFrame(columns=["symbol", "date", "Open", "Close", "High", "Low", "Volume"])

        # ✅ Agregar nueva vela sin sobrescribir el archivo
        vela_df = pd.DataFrame([vela])
        vela_df["symbol"] = vela_df["symbol"].astype(str)

        df_velas = pd.concat([df_velas, vela_df], ignore_index=True)

        # ✅ Guardar sin sobrescribir completamente el archivo
        df_velas.to_csv(archivo_velas, index=False, mode="w", header=True)
        print(f"✅ Vela guardada en {archivo_velas}: {vela}")

    def on_error(self, ws, error):
        """Callback cuando ocurre un error."""
        print(f"⚠️ Error en WebSocket: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Callback cuando la conexión se cierra."""
        print("🔴 Conexión WebSocket cerrada.")
        self.reconnect()

    def on_open(self, ws):
        """Callback cuando la conexión se abre."""
        print("✅ Conectado a Polygon WebSocket")
        ws.send(json.dumps({"action": "auth", "params": API_KEY}))
        for ticker in self.tickers:
            ws.send(json.dumps({"action": "subscribe", "params": f"T.{ticker}"}))
        print(f"📡 Suscrito a {self.tickers}")

    def reconnect(self):
        """Intenta reconectar si la conexión se cae."""
        for attempt in range(1, self.reconnect_attempts + 1):
            print(f"🔄 Intentando reconectar... ({attempt}/{self.reconnect_attempts})")
            time.sleep(5)  
            try:
                self.start()  
                return
            except Exception as e:
                print(f"⚠️ Error al reconectar: {e}")

        print("❌ No se pudo reconectar después de varios intentos.")

    def start(self):
        """Inicia el WebSocket en un hilo separado sin verificación SSL."""
        print("🚀 Intentando conectar con Polygon WebSocket...")

        self.ws = websocket.WebSocketApp(
            POLYGON_WS_URL,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        opts = {"sslopt": {"cert_reqs": ssl.CERT_NONE}}

        print("🔄 Ejecutando WebSocket en un hilo separado...")
        thread = threading.Thread(target=self.ws.run_forever, kwargs=opts, daemon=True)
        thread.start()


# ✅ Para probar la conexión
if __name__ == "__main__":
    ws_client = PolygonWebSocket()
    ws_client.start()

    while True:
        print(f"📊 Último dato recibido: {ws_client.latest_data}")
        time.sleep(1)  
