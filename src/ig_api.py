import requests
import json
import os
import time
from configCurren import IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACCOUNT_TYPE

class IGClient:
    def __init__(self, username, password, api_key, account_type="REAL"):
        self.username = username
        self.password = password
        self.api_key = api_key
        self.account_type = account_type


        # Configuracion para REAL 
        #self.base_url = (
        #    "https://api.ig.com/gateway/deal"
        #    if self.account_type == "REAL"
        #    else "https://demo-api.ig.com/gateway/deal"
        #)
        self.base_url = (
        "https://demo-api.ig.com/gateway/deal"
        if self.account_type == "DEMO"
        else "https://api.ig.com/gateway/deal"
        )

        self.headers = {
            "X-IG-API-KEY": self.api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8"
        }
        self.security_token = None
        self.cst = None
        self.account_id = None
        self.session = requests.Session()  # ← Agregado aquí

    def login(self):
        print("\U0001F4E4 Intentando login con:")
        print("   Usuario:", self.username)
        print("   API KEY:", self.api_key[:6], "****")
        print("   Entorno:", self.base_url)

        url = f"{self.base_url}/session"
        data = {
            "identifier": self.username,
            "password": self.password
        }
        response = requests.post(url, headers=self.headers, data=json.dumps(data))

        if response.status_code == 200:
            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")
            self.account_id = response.json()["accounts"][0]["accountId"]

            self.headers.update({
                "X-SECURITY-TOKEN": self.security_token,
                "CST": self.cst
            })
            print("\u2705 Login exitoso con IG")
        else:
            raise Exception(f"\u274C Login fallido: {response.status_code}, {response.text}")

    #def validar_distancias_con_ig(self, epic, sl_distance, tp_distance):
    #    detalles = self.get_market_details(epic)
    #    if not detalles:
    #        return False

    #    reglas = detalles.get("dealingRules", {})
    #    min_distancia = reglas.get("minStopOrLimitDistance", {}).get("value")
    #    print(f"\U0001F50D Mínima distancia recibida: {min_distancia}")

    #    DEFAULT_MIN_DIST = 10
    #    BUFFER_EXTRA = 2
    #    if min_distancia is None:
    #        print(f"⚠️ minStopOrLimitDistance es None, se usará valor por defecto: {DEFAULT_MIN_DIST}")
    #        min_distancia = DEFAULT_MIN_DIST

    #    min_requerido = min_distancia + BUFFER_EXTRA
    #    if sl_distance < min_requerido or tp_distance < min_requerido:
    #        print(f"❌ SL/TP demasiado cercanos. SL={sl_distance}, TP={tp_distance}, mínimo requerido (con buffer): {min_requerido}")
    #        return False

     #   return True

    def get_open_positions(self):
        url = f"{self.base_url}/positions"
        headers = {**self.headers, "Version": "2"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            positions = response.json().get("positions", [])
            print(f"\n📂 Posiciones abiertas: {len(positions)}")
            for pos in positions:
                epic = pos['market']['epic']
                direction = pos['position']['direction']
                size = pos['position']['size']
                level = pos['position']['level']
                limit_level = pos['position'].get('limitLevel')
                stop_level = pos['position'].get('stopLevel')
                print(f"🔸 {epic}: {direction} {size} @ {level} | SL={stop_level}, TP={limit_level}")
            return positions
        else:
            print(f"❌ Error al consultar posiciones: {response.status_code}, {response.text}")
            return []


    def open_trade_with_sl_tp(self, epic, direction, size, entry_price, sl_pips, tp_pips, symbol=""):
        sl_distance = round(sl_pips)
        tp_distance = round(tp_pips)

        #if not self.validar_distancias_con_ig(epic, sl_distance, tp_distance):
        #    return None

        detalles = self.get_market_details(epic)
        instrument_data = detalles.get("instrument", {})
        currencies = instrument_data.get("currencies", [])

        currency_code = "USD"
        for currency in currencies:
            if currency.get("isDefault"):
                currency_code = currency.get("code", "USD")
                break
        else:
            if currencies:
                currency_code = currencies[0].get("code", "USD")

        if currency_code in ("#.", "$.", "Y."):
            fallback_map = {
                "C:USDJPY": "JPY",
                "C:GBPUSD": "USD",
                "C:EURUSD": "USD",
                "C:EURGBP": "GBP",
                ":CGBPJPY": "JPY"
            }
            currency_code = fallback_map.get(symbol, "USD")

        url = f"{self.base_url}/positions/otc"
        data = {
            "epic": epic,
            "expiry": "-",
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
            "guaranteedStop": False,
            "forceOpen": True,
            "currencyCode": currency_code,
            "stopDistance": str(sl_distance),
            "limitDistance": str(tp_distance),
            "dealReference": f"gpt-auto-trade-{int(time.time())}"
        }
        headers = {**self.headers, "Version": "2"}

        print("\U0001F9EA Payload final a enviar:")
        print(json.dumps(data, indent=2))

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            print(f"✅ Orden {direction} con SL/TP ejecutada: {epic}, size={size}")
            try:
                respuesta = response.json()
                print("\U0001F50D Respuesta:", respuesta)
                print("\U0001F194 dealReference:", respuesta.get("dealReference"))
            except Exception:
                print("⚠️ No se pudo parsear JSON, raw response:")
                print(response.text)
            return respuesta
        else:
            print(f"\u274C Error al ejecutar orden con SL/TP: {response.status_code}, {response.text}")
            print(f"\U0001F9EA Debug info: entry={entry_price}, SLpips={sl_pips}, TPpips={tp_pips}, SLdist={sl_distance}, TPdist={tp_distance}")
            return None

    def close_position(self, deal_id: str, direction: str, size: float):
        """
        Cierra una posición abierta usando dealId, dirección opuesta y tamaño.

        Args:
            deal_id (str): ID de la operación abierta.
            direction (str): Dirección opuesta a la posición ("BUY" o "SELL").
            size (float): Tamaño de la operación a cerrar.
        """
        endpoint = f"{self.base_url}/positions/otc"
        headers = {**self.headers, "Version": "1"}
        print(f"🔑 Headers utilizados en DELETE:")
        print(json.dumps(self.headers, indent=2))



        payload = {
            "dealId": deal_id,
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "timeInForce": "EXECUTE_AND_ELIMINATE"
        }
                # 🔍 Debug: imprimir el payload antes de enviar
        print("📦 Payload de cierre de posición:")
        print(json.dumps(payload, indent=2))

        response = requests.delete(endpoint, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            print(f"✅ Posición cerrada correctamente → dealId: {deal_id}")
            return response.json()
        else:
            raise Exception(f"❌ Error cerrando posición {deal_id}: {response.status_code} → {response.text}")



    def modificar_stop_loss(self, deal_id, stop_level):
        """
        Modifica el stop loss y conserva el TP actual.
        """
        # Paso 1: Obtener detalles actuales de la posición
        positions = self.get_open_positions()
        position = next((p for p in positions if p["position"]["dealId"] == deal_id), None)

        if not position:
            print(f"❌ No se encontró la posición con dealId: {deal_id}")
            return

        current_limit_level = position["position"].get("limitLevel")

        # Paso 2: Armar payload con SL nuevo y TP actual
        url = f"{self.base_url}/positions/otc/{deal_id}"
        headers = {**self.headers, "Version": "2"}

        payload = {
            "stopLevel": str(stop_level)
        }

        if current_limit_level is not None:
            payload["limitLevel"] = str(current_limit_level)

        print("🛠️ Payload para modificar SL:")
        print(json.dumps(payload, indent=2))

        response = requests.put(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"✅ SL modificado para dealId: {deal_id}")
        else:
            print(f"❌ Error modificando SL de {deal_id}: {response.status_code} - {response.text}")



    def check_deal_status(self, deal_reference):
        url = f"{self.base_url}/confirms/{deal_reference}"
        headers = {**self.headers, "Version": "1"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            status = data.get("dealStatus", "UNKNOWN")
            reason = data.get("reason", "")
            deal_id = data.get("dealId", "")
            if status == "ACCEPTED":
                print(f"\U0001F7E2 Orden aceptada con dealId: {deal_id}")
            else:
                print(f"\U0001F534 Orden rechazada: {status} - {reason}")
            return data
        else:
            print(f"⚠️ Error al consultar confirmación: {response.status_code}, {response.text}")
            return None

    def get_market_details(self, epic):
        url = f"{self.base_url}/markets/{epic}"
        headers = {**self.headers, "Version": "1"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            instrument = data.get("instrument", {})
            dealing_rules = data.get("dealingRules", {})
            print("\n\U0001F4CA Detalles del mercado:")
            print("- Nombre:", instrument.get("name"))
            print("- Min deal size:", instrument.get("minDealSize"))
            print("- Max deal size:", instrument.get("maxDealSize"))
            print("- Step size:", instrument.get("stepSize"))
            print("- Min stop distance:", dealing_rules.get("minStopOrLimitDistance"))
            print("- Estado del mercado:", data.get("snapshot", {}).get("marketStatus"))
            print("\U0001F9EA Instrument data:", json.dumps(instrument, indent=2))
            return data
        else:
            print(f"⚠️ Error al obtener detalles del mercado: {response.status_code}, {response.text}")
            return None

    def buscar_epic(self, search_term):
        url = f"{self.base_url}/markets?searchTerm={search_term}"
        headers = {**self.headers, "Version": "1"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            results = response.json().get("markets", [])
            print(f"\n🔍 Resultados de búsqueda para '{search_term}':")
            for market in results:
                print(f"- {market['instrumentName']} -> {market['epic']}")
            return results
        else:
            print(f"❌ Error buscando epic: {response.status_code}, {response.text}")
            return []

    def ejecutar_orden(self, symbol, entry_price, sl_price, tp_price, direction, size=1):
        #epic_map = {
        #    "C:EURGBP": "CS.D.EURGBP.CFD.IP",
        #    "C:EURUSD": "CS.D.EURUSD.CFD.IP",
        #    "C:USDJPY": "CS.D.USDJPY.CFD.IP",
        #    "C:GBPUSD": "CS.D.GBPUSD.CFD.IP"
        #}
        epic_map = {
            "C:EURGBP": "CS.D.EURGBP.MINI.IP",
            "C:EURUSD": "CS.D.EURUSD.MINI.IP",
            "C:USDJPY": "CS.D.USDJPY.MINI.IP",
            "C:GBPUSD": "CS.D.GBPUSD.MINI.IP",
            "C:GBPJPY": "CS.D.GBPJPY.MINI.IP",
            "C:XAUUSD": "CS.D.CFDGOLD.CFM.IP"
        }
        pip_map = {
            "C:USDJPY": 0.01,
            "C:EURUSD": 0.0001,
            "C:GBPUSD": 0.0001,
            "C:EURGBP": 0.0001,
            "C:GBPJPY": 0.01,
            "C:XAUUSD": 0.1
        }

        epic = epic_map.get(symbol)
        pip_size = pip_map.get(symbol, 0.0001)

        if not epic:
            print(f"⚠️ No se encontró epic para {symbol}")
            return

        try:
            entry_price = float(entry_price)
            sl_price = float(sl_price)
            tp_price = float(tp_price)
        except Exception as e:
            print(f"❌ Error al convertir precios a float: {e}")
            return

        sl_pips = abs(entry_price - sl_price) / pip_size
        tp_pips = abs(tp_price - entry_price) / pip_size

        self.login()
        result = self.open_trade_with_sl_tp(
            epic=epic,
            direction=direction,
            size=size,
            entry_price=entry_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            symbol=symbol
        )

        if result and "dealReference" in result:
            confirm = self.check_deal_status(result["dealReference"])
            print("\U0001F9FE Confirmación detallada:", confirm)
