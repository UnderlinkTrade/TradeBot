from ib_insync import *
import time
import random
import os
import csv
from datetime import datetime

class IBClient:
    def __init__(self, ib_instance=None, host='127.0.0.1', port=7497, client_id=None):
        self.ib = ib_instance or IB()
        self.host = host
        self.port = port
        self.client_id = client_id or random.randint(1000, 9999)
        self.current_order_id = 1000
        self._setup_logging()

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/ibkr_orders_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "event", "orderId", "parentId", "type", "action", 
                    "quantity", "status", "filled", "remaining", 
                    "avgFillPrice", "lastFillPrice", "permId", 
                    "symbol", "stop_loss", "take_profit", "errorCode", "errorString"
                ])

        if self.ib.isConnected():
            self.ib.errorEvent += self._on_error
            self.ib.tradeUpdateEvent += self._on_order_status
        else:
            print("⚠️ IB no está conectado aún para suscribirse a eventos.")

    def _on_error(self, reqId, errorCode, errorString, contract):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), "ERROR", reqId, "", "", "", "", "", "", "", "", "", "", "", "", "", errorCode, errorString
            ])

    def _on_order_status(self, trade: Trade):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                "ORDER_STATUS",
                trade.order.orderId,
                trade.order.parentId,
                trade.order.orderType,
                trade.order.action,
                trade.order.totalQuantity,
                trade.orderStatus.status,
                trade.orderStatus.filled,
                trade.orderStatus.remaining,
                trade.orderStatus.avgFillPrice,
                trade.orderStatus.lastFillPrice,
                trade.orderStatus.permId,
                trade.contract.symbol,
                "", "", "", ""
            ])

    def get_next_order_id(self):
        self.current_order_id += 3
        return self.current_order_id

    def login(self):
        try:
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                print(f"✅ Conectado a Interactive Brokers (client_id={self.client_id})")
            return True
        except Exception as e:
            print("❌ Error al conectar con IB:", e)
            return False

    def place_order_with_sl_tp(self, symbol, quantity, action, stop_loss=None, take_profit=None, currency='USD', exchange='SMART', contract_type='FOREX'):
        if quantity <= 0:
            print(f"❌ Quantity inválido: {quantity}. Orden no enviada.")
            return None

        base_order_id = self.get_next_order_id()

        if contract_type.upper() == 'CFD':
            contract = Contract(symbol=symbol, secType='CFD', currency=currency, exchange=exchange)
        elif exchange.upper() == 'IDEALPRO':
            contract = Forex(symbol)
        else:
            contract = Stock(symbol, exchange, currency)

        self.ib.qualifyContracts(contract)

        parent_order = MarketOrder(action, quantity)
        parent_order.orderId = base_order_id
        parent_order.transmit = False







        tp_order = None
        if take_profit is not None and take_profit > 0:
            tp_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit)
            tp_order.orderId = base_order_id + 1
            tp_order.parentId = base_order_id
            tp_order.transmit = False
        else:
            print(f"⚠️ Take Profit no definido o inválido: {take_profit}")

        sl_order = None
        if stop_loss is not None and stop_loss > 0:
            sl_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss)
            sl_order.orderId = base_order_id + 2
            sl_order.parentId = base_order_id
            sl_order.transmit = True
        else:
            print(f"⚠️ Stop Loss no definido o inválido: {stop_loss}")

        self.ib.placeOrder(contract, parent_order)
        if tp_order:
            self.ib.placeOrder(contract, tp_order)
        if sl_order:
            self.ib.placeOrder(contract, sl_order)

        print(f"📤 Bracket Order enviada: {action} {quantity} {symbol} con SL={stop_loss} y TP={take_profit}")
        print(f"🕒 Hora de envío: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), "ORDER_SENT", base_order_id, "", "Market", 
                action, quantity, "", "", "", "", "", "", symbol, stop_loss, take_profit, "", ""
            ])


        return base_order_id

    def send_xtb_style_order(self, symbol, currency, action, capital, leverage, stop_loss, take_profit):
        min_unit = 100
        nominal_value = capital * leverage
        quantity = int(nominal_value // min_unit) * min_unit

        contract = Contract(symbol=symbol, secType='CFD', currency=currency, exchange="SMART")
        self.ib.qualifyContracts(contract)

        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)

        print(f"⏳ Orden enviada: {symbol}.{currency} {quantity} {action}")

        self.ib.sleep(0.5)
        for _ in range(10):
            if trade.orderStatus.status == 'Filled':
                print(f"✅ Orden EJECUTADA: {quantity} {action} {symbol}.{currency}")
                return trade
            elif trade.orderStatus.status == 'Inactive':
                print(f"❌ Orden rechazada o inactiva")
                return trade
            self.ib.sleep(0.5)

        print(f"⚠️ Orden no ejecutada en 5 segundos. Estado: {trade.orderStatus.status}")
        return trade

    def send_xtb_style_order_with_sl_tp(self, symbol, currency, action, capital, leverage, stop_loss=None, take_profit=None):
        if stop_loss is None or stop_loss <= 0 or take_profit is None or take_profit <= 0:
            print(f"❌ SL o TP inválido: SL={stop_loss}, TP={take_profit}")
            return None

        min_unit = 100
        nominal_value = capital * leverage
        quantity = int(nominal_value // min_unit) * min_unit

        print(f"🔢 Calculando orden: {quantity} unidades de {symbol}.{currency} con SL={stop_loss}, TP={take_profit}")

        return self.place_order_with_sl_tp(
            symbol=symbol,
            quantity=quantity,
            action=action,
            stop_loss=stop_loss,
            take_profit=take_profit,
            currency=currency,
            exchange="SMART",
            contract_type='CFD'
        )

    def mostrar_estado_completo(self):
        self.ib.reqOpenOrders()
        time.sleep(0.5)

        print("\n🔍 Estado completo de órdenes (agrupado por ParentID):\n")

        # Obtener todas las órdenes abiertas y trades
        trades = self.ib.openTrades()
        orders = self.ib.orders()

        # Agrupar órdenes por Parent ID
        ordenes_por_parent = {}
        for order in orders:
            pid = order.parentId if order.parentId else order.orderId
            if pid not in ordenes_por_parent:
                ordenes_por_parent[pid] = []
            ordenes_por_parent[pid].append(order)

        if not ordenes_por_parent:
            print("📭 No hay órdenes activas.")
            return

        for parent_id, ordenes in ordenes_por_parent.items():
            print(f"🧩 Orden Principal (ParentID: {parent_id})")

            for orden in ordenes:
                tipo = orden.orderType
                action = orden.action
                qty = orden.totalQuantity
                limit = getattr(orden, "lmtPrice", "-")
                stop = getattr(orden, "auxPrice", "-")
                transmit = orden.transmit

                print(f"   ├─ {action} {qty} | Tipo: {tipo} | Limit: {limit} | Stop: {stop} | Transmit: {transmit}")

            print("")  # Espacio entre grupos