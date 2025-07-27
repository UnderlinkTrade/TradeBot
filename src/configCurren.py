# config.py - Configuración global del bot

# Símbolo del activo a operar (Ejemplo: "AAPL", "C:USDCLP", "BTC/USD")
# Lista de acciones a monitorear
#SYMBOLS = ["C:GBPJPY", "C:EURUSD"]  # SAcams  "C:USDJPY", "C:GBPUSD", "C:EURGBP
#SYMBOLS = ["C:XAUUSD", "C:EURGBP", "C:USDJPY", "C:GBPJPY", "C:GBPUSD", "C:EURUSD"]
SYMBOLS = ["C:EURGBP"]

XTB_MODO_DEMO = True  # 🚀 Cambia a False para operar en real
OPENAI_API_KEY = "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"




TELEGRAM_BOT_TOKEN = "7646940729:AAEXmjYTWUCIlqL723nmSRBRo8tZtPYJ8Vg"
TELEGRAM_CHAT_ID = "-4697025899"
telegram_api_id = 29903244  # número, por ejemplo: 12345678
telegram_api_hash = "81391db086aca4149cc3ff42a2212cec"
telegram_phone = "+56989009574"


        #"username": "sharif1007",
        #"password": "Ig195578",
        #"api_key": "a69712083ff2485bfadbf10af73caa4950bf9ff8",


#IG_USERNAME="SebaBotGPTT"

# API Key de Polygon.io
API_KEY = "MItpr9kHZmufmbqbGvxu_S7FsWF8Sljb"

# Configuración de intervalo de tiempo entre solicitudes
INTERVALO_SEGUNDOS = 15  # Tiempo entre actualizaciones


# Configuración de Stop-Loss y Take-Profit
STOP_LOSS_PERCENT = 0.02  # 2% Stop-Loss
TAKE_PROFIT_PERCENT = 0.05  # 5% Take-Profit

# Configuración de control de señales repetitivas
CAMBIO_MINIMO_PRECIO = 0.002  # 0.2% de cambio para enviar nueva señal
TIEMPO_MINIMO_ENTRE_SEÑALES = 180  # 3 minutos mínimo entre señales


# Configuración de RSI
RSI_NIVELES_CLAVE = [30, 70]  # Niveles para alertas
MARGEN_AVISO_RSI = 2  # Avisar si RSI está cerca del cruce

# Configuración de Stop-Loss y Take-Profit dinámicos
RIESGO_FACTOR = 1.5  # Factor para ajustar la distancia de SL y TP

# Configuración de indicadores adicionales
ATR_WINDOW = 14  # Ventana de cálculo para ATR




# Filtros para evaluacion en live por moneda y dirteccion

FILTROS_POR_SYMBOL = {
    "C:USDJPY": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 45,
            "rsi_rebote_max": 60,
            "min_volumen_relativo": 0.2,
            "min_tp_pips": 0.12,
            "min_sl_pips": 0.18,
            "min_atr": 0.1,
            "adx_min": 30,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 45,
            "rsi_rebote_max": 60,
            "min_volumen_relativo": 0.2,
            "min_tp_pips": 0.12,
            "min_sl_pips": 0.16,
            "min_atr": 0.1,
            "adx_min": 25,
            "ema_conf": False,
            "ema_slope": False
        }
    ],
    "C:EURUSD": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 30,
            "rsi_rebote_max": 50,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.0016,          # objetivo de ~6 pips
            "min_sl_pips": 0.0016,
            "min_atr": 0.0007,         # ATR(14) > 0.0002 => ~2 pips en EUR/USD
            "adx_min": 25,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 99,
            "rsi_rebote_max": 100,   #Para qeu no entre venta
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.0016,
            "min_sl_pips": 0.0018,
            "min_atr": 0.0002,
            "adx_min": 25,
            "ema_conf": False,
            "ema_slope": False
        }
    ],
        "C:GBPJPY": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 45,
            "rsi_rebote_max": 75,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 1,          # objetivo de ~6 pips
            "min_sl_pips": 0.5,
            "min_atr": 0.07,         # ATR(14) > 0.0002 => ~2 pips en EUR/USD
            "adx_min": 30,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 40,
            "rsi_rebote_max": 60,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 1,
            "min_sl_pips": 0.5,
            "min_atr": 0.14,
            "adx_min": 20,
            "ema_conf": False,
            "ema_slope": False
        }
    ],
    "C:GBPUSD": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 30,
            "rsi_rebote_max": 60,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.001,         # objetivo de ~10 pips
            "min_sl_pips": 0.0018,
            "min_atr": 0.0005,         # ATR(14) > 0.0004 => ~4 pips en GBP/USD
            "adx_min": 35,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 30,
            "rsi_rebote_max": 75,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.0016,         # objetivo de ~10 pips
            "min_sl_pips": 0.0018,
            "min_atr": 0.0007,         # ATR(14) > 0.0004 => ~4 pips en GBP/USD
            "adx_min": 35,
            "ema_conf": False,
            "ema_slope": False
        }
    ],
    "C:EURGBP": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0.01,
            "rsi_rebote_min": 99,
            "rsi_rebote_max": 100, #PARA QEU NO ENTRE, no tiene retorno 
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.0010,          # objetivo de ~5 pips
            "min_sl_pips": 0.0018,
            "min_atr": 0.002,         # ATR(14) > 0.0002 => ~2 pips en EUR/GBP
            "adx_min": 15,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 0.001,
            "rsi_rebote_min": 45,
            "rsi_rebote_max": 60,
            "min_volumen_relativo": 0.0,
            "min_tp_pips": 0.0012,
            "min_sl_pips": 0.0016,
            "min_atr": 0.0006,
            "adx_min": 15,
            "ema_conf": False,
            "ema_slope": False
        }
    ],
    "C:XAUUSD": [
        {
            "tipo_operacion": "Compra",
            "min_cambio_pct": 0,
            "rsi_rebote_min": 0,
            "rsi_rebote_max": 100,
            "min_volumen_relativo": 0,
            "min_tp_pips": 1,
            "min_sl_pips": 0.12,
            "min_atr": 0,
            "adx_min": 0,
            "ema_conf": False,
            "ema_slope": False
        },
        {
            "tipo_operacion": "Venta",
            "min_cambio_pct": 100,
            "rsi_rebote_min": 0,
            "rsi_rebote_max": 100,
            "min_volumen_relativo": 0,
            "min_tp_pips": 1,
            "min_sl_pips": 0.8,
            "min_atr": 0,
            "adx_min": 0,
            "ema_conf": False,
            "ema_slope": False
        }
    ]
}

PIP_DECIMALS = {
    "USDJPY": 2,
    "EURUSD": 5,
    "GBPUSD": 5,
    "EURGBP": 5,
    "XAUUSD": 1
}

MIN_TP_PIPS = {
    "USDJPY": 3,
    "EURUSD": 6,
    "GBPUSD": 10,
    "EURGBP": 5,
    "XAUUSD": 10
}


MIN_DISTANCIA_IG_POR_SYMBOL = {
    "C:USDJPY": 0.012,
    "C:EURUSD": 0.00012,
    "C:GBPUSD": 0.00012,
    "C:EURGBP": 0.0001,
    "C:XAUUSD": 0.2
}


EPIC_MAP = {
    "C:USDJPY": "CS.D.USDJPY.CFD.IP",
    "C:EURUSD": "CS.D.EURUSD.CFD.IP",
    "C:GBPUSD": "CS.D.GBPUSD.CFD.IP",
    "C:EURGBP": "CS.D.EURGBP.CFD.IP",
    "C:GBPJPY": "CS.D.GBPJPY.CFD.IP",
    "C:XAUUSD": "CS.D.CFDGOLD.CFM.IP"
}


