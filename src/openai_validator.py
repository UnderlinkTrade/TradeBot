import openai
import os

# Versión nueva del cliente (>=1.0.0)
from configCurren import OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def validar_senal_con_gpt(data):
    try:
        system_prompt = (
            "Eres un analista técnico profesional especializado en scalping intradía con velas de 5 minutos sobre pares de divisas (Forex). "
            "Tu única fuente de decisión es el análisis técnico cuantitativo y la lectura de Order Flow. No debes considerar noticias ni fundamentales. "
            "El analisis debe ser en base al predicho, quiere decir si el predicho es mayor al precio actual, entonces evalua solo compra, en caso contrario, venta"
            "Tu trabajo es identificar oportunidades con ventaja estadística clara, incluso en escenarios imperfectos, tal como lo haría un scalper profesional en tiempo real. "
            "Tu tasa mínima aceptable de aciertos es del 70%, pero no debes ser excesivamente conservador. La inacción injustificada también es un error. "
            "Debes sugerir operar si al menos 2 señales técnicas razonablemente válidas están alineadas. No necesitas esperar una confluencia perfecta. "
            "Evita operar solo si las señales están claramente en conflicto. Asume el rol de quien busca maximizar oportunidades sin sobreoperar ni forzar entradas."
            "Tus decisiones deben estar respaldadas por principios operativos reales, extraídos de los 10 libros más influyentes sobre price action y scalping intradía "
            "(como 'Japanese Candlestick Charting Techniques', 'Technical Analysis of Financial Markets', 'High Probability Trading', entre otros). "
            "Tendrás acceso a variables como: RSI, MACD, ADX, ATR, Bandas de Bollinger, volumen relativo, distancia a SMA20, contexto multitemporal, "
            "y datos de Order Flow (spread, desequilibrio, presión BID/ASK). También se te entregará un resumen de las últimas 30 velas de 5 minutos. "

            "Tus respuestas deben tener este formato exacto, claro y táctico:"
            "- Recomendación: Comprar / Vender / Esperar\n"
            "- Stop Loss sugerido: nivel técnico o ajustado por ATR\n"
            "- Nivel sugerido de stop loss"
            "- Nivel sugerido de take profit"
            "- Calidad de la señal: nota del 0 al 10 (según solidez y claridad técnica)\n"

        )


        user_prompt = f"""
Analiza esta posible señal de trading intradía en velas de 5 minutos para el par {data['symbol']}. 
Evalúa la probabilidad de éxito de una operación de compra o venta en las próximas 3 velas (15 minutos), basándote exclusivamente en análisis técnico.

Contexto actual:
- Precio actual: {data['precio_actual']}
- Precio predicho: {data['precio_predicho']}
- RSI 5m: {data['rsi_5m']}
- RSI 15m: {data.get('rsi_15m', 'N/A')}
- RSI 1h: {data.get('rsi_1h', 'N/A')}
- MACD / Signal: {data['macd']} / {data['macd_signal']}
- ADX 5m: {data['adx_5m']}, +DI: {data['di_plus_5m']}, -DI: {data['di_minus_5m']}
- ATR: {data['atr']}
- Volumen relativo: {data['volumen_relativo']}
- Bandas de Bollinger: High {data['bb_high']}, Low {data['bb_low']}
- Distancia a SMA20: {data['distancia_sma20']}

📊 Resumen de velas (últimas 30 velas):
- Alcistas: {data['resumen_velas']['alcistas']}
- Bajistas: {data['resumen_velas']['bajistas']}
- Neutrales: {data['resumen_velas']['neutrales']}
- Rango promedio: {data['resumen_velas']['rango_promedio']:.5f}
- Último cierre: {data['resumen_velas']['ultimo_cierre']}
- Comportamiento: {data['resumen_velas']['comportamiento']}

🔍 Order Flow:
- Órdenes BID: {data.get('ordenes_bid', 'N/A')}
- Órdenes ASK: {data.get('ordenes_ask', 'N/A')}
- Desequilibrio: {data.get('desequilibrio_ordenes', 'N/A')}
- Último BID: {data.get('ultimo_bid', 'N/A')}
- Último ASK: {data.get('ultimo_ask', 'N/A')}
- Spread: {data.get('spread', 'N/A')}
"""
        if "ultimas_velas" in data:
            user_prompt += "🕒 Últimas 30 velas (timestamp, o, h, l, c):\n"
            for vela in data["ultimas_velas"]:
                user_prompt += f"- {vela['timestamp']}: O:{vela['o']} H:{vela['h']} L:{vela['l']} C:{vela['c']}\n"


        if "stop_loss" in data and "take_profit" in data:
            user_prompt += f"""

Niveles actuales propuestos:
- Stop Loss: {data['stop_loss']}
- Take Profit: {data['take_profit']}
"""

        user_prompt += """
Entrega tu análisis en este formato:
- Recomendación: Comprar / Vender / Esperar
- Entrega tambien el par que estas respuondiendo
- Entrega los pips para SL y TP para ig broker , pero no lo mezcles con el de arriba envia el mismo formato en compra y venta, evita mezclar pip con stop o take en una misma linea
- Nivel de calidad de la señal (entre 0 y 10)
- Sin resumen
"""

        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error al validar señal con GPT: {e}"
