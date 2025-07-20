import requests
from config import API_KEY, SYMBOLS

def obtener_sentimiento_tickers(tickers=SYMBOLS, limite_noticias=5):
    sentimientos_por_ticker = {symbol: [] for symbol in tickers}
    puntajes = {"positive": 1, "neutral": 0, "negative": -1}

    for symbol in tickers:
        url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit={limite_noticias}&apiKey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        noticias = data.get('results', [])

        for noticia in noticias:
            insights = noticia.get('insights', [])
            for insight in insights:
                ticker_insight = insight['ticker']
                if ticker_insight in tickers:
                    sentimientos_por_ticker[ticker_insight].append(
                        puntajes.get(insight['sentiment'], 0)
                    )

    # Calcular promedio de sentimientos
    sentimiento_promedio = {}
    for ticker, scores in sentimientos_por_ticker.items():
        if scores:
            sentimiento_promedio[ticker] = round(sum(scores) / len(scores), 2)
        else:
            sentimiento_promedio[ticker] = 0  # neutral si no hay datos

    return sentimiento_promedio
