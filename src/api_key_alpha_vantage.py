from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import requests  # Vamos a usar requests para obtener más detalles de la conexión

# Tu clave API de Alpha Vantage
api_key = 'JL8H0GC994PMGLTQ'  # Sustituye con tu clave API

# Crear un objeto TimeSeries
ts = TimeSeries(key=api_key, output_format='pandas')

try:
    # Verificar la URL de la API antes de realizar la solicitud
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=^IXIC&interval=1min&apikey={api_key}"
    print(f"🔍 Realizando la solicitud a la API: {url}")
    
    # Probar la conexión con requests para ver si la API responde correctamente
    response = requests.get(url)
    print(f"🔍 Código de estado de la respuesta: {response.status_code}")
    print(f"🔍 Contenido de la respuesta: {response.text[:200]}...")  # Mostrar los primeros 200 caracteres de la respuesta

    # Si la API responde correctamente, obtener los datos
    data, meta_data = ts.get_intraday(symbol="AAPL", interval="1min", outputsize="full")
    
    # Verificar si la respuesta contiene datos
    if data.empty:
        print("⚠️ Los datos recibidos están vacíos.")
    else:
        # Mostrar los últimos datos
        print(data.tail())

except Exception as e:
    print(f"⚠️ Error al obtener los datos: {e}")
