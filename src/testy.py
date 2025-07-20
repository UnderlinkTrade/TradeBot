import requests, json, time, pandas as pd
from datetime import datetime, timedelta, timezone
import configCurren as cfg

LOGIN_URL  = "https://api.ig.com/gateway/deal/session"
CHART_URL  = "https://api.ig.com/gateway/deal/prices/{epic}/MINUTE_5?from={}&to={}&pageSize=1000&pageNumber={}"

EPIC = "CS.D.EURGBP.CFD.IP"   # spot CFD
HEADERS = {
    "X-IG-API-KEY": cfg.IG_API_KEY,
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json; charset=UTF-8",
    "Version": "3"
}

# 1⃣  login REST (obtenemos CST y X-SECURITY-TOKEN)
r = requests.post(LOGIN_URL, headers=HEADERS, json={
    "identifier": cfg.IG_USERNAME,
    "password":   cfg.IG_PASSWORD
})
r.raise_for_status()
auth = r.headers
HEADERS["CST"]  = auth["CST"]
HEADERS["X-SECURITY-TOKEN"] = auth["X-SECURITY-TOKEN"]

# 2⃣  paginado de 2 días = 576 velas por call
end   = datetime.now(timezone.utc)
start = end - timedelta(days=10)      # ← rango que quieras
page  = 1
frames = []

while start < end:
    chunk_end = min(start + timedelta(days=2), end)
    url = CHART_URL.format(EPIC, start.isoformat(), chunk_end.isoformat(), page)
    resp = requests.get(url, headers=HEADERS).json()

    candles = resp["prices"]
    if not candles:
        break

    df = pd.DataFrame([{
        "date":   c["snapshotTimeUTC"],
        "o":      c["openPrice"]["bid"],
        "h":      c["highPrice"]["bid"],
        "l":      c["lowPrice"]["bid"],
        "c":      c["closePrice"]["bid"],
        "v":      c["lastTradedVolume"],
    } for c in candles])
    frames.append(df)

    start += timedelta(days=2)
    page = 1
    time.sleep(0.5)      # evita burst
print("Descargadas:", sum(len(f) for f in frames), "velas")
df_all = pd.concat(frames).sort_values("date")
df_all.to_csv("EURGBP_5m_IG.csv", index=False)
