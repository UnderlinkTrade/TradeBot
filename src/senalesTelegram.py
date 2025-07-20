# ── IMPORTS ────────────────────────────────────────────────────────────────────
import re, logging, asyncio, sys, subprocess
from telethon import TelegramClient, events
from pathlib import Path            # ← AÑADE ESTA LÍNEA
from configCurren import (
    telegram_api_id as api_id, telegram_api_hash as api_hash, telegram_phone as phone,
    TELEGRAM_CHAT_ID,  # lo seguiremos usando, luego lo corregimos
)

client = TelegramClient('bot_ig_session', api_id, api_hash)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Ruta absoluta al script backtest.py
BACKTEST_PATH = Path(__file__).resolve().parent / "backtest.py"

# ── PASO 1: handler sin filtro para ver los IDs ──────────────────────────────
@client.on(events.NewMessage)          # <── sin “chats=…”
async def debug(event):
    logging.info(f"Debug ▶ chat_id={event.chat_id} | text={event.raw_text!r}")

    # cuando veas “backtest”, lanza el script
    if re.search(r"\bbacktest\b", event.raw_text, re.I):
        await event.reply("🔄 Ejecutando back-test…")

        cmd = [sys.executable, str(BACKTEST_PATH)]   # ← usa la ruta absoluta
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = await proc.communicate()

        if proc.returncode == 0:
            await event.reply("✅ Back-test finalizado")
            logging.info("Back-test OK")
        else:
            await event.reply(f"❌ Back-test falló (código {proc.returncode})")
            logging.error(err.decode() if err else "Sin stderr")

if __name__ == "__main__":
    client.start(phone)
    client.run_until_disconnected()
