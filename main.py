import subprocess

print("⚙️ Entrenando o cargando modelos...")
subprocess.run(["python", "src/modelCurren.py"], check=True)

print("⚙️ Descargando data......")
subprocess.run(["python", "src/data_loader.py"], check=True)

print("🚀 Iniciando monitoreo en vivo...")
subprocess.run(["python", "src/live_monitorCurren.py"], check=True)
