import subprocess
import sys
import time

print("Starting FastAPI...")
subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "app:app", "--host", "0.0.0.0", "--port", "8000"
])

time.sleep(2)

print("Starting UI...")
subprocess.Popen([
    sys.executable, "-m", "streamlit",
    "run", "ui.py"
])

print("Service is running")
print("API: http://127.0.0.1:8000/docs")
print("UI: http://localhost:8501")

input("Press ENTER to stop services...")
