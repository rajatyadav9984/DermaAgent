"""
DermaAgent - Flask Web Application Entry Point
"""

import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from threading import Thread
from flask import Flask, send_from_directory

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend directory & project root are in path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routes.predict_route import predict_bp
from routes.health_route import health_bp

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# CORS headers handler for development API calls
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(health_bp)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

def wait_and_open_browser(port=5000):
    """
    Poll /api/health via HTTP until Flask is ready, then open browser.
    Prevents Werkzeug socket resets and ERR_CONNECTION_REFUSED.
    """
    for _ in range(30):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(f"http://127.0.0.1:{port}/")
                    return
        except Exception:
            pass

if __name__ == "__main__":
    print("\n[INFO] Starting DermaAgent Server...")
    print("[INFO] Web App will open automatically once server is ready...\n")
    Thread(target=wait_and_open_browser, daemon=True).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
