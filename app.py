"""
DermaAgent - Root Application Launcher
Allows running from root directory directly via: python app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app import app

if __name__ == "__main__":
    from backend.app import wait_and_open_browser
    from threading import Thread

    print("\n[INFO] Starting DermaAgent from Root Entry Point...")
    Thread(target=wait_and_open_browser, daemon=True).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
