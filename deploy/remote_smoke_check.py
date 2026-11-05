from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from serve_sign_api import create_app


def main() -> int:
    app = create_app()
    response = app.test_client().get("/")
    model_path = Path(app.config["SIGN_MODEL_PATH"])
    print(f"APP_ROOT_STATUS={response.status_code}")
    print(f"SIGN_MODEL_READY={model_path.is_file()}")
    print(f"MEDIAPIPE_ASSETS_READY={(ROOT / 'node_modules/@mediapipe/tasks-vision/wasm').is_dir()}")

    uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
    if not uri:
        print("ATLAS_PING=NOT_CONFIGURED")
        return 2
    try:
        client = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
    except Exception as exc:
        print(f"ATLAS_PING=FAILED ({type(exc).__name__})")
        return 3
    print("ATLAS_PING=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
