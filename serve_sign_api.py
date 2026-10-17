from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
import joblib

from sign_baseline import predict_sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model_output" / "multisigner_run-20261015-181411" / "final_demo_model.joblib"


def _load_terms(path: Path, videos_dir: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("terms file must contain a JSON array")
    terms = []
    for raw in data:
        item = dict(raw)
        filename = item.get("video_url")
        ready = item.get("status") == "ready" and isinstance(filename, str) and bool(filename)
        if ready:
            video_path = videos_dir / filename
            ready = video_path.is_file() and video_path.resolve().parent == videos_dir.resolve()
        if not ready:
            item["status"] = "pending_material" if item.get("status") == "ready" else item.get("status", "draft")
            item["video_url"] = None
        else:
            item["video_url"] = "/videos/" + filename
        terms.append(item)
    return terms


def create_app(model_path: Path | str | None = None, threshold: float = 0.55,
               terms_path: Path | str | None = None, videos_dir: Path | str | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    terms_file = Path(terms_path) if terms_path else ROOT / "data" / "sign-entries.json"
    media_dir = Path(videos_dir) if videos_dir else ROOT / "videos"
    selected_model = Path(model_path) if model_path else DEFAULT_MODEL
    bundle = None
    if selected_model.is_file():
        try:
            bundle = joblib.load(selected_model)
        except Exception:
            bundle = None

    def terms():
        return _load_terms(terms_file, media_dir)

    @app.get("/")
    def index():
        return send_from_directory(ROOT, "index.html")

    @app.get("/api/terms")
    def all_terms():
        return jsonify({"terms": terms()})

    @app.get("/api/terms/<term_id>")
    def one_term(term_id: str):
        item = next((entry for entry in terms() if str(entry.get("id")) == term_id), None)
        if item is None:
            return jsonify({"message": "词条不存在"}), 404
        return jsonify(item)

    @app.post("/api/terms/match")
    def match_terms():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return jsonify({"status": "not_found", "match_status": "not_found", "message": "请输入词语"}), 400
        items = terms()
        exact = [item for item in items if text == item.get("term") or text in (item.get("aliases") or [])]
        if len(exact) == 1:
            return jsonify({"status": exact[0].get("status"), "match_status": "matched", "term_data": exact[0]})
        candidates = [item for item in items if item.get("term") in text or any(alias and alias in text for alias in item.get("aliases", []))]
        unique = {item["id"]: item for item in candidates}
        candidates = list(unique.values())
        if len(candidates) == 1:
            return jsonify({"status": "confirmation_required", "match_status": "not_direct", "message": "检测到一个词条，请确认", "candidates": candidates})
        if candidates:
            return jsonify({"status": "ambiguous", "match_status": "not_direct", "message": "检测到多个词条，请选择", "candidates": candidates})
        return jsonify({"status": "not_found", "match_status": "not_found", "message": "该内容暂未收录"})

    @app.get("/videos/<path:filename>")
    def video(filename: str):
        return send_from_directory(media_dir, filename, conditional=True)

    @app.get("/api/health")
    def health():
        loaded = bundle is not None
        labels = list(bundle.get("labels", [])) if loaded and isinstance(bundle, dict) else []
        assets_ready = (ROOT / "assets" / "hand_landmarker.task").is_file() and (ROOT / "node_modules" / "@mediapipe" / "tasks-vision" / "wasm").is_dir()
        return jsonify({"model_loaded": loaded, "labels": labels, "input_shape": [60, 126], "camera_features_ready": assets_ready}), (200 if loaded else 503)

    @app.post("/api/predict")
    def predict():
        if bundle is None:
            return jsonify({"accepted": False, "label": None, "candidate_label": None, "confidence": 0.0, "message": "模型尚未加载"}), 503
        payload = request.get_json(silent=True) or {}
        try:
            result = predict_sequence(payload.get("sequence"), bundle, threshold=threshold)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.get("/camera_features.mjs")
    def camera_features():
        return send_from_directory(ROOT, "camera_features.mjs")

    @app.get("/vendor/mediapipe/<path:filename>")
    def mediapipe_vendor(filename: str):
        return send_from_directory(ROOT / "node_modules" / "@mediapipe" / "tasks-vision", filename)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=False)
