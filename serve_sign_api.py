from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from datetime import datetime
from uuid import uuid4

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import joblib

from sign_baseline import predict_sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_REJECTION_CONFIG = ROOT / "config" / "rejection_profiles.json"
DEBUG_FRAMES_DIR = ROOT / "debug_frames"


def _save_debug_sequence(sequence: Any, candidate: str | None, confidence: float) -> Path | None:
    """Persist one valid request sequence without allowing it to affect inference."""
    try:
        array = np.asarray(sequence, dtype=np.float32)
        if array.shape != (60, 126) or not np.isfinite(array).all():
            return None
        DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        safe_candidate = re.sub(r"[^\w-]", "_", candidate or "unknown", flags=re.UNICODE)[:32]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = f"debug_{stamp}_{safe_candidate}_confidence-{confidence:.4f}_{uuid4().hex[:8]}.npy"
        path = DEBUG_FRAMES_DIR / filename
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        return path
    except Exception as exc:
        print(f"[debug-capture] save failed: {exc}", flush=True)
        return None


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


def _load_rejection_config(config_path: Path, mode: str | None, threshold_override: float | None) -> tuple[dict[str, Any], str, float]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"无法读取拒识配置 {config_path}: {exc}") from exc
    profiles = config.get("profiles")
    selected_mode = mode or os.environ.get("SIGN_REJECTION_MODE") or config.get("default_mode")
    if not isinstance(profiles, dict) or selected_mode not in profiles:
        raise RuntimeError(f"拒识配置缺少有效模式: {selected_mode}")
    profile = profiles[selected_mode]
    required = ("confidence_threshold", "quality_gate", "min_selected_frames", "top2_margin_threshold", "feature_mode")
    missing = [key for key in required if key not in config and key not in profile]
    if missing:
        raise RuntimeError(f"拒识配置缺少字段: {', '.join(missing)}")
    threshold = float(threshold_override if threshold_override is not None else profile["confidence_threshold"])
    minimum = int(config.get("min_selected_frames", profile.get("min_selected_frames")))
    margin = float(config.get("top2_margin_threshold", profile.get("top2_margin_threshold")))
    if not 0 <= threshold <= 1 or minimum < 1 or margin < 0:
        raise RuntimeError("拒识配置中的阈值或有效帧门槛非法")
    resolved = dict(config)
    resolved["selected_mode"] = selected_mode
    resolved["selected_profile"] = dict(profile)
    resolved["confidence_threshold"] = threshold
    resolved["min_selected_frames"] = minimum
    resolved["top2_margin_threshold"] = margin
    resolved["config_path"] = str(config_path.resolve())
    return resolved, selected_mode, threshold


def create_app(model_path: Path | str | None = None, threshold: float | None = None,
               terms_path: Path | str | None = None, videos_dir: Path | str | None = None,
               config_path: Path | str | None = None, rejection_mode: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    terms_file = Path(terms_path) if terms_path else ROOT / "data" / "sign-entries.json"
    media_dir = Path(videos_dir) if videos_dir else ROOT / "videos"
    config_file = Path(config_path) if config_path else Path(os.environ.get("SIGN_REJECTION_CONFIG", DEFAULT_REJECTION_CONFIG))
    rejection_config, selected_mode, selected_threshold = _load_rejection_config(config_file, rejection_mode, threshold)
    configured_model = Path(rejection_config["model_path"])
    if not configured_model.is_absolute():
        configured_model = (config_file.parent / configured_model).resolve()
    selected_model = Path(model_path).resolve() if model_path else configured_model
    bundle = None
    model_error = None
    if selected_model.is_file():
        try:
            bundle = joblib.load(selected_model)
            if model_path is None:
                feature_config = bundle.get("feature_config", {})
                if (feature_config.get("mode") != rejection_config["feature_mode"] or
                        feature_config.get("model_feature_dimension") != 252):
                    raise RuntimeError("模型特征 metadata 与拒识配置不一致")
                if len(bundle.get("labels", [])) != 10:
                    raise RuntimeError("演示模型类别数不是固定词表的 10 类")
        except Exception as exc:
            bundle = None
            model_error = str(exc)
    else:
        model_error = f"模型文件不存在: {selected_model}"
    app.config["SIGN_MODEL_PATH"] = str(selected_model)
    app.config["SIGN_REJECTION_MODE"] = selected_mode
    app.config["SIGN_CONFIDENCE_THRESHOLD"] = selected_threshold

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
        feature_config = bundle.get("feature_config", {}) if loaded and isinstance(bundle, dict) else {}
        return jsonify({"model_loaded": loaded, "labels": labels, "input_shape": [60, 126],
                        "feature_mode": feature_config.get("mode", rejection_config["feature_mode"]),
                        "model_feature_dimension": feature_config.get("model_feature_dimension"),
                        "selected_frames": "any_valid", "min_selected_frames": rejection_config["min_selected_frames"],
                        "training_min_selected_frames": feature_config.get("min_selected_frames") if loaded else None,
                        "quality_gate": rejection_config["quality_gate"],
                        "confidence_threshold": selected_threshold,
                        "rejection_mode": selected_mode,
                        "rejection_mode_description": rejection_config["selected_profile"].get("description", ""),
                        "top2_margin_threshold": rejection_config["top2_margin_threshold"],
                        "rejection_config_source": rejection_config["config_path"],
                        "calibration_source": rejection_config.get("calibration_source"),
                        "model_path": str(selected_model), "model_error": model_error,
                        "camera_features_ready": assets_ready}), (200 if loaded else 503)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if bundle is None:
            _save_debug_sequence(payload.get("sequence"), None, 0.0)
            return jsonify({"accepted": False, "label": None, "candidate_label": None, "confidence": 0.0, "message": "模型尚未加载"}), 503
        try:
            result = predict_sequence(payload.get("sequence"), bundle, threshold=selected_threshold,
                                      margin_threshold=rejection_config["top2_margin_threshold"],
                                      min_selected_frames=rejection_config["min_selected_frames"])
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        debug_path = _save_debug_sequence(
            payload.get("sequence"), result.get("candidate_label"), float(result.get("confidence", 0.0))
        )
        print(
            "[predict] "
            f"accepted={result.get('accepted')} "
            f"label={result.get('label')} "
            f"candidate={result.get('candidate_label')} "
            f"confidence={float(result.get('confidence', 0.0)):.4f} "
            f"threshold={selected_threshold:.2f} "
            f"margin={float(result.get('top2_margin', 0.0)):.4f} "
            f"valid_frames={result.get('valid_frame_count', 'n/a')} "
            f"message={result.get('message')}",
            flush=True,
        )
        if debug_path:
            print(f"[debug-capture] saved={debug_path}", flush=True)
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
    parser.add_argument("--config", type=Path, default=None, help="拒识配置文件路径")
    parser.add_argument("--rejection-mode", choices=("calibrated", "demo"), default=None,
                        help="默认 calibrated；demo=0.60，未经同等校准")
    args = parser.parse_args()
    app = create_app(config_path=args.config, rejection_mode=args.rejection_mode)
    print(f"model={app.config['SIGN_MODEL_PATH']} rejection_mode={app.config['SIGN_REJECTION_MODE']} threshold={app.config['SIGN_CONFIDENCE_THRESHOLD']}")
    app.run(host=args.host, port=args.port, debug=False)
