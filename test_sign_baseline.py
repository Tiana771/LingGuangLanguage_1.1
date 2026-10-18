from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from serve_sign_api import create_app
from sign_baseline import parse_label, pooled_sequence_features, predict_sequence, scan_dataset, sequence_features


class BaselineTests(unittest.TestCase):
    def test_parse_current_filename_label(self):
        self.assertEqual(parse_label(Path("5_欢迎_10.npy")), "欢迎")

    def test_scan_reports_valid_and_invalid_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "1_谢谢_0.npy", np.ones((60, 126), dtype=np.float64))
            np.save(root / "1_谢谢_1.npy", np.full((59, 126), 2, dtype=np.float32))
            records, summary = scan_dataset(root)
        self.assertEqual(summary["files_found"], 2)
        self.assertEqual(summary["valid_samples"], 1)
        self.assertEqual(summary["invalid_samples"], 1)
        self.assertEqual(records[1].status, "invalid")

    def test_temporal_features_keep_one_row_per_sequence(self):
        output = sequence_features(np.ones((60, 126), dtype=np.float32))
        self.assertEqual(output.shape, (1260,))

    def test_prediction_and_rejection(self):
        model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
        rows = np.vstack([np.zeros((5, 1260)), np.ones((5, 1260))])
        labels = np.array(["甲"] * 5 + ["乙"] * 5)
        model.fit(rows, labels)
        bundle = {"model": model, "labels": model.named_steps["svc"].classes_.tolist()}
        sample = np.zeros((60, 126), dtype=np.float32)
        accepted = predict_sequence(sample, bundle, threshold=0.0)
        rejected = predict_sequence(sample, bundle, threshold=1.0)
        self.assertIn(accepted["label"], {"甲", "乙"})
        self.assertTrue(accepted["accepted"])
        self.assertIsNone(rejected["label"])
        self.assertIsNotNone(rejected["candidate_label"])
        self.assertEqual(rejected["message"], "暂未识别，请重试")

    def test_prediction_uses_classifier_label_not_probability_argmax(self):
        class MismatchedModel:
            def predict(self, _features):
                return np.array(["甲"])

            def predict_proba(self, _features):
                return np.array([[0.3, 0.7]])

        bundle = {"model": MismatchedModel(), "labels": ["甲", "乙"]}
        result = predict_sequence(np.zeros((60, 126)), bundle, threshold=0.5)
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["label"])
        self.assertAlmostEqual(result["confidence"], 0.3)

    def test_probability_columns_follow_classifier_classes_order(self):
        class ReorderedModel:
            classes_ = np.asarray(["乙", "甲"])

            def predict(self, _features):
                return np.asarray(["甲"])

            def predict_proba(self, _features):
                return np.asarray([[0.2, 0.8]])

        bundle = {"model": ReorderedModel(), "labels": ["甲", "乙"]}
        result = predict_sequence(np.zeros((60, 126)), bundle, threshold=0.75)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["label"], "甲")
        self.assertAlmostEqual(result["confidence"], 0.8)

    def test_scheme_b_uses_calibration_top_probability_class(self):
        class ReorderedModel:
            classes_ = np.asarray(["谢谢", "你好"])

            def predict(self, _features):
                return np.asarray(["你好"])

            def predict_proba(self, _features):
                return np.asarray([[0.82, 0.18]])

        sequence = np.zeros((60, 126), dtype=np.float32)
        sequence[:15, 3] = 1
        bundle = {"model": ReorderedModel(), "labels": ["你好", "谢谢"],
                  "feature_config": {"mode": "B", "min_selected_frames": 15}}
        result = predict_sequence(sequence, bundle, threshold=0.78)
        self.assertEqual(result["candidate_label"], "谢谢")
        self.assertEqual(result["label"], "谢谢")
        self.assertAlmostEqual(result["confidence"], 0.82)

    def test_scheme_b_features_and_low_valid_frame_rejection(self):
        class FixedModel:
            classes_ = np.asarray(["你好", "谢谢"])

            def predict(self, _features):
                return np.asarray(["你好"])

            def predict_proba(self, _features):
                return np.asarray([[0.9, 0.1]])

        sequence = np.zeros((60, 126), dtype=np.float32)
        sequence[:15, 4] = np.arange(1, 16)
        features = pooled_sequence_features(sequence, min_selected_frames=15)
        self.assertEqual(features.shape, (252,))
        bundle = {"model": FixedModel(), "labels": ["你好", "谢谢"],
                  "feature_config": {"mode": "B", "min_selected_frames": 15, "model_feature_dimension": 252}}
        accepted = predict_sequence(sequence, bundle, threshold=0.78)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["label"], "你好")
        sequence[14] = 0
        rejected = predict_sequence(sequence, bundle, threshold=0.78)
        self.assertFalse(rejected["accepted"])
        self.assertIsNone(rejected["label"])
        self.assertEqual(rejected["candidate_label"], None)
        self.assertEqual(rejected["valid_frame_count"], 14)

    def test_b_model_bundle_round_trip_and_configured_health(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
            x = np.vstack([np.zeros((5, 252)), np.ones((5, 252))]).astype(np.float32)
            y = np.asarray(["你好"] * 5 + ["谢谢"] * 5)
            model.fit(x, y)
            model_path = Path(temp) / "b-model.joblib"
            joblib.dump({"model": model, "labels": ["你好", "谢谢"],
                         "feature_config": {"mode": "B", "input_shape": [60, 126],
                                            "model_feature_dimension": 252, "min_selected_frames": 30}}, model_path)
            reloaded = joblib.load(model_path)
            self.assertEqual(reloaded["feature_config"]["model_feature_dimension"], 252)
            client = create_app(model_path=model_path, threshold=0.78).test_client()
            health = client.get("/api/health").get_json()
            self.assertEqual(health["feature_mode"], "B")
            self.assertEqual(health["min_selected_frames"], 15)
            self.assertEqual(health["confidence_threshold"], 0.78)
            self.assertEqual(health["rejection_mode"], "calibrated")
            low_quality = np.zeros((60, 126), dtype=float)
            low_quality[:14, 2] = 1
            rejected = client.post("/api/predict", json={"sequence": low_quality.tolist()}).get_json()
            self.assertFalse(rejected["accepted"])
            self.assertIsNone(rejected["label"])
            self.assertIsNone(rejected["candidate_label"])
            self.assertEqual(rejected["message"], "暂未识别，请重试")
            good_quality = np.zeros((60, 126), dtype=float)
            good_quality[:30, 2] = 1
            permissive_client = create_app(model_path=model_path, threshold=0.0).test_client()
            accepted = permissive_client.post("/api/predict", json={"sequence": good_quality.tolist()}).get_json()
            self.assertTrue(accepted["accepted"])
            self.assertIn(accepted["label"], {"你好", "谢谢"})
            strict_client = create_app(model_path=model_path, threshold=1.0).test_client()
            low_confidence = strict_client.post("/api/predict", json={"sequence": good_quality.tolist()}).get_json()
            self.assertFalse(low_confidence["accepted"])
            self.assertIsNone(low_confidence["label"])
            self.assertEqual(low_confidence["message"], "暂未识别，请重试")

    def test_rejection_config_profiles_and_invalid_config(self):
        client = create_app(rejection_mode="demo").test_client()
        health = client.get("/api/health").get_json()
        self.assertEqual(health["rejection_mode"], "demo")
        self.assertEqual(health["confidence_threshold"], 0.60)
        self.assertIn("未经同等校准", health["rejection_mode_description"])
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp) / "invalid.json"
            invalid.write_text('{"default_mode":"calibrated","profiles":{}}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                create_app(config_path=invalid)

    def test_api_model_not_ready_and_request_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            missing_app = create_app(Path(temp) / "missing.joblib")
            missing_client = missing_app.test_client()
            self.assertEqual(missing_client.get("/api/health").status_code, 503)

            model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
            rows = np.vstack([np.zeros((5, 1260)), np.ones((5, 1260))])
            labels = np.array(["甲"] * 5 + ["乙"] * 5)
            model.fit(rows, labels)
            model_path = Path(temp) / "model.joblib"
            joblib.dump({"model": model, "labels": model.named_steps["svc"].classes_.tolist()}, model_path)
            client = create_app(model_path, threshold=0.0).test_client()
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.get_json()["model_loaded"])
            self.assertEqual(health.get_json()["input_shape"], [60, 126])
            self.assertEqual(set(health.get_json()["labels"]), {"甲", "乙"})
            self.assertEqual(client.post("/api/predict", json={"sequence": [[0]]}).status_code, 400)
            response = client.post(
                "/api/predict",
                json={"sequence": np.zeros((60, 126), dtype=float).tolist()},
            )
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertTrue({"accepted", "label", "candidate_label", "confidence", "message"}.issubset(result))
            self.assertIsNotNone(result["candidate_label"])
            reject_client = create_app(model_path, threshold=1.0).test_client()
            rejected = reject_client.post(
                "/api/predict",
                json={"sequence": np.zeros((60, 126), dtype=float).tolist()},
            ).get_json()
            self.assertFalse(rejected["accepted"])
            self.assertIsNone(rejected["label"])
            self.assertEqual(rejected["message"], "暂未识别，请重试")

    def test_local_camera_assets_are_served_without_external_model_fetch(self):
        client = create_app().test_client()
        health = client.get("/api/health")
        self.assertIn("camera_features_ready", health.get_json())
        for path in (
            "/camera_features.mjs",
            "/vendor/mediapipe/vision_bundle.mjs",
            "/vendor/mediapipe/wasm/vision_wasm_internal.js",
            "/vendor/mediapipe/wasm/vision_wasm_internal.wasm",
            "/assets/hand_landmarker.task",
        ):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertGreater(len(response.data), 0)
                response.close()

    def test_term_api_uses_current_ten_labels_and_ready_videos(self):
        client = create_app().test_client()
        for path in ("/", "/app.js", "/styles.css"):
            page_response = client.get(path)
            self.assertEqual(page_response.status_code, 200)
            page_response.close()
        response = client.get("/api/terms")
        self.assertEqual(response.status_code, 200)
        terms = response.get_json()["terms"]
        self.assertEqual([item["term"] for item in terms], ["你好", "谢谢", "请问", "医院", "厕所", "欢迎", "帮助", "报警", "吃饭", "再见"])
        self.assertTrue(all(item["status"] == "ready" for item in terms))
        self.assertTrue(all(item["video_url"].startswith("/videos/") for item in terms))
        for item in terms:
            response = client.get(item["video_url"])
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_term_match_exact_confirmation_ambiguity_and_not_found(self):
        client = create_app().test_client()
        exact = client.post("/api/terms/match", json={"text": "谢谢"}).get_json()
        self.assertEqual(exact["status"], "ready")
        self.assertEqual(exact["match_status"], "matched")
        self.assertEqual(exact["term_data"]["term"], "谢谢")

        sentence = client.post("/api/terms/match", json={"text": "我想去医院"}).get_json()
        self.assertEqual(sentence["status"], "confirmation_required")
        self.assertEqual(sentence["candidates"][0]["term"], "医院")

        ambiguous = client.post("/api/terms/match", json={"text": "再见医院"}).get_json()
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertEqual({item["term"] for item in ambiguous["candidates"]}, {"再见", "医院"})

        missing = client.post("/api/terms/match", json={"text": "早上好"}).get_json()
        self.assertEqual(missing["status"], "not_found")

    def test_video_is_served_only_for_ready_manifest_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            videos = root / "videos"
            videos.mkdir()
            (videos / "clip.mp4").write_bytes(b"test-video")
            terms_file = root / "terms.json"
            terms_file.write_text(
                '[{"id":"x","term":"词","aliases":[],"status":"ready","video_url":"clip.mp4"}]',
                encoding="utf-8",
            )
            client = create_app(terms_path=terms_file, videos_dir=videos).test_client()
            term = client.get("/api/terms/x").get_json()
            self.assertEqual(term["status"], "ready")
            self.assertEqual(term["video_url"], "/videos/clip.mp4")
            video_response = client.get("/videos/clip.mp4")
            self.assertEqual(video_response.status_code, 200)
            video_response.close()
            self.assertEqual(client.get("/videos/../terms.json").status_code, 404)

    def test_ready_term_with_missing_file_stays_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            terms_file = root / "terms.json"
            terms_file.write_text(
                '[{"id":"x","term":"词","aliases":[],"status":"ready","video_url":"missing.mp4"}]',
                encoding="utf-8",
            )
            client = create_app(terms_path=terms_file, videos_dir=root / "videos").test_client()
            term = client.get("/api/terms/x").get_json()
            self.assertEqual(term["status"], "pending_material")
            self.assertIsNone(term["video_url"])


if __name__ == "__main__":
    unittest.main()
