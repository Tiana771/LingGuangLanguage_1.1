import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import serve_sign_api
from serve_sign_api import create_app
from test_assistant_api import FakeMongoStore, login_test_session


class Upstream:
    headers = {"x-request-id": "provider-test-request"}

    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        for event in self.events:
            if event == "[DONE]":
                yield b"data: [DONE]\n"
            else:
                yield ("data: " + json.dumps(event) + "\n").encode()

    def close(self):
        self.closed = True


class PublicFilesTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(mongo_store=FakeMongoStore()).test_client()

    def test_existing_private_files_are_not_public(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [".env", ".git/config", "connection.env", "serve_sign_api.py",
                     "model_audit.py", "config/rejection_profiles.json", "models/private.joblib",
                     "data/sign-entries.json", "requirements.txt", "test_assistant_api.py"]
            for name in paths:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("private sentinel", encoding="utf-8")
            with patch("serve_sign_api.ROOT", root):
                for name in paths:
                    with self.subTest(path=name):
                        response = self.client.get("/" + name)
                        self.assertEqual(response.status_code, 404)
                        self.assertNotIn(b"private sentinel", response.data)

    def test_browser_modules_and_camera_model_still_load(self):
        paths = ["/", "/index.html", "/styles.css", "/app.js", "/auth_client.mjs",
                 "/communication_assistant.mjs", "/deepseek_stream.mjs", "/learning_client.mjs",
                 "/assets/linguang-mark.svg", "/assets/hand_landmarker.task",
                 "/vendor/mediapipe/vision_bundle.mjs",
                 "/vendor/mediapipe/wasm/vision_wasm_internal.js",
                 "/vendor/mediapipe/wasm/vision_wasm_internal.wasm"]
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()
        self.assertEqual(self.client.get("/vendor/mediapipe/package.json").status_code, 404)
        self.assertEqual(self.client.get("/node_modules/@mediapipe/tasks-vision/package.json").status_code, 404)

    def test_public_filename_cannot_escape_through_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            root = parent / "public"
            root.mkdir()
            secret = parent / "private.txt"
            secret.write_text("private sentinel", encoding="utf-8")
            try:
                (root / "app.js").symlink_to(secret)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with patch("serve_sign_api.ROOT", root):
                self.assertEqual(self.client.get("/app.js").status_code, 404)


class ModelUsageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "model-usage.jsonl"
        self.environment = patch.dict(os.environ, {
            "DEEP_SEEK_API_KEY": "private-api-key-sentinel",
            "DEEP_SEEK_KEY_LABEL": "production-test",
            "LINGUANG_MODEL_LOG": str(self.path),
            "LINGUANG_TRUST_PROXY": "1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.store = FakeMongoStore()
        self.client = create_app(mongo_store=self.store).test_client()
        login_test_session(self.client)

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def post(self, path="/api/assistant/stream", payload=None, **kwargs):
        return self.client.post(
            path, json=payload or {"action": "reply", "text": "private-transcript-sentinel"},
            headers={"X-CSRF-Token": "test-csrf", "X-Forwarded-For": "203.0.113.20"}, **kwargs,
        )

    def test_usage_only_chunk_is_saved_without_private_text_or_key(self):
        usage = {"prompt_tokens": 250, "completion_tokens": 20, "total_tokens": 270,
                 "prompt_cache_hit_tokens": 100, "prompt_cache_miss_tokens": 150}
        upstream = Upstream([
            {"id": "chat-test", "model": "deepseek-flash", "choices": [{"delta": {"content": "reply"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage}, "[DONE]",
        ])
        with patch("serve_sign_api.urlopen", return_value=upstream) as mocked:
            response = self.post()
            self.assertIn(b"event: done", response.data)
            response.close()
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["started", "completed"])
        self.assertEqual(records[-1]["usage"], usage)
        self.assertEqual(records[-1]["client_ip"], "203.0.113.20")
        self.assertEqual(records[-1]["user_id"], "user-a")
        self.assertEqual(records[-1]["provider_request_id"], "provider-test-request")
        self.assertEqual(records[-1]["model_returned"], "deepseek-flash")
        self.assertEqual(records[-1]["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(json.loads(mocked.call_args.args[0].data)["stream_options"], {"include_usage": True})
        logged = self.path.read_text(encoding="utf-8")
        self.assertNotIn("private-api-key-sentinel", logged)
        self.assertNotIn("private-transcript-sentinel", logged)
        self.assertTrue(upstream.closed)

    def test_incomplete_stream_keeps_unknown_usage(self):
        with patch("serve_sign_api.urlopen", return_value=Upstream([
            {"choices": [{"delta": {"content": "partial"}}]},
        ])):
            response = self.post()
            self.assertIn(b"event: error", response.data)
            response.close()
        last = self.records()[-1]
        self.assertEqual(last["status"], "stream_incomplete")
        self.assertEqual(last["usage_status"], "unknown")
        self.assertIsNone(last["usage"])

    def test_cancelled_stream_is_logged_once(self):
        upstream = Upstream([{"choices": [{"delta": {"content": "later"}}]}, "[DONE]"])
        with patch("serve_sign_api.urlopen", return_value=upstream):
            response = self.post(buffered=False)
            response.close()
        self.assertEqual([r["status"] for r in self.records()], ["started", "cancelled"])
        self.assertTrue(upstream.closed)

    def test_upstream_error_is_logged_without_retry(self):
        with patch("serve_sign_api.urlopen", side_effect=HTTPError(
            "https://api.deepseek.com/chat/completions", 402, "balance", {}, io.BytesIO(b""),
        )) as mocked:
            response = self.post()
        self.assertEqual(response.status_code, 502)
        mocked.assert_called_once()
        self.assertEqual(self.records()[-1]["http_status"], 402)

    def test_history_uses_same_usage_audit(self):
        self.store.insert_conversation("user-a", {"title": "test", "blocks": [
            {"id": "block-a", "text": "医院明天下午三点复诊"},
        ]})
        usage = {"prompt_tokens": 310, "completion_tokens": 16, "total_tokens": 326}
        with patch("serve_sign_api.urlopen", return_value=Upstream([
            {"choices": [{"delta": {"content": "明天下午三点"}}]},
            {"choices": [], "usage": usage}, "[DONE]",
        ])):
            response = self.post("/api/history/ask", {"question": "医院"})
            self.assertIn(b"event: done", response.data)
            response.close()
        self.assertEqual(self.records()[-1]["action"], "history")
        self.assertEqual(self.records()[-1]["usage"], usage)
        self.assertEqual(len(self.store.list_assistant_chats("user-a")), 1)


if __name__ == "__main__":
    unittest.main()
