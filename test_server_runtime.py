import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from server_runtime import ServerAlreadyRunningError, claim_server_port


class ServerRuntimeTests(unittest.TestCase):
    def test_refuses_existing_listener_including_legacy_reusable_socket(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaises(ServerAlreadyRunningError):
                with claim_server_port("127.0.0.1", listener.getsockname()[1], lock_dir=directory):
                    self.fail("A second server must not start")

    def test_prevents_duplicate_launcher_and_releases_lock_after_exit(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
            with claim_server_port("127.0.0.1", port, lock_dir=directory):
                with self.assertRaises(ServerAlreadyRunningError):
                    with claim_server_port("127.0.0.1", port, lock_dir=directory):
                        self.fail("The launch lock must be exclusive")
            with claim_server_port("127.0.0.1", port, lock_dir=directory):
                pass

    def test_sample_launcher_selects_v2_before_creating_app(self):
        from serve_sign_api import main
        app = MagicMock()
        app.config = {"SIGN_MODEL_PATH": "test", "SIGN_REJECTION_MODE": "calibrated", "SIGN_CONFIDENCE_THRESHOLD": 0.78}
        observed = {}

        def build_app(**kwargs):
            for name in ("BOOK_KNOWLEDGE_COLLECTION", "BOOK_EMBEDDING_INDEX_NAME", "BOOK_CONTENT_EXTERNAL_ALLOWED"):
                observed[name] = os.environ[name]
            return app

        with patch.dict(os.environ, {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks"}), \
                patch("serve_sign_api.create_app", side_effect=build_app), \
                patch("server_runtime.claim_server_port"), patch("builtins.print"):
            self.assertEqual(main(["--sample-books"]), 0)
        self.assertEqual(observed, {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2",
                                    "BOOK_EMBEDDING_INDEX_NAME": "sign_knowledge_vector_v2",
                                    "BOOK_CONTENT_EXTERNAL_ALLOWED": "false"})
        app.run.assert_called_once_with(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

    def test_occupied_port_does_not_create_database_or_second_app(self):
        from serve_sign_api import main
        with patch("server_runtime.claim_server_port", side_effect=ServerAlreadyRunningError("occupied")), \
                patch("serve_sign_api.create_app") as create_app, patch("builtins.print"):
            self.assertEqual(main([]), 1)
        create_app.assert_not_called()

    def test_status_and_search_identify_the_active_knowledge_dataset(self):
        from serve_sign_api import create_app
        store = MagicMock()
        store.knowledge_chunks.count_documents.return_value = 11
        store.search_knowledge_exact.return_value = [{
            "_id": "book4-v2-hospital", "entry_title": "医院 yīyuàn", "book": 4, "page": 111,
            "text": "医院 yīyuàn\n（一）一手拇、食指搭成十字形。\n（二）双手搭成∧形。",
            "image_paths": ["/book-media/book4/hospital.jpeg"],
        }]
        with patch.dict(os.environ, {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}), \
                patch("serve_sign_api._book_external_allowed", return_value=False):
            client = create_app(mongo_store=store).test_client()
            for path, method, kwargs in [("/api/knowledge/status", "get", {}),
                                          ("/api/knowledge/search", "post", {"json": {"question": "医院"}})]:
                response = getattr(client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["knowledgeApiVersion"], 2)
                self.assertEqual(response.json["collection"], "SignKnowledgeChunks_v2")
            result = client.post("/api/knowledge/search", json={"question": "医院"}).json
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page"], 111)
        self.assertEqual(len(result["sources"][0]["images"]), 1)
        store.search_knowledge_keywords.assert_not_called()


if __name__ == "__main__":
    unittest.main()
