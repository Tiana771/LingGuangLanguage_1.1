import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mongo_store import MongoStore, create_mongodb_client


class MongoConnectionTests(unittest.TestCase):
    def test_shared_client_uses_certifi_and_bounded_timeouts_without_disabling_tls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("certifi.where", return_value="trusted-ca.pem"), patch("pymongo.MongoClient") as constructor:
            create_mongodb_client("mongodb://example.invalid/test", root=Path(directory))
        options = constructor.call_args.kwargs
        self.assertEqual(options["tlsCAFile"], "trusted-ca.pem")
        self.assertEqual(options["serverSelectionTimeoutMS"], 15000)
        self.assertEqual(options["socketTimeoutMS"], 30000)
        self.assertNotIn("tlsAllowInvalidCertificates", options)

    def test_custom_ca_and_explicit_development_tls_setting_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("pymongo.MongoClient") as constructor:
            root = Path(directory)
            (root / ".env").write_text("MONGODB_TLS_CA_FILE=custom-ca.pem\nMONGODB_TLS_ALLOW_INVALID_CERTIFICATES=true\n", encoding="utf-8")
            create_mongodb_client("mongodb://example.invalid/test", root=root)
        self.assertEqual(constructor.call_args.kwargs["tlsCAFile"], "custom-ca.pem")
        self.assertTrue(constructor.call_args.kwargs["tlsAllowInvalidCertificates"])

    def test_web_data_store_uses_shared_factory(self):
        uri = "mongodb://example.invalid/test"
        client = MagicMock()
        with patch("mongo_store.create_mongodb_client", return_value=client) as factory, \
                patch.dict(os.environ, {"MONGODB_DATABASE": "test"}):
            store = MongoStore(uri=uri)
            store._database()
        factory.assert_called_once_with(uri)

    def test_importer_uses_same_factory_and_closes_client_when_preparation_fails(self):
        import import_sign_books
        client = MagicMock()
        store = MagicMock()
        store.initialize_staging_knowledge.side_effect = RuntimeError("preparation failed")
        with patch.object(import_sign_books, "load_mongodb_uri", return_value="mongodb://example.invalid/test"), \
                patch.object(import_sign_books, "create_mongodb_client", return_value=client) as factory, \
                patch.object(import_sign_books, "MongoStore", return_value=store), self.assertRaises(RuntimeError):
            import_sign_books._import_store()
        factory.assert_called_once_with("mongodb://example.invalid/test")
        client.close.assert_called_once()

    def test_failed_status_never_reports_an_empty_library_and_can_recover(self):
        from pymongo.errors import ServerSelectionTimeoutError
        from serve_sign_api import create_app
        store = MagicMock()
        store.initialize.side_effect = [ServerSelectionTimeoutError("secret connection details"), None]
        store.knowledge_chunks.count_documents.return_value = 11
        with patch("serve_sign_api._book_external_allowed", return_value=False):
            client = create_app(mongo_store=store).test_client()
            failed = client.get("/api/knowledge/status")
            recovered = client.get("/api/knowledge/status")
        self.assertEqual(failed.status_code, 503)
        self.assertFalse(failed.json["databaseAvailable"])
        self.assertIsNone(failed.json["approvedCount"])
        self.assertNotIn("secret", failed.get_data(as_text=True))
        self.assertEqual(recovered.status_code, 200)
        self.assertTrue(recovered.json["databaseAvailable"])
        self.assertEqual(recovered.json["approvedCount"], 11)


if __name__ == "__main__":
    unittest.main()
