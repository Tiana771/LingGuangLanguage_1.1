import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import import_sign_books_sample as sample
from mongo_store import MongoStore


class SampleImportTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"_id": chunk_id, "content_hash": digest, "entry_title": f"sample {i}",
             "text": f"source {i}", "image_paths": [f"/book-media/{i}.png"],
             "image_evidence": [{"boundary_confirmed": True}], "page": 634,
             "page_end": 635, "review_status": "pending"}
            for i, (chunk_id, digest) in enumerate(sample.REVIEWED_SAMPLE)
        ]

    def test_sample_filters_unrelated_entries_and_rejects_changed_content(self):
        selected = sample.select_reviewed_sample([{"_id": "unrelated"}, *self.chunks])
        self.assertEqual(len(selected), 11)
        self.assertNotIn("unrelated", [c["_id"] for c in selected])
        self.chunks[0]["content_hash"] = "changed"
        with self.assertRaises(ValueError):
            sample.select_reviewed_sample(self.chunks)

    def test_only_eleven_verified_hashes_are_written_and_approved_without_embedding(self):
        store = MagicMock()
        store.review_knowledge_chunk.return_value = True
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}), \
                patch("import_sign_books._retry_database_write", side_effect=lambda op: op()), \
                patch("import_sign_books._retry_embeddings") as embed, patch("builtins.print"):
            sample.upload_sample(self.chunks, store)
        self.assertEqual(store.upsert_staging_knowledge_chunks.call_count, 11)
        self.assertEqual(store.review_knowledge_chunk.call_count, 11)
        self.assertEqual(store.upsert_staging_knowledge_chunks.call_args_list[0].args[0], [self.chunks[0]])
        store.review_knowledge_chunk.assert_any_call(
            self.chunks[0]["_id"], approved=True, expected_content_hash=self.chunks[0]["content_hash"],
        )
        embed.assert_not_called()
        store.upsert_knowledge_chunks.assert_not_called()
        store.knowledge_ids_without_embeddings.assert_not_called()

    def test_legacy_target_and_invalid_sample_fail_before_any_write(self):
        store = MagicMock()
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks"}):
            with self.assertRaises(RuntimeError):
                sample.upload_sample(self.chunks, store)
        self.chunks[0]["content_hash"] = "changed"
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}):
            with self.assertRaises(ValueError):
                sample.upload_sample(self.chunks, store)
        self.assertEqual(store.mock_calls, [])

    def test_wrong_vector_dimension_is_not_written_or_approved(self):
        store = MagicMock()
        store.knowledge_ids_without_embeddings.return_value = {c["_id"] for c in self.chunks}
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}), \
                patch("import_sign_books._retry_database_write", side_effect=lambda op: op()), \
                patch("import_sign_books._retry_embeddings", return_value=[[0.1, 0.2]]) as embed, \
                patch("builtins.print"), self.assertRaises(ValueError):
            sample.upload_sample(self.chunks, store, embed=True)
        embed.assert_called_once_with([self.chunks[0]["text"]], max_retries=3, retry_delay=3)
        store.save_knowledge_embeddings.assert_not_called()
        store.review_knowledge_chunk.assert_not_called()

    def test_embedding_resumes_with_only_missing_sample_and_serial_single_request(self):
        store = MagicMock()
        store.knowledge_ids_without_embeddings.return_value = {self.chunks[-1]["_id"]}
        store.review_knowledge_chunk.return_value = True
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}), \
                patch("import_sign_books._retry_database_write", side_effect=lambda op: op()), \
                patch("import_sign_books._retry_embeddings", return_value=[[0.1, 0.2]]) as embed, \
                patch("builtins.print"):
            sample.upload_sample(self.chunks, store, embed=True, expected_dimensions=2)
        embed.assert_called_once_with([self.chunks[-1]["text"]], max_retries=3, retry_delay=3)
        store.save_knowledge_embeddings.assert_called_once()
        self.assertEqual(store.save_knowledge_embeddings.call_args.args[0][0]["_id"], self.chunks[-1]["_id"])

    def test_local_preview_never_connects_to_cloud_or_changes_review_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in sample.SOURCE_FILES:
                (root / name).touch()
            with patch.object(sample, "ROOT", root), \
                    patch.object(sample, "parse_pdf", return_value=(self.chunks, [])), \
                    patch("import_sign_books._import_store") as connect, \
                    patch("import_sign_books._retry_embeddings") as embed, patch("builtins.print"):
                self.assertEqual(sample.main([]), 0)
            self.assertTrue((root / "book_audit" / "sample_import_preview.json").is_file())
            connect.assert_not_called()
            embed.assert_not_called()
        self.assertTrue(all(c["review_status"] == "pending" for c in self.chunks))

    def test_review_database_filter_requires_exact_verified_content_hash(self):
        store = MongoStore(client=MagicMock())
        store._db = MagicMock()
        collection = store._db.__getitem__.return_value
        collection.update_one.return_value.matched_count = 0
        self.assertFalse(store.review_knowledge_chunk("sample", True, expected_content_hash="verified"))
        self.assertEqual(collection.update_one.call_args.args[0], {"_id": "sample", "content_hash": "verified"})

    def test_staging_import_reuses_equivalent_app_index_without_renaming_or_deleting(self):
        store = MongoStore(client=MagicMock())
        collection = MagicMock()
        collection.list_indexes.return_value = [{"name": "knowledge_book_review", "key": {"book": 1, "review_status": 1}}]
        store._db = MagicMock()
        store._db.__getitem__.return_value = collection
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}):
            store.initialize_staging_knowledge()
        collection.create_index.assert_not_called()
        collection.drop_index.assert_not_called()
        collection.drop.assert_not_called()

    def test_new_staging_and_app_initialization_use_same_index_name(self):
        store = MongoStore(client=MagicMock())
        collection = MagicMock()
        collection.list_indexes.return_value = []
        store._db = MagicMock()
        store._db.__getitem__.return_value = collection
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}):
            store.initialize_staging_knowledge()
        collection.create_index.assert_called_once_with(
            [("book", 1), ("review_status", 1)], name="knowledge_book_review_v2",
        )

    def test_index_creation_race_reuses_equivalent_index_after_conflict(self):
        from pymongo.errors import OperationFailure
        store = MongoStore(client=MagicMock())
        collection = MagicMock()
        collection.list_indexes.side_effect = [[], [{"name": "other_launcher", "key": {"book": 1, "review_status": 1}}]]
        collection.create_index.side_effect = OperationFailure("sensitive details", code=85)
        store._db = MagicMock()
        store._db.__getitem__.return_value = collection
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}):
            self.assertEqual(store._ensure_knowledge_lookup_index(), "other_launcher")
        collection.drop_index.assert_not_called()

    def test_non_equivalent_index_conflict_is_not_silently_hidden(self):
        from pymongo.errors import OperationFailure
        store = MongoStore(client=MagicMock())
        collection = MagicMock()
        collection.list_indexes.return_value = [{"name": "unique_index", "key": {"book": 1, "review_status": 1}, "unique": True}]
        collection.create_index.side_effect = OperationFailure("sensitive details", code=85)
        store._db = MagicMock()
        store._db.__getitem__.return_value = collection
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}), self.assertRaises(OperationFailure):
            store._ensure_knowledge_lookup_index()
        collection.drop_index.assert_not_called()

    def test_database_error_explanation_has_code_and_never_prints_driver_secrets(self):
        from pymongo.errors import OperationFailure
        from import_sign_books import print_import_failure
        error = OperationFailure("mongodb+srv://private:secret@example.invalid sensitive-key", code=85)
        with patch("builtins.print") as output:
            print_import_failure(error, prefix="小批导入未完成")
        text = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("85", text)
        self.assertIn("索引", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("mongodb", text)
        self.assertNotIn("example.invalid", text)


if __name__ == "__main__":
    unittest.main()
