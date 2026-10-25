import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch
from pathlib import Path

from book_knowledge import _link_review_candidates, _safe_member, keyword_search, merge_knowledge_results, parse_epub, parse_pdf
from mongo_store import MongoStore


class BookKnowledgeTests(unittest.TestCase):
    def test_pdf_chinese_and_pinyin_blocks_on_same_row_keep_chinese_title(self):
        from book_knowledge import _pdf_heading_groups
        lines = [
            {"text": "阿尔及利亚", "size": 25.01, "bbox": (66, 716.46, 200.45, 744.41)},
            {"text": "'erjiliya", "size": 25.01, "bbox": (236.46, 716.46, 321.66, 744.41)},
            {"text": "动作说明", "size": 15, "bbox": (66, 900, 300, 920)},
            # Roman fragment sorts slightly earlier, but belongs after Chinese.
            {"text": "aman", "size": 25.01, "bbox": (160, 1000, 212, 1028)},
            {"text": "阿曼", "size": 25.01, "bbox": (66, 1000.5, 123, 1028.5)},
        ]
        groups = _pdf_heading_groups(lines)
        self.assertEqual([g["title"] for g in groups], ["阿尔及利亚 'erjiliya", "阿曼 aman"])
        self.assertEqual(groups[0]["indices"], [0, 1])

    def test_pdf_preserves_short_description_and_joins_wrapped_heading(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "词典第1册.pdf"
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "父亲（爸爸）", fontname="china-s", fontsize=25)
                page.insert_text((66, 130), "fuqin (baba)", fontsize=25)
                page.insert_text((66, 260), "右手伸拇指。", fontname="china-s", fontsize=15)
                doc.save(path)
            chunks, warnings = parse_pdf(path, Path(directory) / "media")
            self.assertEqual(len(chunks), 1)
            self.assertIn("fuqin", chunks[0]["entry_title"])
            self.assertTrue(chunks[0]["text"].startswith(chunks[0]["entry_title"]))
            self.assertIn("右手伸拇指。", chunks[0]["text"])
            self.assertEqual(warnings, [])

    def test_pdf_continuation_owns_next_page_image_and_text_until_next_heading(self):
        import fitz
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "词典第1册.pdf"
            picture = Path(directory) / "illustration.png"
            Image.new("RGB", (120, 100), "blue").save(picture)
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 650), "豆浆", fontname="china-s", fontsize=25)
                page.insert_text((150, 650), "doujiang", fontsize=25)
                page.insert_text((66, 730), "先伸拇指。", fontname="china-s", fontsize=15)
                page = doc.new_page(width=600, height=800)
                page.insert_image(fitz.Rect(200, 50, 320, 150), filename=picture)
                page.insert_text((66, 180), "再模仿喝水。", fontname="china-s", fontsize=15)
                page.insert_text((66, 300), "豆芽", fontname="china-s", fontsize=25)
                page.insert_text((150, 300), "douya", fontsize=25)
                page.insert_text((66, 430), "手向上移动。", fontname="china-s", fontsize=15)
                doc.save(path)
            chunks, warnings = parse_pdf(path, Path(directory) / "media")
            self.assertEqual(len(chunks), 2)
            self.assertIn("先伸拇指", chunks[0]["text"])
            self.assertIn("再模仿喝水", chunks[0]["text"])
            self.assertNotIn("豆芽", chunks[0]["text"])
            self.assertEqual(chunks[0]["page_end"], 2)
            self.assertEqual(len(chunks[0]["image_paths"]), 1)
            self.assertEqual(chunks[1]["image_paths"], [])
            self.assertEqual(warnings, [])

    def test_pdf_index_and_appendix_are_not_merged_into_last_entry(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "词典第4册.pdf"
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "WIFI", fontsize=25)
                page.insert_text((66, 300), "双手移动。", fontname="china-s", fontsize=15)
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "附录 手语语法", fontname="china-s", fontsize=27)
                page.insert_text((66, 300), "附录中无关内容。", fontname="china-s", fontsize=15)
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "APPENDIX EXAMPLE", fontsize=25)
                page.insert_text((66, 300), "这是附录例句而非词条。", fontname="china-s", fontsize=15)
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "笔画索引", fontname="china-s", fontsize=27)
                page.insert_text((66, 160), "一画", fontname="china-s", fontsize=25)
                page.insert_text((66, 300), "妈妈 621", fontname="china-s", fontsize=15)
                doc.save(path)
            chunks, warnings = parse_pdf(path, Path(directory) / "media")
            self.assertEqual([c["entry_title"] for c in chunks], ["WIFI"])
            self.assertNotIn("附录", chunks[0]["text"])
            self.assertNotIn("621", chunks[0]["text"])
            self.assertEqual(warnings, [])

    def test_pdf_crossing_image_remains_pending_with_coordinate_warning(self):
        import fitz
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "词典第1册.pdf"
            picture = Path(directory) / "ambiguous.png"
            Image.new("RGB", (120, 200), "blue").save(picture)
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                for label, roman, y in [("豆腐", "doufu", 100), ("豆浆", "doujiang", 400)]:
                    page.insert_text((66, y), label, fontname="china-s", fontsize=25)
                    page.insert_text((150, y), roman, fontsize=25)
                    page.insert_text((66, y + 50), "手向外移动。", fontname="china-s", fontsize=15)
                page.insert_image(fitz.Rect(300, 250, 420, 450), filename=picture)
                doc.save(path)
            chunks, warnings = parse_pdf(path, Path(directory) / "media")
            self.assertTrue(warnings)
            self.assertEqual(chunks[0]["review_status"], "pending")
            self.assertFalse(chunks[0]["image_evidence"][0]["boundary_confirmed"])

    def test_pdf_reviewed_picture_above_heading_is_not_attached_to_previous_entry(self):
        import fitz
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "词典第4册.pdf"
            picture = Path(directory) / "image.png"
            Image.new("RGB", (100, 100), "green").save(picture)
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                page.insert_text((66, 100), "上条", fontname="china-s", fontsize=25)
                page.insert_text((150, 100), "shangtiao", fontsize=25)
                page.insert_text((66, 150), "上一条动作。", fontname="china-s", fontsize=15)
                page.insert_image(fitz.Rect(200, 220, 300, 320), filename=picture)
                page.insert_text((66, 370), "本条", fontname="china-s", fontsize=25)
                page.insert_text((150, 370), "bentiao", fontsize=25)
                page.insert_text((66, 450), "本条动作。", fontname="china-s", fontsize=15)
                doc.save(path)
            with fitz.open(path) as doc:
                from book_knowledge import _pdf_lines, _pdf_heading_groups
                title = _pdf_heading_groups(_pdf_lines(doc[0]))[1]["title"]
            overrides = {1: [{"entry_title": title, "bbox": [200, 220, 300, 320], "reason": "目视确认"}]}
            with patch("book_knowledge._pdf_reviewed_layout", return_value=overrides):
                chunks, warnings = parse_pdf(path, Path(directory) / "media")
            self.assertEqual(chunks[0]["image_paths"], [])
            self.assertEqual(len(chunks[1]["image_paths"]), 1)
            self.assertEqual(chunks[1]["image_evidence"][0]["association_method"], "reviewed_layout_exception")
            self.assertEqual(chunks[1]["review_status"], "pending")
            self.assertEqual(warnings, [])

    def test_pdf_reviewed_layout_only_applies_to_exact_source_checksum(self):
        import hashlib
        import json
        from book_knowledge import _pdf_reviewed_layout
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "词典第4册.pdf"
            source.write_bytes(b"original source")
            config = {"sources": [{"source_file": source.name,
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "images": [{"page": 147, "entry_title": "英国"}]}]}
            (root / "book_pdf_layout_overrides.json").write_text(json.dumps(config), encoding="utf-8")
            with patch("book_knowledge.ROOT", root):
                self.assertIn(147, _pdf_reviewed_layout(source))
                source.write_bytes(b"different edition")
                self.assertEqual(_pdf_reviewed_layout(source), {})

    def test_embeddings_retry_temporary_disconnect_but_not_bad_credentials(self):
        from import_sign_books import _retry_embeddings
        from urllib.error import HTTPError

        with patch("import_sign_books._book_embeddings", side_effect=[ConnectionResetError(), [[0.2]]]) as embed, \
                patch("import_sign_books.time.sleep"), patch("builtins.print"):
            self.assertEqual(_retry_embeddings(["测试内容"]), [[0.2]])
            self.assertEqual(embed.call_count, 2)
        with patch("import_sign_books._book_embeddings", side_effect=HTTPError("https://example.invalid", 401, "", {}, None)) as embed:
            with self.assertRaises(HTTPError):
                _retry_embeddings(["测试内容"])
            self.assertEqual(embed.call_count, 1)

    def test_embedding_attempt_logs_elapsed_http_status_and_never_logs_input(self):
        from import_sign_books import _retry_embeddings

        log = []
        with patch("import_sign_books._book_embeddings", return_value=([[0.2]], 200)), \
                patch("builtins.print", side_effect=lambda message, **kwargs: log.append(message)):
            self.assertEqual(_retry_embeddings(["DO_NOT_LOG_BOOK_TEXT"]), [[0.2]])
        joined = "\n".join(log)
        self.assertIn("HTTP 状态=200", joined)
        self.assertIn("耗时=", joined)
        self.assertNotIn("DO_NOT_LOG_BOOK_TEXT", joined)

    def test_embedding_diagnostics_are_persisted_without_request_text(self):
        import json
        from import_sign_books import _retry_embeddings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            with patch("import_sign_books.DIAGNOSTIC_LOG_PATH", path), \
                    patch("import_sign_books._book_embeddings", return_value=([[0.2]], 200)), \
                    patch("builtins.print"):
                _retry_embeddings(["DO_NOT_PERSIST_BOOK_TEXT"])
            entry = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(entry["event"], "embedding_succeeded")
        self.assertEqual(entry["http_status"], 200)
        self.assertIn("elapsed_seconds", entry)
        self.assertNotIn("DO_NOT_PERSIST_BOOK_TEXT", str(entry))

    def test_embedding_disconnect_logs_no_http_response_and_exception_type(self):
        from http.client import RemoteDisconnected
        from import_sign_books import _retry_embeddings
        from urllib.error import URLError

        log = []
        with patch("import_sign_books._book_embeddings", side_effect=URLError(RemoteDisconnected())), \
                patch("import_sign_books.time.sleep"), \
                patch("builtins.print", side_effect=lambda message, **kwargs: log.append(message)):
            with self.assertRaises(URLError):
                _retry_embeddings(["DO_NOT_LOG_BOOK_TEXT"])
        joined = "\n".join(log)
        self.assertIn("HTTP 状态=无响应", joined)
        self.assertIn("RemoteDisconnected", joined)
        self.assertIn("耗时=", joined)
        self.assertNotIn("DO_NOT_LOG_BOOK_TEXT", joined)

    def test_embedding_retries_transient_failures_three_times_after_initial_attempt(self):
        from import_sign_books import _retry_embeddings
        from urllib.error import URLError

        with patch("import_sign_books._book_embeddings", side_effect=[
                URLError("disconnected"), URLError("disconnected"),
                URLError("disconnected"), ([[0.2]], 200)]
        ) as embed, patch("import_sign_books.time.sleep") as sleep, patch("builtins.print"):
            self.assertEqual(_retry_embeddings(["sample"], max_retries=3, retry_delay=3), [[0.2]])
        self.assertEqual(embed.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [3, 3, 3])

    def test_invalid_embedding_response_logs_received_http_status(self):
        import json
        from import_sign_books import _retry_embeddings
        from serve_sign_api import BookEmbeddingResponseError

        log = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            with patch("import_sign_books.DIAGNOSTIC_LOG_PATH", path), \
                    patch("import_sign_books._book_embeddings", side_effect=BookEmbeddingResponseError("invalid response", 200)), \
                    patch("builtins.print", side_effect=lambda message, **kwargs: log.append(message)):
                with self.assertRaises(BookEmbeddingResponseError):
                    _retry_embeddings(["DO_NOT_LOG_BOOK_TEXT"])
            entry = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(entry["http_status"], 200)
        self.assertEqual(entry["failure_reason"], "invalid_embedding_response")
        self.assertIn("HTTP 状态=200", "\n".join(log))
        self.assertNotIn("DO_NOT_LOG_BOOK_TEXT", str(entry))

    def test_single_worker_stops_after_first_failed_embedding_batch(self):
        import import_sign_books
        from urllib.error import URLError

        chunks = [
            {"_id": "one", "text": "第一段"},
            {"_id": "two", "text": "第二段"},
        ]
        store = MagicMock()
        store.existing_knowledge_ids.return_value = {"one", "two"}
        store.knowledge_ids_without_embeddings.return_value = {"one", "two"}
        parsed = {"chunks": chunks, "report": {"review_count": 2, "parse_warning_count": 0}}
        with patch.object(import_sign_books, "import_books", return_value=parsed), \
                patch.object(import_sign_books, "_import_store", return_value=store), \
                patch.object(import_sign_books, "_book_api_key", return_value="unit-test-only"), \
                patch.object(import_sign_books, "_retry_embeddings", side_effect=URLError("disconnected")) as embed, \
                patch("sys.argv", ["import_sign_books.py", "--upload", "--embed", "--embedding-workers", "1", "--embedding-batch-size", "1"]), \
                patch.dict("os.environ", {"BOOK_CONTENT_ATLAS_ALLOWED": "true", "BOOK_CONTENT_EXTERNAL_ALLOWED": "true"}), \
                patch("builtins.print"):
            with self.assertRaises(URLError):
                import_sign_books.main()
        embed.assert_called_once_with(["第一段"], max_retries=3, retry_delay=3)

    def test_embedding_resume_skips_saved_vectors_and_preserves_id_pairing(self):
        import import_sign_books

        known = {"_id": "known", "text": "已保存的内容"}
        missing = {"_id": "missing", "text": "需要向量的内容"}
        store = MagicMock()
        store.existing_knowledge_ids.return_value = {"known", "missing"}
        store.knowledge_ids_without_embeddings.return_value = {"missing"}
        parsed = {"chunks": [known, missing], "report": {"review_count": 2, "parse_warning_count": 0}}
        with patch.object(import_sign_books, "import_books", return_value=parsed), \
                patch.object(import_sign_books, "_import_store", return_value=store), \
                patch.object(import_sign_books, "_book_api_key", return_value="unit-test-only"), \
                patch.object(import_sign_books, "_book_embeddings", return_value=[[0.1, 0.2]]) as embed, \
                patch("sys.argv", ["import_sign_books.py", "--upload", "--embed", "--upload-batch-size", "1", "--expected-dimensions", "2"]), \
                patch.dict("os.environ", {"BOOK_CONTENT_ATLAS_ALLOWED": "true", "BOOK_CONTENT_EXTERNAL_ALLOWED": "true"}), \
                patch("builtins.print"):
            self.assertEqual(import_sign_books.main(), 0)
        embed.assert_called_once_with([missing["text"]], include_status=True)
        store.upsert_knowledge_chunks.assert_not_called()
        store.save_knowledge_embeddings.assert_called_once_with([
            {"_id": "missing", "text": "需要向量的内容", "embedding": [0.1, 0.2]},
        ])

    def test_embedding_batch_size_four_sends_one_four_item_request(self):
        import import_sign_books

        chunks = [{"_id": str(index), "text": f"chunk {index}"} for index in range(5)]
        store = MagicMock()
        store.existing_knowledge_ids.return_value = {chunk["_id"] for chunk in chunks}
        store.knowledge_ids_without_embeddings.return_value = {chunk["_id"] for chunk in chunks}
        parsed = {"chunks": chunks, "report": {"review_count": 5, "parse_warning_count": 0}}
        vectors = [[0.1, 0.2] for _ in range(4)]
        with patch.object(import_sign_books, "import_books", return_value=parsed), \
                patch.object(import_sign_books, "_import_store", return_value=store), \
                patch.object(import_sign_books, "_book_api_key", return_value="unit-test-only"), \
                patch.object(import_sign_books, "_book_embeddings", return_value=vectors) as embed, \
                patch("sys.argv", [
                    "import_sign_books.py", "--upload", "--embed", "--embedding-limit", "4",
                    "--embedding-batch-size", "4", "--expected-dimensions", "2",
                ]), \
                patch.dict("os.environ", {
                    "BOOK_CONTENT_ATLAS_ALLOWED": "true", "BOOK_CONTENT_EXTERNAL_ALLOWED": "true",
                }), patch("builtins.print"):
            self.assertEqual(import_sign_books.main(), 0)
        embed.assert_called_once_with(["chunk 0", "chunk 1", "chunk 2", "chunk 3"], include_status=True)
        self.assertEqual(store.save_knowledge_embeddings.call_count, 1)
        self.assertEqual(len(store.save_knowledge_embeddings.call_args.args[0]), 4)

    def test_upload_targets_only_staging_collection_and_is_idempotent(self):
        import import_sign_books

        known, new = {"_id": "known"}, {"_id": "new"}
        store = MagicMock()
        store.upsert_staging_knowledge_chunks.return_value = 2
        parsed = {"chunks": [known, new], "report": {"review_count": 2, "parse_warning_count": 0}}
        with patch.object(import_sign_books, "import_books", return_value=parsed), \
                patch.object(import_sign_books, "_import_store", return_value=store), \
                patch("sys.argv", ["import_sign_books.py", "--upload"]), \
                patch.dict("os.environ", {"BOOK_CONTENT_ATLAS_ALLOWED": "true"}), \
                patch("builtins.print"):
            self.assertEqual(import_sign_books.main(), 0)
        store.upsert_staging_knowledge_chunks.assert_called_once_with([known, new])
        store.upsert_knowledge_chunks.assert_not_called()

    def test_transient_database_failure_can_resume_without_repeating_generation(self):
        from import_sign_books import _retry_database_write
        from pymongo.errors import AutoReconnect

        save = MagicMock(side_effect=[AutoReconnect("connection interrupted"), 16])
        with patch("builtins.print"), patch("import_sign_books.time.sleep") as sleep:
            self.assertEqual(_retry_database_write(save), 16)
        self.assertEqual(save.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_database_retries_are_bounded_and_do_not_hide_failure(self):
        from import_sign_books import _retry_database_write
        from pymongo.errors import AutoReconnect

        save = MagicMock(side_effect=AutoReconnect("connection interrupted"))
        with patch("builtins.print"), patch("import_sign_books.time.sleep"), self.assertRaises(AutoReconnect):
            _retry_database_write(save)
        self.assertEqual(save.call_count, 4)

    def test_knowledge_index_uses_atlas_vector_search_type(self):
        store = MongoStore(client=MagicMock())
        store._db = MagicMock()
        with patch.dict("os.environ", {"BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2"}):
            store.create_knowledge_vector_index("book_vectors", 1024)
        definition = store._db["SignKnowledgeChunks_v2"].create_search_index.call_args.args[0]
        self.assertEqual(definition["type"], "vectorSearch")
        self.assertIn({"type": "filter", "path": "review_status"}, definition["definition"]["fields"])

    def test_knowledge_api_shows_at_most_three_scored_similar_entries(self):
        from serve_sign_api import create_app

        repository = MagicMock()
        repository.has_knowledge_vector_index.return_value = True
        repository.search_knowledge_exact.return_value = []
        repository.search_knowledge_keywords.return_value = []
        repository.search_knowledge_vectors.return_value = [
            {"_id": str(index), "entry_title": f"相似词{index}", "text": "词典原文", "book": 2,
             "page": 10 + index, "image_paths": [], "score": score}
            for index, score in enumerate((0.91, 0.81, 0.72, 0.61), 1)
        ]
        with patch("serve_sign_api._book_external_allowed", return_value=True), \
                patch("serve_sign_api._book_api_key", return_value="test-key"), \
                patch("serve_sign_api._book_embedding", return_value=[0.1, 0.2]), \
                patch("serve_sign_api._qwen_book_answer", return_value="根据原文回答[1]"), \
                patch.dict("os.environ", {
                    "BOOK_KNOWLEDGE_COLLECTION": "SignKnowledgeChunks_v2",
                    "BOOK_EMBEDDING_INDEX_NAME": "sign_knowledge_vector_v2",
                    "BOOK_SIMILARITY_MIN_SCORE": "0.62",
                }):
            app = create_app(mongo_store=repository)
            app.testing = True
            response = app.test_client().post(
                "/api/knowledge/search", json={"question": "水杯"},
            )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["retrieval"], "similar")
        self.assertEqual(result["message"], "暂未找到该词汇，为您推荐以下相似词汇。")
        self.assertEqual([source["title"] for source in result["sources"]], ["相似词1", "相似词2", "相似词3"])
        repository.search_knowledge_vectors.assert_called_once_with(
            [0.1, 0.2], "sign_knowledge_vector_v2", limit=12,
        )

    def test_rejects_archive_path_traversal(self):
        self.assertIsNone(_safe_member("../../outside.png"))
        self.assertIsNone(_safe_member("%2e%2e/outside.png"))
        self.assertEqual(_safe_member("OEBPS/chapter.xhtml"), "OEBPS/chapter.xhtml")

    def test_epub_spine_extracts_text_and_local_image(self):
        with tempfile.TemporaryDirectory() as directory:
            epub = Path(directory) / "国家通用手语词典第2册.epub"
            media = Path(directory) / "media"
            with zipfile.ZipFile(epub, "w") as archive:
                archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
                archive.writestr("OEBPS/book.opf", '''<package><manifest>
                  <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                  <item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/>
                  <item id="pic1" href="hand1.png" media-type="image/png"/>
                  <item id="pic2" href="hand2.png" media-type="image/png"/>
                  </manifest><spine><itemref idref="c1"/></spine></package>''')
                archive.writestr("OEBPS/nav.xhtml", '''<html><body><nav><ol>
                  <li><a href="chapter.xhtml#thanks">谢谢</a></li>
                  <li><a href="chapter.xhtml#hello">你好</a></li>
                  </ol></nav></body></html>''')
                archive.writestr("OEBPS/chapter.xhtml", '''<html><body>
                  <h2 id="thanks" class="sect2">谢谢 xièxie</h2>
                  <p>双手掌心向上，在胸前向外移动。<img src="hand1.png"/></p>
                  <h2 id="hello" class="sect2">你好 nǐhǎo</h2>
                  <p>挥手问候。<img src="hand2.png"/></p>
                  </body></html>''')
                archive.writestr("OEBPS/hand1.png", b"\x89PNG\r\n\x1a\nimage-one")
                archive.writestr("OEBPS/hand2.png", b"\x89PNG\r\n\x1a\nimage-two")
            chunks, review = parse_epub(epub, media)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["book"], 2)
            self.assertIn("双手掌心向上", chunks[0]["text"])
            self.assertEqual(len(chunks[0]["image_paths"]), 1)
            self.assertEqual(chunks[0]["entry_title"], "谢谢 xièxie")
            self.assertEqual(chunks[0]["review_status"], "pending")
            self.assertNotEqual(chunks[0]["image_paths"], chunks[1]["image_paths"])
            self.assertIn("你好", chunks[1]["text"])
            self.assertTrue((media / chunks[0]["image_paths"][0].split("/book-media/")[-1]).exists())
            self.assertEqual(review, [])

    def test_exact_numbered_entries_return_together_or_individually(self):
        chunks = [
            {"_id": "tofu1", "entry_title": "豆腐①", "text": "豆腐①\n动作说明一。"},
            {"_id": "tofu2", "entry_title": "豆腐②", "text": "豆腐②\n动作说明二。"},
            {"_id": "soy", "entry_title": "豆浆", "text": "豆浆\n动作说明三。"},
        ]
        self.assertEqual([item["_id"] for item in keyword_search(chunks, "豆腐")], ["tofu1", "tofu2"])
        self.assertEqual([item["_id"] for item in keyword_search(chunks, "豆腐②")], ["tofu2"])

    def test_natural_language_sign_query_extracts_dictionary_term(self):
        from book_knowledge import _entry_query_term

        self.assertEqual(_entry_query_term("和葡萄有关的手语"), "葡萄")
        self.assertEqual(_entry_query_term("我想知道葡萄相关的手语"), "葡萄")

    def test_pdf_warning_links_only_candidate_chunks_from_its_source_page(self):
        chunks = [
            {"_id": "p637-a", "source_file": "book.pdf", "entry_title": "毒辣", "page": 637,
             "anchor": None, "image_paths": ["/book-media/a.jpg"], "text": "说明一"},
            {"_id": "p637-b", "source_file": "book.pdf", "entry_title": "毒品", "page": 637,
             "anchor": None, "image_paths": ["/book-media/b.jpg"], "text": "说明二"},
            {"_id": "p638-a", "source_file": "book.pdf", "entry_title": "独裁", "page": 638,
             "anchor": None, "image_paths": [], "text": "说明三"},
        ]
        report = _link_review_candidates(
            [{"file": "book.pdf", "reason": "PDF 图片未落入可识别词条区域：第 637 页"}], chunks,
        )
        self.assertEqual([item["id"] for item in report[0]["related_chunks"]], ["p637-a", "p637-b"])

    def test_keyword_search_returns_grounded_chunk_and_sorts_matches(self):
        chunks = [
            {"_id": "a", "entry_title": "谢谢", "text": "双手掌心向上，向外移动。", "book": 2, "page": None},
            {"_id": "b", "entry_title": "你好", "text": "挥手问候。", "book": 1, "page": 2},
        ]
        result = keyword_search(chunks, "谢谢怎么做")
        self.assertEqual([item["_id"] for item in result], ["a"])

    def test_hybrid_search_prioritizes_exact_dictionary_title_over_semantic_neighbor(self):
        vector = [{"_id": "near", "entry_title": "酬谢 chóuxiè", "text": "近义词", "score": 0.9}]
        keyword = [{"_id": "exact", "entry_title": "谢谢 xiè·xie", "text": "谢谢的动作", "score": 1}]
        results = merge_knowledge_results(vector, keyword, "谢谢怎么做", limit=5)
        self.assertEqual([item["_id"] for item in results], ["exact"])

    def test_exact_entry_search_does_not_return_chunks_that_only_mention_the_word(self):
        chunks = [
            {"_id": "mom", "entry_title": "妈妈 māma", "text": "双手动作说明。"},
            {"_id": "family", "entry_title": "家庭", "text": "妈妈、爸爸和孩子组成家庭。"},
            {"_id": "motherhood", "entry_title": "母亲节", "text": "送给妈妈的礼物。"},
        ]
        results = keyword_search(chunks, "妈妈", limit=5)
        self.assertEqual([item["_id"] for item in results], ["mom"])

    def test_exact_entry_query_with_intent_suffix_excludes_vector_neighbors(self):
        vector = [
            {"_id": "mom", "entry_title": "妈妈 māma", "text": "妈妈动作。", "score": 0.8},
            {"_id": "family", "entry_title": "家庭", "text": "家庭动作。", "score": 0.95},
        ]
        keyword = [{"_id": "mom", "entry_title": "妈妈 māma", "text": "妈妈动作。", "score": 1}]
        results = merge_knowledge_results(vector, keyword, "妈妈怎么做", limit=5)
        self.assertEqual([item["_id"] for item in results], ["mom"])

    def test_mongo_keyword_search_checks_exact_title_before_body_matches(self):
        store = MongoStore(client=MagicMock())
        exact_cursor = MagicMock()
        exact_cursor.limit.return_value = [{
            "_id": "mom", "entry_title": "妈妈 māma", "text": "妈妈动作说明。",
        }]
        collection = MagicMock()
        collection.find.return_value = exact_cursor
        store._db = {"SignKnowledgeChunks": collection}
        results = store.search_knowledge_keywords("妈妈", limit=5)
        self.assertEqual([item["_id"] for item in results], ["mom"])
        self.assertEqual(collection.find.call_count, 1)

    def test_keyword_search_rejects_mixed_page_chunks_even_when_body_mentions_query(self):
        chunk = {
            "_id": "mixed", "entry_title": "妇科 293", "book": 3,
            "text": "妇科 293\n妇联 293\n好 364\n妈妈\n双手在胸前交叉，表示母亲。\n戏① 924\n戏弄 124",
        }
        result = keyword_search([chunk], "妈妈")
        self.assertEqual(result, [])

    def test_keyword_search_rejects_index_only_hit(self):
        chunk = {
            "_id": "index", "entry_title": "妇科 293", "book": 3,
            "text": "妇科 293\n妇联 293\n妈妈 621\n戏① 924",
        }
        self.assertEqual(keyword_search([chunk], "妈妈"), [])

    def test_keyword_search_rejects_unstructured_multi_entry_excerpt(self):
        chunk = {
            "_id": "mixed", "entry_title": "家庭", "book": 2,
            "text": "其他词条说明。\n妈妈\n双手在胸前交叉，表示母亲。\n后续词条说明。",
        }
        result = keyword_search([chunk], "妈妈怎么做")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
