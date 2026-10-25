"""Import only eleven source-checked PDF entries into the isolated v2 collection.

Default: local preview only. --upload writes the eleven entries and approves exactly
their verified content hashes. --embed additionally generates real embeddings.
No collection is deleted, and application configuration is never persisted.
"""
from __future__ import annotations

import argparse
import json
import os

from book_knowledge import ROOT, STAGING_COLLECTION, parse_pdf


# These exact contents were compared with their PDF source pages.
# A parser/source change must be reviewed again, not silently auto-approved.
REVIEWED_SAMPLE = (
    ("book1-v2-bdda8e69130c8f6705209546", "b94c00fee9ae39f09676f4dee5e3c1c909c389bb9dc63e7dac055916feebef4c"),
    ("book1-v2-6f8c952ded15398a1e1b1d24", "ddf32e2363cae0081e1a74335b6dca56b922cb0575435e484f520fef8c52c2c4"),
    ("book1-v2-adea0cf4f3d8585d554a2390", "cbd9c007e456b870355f0ec888dbf27bb09d72316d9eeb3bafcd3eb7c18c0b57"),
    ("book1-v2-d92b95515c57822a970f7fd7", "c8f74b71ec2e123c24a6920b35ba68cbd496fced726b1a95d3b2c2e388bc1f25"),
    ("book1-v2-2a2c5671652cd47827af517a", "80332c966ae9d9b987c20408374c1acc216c6de7ec6af9eeecbef2eb4683cf52"),
    ("book1-v2-b8bbdb5602aeb2d09822d03a", "87a49317511f7b42441140568700a5718ab857462722a3980c8ef0992a52f31a"),
    ("book1-v2-8725fe0e95eab7f1d17cb76c", "530d09ac7c70b17d6f8b0d463285ae473b77fca432acff1aff59435370e0b40e"),
    ("book1-v2-1f5c63504cd8498c92353f43", "2f073dca139baa4979290248561ef84b55d67edc1dbeb90b251a003812373a0d"),
    ("book4-v2-70bb3e08367c7dd685a20983", "50e061bcaf235e1306929c140b18e5509a423e14dd72bf8ccba5063e2a0f74ea"),
    ("book4-v2-9e64cd05044a515bb71ca59e", "5646062faac59b9776992463311ac00346c5791bd66c7c064e421325851cdb07"),
    ("book4-v2-e34dc59e5a55d1fed6ca983e", "92ad8de7da691b4f7ece9bc5ec0e7d3968fc239b6938ef06d2bf1d05e01b190a"),
)
SOURCE_FILES = (
    "9787508096483  国家通用手语词典 第1册  (2019, 华夏出版社)  (1).pdf",
    "9787508096483  国家通用手语词典 第4册  (2019, 华夏出版社)  .pdf",
)
SEARCH_TERMS = ("豆腐", "豆腐①", "豆腐②", "豆浆", "毒辣", "毒品", "独裁", "父亲", "父亲节", "医院", "英国", "账户")


def select_reviewed_sample(chunks):
    by_id = {chunk["_id"]: chunk for chunk in chunks}
    selected = []
    for chunk_id, content_hash in REVIEWED_SAMPLE:
        chunk = by_id.get(chunk_id)
        if chunk is None or chunk.get("content_hash") != content_hash:
            raise ValueError("样本与已核对版本不同，已停止；请重新核对源文件和解析器。")
        if not chunk.get("image_paths") or not chunk.get("image_evidence") or any(
            not evidence.get("boundary_confirmed") for evidence in chunk["image_evidence"]
        ):
            raise ValueError("样本图片关联证据不完整，已停止。")
        # Keep local parser output pending. Approval is scoped to uploaded hashes.
        selected.append(dict(chunk, review_status="pending"))
    return selected


def upload_sample(chunks, store, *, embed=False, expected_dimensions=1024):
    from import_sign_books import _retry_database_write, _retry_embeddings
    from mongo_store import knowledge_collection_name

    if knowledge_collection_name() != STAGING_COLLECTION:
        raise RuntimeError("样本导入仅允许写入 SignKnowledgeChunks_v2。")
    # Recheck the allowlist at the write boundary; never upload unrelated chunks.
    chunks = select_reviewed_sample(chunks)
    for number, chunk in enumerate(chunks, 1):
        _retry_database_write(lambda: store.upsert_staging_knowledge_chunks([chunk]))
        print(f"[{number}/{len(chunks)}] 词条写入成功：{chunk['entry_title']}", flush=True)
    if embed:
        missing = store.knowledge_ids_without_embeddings([c["_id"] for c in chunks])
        for number, chunk in enumerate(chunks, 1):
            if chunk["_id"] not in missing:
                print(f"[{number}/{len(chunks)}] 已有向量，跳过。", flush=True)
                continue
            vectors = _retry_embeddings([chunk["text"]], max_retries=3, retry_delay=3)
            if len(vectors) != 1 or len(vectors[0]) != expected_dimensions:
                raise ValueError("真实 Embedding 数量或维度不符，未写入本条向量。")
            _retry_database_write(lambda: store.save_knowledge_embeddings([
                {**chunk, "embedding": vectors[0]},
            ]))
            print(f"[{number}/{len(chunks)}] 真实向量写入成功：{expected_dimensions} 维。", flush=True)
    for chunk in chunks:
        if not store.review_knowledge_chunk(
            chunk["_id"], approved=True, expected_content_hash=chunk["content_hash"],
        ):
            raise RuntimeError("样本不存在或内容发生变化；未将该版本标为通过，请重跑。")
    print(f"{len(chunks)}条已核对样本已设为可检索。其他词条审核状态未改变。", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="11条已核对 PDF 词条的小批试导入；默认仅本地预览。")
    parser.add_argument("--upload", action="store_true", help="只上传这11条到 v2，并批准其已核对版本供搜索")
    parser.add_argument("--embed", action="store_true", help="另外串行生成真实向量；需要现有 Embedding 配置")
    parser.add_argument("--expected-dimensions", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.embed and not args.upload:
        parser.error("--embed 必须和 --upload 一起使用")
    if args.expected_dimensions < 1:
        parser.error("--expected-dimensions 必须大于0")
    paths = [ROOT / name for name in SOURCE_FILES]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("项目根目录缺少已核对的第1册或第4册 PDF，未连接 Atlas。")
    print("正在本地解析第1、4册 PDF 并核对11条样本，不解析 EPUB……", flush=True)
    chunks = []
    for path in paths:
        parsed, _ = parse_pdf(path)
        chunks.extend(parsed)
    selected = select_reviewed_sample(chunks)
    for chunk in selected:
        print(f"样本：{chunk['entry_title']}；PDF {chunk['page']}—{chunk['page_end']} 页；{len(chunk['image_paths'])} 张图。")
    preview = ROOT / "book_audit" / "sample_import_preview.json"
    preview.parent.mkdir(exist_ok=True)
    preview.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"本地样本原文和图片路径：{preview}", flush=True)
    if not args.upload:
        print(f"仅预览，未连接 Atlas 或模型。加 --upload 上传这{len(selected)}条；加 --embed 同时生成向量。")
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    # Override inherited legacy settings only within this importer process.
    os.environ["BOOK_KNOWLEDGE_COLLECTION"] = STAGING_COLLECTION
    os.environ["BOOK_EMBEDDING_INDEX_NAME"] = "sign_knowledge_vector_v2"
    if args.embed:
        os.environ["BOOK_CONTENT_EXTERNAL_ALLOWED"] = "true"
        from serve_sign_api import _book_api_key
        if not _book_api_key():
            raise RuntimeError("缺少真实 Embedding API Key，未连接 Atlas 或写入任何词条。")
    from import_sign_books import _import_store
    print("正在连接 Atlas，并检查新集合的已有索引……", flush=True)
    store = _import_store()
    try:
        print(f"目标集合：{STAGING_COLLECTION}；本次只上传{len(selected)}条，不清空任何集合。", flush=True)
        upload_sample(selected, store, embed=args.embed, expected_dimensions=args.expected_dimensions)
        if args.embed:
            from mongo_store import knowledge_index_name
            name = knowledge_index_name()
            try:
                if not store.has_knowledge_vector_index(name):
                    store.create_knowledge_vector_index(name, args.expected_dimensions)
                    print(f"已提交索引创建请求：{name}；就绪前请用词条精确查询。")
                else:
                    print(f"已有索引：{name}；请确认维度和就绪状态。")
            except Exception as exc:
                print(f"向量已保存，索引未确认（{type(exc).__name__}）；请按 RAG 文档在 Atlas 检查。")
    finally:
        store._client.close()
    print("完成。查看新资料请运行：.\\.venv\\Scripts\\python.exe start_sample_system.py")
    print("如果新版系统已经在运行，直接刷新页面搜索即可，不必重复启动。")
    print("测试词：" + "、".join(SEARCH_TERMS) + "。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Driver/service exceptions may contain credentials: print only the type.
        from import_sign_books import print_import_failure
        print_import_failure(exc, prefix="小批导入未完成")
        raise SystemExit(1)
