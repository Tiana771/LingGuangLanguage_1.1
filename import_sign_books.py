from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure

from book_knowledge import REPORT_PATH, STAGING_COLLECTION, import_books
from mongo_store import MongoStore, create_mongodb_client, knowledge_index_name, load_mongodb_uri
from serve_sign_api import _book_api_key, _book_embeddings


DIAGNOSTIC_LOG_PATH = Path(__file__).resolve().with_name("book_embedding_import.log")
_DIAGNOSTIC_LOG_LOCK = threading.Lock()


def print_import_failure(exc, *, prefix="导入未完成") -> None:
    """Explain allowlisted failures without printing driver messages or URLs."""
    from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

    code = getattr(exc, "code", None)
    suffix = f"，MongoDB错误编号={code}" if isinstance(code, int) else ""
    print(f"{prefix}（{type(exc).__name__}{suffix}）。", flush=True)
    if isinstance(exc, OperationFailure):
        reasons = {
            18: "数据库账号验证失败。请检查本机连接配置中的数据库用户名和密码。",
            13: "数据库账号权限不足。请在 Atlas 检查该账号是否有目标数据库的 readWrite 权限。",
            85: "数据库索引配置冲突。脚本已改为复用同结构索引；若仍失败，请保留错误编号继续排查。",
            86: "数据库索引结构冲突。没有删除或重建原索引；需要核对现有索引配置。",
            26: "目标集合尚不存在，数据库拒绝了本次操作。",
            8000: "Atlas 拒绝了数据库操作，可能涉及账号认证或权限；请保留错误编号继续排查。",
        }
        print(reasons.get(code, "MongoDB 拒绝了本次操作；请用上面的错误编号排查，切勿发送连接串或密码。"), flush=True)
    elif isinstance(exc, ServerSelectionTimeoutError):
        print("未能在规定时间内连接 Atlas。请检查网络、Atlas 网络访问设置和证书配置。", flush=True)
    elif isinstance(exc, ConnectionFailure):
        print("Atlas 连接中断；已保存的数据保留，可以重跑续传。", flush=True)
    print("已保存的数据保留；旧知识集合未修改。", flush=True)


def _record_import_event(event: str, **details) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    try:
        with _DIAGNOSTIC_LOG_LOCK:
            with DIAGNOSTIC_LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        print("诊断日志无法写入本地文件；详细状态仍显示在终端。", flush=True)


def _import_store() -> MongoStore:
    uri = load_mongodb_uri()
    if not uri:
        raise RuntimeError("MongoDB connection is not configured")
    client = create_mongodb_client(uri)
    store = MongoStore(uri=uri, client=client)
    try:
        store.initialize_staging_knowledge()
    except Exception:
        client.close()
        raise
    return store


def _retry_database_write(operation, *, max_retries: int = 3, retry_delay: int = 3):
    # Upserts and embedding replacements are idempotent, including after an
    # acknowledgement is lost. Never retry book generation or review decisions.
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            result = operation()
            elapsed = time.perf_counter() - started
            _record_import_event("atlas_write_succeeded", elapsed_seconds=round(elapsed, 3))
            print(f"Atlas 写入成功：耗时 {elapsed:.2f}s。", flush=True)
            return result
        except ConnectionFailure as exc:
            elapsed = time.perf_counter() - started
            _record_import_event(
                "atlas_write_failed", elapsed_seconds=round(elapsed, 3),
                status="connection_failure", error_type=type(exc).__name__, attempt=attempt + 1,
            )
            print(
                f"Atlas 写入失败：耗时 {elapsed:.2f}s，状态=连接失败，"
                f"错误类型={type(exc).__name__}，尝试 {attempt + 1}/3。",
                flush=True,
            )
            if attempt >= max_retries:
                raise
            print(f"Atlas 写入将在 {retry_delay}s 后重试。", flush=True)
            time.sleep(retry_delay)


def _retry_embeddings(texts, *, max_retries: int = 3, retry_delay: int = 3):
    for attempt in range(max_retries + 1):
        retryable = True
        started = time.perf_counter()
        try:
            result = _book_embeddings(texts, include_status=True)
            if isinstance(result, tuple):
                vectors, status = result
            else:
                vectors, status = result, None
            elapsed = time.perf_counter() - started
            status_text = str(status) if status is not None else "未返回"
            _record_import_event(
                "embedding_succeeded", batch_size=len(texts), elapsed_seconds=round(elapsed, 3),
                http_status=status, attempt=attempt + 1,
            )
            print(
                f"Embedding 请求成功：批量={len(texts)}，耗时={elapsed:.2f}s，"
                f"HTTP 状态={status_text}，尝试={attempt + 1}/3。",
                flush=True,
            )
            return vectors
        except HTTPError as exc:
            elapsed = time.perf_counter() - started
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            _record_import_event(
                "embedding_failed", batch_size=len(texts), elapsed_seconds=round(elapsed, 3),
                http_status=exc.code, error_type="HTTPError", retryable=retryable, attempt=attempt + 1,
            )
            print(
                f"Embedding 请求失败：批量={len(texts)}，耗时={elapsed:.2f}s，"
                f"HTTP 状态={exc.code}，错误类型=HTTPError，"
                f"可重试={str(retryable).lower()}，尝试={attempt + 1}/3。",
                flush=True,
            )
            if not retryable:
                raise
            failure = exc
        except (ConnectionError, TimeoutError, URLError) as exc:
            elapsed = time.perf_counter() - started
            reason = getattr(exc, "reason", None)
            error_type = type(reason).__name__ if reason is not None else type(exc).__name__
            _record_import_event(
                "embedding_failed", batch_size=len(texts), elapsed_seconds=round(elapsed, 3),
                http_status=None, error_type=error_type, retryable=True, attempt=attempt + 1,
            )
            print(
                f"Embedding 请求失败：批量={len(texts)}，耗时={elapsed:.2f}s，"
                f"HTTP 状态=无响应，错误类型={error_type}，尝试={attempt + 1}/3。",
                flush=True,
            )
            failure = exc
        except Exception as exc:
            elapsed = time.perf_counter() - started
            http_status = getattr(exc, "http_status", None)
            failure_reason = (
                "invalid_embedding_response" if http_status is not None else type(exc).__name__
            )
            _record_import_event(
                "embedding_failed", batch_size=len(texts), elapsed_seconds=round(elapsed, 3),
                http_status=http_status, error_type=type(exc).__name__, failure_reason=failure_reason,
                retryable=False, attempt=attempt + 1,
            )
            print(
                f"Embedding 请求失败：批量={len(texts)}，耗时={elapsed:.2f}s，"
                f"HTTP 状态={http_status if http_status is not None else '未知'}，"
                f"失败原因={failure_reason}，错误类型={type(exc).__name__}，不可重试。",
                flush=True,
            )
            raise
        if attempt >= max_retries:
            raise failure
        print(f"Embedding 将在 {retry_delay}s 后重试。", flush=True)
        _record_import_event(
            "embedding_retry_scheduled", batch_size=len(texts), delay_seconds=retry_delay,
        )
        time.sleep(retry_delay)


def main() -> int:
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    parser = argparse.ArgumentParser(description="导入国家通用手语词典，自动提取文字与原图并生成复核清单。")
    parser.add_argument("--upload", action="store_true", help="上传书籍内容到 Atlas；必须显式启用授权开关")
    parser.add_argument("--embed", action="store_true", help="在外部处理授权开关开启时生成并保存真实 Embedding")
    parser.add_argument("--allow-atlas-upload", action="store_true",
                        help="明确允许本次将解析出的书籍内容保存到 Atlas")
    parser.add_argument("--allow-cloud-processing", action="store_true",
                        help="明确允许本次将书籍片段发送给 Embedding 服务")
    parser.add_argument("--upload-batch-size", type=int, choices=(1, 4, 16), default=16,
                        help="Atlas 每批写入数量；网络不稳定时可选 1")
    parser.add_argument("--upload-workers", type=int, choices=(1, 2, 4), default=1,
                        help="独立片段的上传并发数，最多 4 个")
    parser.add_argument("--embedding-workers", type=int, choices=(1,), default=1,
                        help="Embedding 固定单并发")
    parser.add_argument("--embedding-batch-size", type=int, choices=(1, 4, 8), default=1,
                        help="每次模型请求处理 1、4 或 8 段；较大批次需先确认模型支持")
    parser.add_argument("--embedding-limit", type=int, default=None,
                        help="本次最多生成的向量数量；适合先小批验证，省略则处理全部缺失项")
    parser.add_argument("--retry-delay", type=int, default=3,
                        help="可重试错误之间等待秒数，默认 3 秒")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="每条 Embedding 请求最多重试次数，不含首次请求，默认 3 次")
    parser.add_argument("--collection", choices=(STAGING_COLLECTION,), default=STAGING_COLLECTION,
                        help="本次导入的隔离暂存集合；不会写入或删除旧集合")
    parser.add_argument("--expected-dimensions", type=int, default=1024,
                        help="写入前要求的向量维度，默认 1024；0 表示接受模型实际返回维度")
    parser.add_argument("--approve", metavar="CHUNK_ID", help="审核通过一段内容")
    parser.add_argument("--reject", metavar="CHUNK_ID", help="拒绝一段内容")
    args = parser.parse_args()
    if args.collection != STAGING_COLLECTION:
        parser.error("导入脚本只允许写入 SignKnowledgeChunks_v2")
    if args.embedding_limit is not None and args.embedding_limit < 1:
        parser.error("--embedding-limit 必须大于 0")
    if args.retry_delay < 0 or args.max_retries < 0:
        parser.error("--retry-delay 和 --max-retries 不能小于 0")
    if args.expected_dimensions < 0:
        parser.error("--expected-dimensions 不能小于 0")
    if args.allow_cloud_processing:
        os.environ["BOOK_CONTENT_EXTERNAL_ALLOWED"] = "true"
    os.environ["BOOK_KNOWLEDGE_COLLECTION"] = STAGING_COLLECTION
    if args.approve or args.reject:
        store = _import_store()
        chunk_id = args.approve or args.reject
        if not store.review_knowledge_chunk(chunk_id, approved=bool(args.approve)):
            print("找不到该资料片段。")
            return 1
        print("审核状态已更新。")
        return 0

    if args.embed:
        if (os.environ.get("BOOK_CONTENT_EXTERNAL_ALLOWED", "").strip().lower() not in {"1", "true", "yes"}
                and not args.allow_cloud_processing):
            print("未生成 Embedding：BOOK_CONTENT_EXTERNAL_ALLOWED 未启用。")
            return 2
        if not _book_api_key():
            print("未生成 Embedding：未找到 DashScope API Key 环境变量。")
            return 2

    result = import_books()
    print(f"本机解析到 {len(result['chunks'])} 个词条，全部保持待审核。", flush=True)
    print(f"复核清单：{REPORT_PATH}", flush=True)
    print(f"需要人工复核的解析提示：{result['report']['parse_warning_count']} 项。", flush=True)
    if not args.upload:
        print("本次没有连接 Atlas，也没有上传书籍内容。确认授权后使用 --upload。")
        return 0
    if (os.environ.get("BOOK_CONTENT_ATLAS_ALLOWED", "").strip().lower() not in {"1", "true", "yes"}
            and not args.allow_atlas_upload):
        print("未上传 Atlas：BOOK_CONTENT_ATLAS_ALLOWED 未启用。")
        return 2
    print("正在连接 Atlas（使用 CA 证书校验及操作超时）……", flush=True)
    store = _import_store()
    print(f"目标 Atlas 集合：{args.collection}。旧集合不会被读取、覆盖或删除。", flush=True)
    new_chunks = result["chunks"]
    print(f"准备幂等更新/补写 {len(new_chunks)} 个词条；文本变化会重置向量并恢复待审核。", flush=True)
    inserted = 0
    completed_chunks = 0

    def write_batch(batch):
        return _retry_database_write(lambda: store.upsert_staging_knowledge_chunks(batch))

    # MongoClient is thread safe. Batches contain disjoint stable IDs; review
    # state is preserved by $setOnInsert even after a partially acknowledged run.
    pool = ThreadPoolExecutor(max_workers=args.upload_workers)
    futures = {
        pool.submit(write_batch, new_chunks[start:start + args.upload_batch_size]):
            len(new_chunks[start:start + args.upload_batch_size])
        for start in range(0, len(new_chunks), args.upload_batch_size)
    }
    try:
        for future in as_completed(futures):
            inserted += future.result()
            completed_chunks += futures[future]
            if completed_chunks % 16 == 0 or completed_chunks == len(new_chunks):
                print(f"Atlas 补写进度：{completed_chunks}/{len(new_chunks)}", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    print(f"新增写入 Atlas {inserted} 段。", flush=True)
    if args.embed:
        missing_ids = store.knowledge_ids_without_embeddings([chunk["_id"] for chunk in result["chunks"]])
        all_pending = [chunk for chunk in result["chunks"] if chunk["_id"] in missing_ids]
        pending = all_pending[:args.embedding_limit] if args.embedding_limit is not None else all_pending
        print(f"缺少向量 {len(all_pending)} 段，已有向量将跳过；本次处理 {len(pending)} 段。", flush=True)
        completed_before_run = len(result["chunks"]) - len(all_pending)
        completed = 0
        batch_size = args.embedding_batch_size
        dimensions = None

        def embed_batch(batch):
            vectors = _retry_embeddings(
                [chunk["text"] for chunk in batch],
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            if args.expected_dimensions and any(
                len(vector) != args.expected_dimensions for vector in vectors
            ):
                raise RuntimeError(
                    f"Embedding 返回维度与要求不符；要求 {args.expected_dimensions} 维，"
                    f"实际 {len(vectors[0])} 维，未写入本批。"
                )
            documents = [
                {**chunk, "embedding": vector}
                for chunk, vector in zip(batch, vectors)
            ]
            for start in range(0, len(documents), args.upload_batch_size):
                part = documents[start:start + args.upload_batch_size]
                save_started = time.perf_counter()
                _retry_database_write(
                    lambda: store.save_knowledge_embeddings(part),
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
                print(
                    f"向量写入完成：数量={len(part)}，耗时={time.perf_counter() - save_started:.2f}s。",
                    flush=True,
                )
            return len(batch), len(vectors[0])

        if args.embedding_workers == 1:
            batches = (pending[start:start + batch_size] for start in range(0, len(pending), batch_size))
            for batch in batches:
                count, batch_dimensions = embed_batch(batch)
                if dimensions is not None and dimensions != batch_dimensions:
                    raise RuntimeError("embedding dimensions changed between batches")
                dimensions = batch_dimensions
                completed += count
                progress = completed_before_run + completed
                print(
                    f"[{progress}/{len(result['chunks'])}] 写入成功；"
                    f"本次完成 {completed}/{len(pending)}，向量维度={dimensions}。",
                    flush=True,
                )
        else:
            pool = ThreadPoolExecutor(max_workers=args.embedding_workers)
            futures = [
                pool.submit(embed_batch, pending[start:start + batch_size])
                for start in range(0, len(pending), batch_size)
            ]
            try:
                for future in as_completed(futures):
                    count, batch_dimensions = future.result()
                    if dimensions is not None and dimensions != batch_dimensions:
                        raise RuntimeError("embedding dimensions changed between batches")
                    dimensions = batch_dimensions
                    completed += count
                    progress = completed_before_run + completed
                    print(
                        f"[{progress}/{len(result['chunks'])}] 写入成功；"
                        f"本次完成 {completed}/{len(pending)}，向量维度={dimensions}。",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        print(f"本次新生成并保存 {completed} 个真实向量；已有向量已跳过。", flush=True)
        if pending:
            index_name = knowledge_index_name()
            try:
                if not store.has_knowledge_vector_index(index_name):
                    store.create_knowledge_vector_index(index_name, dimensions)
                    print(f"已向 Atlas 提交 Vector Search 索引创建请求：{index_name}", flush=True)
                else:
                    print(f"Atlas Vector Search 索引已存在：{index_name}", flush=True)
            except Exception as exc:
                print(f"Atlas 向量已保存，但自动创建索引失败（{type(exc).__name__}）；请按 SIGN_KNOWLEDGE_RAG.md 手动创建索引。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Driver exceptions may contain server addresses or credential-bearing
        # request details. Keep CLI failures useful without dumping secrets.
        print_import_failure(exc)
        raise SystemExit(1)
