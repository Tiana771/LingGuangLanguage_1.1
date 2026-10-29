from __future__ import annotations

import json
import os
import re
import secrets
import unicodedata
import base64
import binascii
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, session, stream_with_context
from flask.sessions import SessionInterface, SessionMixin
from werkzeug.datastructures import CallbackDict
import joblib
from werkzeug.security import check_password_hash, generate_password_hash

from mongo_store import MongoStore, knowledge_collection_name, knowledge_index_name
from sign_baseline import predict_sequence
from book_knowledge import MEDIA_DIR


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
DEFAULT_REJECTION_CONFIG = ROOT / "config" / "rejection_profiles.json"
DEBUG_FRAMES_DIR = ROOT / "debug_frames"
DEEPSEEK_MAX_TEXT_CHARS = 12000
SESSION_USER_KEY = "linguang_user_id"
SESSION_NAME_KEY = "linguang_username"
CSRF_SESSION_KEY = "linguang_csrf_token"


class MongoBackedSession(CallbackDict, SessionMixin):
    def __init__(self, initial: dict[str, Any] | None = None, sid: str | None = None, new: bool = False):
        def on_update(_):
            self.modified = True

        super().__init__(initial, on_update)
        self.sid = sid or secrets.token_urlsafe(32)
        self.new = new
        self.modified = False

    def regenerate(self) -> None:
        self.previous_sid = self.sid
        self.sid = secrets.token_urlsafe(32)
        self.new = True
        self.modified = True


class MongoSessionInterface(SessionInterface):
    session_class = MongoBackedSession

    def __init__(self, get_store):
        self.get_store = get_store

    def open_session(self, app, req):
        session_id = req.cookies.get(self.get_cookie_name(app))
        if not session_id or len(session_id) > 128:
            return self.session_class()
        try:
            data = self.get_store().get_session(session_id)
        except Exception as exc:
            app.logger.error("server session load failed: %s", type(exc).__name__)
            return self.session_class()
        return self.session_class(data, sid=session_id, new=False) if data else self.session_class()

    def save_session(self, app, session_obj, response):
        name = self.get_cookie_name(app)
        if not session_obj:
            if session_obj.modified and not session_obj.new:
                try:
                    self.get_store().delete_session(session_obj.sid)
                except Exception as exc:
                    app.logger.error("server session delete failed: %s", type(exc).__name__)
            if request.cookies.get(name):
                response.delete_cookie(
                    name,
                    path=self.get_cookie_path(app),
                    domain=self.get_cookie_domain(app),
                    secure=self.get_cookie_secure(app),
                    httponly=self.get_cookie_httponly(app),
                    samesite=self.get_cookie_samesite(app),
                )
            return
        if not session_obj.modified and not session_obj.new:
            return
        expires_at = datetime.now(timezone.utc) + app.permanent_session_lifetime
        try:
            self.get_store().save_session(session_obj.sid, dict(session_obj), expires_at)
            if getattr(session_obj, "previous_sid", None):
                self.get_store().delete_session(session_obj.previous_sid)
        except Exception as exc:
            app.logger.error("server session save failed: %s", type(exc).__name__)
            raise
        response.set_cookie(
            name,
            session_obj.sid,
            expires=expires_at,
            httponly=self.get_cookie_httponly(app),
            secure=self.get_cookie_secure(app),
            samesite=self.get_cookie_samesite(app),
            path=self.get_cookie_path(app),
            domain=self.get_cookie_domain(app),
        )


def _normalize_username(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    username = unicodedata.normalize("NFKC", value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", username):
        return None
    return username, username.casefold()


def _json_error(message: str, status: int):
    return jsonify({"message": message}), status


def _history_messages(question: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence = []
    for index, source in enumerate(sources, 1):
        evidence.append(
            f"[{index}] 沟通：{source['conversationTitle']}；主题：{source['blockTitle']}；"
            f"时间：{source['createdAt']}；原文：{source['originalText']}"
        )
    return [
        {
            "role": "system",
            "content": (
                "你是个人沟通记录查找助手。只能依据下方提供的用户本人记录回答，不得补充记录中没有的事实。"
                "回答时引用来源编号，如[1]；若证据不足，明确说明记录里没有找到答案。"
                "不得回答手语动作或把文字说成手语翻译。"
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\n检索到的本人记录：\n" + "\n".join(evidence)},
    ]


def _iter_history_sse(upstream, repository, owner_id: str, question: str, sources: list[dict[str, Any]]):
    answer = []
    completed = False
    try:
        yield _sse_event("sources", {"sources": [{key: value for key, value in item.items() if key != "score"} for item in sources]})
        yield _sse_event("meta", {"message": "正在根据命中记录整理回答…"})
        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                completed = True
                if answer:
                    repository.save_assistant_turn(owner_id, question, "".join(answer), sources)
                yield _sse_event("done", {})
                break
            try:
                payload = json.loads(data)
                if payload.get("error"):
                    yield _sse_event("error", {"message": "历史记录 AI 回答失败，请重试。"})
                    return
                choices = payload.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content", "")
                if isinstance(content, str) and content:
                    answer.append(content)
                    yield _sse_event("token", {"text": content})
            except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                continue
        if not completed:
            yield _sse_event("error", {"message": "历史记录 AI 流提前结束，结果未完成。"})
    except GeneratorExit:
        raise
    except Exception:
        yield _sse_event("error", {"message": "历史记录 AI 响应中断，请重试。"})
    finally:
        upstream.close()


def _deepseek_endpoint() -> str:
    base_url = os.environ.get("DEEP_SEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _deepseek_messages(action: str, text: str) -> list[dict[str, str]]:
    if action == "summarize":
        instruction = (
            "你是无障碍沟通记录整理助手。只根据用户提供的转写内容，用简体中文提炼明确的信息和待确认事项。"
            "不得补充原文没有的事实；不确定或听写含混的内容应标记为待确认。用简洁条目输出。"
        )
        request_text = f"请总结以下沟通转写的重点：\n\n{text}"
    else:
        instruction = (
            "你是无障碍沟通助手。请基于用户提供的沟通转写，给出一条简短、礼貌、可供用户修改的建议回复。"
            "不得虚构用户的行程、承诺、身份或原文没有的信息；信息不足时提出澄清问题。"
            "只输出建议回复正文，不要声称已发送，也不要把文字说成手语翻译。"
        )
        request_text = f"请根据以下沟通内容拟一条建议回复：\n\n{text}"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": request_text},
    ]


def _iter_deepseek_sse(upstream):
    completed = False
    try:
        yield _sse_event("meta", {"message": "已连接，等待模型首段内容…"})
        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                completed = True
                yield _sse_event("done", {})
                break
            try:
                payload = json.loads(data)
                if payload.get("error"):
                    yield _sse_event("error", {"message": "DeepSeek 生成失败，请重试。"})
                    return
                choices = payload.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content", "")
                if isinstance(content, str) and content:
                    yield _sse_event("token", {"text": content})
            except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                continue
        if not completed:
            yield _sse_event("error", {"message": "DeepSeek 流提前结束，结果未完成。"})
    except GeneratorExit:
        raise
    except Exception:
        yield _sse_event("error", {"message": "DeepSeek 流式响应中断，请重试。"})
    finally:
        upstream.close()


def _sse_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _book_external_allowed() -> bool:
    return os.environ.get("BOOK_CONTENT_EXTERNAL_ALLOWED", "").strip().lower() in {"1", "true", "yes"}


def _book_api_key() -> str:
    return (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASH_SCOPE_API_KEY") or "").strip()


class BookEmbeddingResponseError(RuntimeError):
    def __init__(self, message: str, http_status: int | None):
        super().__init__(message)
        self.http_status = http_status


def _book_embeddings(texts: list[str], *, include_status: bool = False):
    if not _book_external_allowed():
        raise RuntimeError("book content external processing is disabled")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embedding input must contain non-empty strings")
    key = _book_api_key()
    if not key:
        raise RuntimeError("embedding credentials are not configured")
    base_url = os.environ.get(
        "BOOK_EMBEDDING_BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/api/v1"
    ).strip().rstrip("/")
    api_style = os.environ.get("BOOK_EMBEDDING_API_STYLE", "native").strip().lower()
    model = os.environ.get("BOOK_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash").strip()
    if api_style == "compatible":
        endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
        payload = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
    else:
        endpoint = base_url if base_url.endswith("/text-embedding") else f"{base_url}/services/embeddings/text-embedding/text-embedding"
        payload = {"model": model, "input": {"texts": texts}}
    req = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    }, method="POST")
    with urlopen(req, timeout=45) as response:
        status = getattr(response, "status", None) or response.getcode()
        body = json.loads(response.read().decode("utf-8"))
    vectors = []
    if isinstance(body.get("data"), list) and body["data"]:
        indexed = body["data"]
        indexed.sort(key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") or item.get("embeddings") for item in indexed]
    if not vectors:
        output = body.get("output") or {}
        embeddings = output.get("embeddings") or []
        embeddings.sort(key=lambda item: item.get("text_index", 0))
        vectors = [item.get("embedding") for item in embeddings]
    if len(vectors) != len(texts) or any(
        not isinstance(vector, list) or not vector
        or any(not isinstance(value, (int, float)) for value in vector)
        for vector in vectors
    ):
        raise BookEmbeddingResponseError("embedding response did not contain a numeric vector", status)
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise BookEmbeddingResponseError("embedding response contains inconsistent vector dimensions", status)
    parsed_vectors = [[float(value) for value in vector] for vector in vectors]
    return (parsed_vectors, status) if include_status else parsed_vectors


def _book_embedding(text: str) -> list[float]:
    return _book_embeddings([text])[0]


def _qwen_book_answer(question: str, sources: list[dict[str, Any]]) -> str:
    key = (os.environ.get("QWEN_API_KEY") or _book_api_key()).strip()
    if not key:
        raise RuntimeError("Qwen credentials are not configured")
    base_url = os.environ.get(
        "BOOK_QWEN_BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1"
    ).strip().rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    evidence = "\n\n".join(
        f"[{index}] 第{source.get('book')}册，{source.get('entry_title') or '词条片段'}，"
        f"页序{source.get('page') or '未标页码'}：\n{source.get('text', '')}"
        for index, source in enumerate(sources, 1)
    )
    body = json.dumps({
        "model": os.environ.get("BOOK_QWEN_MODEL", "qwen3.8-flash").strip(),
        "messages": [
            {"role": "system", "content": (
                "你是手语词典检索问答助手。只根据提供的国家通用手语词典原文回答。"
                "不得编造动作、补充书外知识或把有限词条说成句子翻译。证据不足时明确说资料没有说明。"
                "回答动作描述时尽量保留原文，并在相关陈述后标注来源编号如[1]。"
                "不要声称生成或展示了图片。"
            )},
            {"role": "user", "content": f"问题：{question}\n\n检索到的已审核原文：\n{evidence}"},
        ],
        "temperature": 0.1,
        "max_tokens": 700,
    }, ensure_ascii=False).encode("utf-8")
    req = Request(endpoint, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    }, method="POST")
    with urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    answer = result["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Qwen response was empty")
    return answer.strip()


def _qwen_book_json(messages: list[dict[str, str]], *, max_tokens: int = 600) -> dict[str, Any]:
    key = (os.environ.get("QWEN_API_KEY") or _book_api_key()).strip()
    if not key:
        raise RuntimeError("Qwen credentials are not configured")
    base_url = os.environ.get(
        "BOOK_QWEN_BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1"
    ).strip().rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    body = json.dumps({
        "model": os.environ.get("BOOK_QWEN_MODEL", "qwen3.8-flash").strip(),
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }, ensure_ascii=False).encode("utf-8")
    req = Request(endpoint, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    }, method="POST")
    with urlopen(req, timeout=45) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen response was empty")
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen response did not contain a JSON object")
    parsed = json.loads(raw[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Qwen response was not a JSON object")
    return parsed


def _qwen_knowledge_plan(question: str) -> dict[str, Any]:
    result = _qwen_book_json([
        {"role": "system", "content": (
            "你负责理解手语词典搜索请求并生成检索计划，不回答用户问题。"
            "用户可能查单个词，也可能按场景找词。只输出 JSON："
            "{\"intent\":\"term|scenario\",\"topic\":\"主题名称\",\"search_terms\":[\"词或短语\"],\"list_title\":\"建议词单名\"}。"
            "检索词必须是中文词语或短语，最多8个，去掉提问套话。"
            "场景意图要给出场景常见表达、同义说法和可能相关词，但不能声称这些词已存在词典。"
            "单词查询应保留被查词，不要扩写成不相关内容。主题名称和词单名简短。"
        )},
        {"role": "user", "content": question},
    ], max_tokens=260)
    intent = "scenario" if result.get("intent") == "scenario" else "term"
    terms = result.get("search_terms", [])
    if not isinstance(terms, list):
        terms = []
    clean_terms = []
    for term in terms:
        if isinstance(term, str):
            term = term.strip()[:40]
            if term and term not in clean_terms:
                clean_terms.append(term)
        if len(clean_terms) >= 8:
            break
    topic = result.get("topic") if isinstance(result.get("topic"), str) else ""
    list_title = result.get("list_title") if isinstance(result.get("list_title"), str) else ""
    return {"intent": intent, "topic": topic.strip()[:40], "search_terms": clean_terms,
            "list_title": list_title.strip()[:40]}


def _qwen_select_knowledge(question: str, candidates: list[dict[str, Any]], topic: str) -> dict[str, Any]:
    compact = [{"id": item["id"], "title": item["title"], "text": item["text"][:420],
                "book": item.get("book"), "page": item.get("page")}
               for item in candidates]
    result = _qwen_book_json([
        {"role": "system", "content": (
            "你是手语词典场景词汇筛选器。只从候选数组选择与用户场景有关的真实词条。"
            "不得创造、改写或补充词条及动作说明；每个 id 必须原样来自候选数组。"
            "最多选择12个，优先最实用、直接相关的词。无关则不选。"
            "只输出 JSON：{\"summary\":\"一句简短说明\",\"items\":[{\"id\":\"原候选id\",\"reason\":\"简短相关理由\"}]}。"
        )},
        {"role": "user", "content": json.dumps({"question": question, "topic": topic,
                                                   "candidates": compact}, ensure_ascii=False)},
    ], max_tokens=700)
    allowed = {item["id"] for item in candidates}
    seen = set()
    selected = []
    raw_items = result.get("items", [])
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if item_id not in allowed or item_id in seen:
                continue
            reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
            selected.append({"id": item_id, "reason": reason.strip()[:120]})
            seen.add(item_id)
            if len(selected) == 12:
                break
    summary = result.get("summary") if isinstance(result.get("summary"), str) else ""
    return {"items": selected, "summary": summary.strip()[:240]}


def _keyword_scenario_candidates(
    candidates: list[dict[str, Any]], search_terms: list[str], topic: str, limit: int = 12,
) -> list[dict[str, Any]]:
    """Keep only retrieved entries with visible lexical evidence for the requested scene."""
    terms = list(dict.fromkeys([topic, *search_terms]))
    terms = [re.sub(r"\s+", "", term).casefold() for term in terms if isinstance(term, str) and term.strip()]
    ranked = []
    for item in candidates:
        title = re.sub(r"\s+", "", str(item.get("title") or "")).casefold()
        text = re.sub(r"\s+", "", str(item.get("text") or "")).casefold()
        title_hits = [term for term in terms if term and term in title]
        text_hits = [term for term in terms if term and term in text]
        if not title_hits and not text_hits:
            continue
        ranked.append((len(title_hits) * 3 + len(text_hits), item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def _fallback_knowledge_plan(question: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", question)
    scene_words = ("场景", "哪些词", "有什么词", "会用到", "相关词", "有关的词", "常用词")
    intent = "scenario" if any(marker in compact for marker in scene_words) else "term"
    topic = ""
    for marker in ("场景", "相关", "有关", "会用到", "有哪些", "有什么"):
        if marker in compact:
            topic = compact.split(marker, 1)[0].replace("关于", "").strip("，。！？")
            topic = re.sub(r"(里面|里|之内|内)$", "", topic)
            if topic:
                break
    terms = [topic] if topic else [compact]
    return {"intent": intent, "topic": topic[:40], "search_terms": terms[:8],
            "list_title": f"{topic}场景"[:40] if topic else ""}


def _book_similarity_min_score() -> float:
    try:
        value = float(os.environ.get("BOOK_SIMILARITY_MIN_SCORE", "0.62"))
    except (TypeError, ValueError):
        return 0.62
    return min(max(value, 0.0), 1.0)


def _public_knowledge_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("_id", "")),
        "book": item.get("book"),
        "title": item.get("entry_title") or item.get("source_title") or "词条片段",
        "text": str(item.get("text", ""))[:1400],
        "page": item.get("page"),
        "sourceFile": item.get("source_file", ""),
        "images": [str(path) for path in item.get("image_paths", []) if isinstance(path, str)],
        "score": item.get("score"),
    }


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
               config_path: Path | str | None = None, rejection_mode: str | None = None,
               mongo_store: MongoStore | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(48)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        MAX_CONTENT_LENGTH=1024 * 1024,
    )
    app.extensions["linguang_mongo_store"] = mongo_store or MongoStore()
    app.extensions["linguang_mongo_ready"] = False

    def store():
        repository = app.extensions["linguang_mongo_store"]
        if not app.extensions["linguang_mongo_ready"]:
            repository.initialize()
            app.extensions["linguang_mongo_ready"] = True
        return repository

    app.session_interface = MongoSessionInterface(store)

    protected_prefixes = ("/api/communications", "/api/history", "/api/assistant/stream", "/api/auth/password", "/api/auth/profile", "/api/learning")
    csrf_prefixes = ("/api/auth/register", "/api/auth/login", "/api/auth/logout", "/api/auth/password", "/api/auth/profile",
                     "/api/communications", "/api/history", "/api/assistant/stream", "/api/learning")

    @app.before_request
    def enforce_session_security():
        if request.path.startswith(protected_prefixes) and not session.get(SESSION_USER_KEY):
            return _json_error("请先登录后使用此功能。", 401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.path.startswith(csrf_prefixes):
            expected = session.get(CSRF_SESSION_KEY)
            provided = request.headers.get("X-CSRF-Token", "")
            if not expected or not secrets.compare_digest(expected, provided):
                return _json_error("请求安全校验失败，请刷新页面后重试。", 403)

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

    @app.get("/api/auth/csrf")
    def auth_csrf():
        token = session.get(CSRF_SESSION_KEY)
        if not token:
            token = secrets.token_urlsafe(32)
            session[CSRF_SESSION_KEY] = token
        return jsonify({"csrfToken": token})

    @app.get("/api/auth/me")
    def auth_me():
        user_id = session.get(SESSION_USER_KEY)
        if not user_id:
            return jsonify({"authenticated": False})
        return jsonify({"authenticated": True, "user": {"id": user_id, "username": session.get(SESSION_NAME_KEY, "")}})

    @app.get("/api/auth/profile")
    def auth_profile():
        user_id = session.get(SESSION_USER_KEY)
        try:
            user = store().find_user_by_id(user_id)
            if not user:
                return _json_error("账号信息暂不可用，请重新登录。", 401)
            return jsonify({"profile": {
                "id": str(user.get("_id", "")),
                "username": str(user.get("username", "")),
                "displayName": str(user.get("display_name") or user.get("username", "")),
                "bio": str(user.get("bio", "")),
                "avatar": str(user.get("avatar_data_url", "")),
            }})
        except Exception as exc:
            app.logger.error("profile load failed: %s", type(exc).__name__)
            return _json_error("个人资料暂时无法读取，请稍后重试。", 503)

    @app.put("/api/auth/profile")
    def auth_update_profile():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("个人资料格式无效。", 400)
        display_name = payload.get("displayName")
        bio = payload.get("bio", "")
        avatar = payload.get("avatar", "")
        if not isinstance(display_name, str):
            return _json_error("请填写昵称。", 400)
        display_name = unicodedata.normalize("NFKC", display_name).strip()
        if not display_name or len(display_name) > 40:
            return _json_error("昵称需为 1–40 个字符。", 400)
        if not isinstance(bio, str) or len(bio.strip()) > 240:
            return _json_error("个人简介最多 240 个字符。", 400)
        if not isinstance(avatar, str) or len(avatar) > 360_000:
            return _json_error("头像文件过大，请选择小于 256 KB 的图片。", 400)
        if avatar:
            match = re.fullmatch(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]+={0,2})", avatar)
            if not match:
                return _json_error("头像仅支持 JPEG、PNG 或 WebP 图片。", 400)
            try:
                image_bytes = base64.b64decode(match.group(2), validate=True)
            except (binascii.Error, ValueError):
                return _json_error("头像图片数据无效，请重新选择。", 400)
            mime = match.group(1)
            valid_signature = (
                (mime == "image/jpeg" and image_bytes.startswith(b"\xff\xd8\xff"))
                or (mime == "image/png" and image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
                or (mime == "image/webp" and len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP")
            )
            if not valid_signature or len(image_bytes) > 256 * 1024:
                return _json_error("头像图片无效或超过 256 KB。", 400)
        try:
            profile = store().update_user_profile(session[SESSION_USER_KEY], display_name, bio.strip(), avatar)
            if not profile:
                return _json_error("个人资料暂时无法保存，请稍后重试。", 503)
            return jsonify({"profile": profile})
        except Exception as exc:
            app.logger.error("profile update failed: %s", type(exc).__name__)
            return _json_error("个人资料暂时无法保存，请稍后重试。", 503)

    @app.post("/api/auth/register")
    def auth_register():
        payload = request.get_json(silent=True) or {}
        normalized = _normalize_username(payload.get("username"))
        password = payload.get("password")
        if not normalized:
            return _json_error("用户名需为 3–32 位字母、数字、下划线或连字符。", 400)
        if not isinstance(password, str) or len(password) < 8 or len(password) > 256:
            return _json_error("密码长度需为 8–256 位。", 400)
        try:
            repository = store()
            if repository.find_user_by_name(normalized[1]):
                return _json_error("用户名不可用，请更换后重试。", 409)
            user = repository.insert_user(normalized[0], normalized[1], generate_password_hash(password))
        except Exception as exc:
            if exc.__class__.__name__ == "DuplicateKeyError":
                return _json_error("用户名不可用，请更换后重试。", 409)
            app.logger.error("account registration failed: %s", type(exc).__name__)
            return _json_error("账号服务暂不可用，请检查 MongoDB Atlas 连接配置。", 503)
        session.clear()
        session.regenerate()
        session[SESSION_USER_KEY] = str(user["_id"])
        session[SESSION_NAME_KEY] = normalized[0]
        session.permanent = True
        session[CSRF_SESSION_KEY] = secrets.token_urlsafe(32)
        return jsonify({"authenticated": True, "user": {"id": str(user["_id"]), "username": normalized[0]},
                        "csrfToken": session[CSRF_SESSION_KEY]}), 201

    @app.post("/api/auth/login")
    def auth_login():
        payload = request.get_json(silent=True) or {}
        normalized = _normalize_username(payload.get("username"))
        password = payload.get("password")
        if not normalized or not isinstance(password, str):
            return _json_error("用户名或密码错误。", 401)
        try:
            user = store().find_user_by_name(normalized[1])
            valid = bool(user and check_password_hash(user.get("password_hash", ""), password))
        except Exception as exc:
            app.logger.error("account login service failed: %s", type(exc).__name__)
            return _json_error("账号服务暂不可用，请稍后重试。", 503)
        if not valid:
            return _json_error("用户名或密码错误。", 401)
        session.clear()
        session.regenerate()
        session[SESSION_USER_KEY] = str(user["_id"])
        session[SESSION_NAME_KEY] = user["username"]
        session.permanent = True
        session[CSRF_SESSION_KEY] = secrets.token_urlsafe(32)
        return jsonify({"authenticated": True, "user": {"id": session[SESSION_USER_KEY],
                                                           "username": session[SESSION_NAME_KEY]},
                        "csrfToken": session[CSRF_SESSION_KEY]})

    @app.post("/api/auth/logout")
    def auth_logout():
        session.clear()
        return jsonify({"authenticated": False})

    @app.post("/api/auth/password")
    def auth_change_password():
        user_id = session.get(SESSION_USER_KEY)
        payload = request.get_json(silent=True) or {}
        current_password = payload.get("currentPassword")
        new_password = payload.get("newPassword")
        if not isinstance(current_password, str) or not isinstance(new_password, str) or len(new_password) < 8 or len(new_password) > 256:
            return _json_error("请提供当前密码和长度为 8–256 位的新密码。", 400)
        try:
            repository = store()
            user = repository.find_user_by_id(user_id)
            if not user or not check_password_hash(user.get("password_hash", ""), current_password):
                return _json_error("当前密码不正确。", 400)
            repository.change_password(user_id, generate_password_hash(new_password))
        except Exception as exc:
            app.logger.error("password update failed: %s", type(exc).__name__)
            return _json_error("密码暂时无法更新，请稍后重试。", 503)
        return jsonify({"message": "密码已更新。"})

    @app.get("/api/communications")
    def list_communications():
        try:
            return jsonify({"records": store().list_conversations(session[SESSION_USER_KEY])})
        except Exception as exc:
            app.logger.error("conversation list failed: %s", type(exc).__name__)
            return _json_error("沟通记录服务暂不可用，请检查 Atlas 连接。", 503)

    @app.post("/api/communications")
    def create_communication():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("沟通记录格式无效。", 400)
        try:
            record = store().insert_conversation(session[SESSION_USER_KEY], payload)
            return jsonify({"record": record}), 201
        except Exception as exc:
            app.logger.error("conversation create failed: %s", type(exc).__name__)
            return _json_error("沟通记录无法保存，请稍后重试。", 503)

    @app.put("/api/communications/<record_id>")
    def update_communication(record_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("沟通记录格式无效。", 400)
        try:
            record = store().update_conversation(session[SESSION_USER_KEY], record_id, payload)
        except Exception as exc:
            app.logger.error("conversation update failed: %s", type(exc).__name__)
            return _json_error("沟通记录无法保存，请稍后重试。", 503)
        if not record:
            return _json_error("沟通记录不存在。", 404)
        return jsonify({"record": record})

    @app.delete("/api/communications/<record_id>")
    def delete_communication(record_id: str):
        try:
            deleted = store().delete_conversation(session[SESSION_USER_KEY], record_id)
        except Exception as exc:
            app.logger.error("conversation delete failed: %s", type(exc).__name__)
            return _json_error("沟通记录无法删除，请稍后重试。", 503)
        if not deleted:
            return _json_error("沟通记录不存在。", 404)
        return jsonify({"deleted": True})

    @app.delete("/api/communications")
    def clear_communications():
        try:
            deleted_count = store().delete_all_conversations(session[SESSION_USER_KEY])
            return jsonify({"deletedCount": deleted_count})
        except Exception as exc:
            app.logger.error("conversation clear failed: %s", type(exc).__name__)
            return _json_error("沟通记录无法清空，请稍后重试。", 503)

    @app.get("/api/learning/today")
    def learning_today():
        owner_id = session[SESSION_USER_KEY]
        try:
            repository = store()
            settings = repository.get_learning_settings(owner_id)
            plan = repository.get_learning_today(owner_id, settings["dailyNewTarget"])
            return jsonify({**plan, "settings": settings})
        except Exception as exc:
            app.logger.error("learning today load failed: %s", type(exc).__name__)
            return _json_error("今日学习内容暂时无法读取，请稍后重试。", 503)

    @app.get("/api/learning/settings")
    def learning_settings():
        try:
            return jsonify({"settings": store().get_learning_settings(session[SESSION_USER_KEY])})
        except Exception as exc:
            app.logger.error("learning settings load failed: %s", type(exc).__name__)
            return _json_error("学习设置暂时无法读取，请稍后重试。", 503)

    @app.put("/api/learning/settings")
    def update_learning_settings():
        payload = request.get_json(silent=True)
        target = payload.get("dailyNewTarget") if isinstance(payload, dict) else None
        if isinstance(target, bool) or target not in {3, 5, 10}:
            return _json_error("每日新词目标请选择 3、5 或 10 个。", 400)
        try:
            settings = store().save_learning_settings(session[SESSION_USER_KEY], target)
        except Exception as exc:
            app.logger.error("learning settings save failed: %s", type(exc).__name__)
            return _json_error("学习设置暂时无法保存，请稍后重试。", 503)
        if not settings:
            return _json_error("账号信息暂时不可用，请重新登录。", 401)
        return jsonify({"settings": settings})

    @app.get("/api/learning/entries")
    def learning_entries():
        query = request.args.get("q", "").strip()
        if len(query) > 100:
            return _json_error("搜索内容不能超过 100 个字符。", 400)
        try:
            book_value = request.args.get("book", "").strip()
            book = int(book_value) if book_value else None
            if book is not None and book not in {1, 2, 3, 4}:
                raise ValueError
            page = max(1, min(int(request.args.get("page", "1")), 10000))
        except ValueError:
            return _json_error("册次或页码无效。", 400)
        limit = 30
        owner_id = session[SESSION_USER_KEY]
        try:
            repository = store()
            items = repository.search_learning_entries(query, book, limit=limit, skip=(page - 1) * limit)
            total = repository.count_learning_entries(query, book)
            flags = repository.learning_entry_flags(owner_id, [item["id"] for item in items])
        except Exception as exc:
            app.logger.error("learning entries load failed: %s", type(exc).__name__)
            return _json_error("已审核词库暂时无法读取，请稍后重试。", 503)
        return jsonify({"entries": [{**item, **flags.get(item["id"], {"favorite": False, "learningActive": False})}
                                    for item in items], "page": page, "pageSize": limit, "total": total})

    @app.get("/api/learning/entries/<entry_id>")
    def learning_entry(entry_id: str):
        owner_id = session[SESSION_USER_KEY]
        try:
            repository = store()
            entry = repository.find_learning_entry(entry_id)
            if not entry:
                return _json_error("没有找到可学习的已审核词条。", 404)
            flags = repository.learning_entry_flags(owner_id, [entry_id]).get(
                entry_id, {"favorite": False, "learningActive": False})
            return jsonify({"entry": {**entry, **flags}})
        except Exception as exc:
            app.logger.error("learning entry load failed: %s", type(exc).__name__)
            return _json_error("词条暂时无法读取，请稍后重试。", 503)

    @app.post("/api/learning/entries/<entry_id>/favorite")
    def learning_favorite(entry_id: str):
        payload = request.get_json(silent=True)
        favorite = payload.get("favorite") if isinstance(payload, dict) else None
        if not isinstance(favorite, bool):
            return _json_error("收藏状态无效。", 400)
        try:
            saved = store().update_learning_flag(session[SESSION_USER_KEY], entry_id, "favorite", favorite)
        except Exception as exc:
            app.logger.error("learning favorite update failed: %s", type(exc).__name__)
            return _json_error("收藏状态暂时无法保存，请稍后重试。", 503)
        if not saved:
            return _json_error("没有找到可收藏的已审核词条。", 404)
        return jsonify({"favorite": favorite})

    @app.route("/api/learning/entries/<entry_id>/plan", methods=["POST", "DELETE"])
    def learning_plan_entry(entry_id: str):
        active = request.method == "POST"
        try:
            saved = store().update_learning_flag(session[SESSION_USER_KEY], entry_id, "learning_active", active)
        except Exception as exc:
            app.logger.error("learning plan update failed: %s", type(exc).__name__)
            return _json_error("学习计划暂时无法更新，请稍后重试。", 503)
        if not saved:
            return _json_error("没有找到可学习的已审核词条。", 404)
        return jsonify({"learningActive": active})

    @app.post("/api/learning/review")
    def learning_review():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("练习记录格式无效。", 400)
        entry_id, rating, event_id = payload.get("entryId"), payload.get("rating"), payload.get("eventId")
        if not isinstance(entry_id, str) or not entry_id or len(entry_id) > 160:
            return _json_error("词条编号无效。", 400)
        if rating not in {"remembered", "fuzzy", "forgot"}:
            return _json_error("请选择记得、有点模糊或忘记了。", 400)
        if not isinstance(event_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", event_id):
            return _json_error("练习记录编号无效，请重新提交。", 400)
        try:
            result = store().record_learning_review(session[SESSION_USER_KEY], entry_id, rating, event_id)
        except ValueError:
            return _json_error("练习反馈无效。", 400)
        except Exception as exc:
            app.logger.error("learning review save failed: %s", type(exc).__name__)
            return _json_error("练习结果暂时无法保存，请稍后重试。", 503)
        if not result:
            return _json_error("没有找到可练习的已审核词条。", 404)
        progress = result["progress"]
        return jsonify({"saved": True, "duplicate": result["duplicate"],
                        "nextReviewAt": progress.get("next_review_at"),
                        "intervalDays": progress.get("interval_days", 0),
                        "dayComplete": result.get("dayComplete", progress.get("day_session_complete", True)),
                        "repeatToday": result.get("repeatToday", False),
                        "masterySuccesses": result.get("masterySuccesses", progress.get("day_session_successes", 0)),
                        "masteryRequired": result.get("masteryRequired", progress.get("day_session_required", 0)),
                        "initialRating": result.get("initialRating", progress.get("day_session_initial_rating"))})

    @app.get("/api/learning/lists")
    def learning_lists():
        try:
            return jsonify(store().list_learning_lists(session[SESSION_USER_KEY]))
        except Exception as exc:
            app.logger.error("learning lists load failed: %s", type(exc).__name__)
            return _json_error("个人词单暂时无法读取，请稍后重试。", 503)

    @app.post("/api/learning/lists")
    def create_learning_list():
        payload = request.get_json(silent=True)
        title = payload.get("title") if isinstance(payload, dict) else None
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 40:
            return _json_error("词单名称需为 1–40 个字符。", 400)
        try:
            item = store().create_learning_list(session[SESSION_USER_KEY], title.strip())
            return jsonify({"list": item}), 201
        except Exception as exc:
            app.logger.error("learning list create failed: %s", type(exc).__name__)
            return _json_error("词单暂时无法创建，请稍后重试。", 503)

    @app.route("/api/learning/lists/<list_id>", methods=["PATCH", "DELETE"])
    def update_or_delete_learning_list(list_id: str):
        repository = store()
        owner_id = session[SESSION_USER_KEY]
        if request.method == "DELETE":
            try:
                deleted = repository.delete_learning_list(owner_id, list_id)
            except Exception as exc:
                app.logger.error("learning list delete failed: %s", type(exc).__name__)
                return _json_error("词单暂时无法删除，请稍后重试。", 503)
            return jsonify({"deleted": True}) if deleted else _json_error("词单不存在。", 404)
        payload = request.get_json(silent=True)
        title = payload.get("title") if isinstance(payload, dict) else None
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 40:
            return _json_error("词单名称需为 1–40 个字符。", 400)
        try:
            renamed = repository.rename_learning_list(owner_id, list_id, title.strip())
        except Exception as exc:
            app.logger.error("learning list rename failed: %s", type(exc).__name__)
            return _json_error("词单暂时无法修改，请稍后重试。", 503)
        return jsonify({"title": title.strip()}) if renamed else _json_error("词单不存在。", 404)

    @app.route("/api/learning/lists/<list_id>/entries/<entry_id>", methods=["POST", "DELETE"])
    def update_learning_list_entry(list_id: str, entry_id: str):
        try:
            saved = store().update_learning_list_entry(
                session[SESSION_USER_KEY], list_id, entry_id, request.method == "POST")
        except Exception as exc:
            app.logger.error("learning list entry update failed: %s", type(exc).__name__)
            return _json_error("词单内容暂时无法更新，请稍后重试。", 503)
        if not saved:
            return _json_error("词单或已审核词条不存在。", 404)
        return jsonify({"updated": True})

    @app.post("/api/learning/batch-save")
    def batch_save_learning_entries():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("批量保存内容格式无效。", 400)
        action = payload.get("action")
        entry_ids = payload.get("entryIds")
        if action not in {"favorite", "list", "plan"}:
            return _json_error("请选择加入收藏、学习计划或词单。", 400)
        if (not isinstance(entry_ids, list) or not entry_ids or len(entry_ids) > 50
                or any(not isinstance(item, str) or not item or len(item) > 160 for item in entry_ids)):
            return _json_error("请选择 1–50 个有效词条。", 400)
        list_id = payload.get("listId", "")
        list_title = payload.get("listTitle", "")
        if not isinstance(list_id, str) or not isinstance(list_title, str):
            return _json_error("词单信息无效。", 400)
        if action == "list":
            if len(list_id) > 100:
                return _json_error("词单编号无效。", 400)
            if len(list_title.strip()) > 40:
                return _json_error("主题名称不能超过 40 个字符。", 400)
            if not list_id and not list_title.strip():
                return _json_error("请选择已有词单或填写新主题名称。", 400)
        try:
            result = store().batch_save_learning_entries(
                session[SESSION_USER_KEY], entry_ids, action=action,
                list_id=list_id.strip(), list_title=list_title.strip(),
            )
        except ValueError:
            return _json_error("批量保存操作无效。", 400)
        except Exception as exc:
            app.logger.error("learning batch save failed: %s", type(exc).__name__)
            return _json_error("批量保存失败，请稍后重试。", 503)
        if result.get("missingList"):
            return _json_error("所选词单不存在或不属于当前账号，请刷新后重试。", 404)
        return jsonify(result)

    @app.get("/api/learning/stats")
    def learning_stats():
        try:
            days = int(request.args.get("days", "7"))
            if days not in {7, 30}:
                raise ValueError
        except ValueError:
            return _json_error("统计周期只能为 7 天或 30 天。", 400)
        try:
            return jsonify(store().learning_stats(session[SESSION_USER_KEY], days))
        except Exception as exc:
            app.logger.error("learning stats load failed: %s", type(exc).__name__)
            return _json_error("学习统计暂时无法读取，请稍后重试。", 503)

    @app.get("/api/learning/stats/words")
    def learning_stat_words():
        category = request.args.get("category", "")
        if category not in {"today", "learned", "remembered", "fuzzy", "forgot", "due"}:
            return _json_error("统计分类无效。", 400)
        try:
            days = int(request.args.get("days", "7"))
            if days not in {7, 30}:
                raise ValueError
        except ValueError:
            return _json_error("统计周期只能为 7 天或 30 天。", 400)
        try:
            groups = store().learning_stat_words(session[SESSION_USER_KEY], category, days)
            return jsonify({"category": category, "days": days, "groups": groups})
        except Exception as exc:
            app.logger.error("learning stat words failed: %s", type(exc).__name__)
            return _json_error("对应词汇暂时无法读取，请稍后重试。", 503)

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

    @app.get("/book-media/<path:filename>")
    def book_media(filename: str):
        return send_from_directory(MEDIA_DIR, filename, conditional=True)

    @app.get("/api/knowledge/status")
    def knowledge_status():
        index_name = knowledge_index_name()
        vector_ready = bool(index_name and _book_api_key() and _book_external_allowed())
        content_allowed = _book_external_allowed()
        try:
            repository = store()
            pending_count = repository.knowledge_chunks.count_documents({"review_status": "pending"})
            approved_count = repository.knowledge_chunks.count_documents({"review_status": "approved"})
            vector_ready = vector_ready and repository.has_knowledge_vector_index(index_name)
        except Exception as exc:
            app.logger.error("book knowledge status failed: %s", type(exc).__name__)
            return jsonify({
                "knowledgeApiVersion": 2,
                "collection": knowledge_collection_name(),
                "databaseAvailable": False,
                "pendingCount": None,
                "approvedCount": None,
                "message": "暂时连不上资料库，请稍后重试。当前无法读取词条数量，无需重新导入。",
            }), 503
        return jsonify({
            "retrieval": "atlas_vector" if vector_ready and content_allowed else "keyword",
            "knowledgeApiVersion": 2,
            "collection": knowledge_collection_name(),
            "databaseAvailable": True,
            "vectorConfigured": vector_ready,
            "externalProcessingAllowed": content_allowed,
            "pendingCount": pending_count,
            "approvedCount": approved_count,
        })

    @app.post("/api/knowledge/search")
    def knowledge_search():
        payload = request.get_json(silent=True)
        question = payload.get("question", "") if isinstance(payload, dict) else ""
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            return _json_error("请输入不超过 500 字的问题。", 400)
        question = question.strip()
        try:
            repository = store()
            use_vectors = bool(_book_external_allowed() and _book_api_key() and knowledge_index_name())
            if use_vectors:
                use_vectors = repository.has_knowledge_vector_index(knowledge_index_name())
        except Exception as exc:
            app.logger.error("book knowledge store unavailable: %s", type(exc).__name__)
            return _json_error("手语书籍资料暂不可用，请检查 Atlas 连接。", 503)

        try:
            exact_sources = repository.search_knowledge_exact(question)
            if exact_sources:
                answer = "\n\n".join(item["text"][:900] for item in exact_sources)
                answer_mode = "原书词条"
                if _book_external_allowed():
                    try:
                        answer = _qwen_book_answer(question, exact_sources)
                        answer_mode = "Qwen 基于原文回答"
                    except Exception as exc:
                        app.logger.error("Qwen book answer failed: %s", type(exc).__name__)
                        answer_mode = "原书内容（模型暂不可用）"
                sources = [_public_knowledge_source(item) for item in exact_sources]
                return jsonify({
                    "knowledgeApiVersion": 2, "collection": knowledge_collection_name(),
                    "retrieval": "exact_title", "intent": "term", "answerMode": answer_mode,
                    "recommendationMode": "exact", "topic": "", "suggestedListTitle": "",
                    "message": "已按词条标题精确匹配，展示全部对应词条。",
                    "answer": answer, "sources": sources,
                })

            plan = _fallback_knowledge_plan(question)
            plan_mode = "keyword_fallback"
            if _book_external_allowed() and (os.environ.get("QWEN_API_KEY") or _book_api_key()):
                try:
                    plan = _qwen_knowledge_plan(question)
                    plan_mode = "qwen"
                except Exception as exc:
                    app.logger.error("Qwen book query planning failed: %s", type(exc).__name__)

            candidates: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            def append_candidates(found):
                for item in found:
                    item_id = str(item.get("_id", ""))
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        candidates.append(item)

            search_terms = list(dict.fromkeys([question, *plan.get("search_terms", [])]))[:9]
            retrieval_methods = []
            if use_vectors:
                vectors = _book_embeddings(search_terms[:5])
                for vector in vectors:
                    found = repository.search_knowledge_vectors(vector, knowledge_index_name(), limit=20)
                    append_candidates([
                        item for item in found
                        if isinstance(item.get("score"), (int, float))
                        and item["score"] >= min(_book_similarity_min_score(), 0.42)
                    ])
                if candidates:
                    retrieval_methods.append("Atlas Vector Search")
            for term in search_terms:
                append_candidates(repository.search_knowledge_keywords(term, limit=12))
            if candidates and not retrieval_methods:
                retrieval_methods.append("Atlas 关键词检索")
            candidates = candidates[:50]

            if not candidates:
                return jsonify({
                    "knowledgeApiVersion": 2, "collection": knowledge_collection_name(),
                    "retrieval": "none", "intent": plan.get("intent", "term"),
                    "answerMode": "检索候选为空", "recommendationMode": plan_mode,
                    "topic": plan.get("topic", ""), "suggestedListTitle": plan.get("list_title", ""),
                    "message": "已审核词典中没有找到足够相关的词条。可以换个说法或试试单个词语。",
                    "answer": "", "sources": [],
                })

            if plan.get("intent") == "scenario":
                selected_ids = []
                reasons = {}
                recommendation_mode = "retrieved_candidates"
                if _book_external_allowed() and (os.environ.get("QWEN_API_KEY") or _book_api_key()):
                    try:
                        selection = _qwen_select_knowledge(question, [_public_knowledge_source(item) for item in candidates],
                                                           plan.get("topic", ""))
                        selected_ids = [item["id"] for item in selection["items"]]
                        reasons = {item["id"]: item["reason"] for item in selection["items"]}
                        recommendation_mode = "qwen_curated"
                    except Exception as exc:
                        app.logger.error("Qwen book candidate selection failed: %s", type(exc).__name__)
                if not selected_ids:
                    lexical_matches = _keyword_scenario_candidates(
                        [_public_knowledge_source(item) for item in candidates],
                        plan.get("search_terms", []), plan.get("topic", ""),
                    )
                    selected_ids = [item["id"] for item in lexical_matches]
                    if recommendation_mode == "retrieved_candidates":
                        recommendation_mode = "keyword_fallback"
                selected_set = set(selected_ids)
                sources = []
                for item in candidates:
                    source = _public_knowledge_source(item)
                    if source["id"] not in selected_set:
                        continue
                    source["recommendationReason"] = reasons.get(source["id"], "与检索场景相关的已审核词条")
                    sources.append(source)
                answer = f"从已审核词典中筛出 {len(sources)} 个可能适用于“{plan.get('topic') or '该场景'}”的词条。请核对后勾选保存。"
                if not sources:
                    answer = "当前已审核词典中没有找到能明确对应这个场景的词条。可以换个场景说法，或搜索具体词语。"
                return jsonify({
                    "knowledgeApiVersion": 2, "collection": knowledge_collection_name(),
                    "retrieval": "scenario", "intent": "scenario", "answerMode": recommendation_mode,
                    "recommendationMode": recommendation_mode, "retrievalMethods": retrieval_methods,
                    "topic": plan.get("topic", ""), "suggestedListTitle": plan.get("list_title", ""),
                    "message": "下面仅列出已审核词典中的候选词；请核对后勾选保存。",
                    "answer": answer, "sources": sources,
                })

            sources = [_public_knowledge_source(item) for item in candidates[:12]]
            answer = "\n\n".join(f"{item['title']}：{item['text'][:500]}" for item in sources[:5])
            return jsonify({
                "knowledgeApiVersion": 2, "collection": knowledge_collection_name(),
                "retrieval": "keyword", "intent": "term", "answerMode": "审核词条原文",
                "recommendationMode": plan_mode, "topic": plan.get("topic", ""),
                "suggestedListTitle": plan.get("list_title", ""),
                "message": "未命中完全相同的词条标题，以下是相关的已审核词条。",
                "answer": answer, "sources": sources,
            })
        except Exception as exc:
            app.logger.error("book knowledge retrieval failed: %s", type(exc).__name__)
            return _json_error("手语书籍检索暂时失败，请检查 Atlas、Embedding 服务和向量索引配置。", 503)

    @app.get("/videos/<path:filename>")
    def video(filename: str):
        return send_from_directory(media_dir, filename, conditional=True)

    @app.post("/api/assistant/stream")
    def assistant_stream():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"message": "请求内容格式无效。"}), 400
        action = payload.get("action")
        text = payload.get("text")
        if action not in {"summarize", "reply"}:
            return jsonify({"message": "不支持的 AI 操作。"}), 400
        if not isinstance(text, str) or not text.strip():
            return jsonify({"message": "请先提供沟通转写内容。"}), 400
        text = text.strip()
        if len(text) > DEEPSEEK_MAX_TEXT_CHARS:
            return jsonify({"message": "当前记录过长，请先缩短或分段处理。"}), 413
        api_key = os.environ.get("DEEP_SEEK_API_KEY", "").strip()
        if not api_key:
            return jsonify({"message": "DeepSeek API 尚未配置，请检查服务端环境变量。"}), 503

        model = os.environ.get("DEEP_SEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
        body = json.dumps({
            "model": model,
            "messages": _deepseek_messages(action, text),
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 480 if action == "summarize" else 220,
        }, ensure_ascii=False).encode("utf-8")
        upstream_request = Request(
            _deepseek_endpoint(),
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            upstream = urlopen(upstream_request, timeout=90)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status in {401, 403}:
                message = "DeepSeek API 认证失败，请检查服务端密钥。"
            elif status == 429:
                message = "DeepSeek API 请求过于频繁或额度不足，请稍后重试。"
            else:
                message = f"DeepSeek API 请求失败（HTTP {status}）。"
            return jsonify({"message": message}), 502
        except (URLError, TimeoutError, OSError):
            return jsonify({"message": "无法连接 DeepSeek API，请检查网络后重试。"}), 502

        return Response(
            stream_with_context(_iter_deepseek_sse(upstream)),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/history/chats")
    def history_chats():
        try:
            return jsonify({"chats": store().list_assistant_chats(session[SESSION_USER_KEY])})
        except Exception as exc:
            app.logger.error("history chat list failed: %s", type(exc).__name__)
            return _json_error("历史问答暂不可用，请稍后重试。", 503)

    @app.post("/api/history/ask")
    def ask_history():
        payload = request.get_json(silent=True)
        question = payload.get("question", "") if isinstance(payload, dict) else ""
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            return _json_error("请输入不超过 500 字的问题。", 400)
        question = question.strip()
        owner_id = session[SESSION_USER_KEY]
        try:
            repository = store()
            sources = repository.search_conversations(owner_id, question)
        except Exception as exc:
            app.logger.error("history search failed: %s", type(exc).__name__)
            return _json_error("历史记录检索暂不可用，请稍后重试。", 503)

        if not sources:
            answer = "在当前账号保存的沟通记录中没有找到相关内容。"
            try:
                repository.save_assistant_turn(owner_id, question, answer, [])
            except Exception as exc:
                app.logger.error("empty history answer save failed: %s", type(exc).__name__)
                return _json_error("问答记录无法保存，请稍后重试。", 503)

            def empty_history_answer():
                yield _sse_event("sources", {"sources": []})
                yield _sse_event("token", {"text": answer})
                yield _sse_event("done", {})

            return Response(stream_with_context(empty_history_answer()), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

        api_key = os.environ.get("DEEP_SEEK_API_KEY", "").strip()
        if not api_key:
            return _json_error("DeepSeek API 尚未配置，请检查服务端环境变量。", 503)
        body = json.dumps({
            "model": os.environ.get("DEEP_SEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
            "messages": _history_messages(question, sources),
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 600,
        }, ensure_ascii=False).encode("utf-8")
        upstream_request = Request(_deepseek_endpoint(), data=body, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }, method="POST")
        try:
            upstream = urlopen(upstream_request, timeout=90)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            message = "DeepSeek API 认证失败，请检查服务端密钥。" if status in {401, 403} else "DeepSeek API 请求失败，请稍后重试。"
            return _json_error(message, 502)
        except (URLError, TimeoutError, OSError):
            return _json_error("无法连接 DeepSeek API，请检查网络后重试。", 502)
        return Response(
            stream_with_context(_iter_history_sse(upstream, repository, owner_id, question, sources)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

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


def main(argv=None):
    import argparse
    from server_runtime import ServerAlreadyRunningError, claim_server_port

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--config", type=Path, default=None, help="拒识配置文件路径")
    parser.add_argument("--rejection-mode", choices=("calibrated", "demo"), default=None,
                        help="默认 calibrated；demo=0.60，未经同等校准")
    parser.add_argument("--sample-books", action="store_true", help="使用新版词条集合，直接显示原书图文供测试")
    parser.add_argument("--semantic-books", action="store_true", help="使用新版词条集合和 Atlas 向量检索")
    args = parser.parse_args(argv)
    if args.sample_books and args.semantic_books:
        parser.error("--sample-books 与 --semantic-books 不能同时使用")
    if args.sample_books:
        os.environ["BOOK_KNOWLEDGE_COLLECTION"] = "SignKnowledgeChunks_v2"
        os.environ["BOOK_EMBEDDING_INDEX_NAME"] = "sign_knowledge_vector_v2"
        os.environ["BOOK_CONTENT_EXTERNAL_ALLOWED"] = "false"
    elif args.semantic_books:
        os.environ["BOOK_KNOWLEDGE_COLLECTION"] = "SignKnowledgeChunks_v2"
        os.environ["BOOK_EMBEDDING_INDEX_NAME"] = "sign_knowledge_vector_v2"
        os.environ["BOOK_CONTENT_EXTERNAL_ALLOWED"] = "true"
    try:
        with claim_server_port(args.host, args.port):
            app = create_app(config_path=args.config, rejection_mode=args.rejection_mode)
            print(f"model={app.config['SIGN_MODEL_PATH']} rejection_mode={app.config['SIGN_REJECTION_MODE']} threshold={app.config['SIGN_CONFIDENCE_THRESHOLD']}")
            print(f"词典读取集合：{knowledge_collection_name()}。", flush=True)
            if args.sample_books:
                print("新版词条测试已启用：直接显示原书文字和图片，不调用回答模型。", flush=True)
            elif args.semantic_books:
                print("新版词条语义检索已启用：使用 Atlas 向量索引；查询内容将发送到已配置的 Embedding 和回答模型服务。", flush=True)
            app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except ServerAlreadyRunningError as exc:
        print(str(exc), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
