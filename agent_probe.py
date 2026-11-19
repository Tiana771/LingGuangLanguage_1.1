"""Agent 工具调用模块（灰度开关 AGENT_ENABLED，默认关闭）。

职责：把 DeepSeek 的 tool_calls 循环跑通——模型决定调哪个工具、我们执行、
把结果回填、直到它愿意作答，全过程以 SSE 事件吐出，前端据此画“执行轨迹”。

三个工具的分工：
    lookup_term     查一个具体词条（精确匹配，返回册号页码与原书图）
    search_scene    按一句话或一个场景找相关词（走向量 + 关键词）
    get_sign_video  取某个词的示范视频与动作说明

实现上有意保持克制：工具轮用非流式请求（便于拿到完整参数），最后一轮把答案
切成小段吐出；答案文字本身仍然由模型生成，但事实必须来自工具返回。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_ROUNDS = max(1, int(os.environ.get("AGENT_MAX_ROUNDS", "3") or 3))
REQUEST_TIMEOUT = float(os.environ.get("AGENT_REQUEST_TIMEOUT", "60") or 60)

SYSTEM_PROMPT = (
    "你是“聆光手语”平台里的助手，服务听障用户与健听用户之间的沟通。"
    "你可以调用工具：lookup_term 查一个具体词条，search_scene 按场景或一句话找相关词，"
    "get_sign_video 取某个词的示范视频。"
    "只要问题涉及词条释义、动作做法、册号页码或示范视频，就先调用工具，不要凭记忆回答。"
    "推荐替代说法时只能使用工具返回过的词；没有任何可用结果时，直接说明词典未收录，"
    "并反问用户是否要查某个具体的词，不要自己编词。"
    "回答用简体中文，不超过 200 字，先给结论再给依据。"
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_term",
            "description": "在已审核的《国家通用手语词典》中查询一个手语词条，返回释义、动作描述、册号页码与原书图片。适合确定要查哪个词时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "要查询的词，例如：复诊、台灯、谢谢。只传词本身，不要带句子。",
                    }
                },
                "required": ["term"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_scene",
            "description": "按一句话或一个场景检索已审核词典，返回一批可能适用的词条。适合“去医院会用到哪些词”“和购物有关的词”这类不确定具体词条的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "场景或一句话，例如：去医院看病、在餐厅点餐。不要只传单个词。",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sign_video",
            "description": "取某个词的手语示范视频地址、动作说明与释义。用户问“怎么打”“有没有视频”时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "要取视频的词，例如：你好、谢谢、医院。",
                    }
                },
                "required": ["term"],
                "additionalProperties": False,
            },
        },
    },
]


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _post_chat(url: str, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # pragma: no cover - 仅用于诊断信息
            detail = ""
        return f"HTTP {exc.code} {detail}".strip()
    if isinstance(exc, URLError):
        return f"网络错误：{exc.reason}"
    return f"{type(exc).__name__}: {exc}"


Executor = Callable[[dict[str, Any]], dict[str, Any]]


def _run_tool(name: str, arguments: dict[str, Any], executors: dict[str, Executor]) -> dict[str, Any]:
    executor = executors.get(name)
    if executor is None:
        return {"ok": False, "error": f"未知工具：{name}"}
    try:
        result = executor(arguments)
    except Exception as exc:  # 工具失败要让模型知道，而不是把整个流打断
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "工具返回格式不正确"}
    return result


def _chunks(text: str, size: int = 28) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]


def _event_payload(name: str, arguments: dict[str, Any], result: dict[str, Any], took_ms: int) -> dict[str, Any]:
    items = result.get("items") or []
    return {
        "name": name,
        "argument": arguments.get("term") or arguments.get("query") or "",
        "count": result.get("count", len(items)),
        "took_ms": took_ms,
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "titles": [item.get("title") or item.get("term") for item in items][:6],
    }


def iter_agent_stream(
    *,
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    executors: dict[str, Executor],
) -> Iterator[str]:
    """跑一次完整的“工具调用 → 回填 → 作答”，以 SSE 文本逐条吐出。"""

    started = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    yield sse("start", {"question": question, "model": model, "max_rounds": MAX_ROUNDS})

    rounds = 0
    tool_calls_total = 0
    finish = "answered"

    while rounds < MAX_ROUNDS:
        rounds += 1
        round_started = time.time()
        try:
            data = _post_chat(endpoint, api_key, {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": 600,
                "stream": False,
            })
        except Exception as exc:  # 探针阶段把真实原因带回前端，便于对照
            yield sse("error", {"stage": "request", "message": _error_text(exc)})
            yield sse("done", {"rounds": rounds, "tools": tool_calls_total,
                               "finish": "error", "elapsed_ms": int((time.time() - started) * 1000)})
            return

        choice = (data.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        yield sse("round", {
            "round": rounds,
            "latency_ms": int((time.time() - round_started) * 1000),
            "tool_calls": len(calls),
            "finish_reason": choice.get("finish_reason"),
        })

        if not calls:
            text = str(message.get("content") or "").strip()
            for piece in _chunks(text):
                yield sse("token", {"text": piece})
            yield sse("done", {
                "rounds": rounds,
                "tools": tool_calls_total,
                "finish": finish,
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            return

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:
            tool_calls_total += 1
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
            yield sse("tool_start", {
                "name": name,
                "raw_arguments": raw_arguments,
                "call_id": call.get("id"),
            })
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
            except json.JSONDecodeError:
                arguments = {"_parse_error": str(raw_arguments)[:200]}

            tool_started = time.time()
            result = _run_tool(name, arguments if isinstance(arguments, dict) else {}, executors)
            took_ms = int((time.time() - tool_started) * 1000)
            yield sse("tool_result", _event_payload(name, arguments if isinstance(arguments, dict) else {}, result, took_ms))
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result, ensure_ascii=False),
            })

    finish = "max_rounds"
    try:
        data = _post_chat(endpoint, api_key, {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 600,
            "stream": False,
        })
        text = str((((data.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or "").strip()
    except Exception as exc:
        yield sse("error", {"stage": "final", "message": _error_text(exc)})
        text = ""
    for piece in _chunks(text):
        yield sse("token", {"text": piece})
    yield sse("done", {
        "rounds": rounds,
        "tools": tool_calls_total,
        "finish": finish,
        "elapsed_ms": int((time.time() - started) * 1000),
    })
