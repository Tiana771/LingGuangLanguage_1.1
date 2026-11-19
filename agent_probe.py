"""Agent 探针（仅用于验证工具调用链路，默认关闭）。

这个模块只回答一个问题：DeepSeek 能不能按我们给的 schema 真的发起 tool_calls，
我们能不能把结果回填、再让它据此作答，以及这一圈要花多久。

实现上有意保持最小：一个工具、最多三轮、工具轮用非流式请求（便于拿到完整的
tool_calls 参数），最后一轮把答案切成小段以 SSE 吐出，形状与前端将来的
tool_start / tool_result / token / done 事件一致。

通过环境变量控制：
    AGENT_ENABLED=1     在 serve_sign_api 里注册 /api/agent/probe
    AGENT_MAX_ROUNDS=3  最多几轮工具调用
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
    "你可以调用工具查询已审核的《国家通用手语词典》。"
    "需要词条释义、动作做法、册号页码或示范视频时，先调用 lookup_term，不要凭记忆回答。"
    "工具返回为空时，明确说明词典中没有收录，不要编造。"
    "回答用简体中文，不超过 200 字，先给结论再给依据。"
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_term",
            "description": "在已审核的《国家通用手语词典》中查询一个手语词条，返回释义、动作描述、册号页码与原书图片。",
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
    }
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


def _run_tool(name: str, arguments: dict[str, Any], search: Callable[[str], list[dict]]) -> dict[str, Any]:
    if name != "lookup_term":
        return {"ok": False, "error": f"未知工具：{name}"}
    term = str(arguments.get("term") or "").strip()
    if not term:
        return {"ok": False, "error": "term 为空"}
    items = search(term) or []
    return {
        "ok": True,
        "term": term,
        "count": len(items),
        "items": [
            {
                "title": item.get("title"),
                "book": item.get("book"),
                "page": item.get("page"),
                "text": str(item.get("text") or "")[:600],
                "images": (item.get("images") or [])[:2],
            }
            for item in items[:3]
        ],
    }


def _chunks(text: str, size: int = 28) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]


def iter_agent_stream(
    *,
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    search: Callable[[str], list[dict]],
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

        message = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
        calls = message.get("tool_calls") or []
        yield sse("round", {
            "round": rounds,
            "latency_ms": int((time.time() - round_started) * 1000),
            "tool_calls": len(calls),
            "finish_reason": ((data.get("choices") or [{}])[0] or {}).get("finish_reason"),
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
            result = _run_tool(name, arguments if isinstance(arguments, dict) else {}, search)
            took_ms = int((time.time() - tool_started) * 1000)
            yield sse("tool_result", {
                "name": name,
                "term": result.get("term"),
                "count": result.get("count", 0),
                "took_ms": took_ms,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "titles": [item.get("title") for item in result.get("items", [])],
            })
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
