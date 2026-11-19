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
    "你是“聆光手语”平台里的查询助手。用户来查手语怎么说、问某个场景会用到哪些词，"
    "你把已审核的词典内容找出来并标明出处。可以调用：lookup_term 查具体词条（返回值里已带示范视频），"
    "search_scene 按场景或一句话找一批相关词，get_sign_video 单独取某个词的视频。"
    "\n\n遵守以下规则："
    "\n1. 问题涉及释义、动作做法、册号页码或视频时，先调用工具，不要凭记忆回答。"
    "\n2. 不要反问“要不要看视频”“要不要看详细说明”。工具返回的视频、页码、原书图、相近词，"
    "直接写进回答或直接呈现出来，用户想看自己会点。"
    "\n3. 不要寒暄，不要征求许可，不要说“如果需要我可以……”。这类句子对用户没有信息量。"
    "\n4. 用户只发来一个词（例如“要”“医院”）时，按查词处理，不要当成对上一句的回应。"
    "\n5. 推荐替代说法只能使用工具本次返回过的词；一条也没有时，说明词典未收录，并列出可查的相近词。"
    "\n6. 不要输出文件路径、图片地址或视频地址，这些由界面呈现，正文只写文字结论。"
    "\n7. 回答用简体中文，不超过 200 字，先给结论，再给册号与页码。"
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


def _cards_from(name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """把一次工具返回整理成前端能直接渲染的卡片。"""
    cards: list[dict[str, Any]] = []
    for item in (result.get("items") or [])[:6]:
        demo = item.get("demo") or {}
        video = str(demo.get("video_url") or item.get("video_url") or "")
        images = [str(url) for url in (item.get("images") or [])[:2] if isinstance(url, str)]
        title = str(item.get("title") or item.get("term") or "").strip()
        if not title and not images and not video:
            continue
        cards.append({
            "kind": "video" if video else "term",
            "title": title,
            "book": item.get("book"),
            "page": item.get("page"),
            "text": str(item.get("text") or "")[:240],
            "images": images,
            "video": video,
        })
    return cards


def iter_agent_stream(
    *,
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    executors: dict[str, Executor],
    history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """跑一次完整的“工具调用 → 回填 → 作答”，以 SSE 文本逐条吐出。"""

    started = time.time()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or [])[-8:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": question})
    yield sse("start", {"question": question, "model": model, "max_rounds": MAX_ROUNDS})

    rounds = 0
    tool_calls_total = 0
    cards: list[dict[str, Any]] = []
    seen_cards: set[tuple[str, str]] = set()
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
            if cards:
                yield sse("cards", {"cards": cards})
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
            for card in _cards_from(name, result):
                key = (card["title"], str(card.get("page") or ""))
                if key in seen_cards:
                    continue
                seen_cards.add(key)
                cards.append(card)
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
    if cards:
        yield sse("cards", {"cards": cards})
    yield sse("done", {
        "rounds": rounds,
        "tools": tool_calls_total,
        "finish": finish,
        "elapsed_ms": int((time.time() - started) * 1000),
    })

AGENT_DEMO_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 验收 · 聆光手语</title>
<style>
  :root { --ink:#123c33; --mint:#2f7d63; --soft:#eef6f1; --line:#d5e7dd; --muted:#6d8a80; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:linear-gradient(160deg,#f2f9f5,#e6f2ec); color:var(--ink); min-height:100vh; }
  .wrap { max-width:760px; margin:0 auto; padding:34px 20px 80px; }
  h1 { font-size:19px; margin:0 0 6px; font-weight:600; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .panel { background:#fff; border:1px solid var(--line); border-radius:16px;
           box-shadow:0 12px 32px rgba(18,60,51,.07); overflow:hidden; }
  .ask { display:flex; gap:10px; padding:16px; border-bottom:1px solid var(--line); }
  .ask input { flex:1; border:1px solid var(--line); border-radius:10px; padding:11px 13px; font-size:15px; outline:none; }
  .ask input:focus { border-color:var(--mint); box-shadow:0 0 0 3px rgba(47,125,99,.12); }
  .ask button { border:0; border-radius:10px; background:var(--mint); color:#fff; padding:0 20px; font-size:15px; cursor:pointer; }
  .ask button:disabled { opacity:.55; cursor:default; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; padding:11px 16px; border-bottom:1px solid var(--line); background:var(--soft); }
  .chip { border:1px solid var(--line); background:#fff; border-radius:999px; padding:5px 11px; font-size:13px; color:var(--mint); cursor:pointer; }
  .steps { padding:14px 16px 4px; }
  .steps:empty { display:none; }
  .step { display:flex; align-items:flex-start; gap:9px; font-size:13px; color:var(--muted); padding:5px 0; }
  .step .dot { width:7px; height:7px; border-radius:50%; background:#b7cec5; margin-top:6px; flex:none; }
  .step.running .dot { background:var(--mint); animation:pulse 1.1s infinite; }
  .step.done .dot { background:#c7dbd2; }
  .step .label { cursor:pointer; }
  .step .detail { display:none; margin:6px 0 2px; font-family:Consolas,monospace; font-size:11px;
                  color:#5c7a70; background:#f6fbf8; border:1px solid var(--line); border-radius:8px;
                  padding:8px; white-space:pre-wrap; word-break:break-all; }
  .step.open .detail { display:block; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.25 } }
  .cards { display:grid; gap:10px; padding:8px 16px 16px; }
  .cards:empty { display:none; }
  .card { border:1px solid var(--line); border-radius:12px; padding:12px; display:flex; gap:13px; background:#fbfefc; }
  .card video { flex:none; width:210px; border-radius:8px; background:#000; }
  .card img { flex:none; width:126px; height:94px; object-fit:cover; border-radius:8px; background:#e8f1ec; }
  .card .meta { font-size:12px; color:var(--muted); margin-bottom:3px; letter-spacing:.02em; }
  .card .name { font-size:15px; font-weight:600; margin-bottom:5px; }
  .card .desc { font-size:13px; line-height:1.65; color:#33574c; }
  .answer { padding:6px 16px 18px; }
  .answer .text { white-space:pre-wrap; line-height:1.8; font-size:15px; }
  .answer .text:empty { display:none; }
  .foot { display:flex; justify-content:space-between; align-items:center; padding:10px 16px;
          border-top:1px solid var(--line); font-size:12px; color:#9db5ac; }
  .foot a { color:#9db5ac; cursor:pointer; text-decoration:none; }
  .foot a:hover { color:var(--mint); }
  .log { display:none; padding:12px 16px; border-top:1px dashed var(--line); font-family:Consolas,monospace;
         font-size:11px; color:#5c7a70; max-height:280px; overflow:auto; white-space:pre-wrap; }
  .log.open { display:block; }
  .err { color:#b4423a; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Agent 验收页</h1>
  <div class="sub">B 版：用户只看一行进度与结果卡片，完整轨迹收在下面。</div>
  <div class="panel">
    <div class="ask">
      <input id="q" placeholder="问点什么，例如：谢谢的手语怎么打">
      <button id="go">发送</button>
    </div>
    <div class="chips">
      <span class="chip">谢谢的手语怎么打</span>
      <span class="chip">台灯的手语怎么打</span>
      <span class="chip">去医院看病会用到哪些词</span>
      <span class="chip">复诊的手语怎么打</span>
    </div>
    <div class="steps" id="steps"></div>
    <div class="cards" id="cards"></div>
    <div class="answer"><div class="text" id="answer"></div></div>
    <div class="foot">
      <span id="status">待提问</span>
      <a id="toggle">显示完整轨迹</a>
    </div>
    <div class="log" id="log"></div>
  </div>
</div>
<script>
const LABELS = { lookup_term: '查词典', search_scene: '按场景检索', get_sign_video: '调取示范视频' };
const stepsBox = document.getElementById('steps');
const cardsBox = document.getElementById('cards');
const answerBox = document.getElementById('answer');
const logBox = document.getElementById('log');
const statusBox = document.getElementById('status');
const input = document.getElementById('q');
const go = document.getElementById('go');
const SEP = String.fromCharCode(10, 10);

let started = 0, buffer = '', turns = [], running = null;

document.getElementById('toggle').addEventListener('click', function () {
  logBox.classList.toggle('open');
  this.textContent = logBox.classList.contains('open') ? '收起完整轨迹' : '显示完整轨迹';
});

document.querySelectorAll('.chip').forEach(function (chip) {
  chip.addEventListener('click', function () { input.value = chip.textContent; input.focus(); });
});

function raw(text) {
  logBox.textContent += text + String.fromCharCode(10);
  logBox.scrollTop = logBox.scrollHeight;
}

function addStep(text) {
  const row = document.createElement('div');
  row.className = 'step running';
  row.innerHTML = '<span class="dot"></span><span class="label"></span><div class="detail"></div>';
  row.querySelector('.label').textContent = text;
  row.querySelector('.label').addEventListener('click', function () { row.classList.toggle('open'); });
  stepsBox.appendChild(row);
  stepsBox.scrollTop = stepsBox.scrollHeight;
  return row;
}

function renderCards(cards) {
  cardsBox.innerHTML = '';
  cards.forEach(function (card) {
    const row = document.createElement('div');
    row.className = 'card';
    if (card.video) {
      const video = document.createElement('video');
      video.src = card.video; video.controls = true; video.preload = 'metadata';
      row.appendChild(video);
    } else if (card.images && card.images.length) {
      const image = document.createElement('img');
      image.src = card.images[0]; image.loading = 'lazy';
      row.appendChild(image);
    }
    const body = document.createElement('div');
    body.innerHTML = '<div class="meta"></div><div class="name"></div><div class="desc"></div>';
    body.querySelector('.meta').textContent =
      [card.book ? '第 ' + card.book + ' 册' : '', card.page ? '第 ' + card.page + ' 页' : '']
        .filter(Boolean).join(' · ') || '已审核词典';
    body.querySelector('.name').textContent = card.title || '';
    body.querySelector('.desc').textContent = card.text || '';
    row.appendChild(body);
    cardsBox.appendChild(row);
  });
}

function reset() {
  buffer = '';
  running = null;
  stepsBox.innerHTML = '';
  cardsBox.innerHTML = '';
  answerBox.textContent = '';
  logBox.textContent = '';
  statusBox.textContent = '正在请求…';
  started = Date.now();
}

async function ask() {
  const question = input.value.trim();
  if (!question || go.disabled) return;
  go.disabled = true;
  reset();
  try {
    const response = await fetch('/api/agent/probe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, history: turns.slice(-8) })
    });
    if (!response.ok || !response.body) {
      statusBox.innerHTML = '<span class="err">请求失败：HTTP ' + response.status + '</span>';
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      let index;
      while ((index = pending.indexOf(SEP)) >= 0) {
        handle(pending.slice(0, index));
        pending = pending.slice(index + SEP.length);
      }
    }
  } catch (error) {
    statusBox.innerHTML = '<span class="err">异常：' + error + '</span>';
  } finally {
    go.disabled = false;
  }
}

function handle(block) {
  let event = '', data = '';
  block.split(String.fromCharCode(10)).forEach(function (row) {
    if (row.startsWith('event:')) event = row.slice(6).trim();
    else if (row.startsWith('data:')) data += row.slice(5).trim();
  });
  if (!event) return;
  raw('[' + event + '] ' + data);
  let payload = {};
  try { payload = JSON.parse(data); } catch (error) { return; }
  const ms = Date.now() - started;

  if (event === 'start') { statusBox.textContent = '正在处理…'; }
  else if (event === 'tool_start') {
    running = addStep('正在' + (LABELS[payload.name] || payload.name) + '…');
    running.querySelector('.detail').textContent = payload.raw_arguments || '';
  }
  else if (event === 'tool_result') {
    const row = running || addStep('');
    running = null;
    row.classList.remove('running');
    row.classList.add('done');
    const label = LABELS[payload.name] || payload.name;
    row.querySelector('.label').textContent = payload.ok
      ? (label + '：找到 ' + (payload.count || 0) + ' 条' + (payload.titles && payload.titles.length ? ' · ' + payload.titles.join('、') : ''))
      : (label + '：未成功' + (payload.error ? '（' + payload.error + '）' : ''));
    row.querySelector('.detail').textContent =
      (payload.raw_arguments || '') + String.fromCharCode(10) + '耗时 ' + payload.took_ms + 'ms · 命中 ' + (payload.count || 0) + ' 条';
  }
  else if (event === 'token') { buffer += payload.text || ''; answerBox.textContent = buffer; }
  else if (event === 'cards') renderCards(payload.cards || []);
  else if (event === 'error') statusBox.innerHTML = '<span class="err">出错：' + (payload.stage || '') + ' ' + payload.message + '</span>';
  else if (event === 'done') {
    statusBox.textContent = '用时 ' + (payload.elapsed_ms / 1000).toFixed(1) + ' 秒 · ' + payload.tools + ' 次工具调用';
    if (buffer) {
      turns.push({ role: 'user', content: input.value.trim() });
      turns.push({ role: 'assistant', content: buffer });
      if (turns.length > 8) turns = turns.slice(-8);
    }
  }
}

go.addEventListener('click', ask);
input.addEventListener('keydown', function (event) { if (event.key === 'Enter') ask(); });
window.onerror = function (message, source, line) {
  statusBox.innerHTML = '<span class="err">页面脚本出错：' + message + '（第 ' + line + ' 行）</span>';
};
</script>
</body>
</html>
"""
