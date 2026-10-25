export class SseDecoder {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.lineBuffer = "";
    this.eventName = "message";
    this.dataLines = [];
  }

  push(chunk, final = false) {
    this.lineBuffer += chunk;
    let newline;
    while ((newline = this.lineBuffer.indexOf("\n")) !== -1) {
      const line = this.lineBuffer.slice(0, newline).replace(/\r$/, "");
      this.lineBuffer = this.lineBuffer.slice(newline + 1);
      this.#consumeLine(line);
    }
    if (final && this.lineBuffer) {
      this.#consumeLine(this.lineBuffer.replace(/\r$/, ""));
      this.lineBuffer = "";
    }
    if (final) this.#dispatch();
  }

  #consumeLine(line) {
    if (!line) {
      this.#dispatch();
      return;
    }
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const value = separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "event") this.eventName = value || "message";
    if (field === "data") this.dataLines.push(value);
  }

  #dispatch() {
    if (this.dataLines.length) {
      const raw = this.dataLines.join("\n");
      let data = raw;
      try {
        data = JSON.parse(raw);
      } catch {
        // Keep non-JSON SSE payloads available to the caller.
      }
      this.onEvent?.(this.eventName, data);
    }
    this.eventName = "message";
    this.dataLines = [];
  }
}

export async function streamAssistant({ url, action, text, payload, signal, onToken, onMeta, onSources }) {
  const headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
  if (typeof window !== "undefined" && window.LINGUANG_CSRF_TOKEN) headers["X-CSRF-Token"] = window.LINGUANG_CSRF_TOKEN;
  const userId = typeof window !== "undefined" ? window.linguangAuth?.currentUser?.()?.id : "";
  if (userId) headers["X-Linguang-User-ID"] = userId;
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(payload || { action, text }),
    signal,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.message || `AI 服务暂不可用（${response.status}）`);
  }
  if (!response.body) throw new Error("浏览器无法读取流式响应");

  let sawDone = false;
  let streamError = null;
  const decoder = new TextDecoder();
  const parser = new SseDecoder((eventName, data) => {
    if (eventName === "token") onToken?.(typeof data?.text === "string" ? data.text : "");
    if (eventName === "meta") onMeta?.(typeof data?.message === "string" ? data.message : "已连接，等待模型输出…");
    if (eventName === "sources") onSources?.(Array.isArray(data?.sources) ? data.sources : []);
    if (eventName === "error") streamError = new Error(data?.message || "AI 生成中断");
    if (eventName === "done") sawDone = true;
  });
  const reader = response.body.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
      if (streamError) throw streamError;
    }
    parser.push(decoder.decode(), true);
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (streamError) throw streamError;
  if (!sawDone) throw new Error("AI 连接提前结束，结果未完成");
}
