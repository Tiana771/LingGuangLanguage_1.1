import assert from "node:assert/strict";
import { SseDecoder, streamAssistant } from "./deepseek_stream.mjs";

const events = [];
const parser = new SseDecoder((name, data) => events.push({ name, data }));
parser.push("event: tok");
parser.push("en\ndata: {\"text\":\"你\"}\r\n\r\nevent: token\ndata: {\"text\":\"好\"}\n\nevent: done\ndata: {}\n\n", true);
assert.deepEqual(events, [
  { name: "token", data: { text: "你" } },
  { name: "token", data: { text: "好" } },
  { name: "done", data: {} },
]);

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
globalThis.window = {
  LINGUANG_CSRF_TOKEN: "test-csrf",
  linguangAuth: { currentUser: () => ({ id: "user-123" }) },
};
const chunks = [
  "event: meta\ndata: {\"message\":\"已连接\"}\n\n",
  "event: token\ndata: {\"text\":\"第一段\"}\n\n",
  "event: token\ndata: {\"text\":\"第二段\"}\n\n",
  "event: done\ndata: {}\n\n",
].map((chunk) => new TextEncoder().encode(chunk));
let requestOptions;
globalThis.fetch = async (_url, options) => {
  requestOptions = options;
  return ({
  ok: true,
  body: {
    getReader: () => ({
      read: async () => chunks.length ? { value: chunks.shift(), done: false } : { value: undefined, done: true },
      cancel: async () => {},
      releaseLock: () => {},
    }),
  },
  });
};
const received = [];
await streamAssistant({
  url: "/api/assistant/stream",
  action: "summarize",
  text: "用户明确选择的内容",
  onMeta: (message) => received.push({ type: "meta", message }),
  onToken: (text) => received.push({ type: "token", text }),
});
globalThis.fetch = originalFetch;
globalThis.window = originalWindow;
assert.equal(requestOptions.headers["X-CSRF-Token"], "test-csrf");
assert.equal(requestOptions.headers["X-Linguang-User-ID"], "user-123");
assert.deepEqual(received, [
  { type: "meta", message: "已连接" },
  { type: "token", text: "第一段" },
  { type: "token", text: "第二段" },
]);
console.log("DeepSeek SSE parser and progressive client tests passed");
