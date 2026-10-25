import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [html, css, communication, auth, app] = await Promise.all([
  readFile(new URL("./index.html", import.meta.url), "utf8"),
  readFile(new URL("./styles.css", import.meta.url), "utf8"),
  readFile(new URL("./communication_assistant.mjs", import.meta.url), "utf8"),
  readFile(new URL("./auth_client.mjs", import.meta.url), "utf8"),
  readFile(new URL("./app.js", import.meta.url), "utf8"),
]);

const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
for (const [, id] of communication.matchAll(/el\("#([a-z0-9-]+)"\)/g)) {
  assert.ok(ids.has(id), `communication client references missing #${id}`);
}
for (const [, id] of auth.matchAll(/el\("#([a-z0-9-]+)"\)/g)) {
  assert.ok(ids.has(id), `auth client references missing #${id}`);
}

assert.match(html, /id="communication-record-layout"/);
assert.match(html, /id="history-assistant-panel"/);
assert.match(css, /\.communication-record-layout\.is-history-assistant\s+\.communication-record-detail/);
assert.match(communication, /currentHistorySources = sources/);
assert.match(communication, /data-conversation-id/);
assert.match(auth, /credentials: "same-origin"/);
assert.doesNotMatch(auth, /X-Linguang-User-ID/);
assert.match(html, /data-mode="profile"/);
assert.match(html, /id="profile-avatar-file"/);
assert.match(html, /id="account-logout"/);
assert.match(app, /profilePanel\.classList\.toggle\("hidden", mode !== "profile"\)/);
assert.match(html, /id="knowledge-search-history-list"/);
assert.match(html, /id="knowledge-history-clear"/);
assert.match(app, /KNOWLEDGE_HISTORY_LIMIT = 20/);
assert.match(app, /KNOWLEDGE_HISTORY_TTL_MS = 24 \* 60 \* 60 \* 1000/);
assert.match(app, /showKnowledgeResult\(entry\.data, \{ cachedAt: entry\.savedAt \}\)/);
assert.match(app, /localStorage\.removeItem\(KNOWLEDGE_HISTORY_KEY\)/);
assert.match(html, /id="transcript-speech-status"/);
assert.match(communication, /className = "transcript-speak"/);
assert.match(communication, /renderTranscriptText\(editor, block\.text \|\| ""\)/);
assert.match(app, /className = "book-source-body"/);
assert.match(app, /className = "book-source-tag"/);
assert.match(css, /\.book-source-body\{display:grid;grid-template-columns:minmax\(0,1fr\) minmax\(180px,\.72fr\)/);
assert.match(css, /\.ai-actions\{position:sticky/);
console.log("Frontend DOM contract checks passed");
