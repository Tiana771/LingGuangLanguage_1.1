import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [html, css, communication, auth, app, learning, speechRetry] = await Promise.all([
  readFile(new URL("./index.html", import.meta.url), "utf8"),
  readFile(new URL("./styles.css", import.meta.url), "utf8"),
  readFile(new URL("./communication_assistant.mjs", import.meta.url), "utf8"),
  readFile(new URL("./auth_client.mjs", import.meta.url), "utf8"),
  readFile(new URL("./app.js", import.meta.url), "utf8"),
  readFile(new URL("./learning_client.mjs", import.meta.url), "utf8"),
  readFile(new URL("./speech_retry.mjs", import.meta.url), "utf8"),
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
assert.match(auth, /requireLoginAtStart = options\.requireLoginAtStart === true/);
assert.match(auth, /if \(requireLoginAtStart && !currentUser\) return/);
assert.match(auth, /auth-close.*classList\.toggle\("hidden", requireLoginAtStart && !currentUser\)/);
assert.doesNotMatch(auth, /X-Linguang-User-ID/);
assert.match(html, /data-mode="profile"/);
assert.match(html, /id="profile-avatar-file"/);
assert.match(html, /id="account-logout"/);
assert.match(app, /profilePanel\.classList\.toggle\("hidden", mode !== "profile"\)/);
assert.match(app, /requireLoginAtStart: true/);
assert.match(html, /id="knowledge-search-history-list"/);
assert.match(html, /id="knowledge-history-clear"/);
assert.match(html, /id="knowledge-selection-toolbar"/);
assert.match(html, /id="knowledge-add-favorites"/);
assert.match(html, /id="knowledge-add-today-plan"/);
assert.match(html, /id="app-toast"/);
assert.match(html, /id="knowledge-list-select"/);
assert.match(html, /id="knowledge-create-topic-list"/);
assert.match(app, /data-knowledge-entry-id/);
assert.match(app, /\/api\/learning\/batch-save/);
assert.match(app, /saveSelectedKnowledge\("plan"\)/);
assert.match(app, /requireLogin\("knowledge"\)/);
assert.match(css, /\.knowledge-selection-toolbar\{/);
assert.match(css, /\.book-source-select input/);
assert.match(app, /KNOWLEDGE_HISTORY_LIMIT = 20/);
assert.match(app, /KNOWLEDGE_HISTORY_TTL_MS = 24 \* 60 \* 60 \* 1000/);
assert.match(app, /showKnowledgeResult\(entry\.data, \{ cachedAt: entry\.savedAt \}\)/);
assert.match(app, /localStorage\.removeItem\(KNOWLEDGE_HISTORY_KEY\)/);
assert.match(html, /id="transcript-speech-status"/);
assert.match(communication, /className = "transcript-speak"/);
assert.match(communication, /renderTranscriptText\(editor, block\.text \|\| ""\)/);
assert.match(communication, /语音服务连接中断/);
assert.match(communication, /已确认的字幕已保留/);
assert.match(communication, /speechRetryDelay\("network"/);
assert.match(speechRetry, /MAX_SPEECH_NETWORK_RETRIES = RETRY_DELAYS_MS\.length/);
assert.match(app, /className = "book-source-body"/);
assert.match(app, /className = "book-source-tag"/);
assert.match(css, /\.book-source-body\{display:grid;grid-template-columns:minmax\(0,1fr\) minmax\(180px,\.72fr\)/);
assert.match(css, /\.ai-actions\{position:sticky/);
for (const [, id] of learning.matchAll(/\$\("#([a-z0-9-]+)"\)/g)) {
  assert.ok(ids.has(id), `learning client references missing #${id}`);
}
assert.match(html, /id="learning-panel"/);
assert.match(html, /data-learning-view="today"/);
assert.match(html, /id="learning-entry-page"/);
assert.match(html, /id="learning-detail-back"/);
assert.match(html, /id="learning-stat-detail-close"/);
assert.match(html, /data-learning-view="favorites"/);
assert.match(html, /id="learning-favorites-view"/);
assert.doesNotMatch(html, /本地演示数据|视频和词条数据由 data\/sign-entries\.json/);
assert.doesNotMatch(html, /id="lookup-aside"|id="aside-title"|id="aside-copy"|id="term-count"/);
assert.doesNotMatch(app, /词条由 Flask 本地接口提供|Atlas 向量检索/);
assert.match(learning, /function renderWordRow/);
assert.match(learning, /learning-history-word-preview/);
assert.match(learning, /learning-word-detail-button/);
assert.match(learning, /learning-word-remove/);
assert.match(learning, /function openEntryDetail/);
assert.match(learning, /function returnFromEntryDetail/);
assert.match(learning, /function loadFavorites/);
assert.match(learning, /\/api\/learning\/stats\/words\?category=learned/);
assert.match(learning, /今日新词/);
assert.match(learning, /今日复习/);
assert.match(learning, /\/api\/learning\/review/);
assert.match(learning, /\/api\/learning\/entries/);
assert.match(learning, /renderGroupedEntries\(favorites/);
assert.match(learning, /loadStatWords\(category/);
assert.match(learning, /masteryRequired/);
console.log("Frontend DOM contract checks passed");
