import { streamAssistant } from "./deepseek_stream.mjs";
import { getTranscriptSelection } from "./transcript_segments.mjs";
import { appendTranscriptText, migrateLegacySegments } from "./transcript_blocks.mjs";
import { authenticatedRequest } from "./auth_client.mjs";

const timeFormatter = new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
const fullDateFormatter = new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" });

let apiBase = "";
let records = [];
const persistedRecordIds = new Set();
const recordSaveChains = new Map();
let selectedRecordId = null;
let speechRecognition = null;
let shouldListen = false;
let listening = false;
let interimText = "";
let retryTimer = null;
let generationController = null;
let generationRecordId = null;
let persistTimer = null;
let selectedTranscriptText = "";
let selectedTextRecordId = null;
let selectedTextBlockId = null;
let preservingSelectionForAction = false;
let activeBlockId = null;

const el = (selector) => document.querySelector(selector);

async function getAllRecords() {
  const result = await authenticatedRequest("/api/communications");
  const loaded = Array.isArray(result.records) ? result.records : [];
  persistedRecordIds.clear();
  loaded.forEach((record) => persistedRecordIds.add(record.id));
  return loaded;
}

function saveRecord(record) {
  const previous = recordSaveChains.get(record.id) || Promise.resolve();
  const next = previous.catch(() => {}).then(async () => {
    const isPersisted = persistedRecordIds.has(record.id);
    const result = await authenticatedRequest(isPersisted ? `/api/communications/${encodeURIComponent(record.id)}` : "/api/communications", {
      method: isPersisted ? "PUT" : "POST",
      body: JSON.stringify(record),
    });
    if (result.record?.id) {
      record.id = result.record.id;
      persistedRecordIds.add(record.id);
    }
    return record;
  });
  recordSaveChains.set(record.id, next);
  next.finally(() => {
    if (recordSaveChains.get(record.id) === next) recordSaveChains.delete(record.id);
  }).catch(() => {});
  return next;
}

async function deleteRecordById(id) {
  await authenticatedRequest(`/api/communications/${encodeURIComponent(id)}`, { method: "DELETE", body: "{}" });
  persistedRecordIds.delete(id);
}

async function clearRecords() {
  await authenticatedRequest("/api/communications", { method: "DELETE", body: "{}" });
  persistedRecordIds.clear();
}

function selectedRecord() {
  return records.find((record) => record.id === selectedRecordId) || null;
}

function makeId() {
  return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newRecord() {
  const createdAt = new Date().toISOString();
  return {
    id: makeId(),
    title: `沟通 ${timeFormatter.format(new Date(createdAt))}`,
    createdAt,
    updatedAt: createdAt,
    blocks: [],
    summary: null,
    reply: null,
  };
}

function sortRecords() {
  records.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

function ensureRecordBlocks(record) {
  const wasLegacy = !Array.isArray(record.blocks);
  const blocks = migrateLegacySegments(record, makeId);
  if (wasLegacy) saveRecord(record).catch(showStorageError);
  return blocks;
}

function formatPreview(record) {
  const text = [...ensureRecordBlocks(record)].reverse()
    .map((block) => block.text.trim() || block.summary?.trim() || block.reply?.trim())
    .find(Boolean);
  return text || "暂无转写内容";
}

function renderRecordList() {
  const list = el("#communication-record-list");
  const query = el("#record-search").value.trim().toLocaleLowerCase();
  const visible = records.filter((record) => {
    const haystack = `${record.title} ${ensureRecordBlocks(record).map((block) => `${block.title} ${block.text} ${block.summary || ""} ${block.reply || ""}`).join(" ")}`.toLocaleLowerCase();
    return !query || haystack.includes(query);
  });
  const items = visible.map((record) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `record-item${record.id === selectedRecordId ? " active" : ""}`;
    button.dataset.recordId = record.id;
    const title = document.createElement("strong");
    title.textContent = record.title || "未命名沟通";
    const time = document.createElement("time");
    time.dateTime = record.updatedAt;
    time.textContent = timeFormatter.format(new Date(record.updatedAt));
    const preview = document.createElement("span");
    preview.textContent = formatPreview(record);
    button.append(title, time, preview);
    li.append(button);
    return li;
  });
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "record-empty";
    empty.textContent = query ? "没有匹配的记录" : "还没有沟通记录";
    items.push(empty);
  }
  list.replaceChildren(...items);
  el("#record-count").textContent = query ? `找到 ${visible.length} 条记录` : `${records.length} 条记录`;
}

function renderTranscript(record) {
  const transcript = el("#communication-transcript");
  const blocks = ensureRecordBlocks(record);
  if (!blocks.length) {
    const empty = document.createElement("p");
    empty.className = "transcript-empty";
    empty.textContent = "还没有语音框。点击“新增语音框”整理一个新主题，或直接开始收听。";
    transcript.replaceChildren(empty);
    renderTranscriptNavigation(record);
    return;
  }
  transcript.replaceChildren(...blocks.map((block, index) => makeBlockElement(block, index)));
  renderTranscriptNavigation(record);
}

function makeBlockElement(block, index) {
  const article = document.createElement("article");
  article.className = `transcript-block${block.id === activeBlockId ? " active" : ""}`;
  article.dataset.blockId = block.id;
  const header = document.createElement("header");
  header.className = "transcript-block-header";
  const title = document.createElement("input");
  title.className = "transcript-block-title";
  title.value = block.title || `语音记录 ${index + 1}`;
  title.maxLength = 60;
  title.setAttribute("aria-label", `语音框 ${index + 1} 的主题名称`);
  const time = document.createElement("time");
  time.dateTime = block.createdAt;
  time.textContent = timeFormatter.format(new Date(block.createdAt));
    const speak = document.createElement("button");
    speak.type = "button";
    speak.className = "transcript-speak";
    speak.setAttribute("aria-label", `朗读${block.title || `语音记录 ${index + 1}`}的转写内容`);
    speak.title = "朗读这段转写内容";
    speak.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 10v4h4l5 4V6l-5 4H4Z"/><path d="M16 9.5a4 4 0 0 1 0 5M18.5 7a7.5 7.5 0 0 1 0 10"/></svg>';
    header.append(title, speak, time);
  const editor = document.createElement("div");
  editor.className = "transcript-block-editor";
  editor.contentEditable = "true";
  editor.setAttribute("role", "textbox");
  editor.setAttribute("aria-multiline", "true");
  editor.setAttribute("aria-label", `语音框 ${index + 1} 转写内容，可编辑`);
  editor.dataset.blockId = block.id;
  renderTranscriptText(editor, block.text || "");
  editor.dataset.placeholder = "点击此框后开始收听，识别文字会放在这里";
  const notes = document.createElement("div");
  notes.className = "transcript-block-notes";
  for (const [field, title, placeholder] of [
    ["summary", "总结重点", "生成或编辑此语音框对应的沟通重点"],
    ["reply", "回复", "保存或编辑此主题的回复内容"],
  ]) {
    const group = document.createElement("label");
    group.className = "transcript-block-note";
    const caption = document.createElement("span");
    caption.textContent = title;
    const textarea = document.createElement("textarea");
    textarea.className = "transcript-block-note-editor";
    textarea.dataset.blockId = block.id;
    textarea.dataset.blockField = field;
    textarea.rows = field === "summary" ? 2 : 2;
    textarea.placeholder = placeholder;
    textarea.value = block[field] || "";
    group.append(caption, textarea);
    notes.append(group);
  }
  article.append(header, editor, notes);
  return article;
}

function renderTranscriptText(editor, text) {
  editor.replaceChildren();
  const sentences = text.match(/[^。！？!?；;]+[。！？!?；;]*|[。！？!?；;]+/gu) || [];
  const parts = sentences.map((part) => part.trim()).filter(Boolean);
  for (const part of parts) {
    const paragraph = document.createElement("div");
    paragraph.className = "transcript-paragraph";
    paragraph.textContent = part;
    editor.append(paragraph);
  }
}

function renderTranscriptNavigation(record) {
  const map = el("#transcript-map");
  const blocks = ensureRecordBlocks(record);
  map.replaceChildren(...blocks.map((block, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const preview = block.text.trim().replace(/\s+/g, " ") || block.summary?.trim().replace(/\s+/g, " ") || block.reply?.trim().replace(/\s+/g, " ") || "暂无转写内容";
    const shortPreview = [...preview].slice(0, 96).join("") + (preview.length > 96 ? "…" : "");
    button.type = "button";
    button.className = `transcript-map-marker${block.id === activeBlockId ? " active" : ""}`;
    button.dataset.blockId = block.id;
    button.dataset.preview = shortPreview;
    button.title = `${block.title || `语音记录 ${index + 1}`}：${preview}`;
    button.setAttribute("aria-label", `跳转到${block.title || `语音记录 ${index + 1}`}：${preview.slice(0, 80)}`);
    item.append(button);
    return item;
  }));
}

function setOutput(outputName, value, status = "") {
  const summary = outputName === "summary";
  const section = el(summary ? "#summary-output" : "#reply-output");
  const textarea = el(summary ? "#summary-text" : "#reply-text");
  const state = el(summary ? "#summary-state" : "#reply-state");
  section.classList.toggle("hidden", !value && !status);
  textarea.value = value || "";
  state.textContent = status;
}

function setOutputBlockTarget(outputName, record, blockId) {
  const button = el(outputName === "summary" ? "#summary-save-to-block" : "#reply-save-to-block");
  const block = ensureRecordBlocks(record).find((item) => item.id === blockId);
  button.dataset.blockId = block?.id || "";
  button.textContent = block ? `存入「${block.title || "当前语音框"}」` : "存入语音框";
  button.disabled = !block;
}

function renderBlockOutputs(record, blockId) {
  const block = ensureRecordBlocks(record).find((item) => item.id === blockId);
  setOutput("summary", block?.summary || "", block?.summary ? "已保存在此语音框" : "");
  setOutput("reply", block?.reply || "", block?.reply ? "已保存在此语音框" : "");
  setOutputBlockTarget("summary", record, blockId);
  setOutputBlockTarget("reply", record, blockId);
}

function renderSelectedRecord() {
  const record = selectedRecord();
  const empty = el("#communication-empty");
  const detail = el("#communication-record-layout");
  empty.classList.toggle("hidden", Boolean(record));
  detail.classList.toggle("hidden", !record);
  renderRecordList();
  if (!record) return;
  el("#communication-title").value = record.title;
  el("#communication-timestamp").textContent = `创建于 ${fullDateFormatter.format(new Date(record.createdAt))}`;
  el("#speech-interim").classList.toggle("hidden", !interimText);
  el("#speech-interim").textContent = interimText ? `${interimText}（识别中）` : "";
  selectedTranscriptText = "";
  selectedTextRecordId = record?.id || null;
  selectedTextBlockId = null;
  const blocks = record ? ensureRecordBlocks(record) : [];
  if (!blocks.some((block) => block.id === activeBlockId)) activeBlockId = blocks[0]?.id || null;
  renderBlockOutputs(record, activeBlockId);
  if (record) renderTranscript(record);
  updateTranscriptSelection();
}

async function refreshRecords() {
  records = await getAllRecords();
  sortRecords();
  if (!records.some((record) => record.id === selectedRecordId)) selectedRecordId = records[0]?.id || null;
  renderSelectedRecord();
}

async function persistCurrentRecord() {
  const record = selectedRecord();
  if (!record) return;
  record.updatedAt = new Date().toISOString();
  await saveRecord(record);
  sortRecords();
  renderRecordList();
}

function queuePersist(recordId = selectedRecordId) {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(async () => {
    const record = records.find((item) => item.id === recordId);
    if (!record) return;
    record.updatedAt = new Date().toISOString();
    try {
      await saveRecord(record);
      sortRecords();
      renderRecordList();
    } catch (error) {
      showStorageError(error);
    }
  }, 250);
}

function showStorageError(error) {
  el("#speech-status").textContent = error.message || "Atlas 云端记录暂时无法保存";
}

function setSpeechControls(state) {
  const active = state === "listening";
  const paused = state === "paused";
  const ended = state === "ended";
  el("#speech-status").classList.toggle("is-active", active);
  const activeBlock = ensureRecordBlocks(selectedRecord() || { blocks: [] }).find((block) => block.id === activeBlockId);
  el("#speech-status").textContent = active ? `正在收听 · ${activeBlock?.title || "语音框"}` : paused ? "已暂停" : ended ? "已结束" : state === "idle" ? "麦克风未开启" : state;
  el("#communication-start").classList.toggle("hidden", active || paused);
  el("#communication-pause").classList.toggle("hidden", !active);
  el("#communication-resume").classList.toggle("hidden", !paused);
  el("#communication-stop").classList.toggle("hidden", ended || (!active && !paused));
  el("#communication-start-empty").disabled = active;
}

function renderInterim(text = "") {
  interimText = text.trim();
  const interim = el("#speech-interim");
  interim.classList.toggle("hidden", !interimText);
  const block = ensureRecordBlocks(selectedRecord() || { blocks: [] }).find((item) => item.id === activeBlockId);
  interim.textContent = interimText ? `${interimText}（识别中 · 将放入${block?.title || "当前语音框"}）` : "";
}

async function appendFinalTranscript(text) {
  const clean = text.trim();
  const record = selectedRecord();
  if (!clean || !record) return;
  const block = ensureRecordBlocks(record).find((item) => item.id === activeBlockId);
  if (!block) return;
  const current = block.text;
  block.text = appendTranscriptText(current, clean);
  record.updatedAt = new Date().toISOString();
  await saveRecord(record);
  sortRecords();
  if (selectedRecordId === record.id) {
    const editor = [...el("#communication-transcript").querySelectorAll(".transcript-block-editor")]
      .find((item) => item.dataset.blockId === block.id);
    if (editor) {
      renderTranscriptText(editor, block.text);
    }
    renderTranscriptNavigation(record);
  }
  renderRecordList();
  const transcript = el("#communication-transcript");
  if (transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 48) {
    transcript.scrollTop = transcript.scrollHeight;
  }
}

function ensureSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) throw new Error("此 Edge 环境不支持 Web Speech 语音识别，请更新浏览器后重试");
  speechRecognition = new SpeechRecognition();
  speechRecognition.lang = "zh-CN";
  speechRecognition.continuous = true;
  speechRecognition.interimResults = true;
  speechRecognition.maxAlternatives = 1;
  speechRecognition.onresult = (event) => {
    const interim = [];
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const transcript = result[0]?.transcript || "";
      if (result.isFinal) appendFinalTranscript(transcript).catch(showStorageError);
      else interim.push(transcript);
    }
    const pending = [];
    for (let index = 0; index < event.results.length; index += 1) {
      if (!event.results[index].isFinal) pending.push(event.results[index][0]?.transcript || "");
    }
    renderInterim(pending.length ? pending.join(" ") : interim.join(" "));
  };
  speechRecognition.onerror = (event) => {
    const messages = {
      "not-allowed": "麦克风权限未获允许",
      "service-not-allowed": "浏览器语音识别服务不可用",
      network: "语音识别网络连接失败",
      "audio-capture": "没有检测到可用麦克风",
      "no-speech": "暂未检测到语音，可继续收听",
    };
    if (["not-allowed", "service-not-allowed", "audio-capture"].includes(event.error)) {
      shouldListen = false;
      listening = false;
      renderInterim("");
      setSpeechControls("ended");
    }
    if (event.error === "network") {
      shouldListen = false;
      listening = false;
      setSpeechControls("ended");
    }
    el("#speech-status").textContent = messages[event.error] || `语音识别暂不可用（${event.error}）`;
  };
  speechRecognition.onend = () => {
    listening = false;
    if (shouldListen) {
      clearTimeout(retryTimer);
      retryTimer = setTimeout(() => {
        if (!shouldListen || !speechRecognition) return;
        try {
          speechRecognition.start();
          listening = true;
          setSpeechControls("listening");
        } catch {
          shouldListen = false;
          setSpeechControls("ended");
          el("#speech-status").textContent = "语音识别已中断，请重新开始收听";
        }
      }, 300);
    } else {
      renderInterim("");
    }
  };
}

async function startListening() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    setSpeechControls("此 Edge 环境不支持 Web Speech API");
    return;
  }
  if (!selectedRecordId) {
    const record = newRecord();
    records.unshift(record);
    selectedRecordId = record.id;
    renderSelectedRecord();
    saveRecord(record).catch(showStorageError);
  }
  const record = selectedRecord();
  const blocks = ensureRecordBlocks(record);
  if (!blocks.some((block) => block.id === activeBlockId)) {
    createTranscriptBlock(record, false);
  }
  if (!speechRecognition) ensureSpeechRecognition();
  clearTimeout(retryTimer);
  shouldListen = true;
  try {
    speechRecognition.start();
    listening = true;
    setSpeechControls("listening");
  } catch (error) {
    shouldListen = false;
    listening = false;
    el("#speech-status").textContent = error.name === "NotAllowedError" ? "麦克风权限未获允许" : "无法启动语音识别，请稍后重试";
  }
}

function pauseListening() {
  shouldListen = false;
  clearTimeout(retryTimer);
  if (speechRecognition && listening) speechRecognition.stop();
  listening = false;
  renderInterim("");
  setSpeechControls("paused");
}

function finishListening() {
  shouldListen = false;
  clearTimeout(retryTimer);
  if (speechRecognition && listening) speechRecognition.stop();
  listening = false;
  renderInterim("");
  setSpeechControls("ended");
}

export function stopCommunicationListening() {
  if (!shouldListen && !listening) return;
  shouldListen = false;
  clearTimeout(retryTimer);
  try { speechRecognition?.stop(); } catch { /* Recognition may already be stopped. */ }
  listening = false;
  renderInterim("");
  if (el("#speech-status")) setSpeechControls("paused");
}

async function createCommunication() {
  if (shouldListen || listening) finishListening();
  const record = newRecord();
  records.unshift(record);
  selectedRecordId = record.id;
  activeBlockId = null;
  await saveRecord(record);
  renderSelectedRecord();
  setSpeechControls("idle");
  el("#communication-title").focus();
}

function createTranscriptBlock(record, shouldFocus = true) {
  const blocks = ensureRecordBlocks(record);
  const block = {
    id: makeId(),
    title: `语音记录 ${blocks.length + 1}`,
    createdAt: new Date().toISOString(),
    text: "",
  };
  blocks.push(block);
  activeBlockId = block.id;
  record.updatedAt = new Date().toISOString();
  renderBlockOutputs(record, block.id);
  renderTranscript(record);
  queuePersist(record.id);
  if (shouldListen || listening) {
    setSpeechControls("listening");
    renderInterim(interimText);
  }
  if (shouldFocus) {
    const editor = [...el("#communication-transcript").querySelectorAll(".transcript-block-editor")]
      .find((item) => item.dataset.blockId === block.id);
    editor?.focus();
  }
  return block;
}

function updateTranscriptBlock(event) {
  const record = selectedRecord();
  if (!record) return;
  const blockId = event.target.closest("[data-block-id]")?.dataset.blockId;
  const block = ensureRecordBlocks(record).find((item) => item.id === blockId);
  if (!block) return;
  if (event.target.matches(".transcript-block-title")) block.title = event.target.value.trim() || "未命名主题";
  else if (event.target.matches(".transcript-block-editor")) block.text = event.target.innerText.replace(/\r\n/g, "\n");
  else if (event.target.matches(".transcript-block-note-editor")) block[event.target.dataset.blockField] = event.target.value;
  renderTranscriptNavigation(record);
  queuePersist(record.id);
}

function activateTranscriptBlock(event) {
  const article = event.target.closest(".transcript-block[data-block-id]");
  if (!article || article.dataset.blockId === activeBlockId) return;
  activeBlockId = article.dataset.blockId;
  renderBlockOutputs(selectedRecord(), activeBlockId);
  el("#communication-transcript").querySelectorAll(".transcript-block").forEach((block) => {
    block.classList.toggle("active", block.dataset.blockId === activeBlockId);
  });
  renderTranscriptNavigation(selectedRecord());
  if (shouldListen || listening) setSpeechControls("listening");
  if (interimText) renderInterim(interimText);
}

function updateTranscriptSelection() {
  const record = selectedRecord();
  const transcript = el("#communication-transcript");
  const selection = window.getSelection();
  const selected = getTranscriptSelection(selection, transcript);
  const selectionRange = selection?.rangeCount ? selection.getRangeAt(0) : null;
  const startElement = selectionRange?.startContainer.nodeType === Node.ELEMENT_NODE
    ? selectionRange.startContainer
    : selectionRange?.startContainer.parentElement;
  const endElement = selectionRange?.endContainer.nodeType === Node.ELEMENT_NODE
    ? selectionRange.endContainer
    : selectionRange?.endContainer.parentElement;
  const selectedBlock = startElement?.closest(".transcript-block-editor");
  const sameBlock = selectedBlock && selectedBlock === endElement?.closest(".transcript-block-editor");
  if (selected && record && sameBlock) {
    selectedTranscriptText = selected;
    selectedTextRecordId = record.id;
    selectedTextBlockId = selectedBlock.dataset.blockId;
  } else if (!preservingSelectionForAction) {
    selectedTranscriptText = "";
    selectedTextRecordId = record?.id || null;
    selectedTextBlockId = null;
  }
  if (selectedTextRecordId !== record?.id) selectedTranscriptText = "";
  const count = [...selectedTranscriptText].length;
  el("#ai-selection-status").textContent = count
    ? `已选择 ${count} 个字，AI 只会处理这段字幕。`
    : "在一个语音框中选择要处理的文字；所选内容会发送给 DeepSeek。";
  el("#communication-summarize").disabled = !count || Boolean(generationController);
  el("#communication-reply").disabled = !count || Boolean(generationController);
}

async function generateWithDeepSeek(action) {
  const record = selectedRecord();
  if (!record) return;
  const text = selectedTextRecordId === record.id ? selectedTranscriptText.trim() : "";
  if (!text) {
    el("#ai-selection-status").textContent = "请先在字幕中选择要处理的文字。";
    return;
  }
  if (generationController) return;
  const controller = new AbortController();
  generationController = controller;
  generationRecordId = record.id;
  const outputName = action === "summarize" ? "summary" : "reply";
  const section = el(outputName === "summary" ? "#summary-output" : "#reply-output");
  const textarea = el(outputName === "summary" ? "#summary-text" : "#reply-text");
  const state = el(outputName === "summary" ? "#summary-state" : "#reply-state");
  const targetBlockId = selectedTextBlockId;
  setOutputBlockTarget(outputName, record, targetBlockId);
  section.classList.remove("hidden");
  textarea.value = "";
  state.textContent = "正在连接 DeepSeek…";
  el("#communication-summarize").disabled = true;
  el("#communication-reply").disabled = true;
  el("#communication-cancel-ai").classList.remove("hidden");
  let output = "";
  let pendingOutput = "";
  let outputFrame = 0;
  let firstTokenReceived = false;
  const requestStartedAt = performance.now();
  const flushOutput = () => {
    if (outputFrame) cancelAnimationFrame(outputFrame);
    outputFrame = 0;
    if (selectedRecordId === record.id && pendingOutput) {
      textarea.value += pendingOutput;
      pendingOutput = "";
    }
  };
  try {
    await streamAssistant({
      url: `${apiBase}/api/assistant/stream`,
      action,
      text,
      signal: controller.signal,
      onMeta: (message) => {
        if (selectedRecordId === record.id) state.textContent = message;
      },
      onToken: (chunk) => {
        if (!firstTokenReceived && chunk) {
          firstTokenReceived = true;
          if (selectedRecordId === record.id) {
            state.textContent = `首段已到，正在逐段接收（${Math.round(performance.now() - requestStartedAt)} 毫秒）`;
          }
        }
        output += chunk;
        pendingOutput += chunk;
        if (!outputFrame) outputFrame = requestAnimationFrame(flushOutput);
      },
    });
    flushOutput();
    if (selectedRecordId === record.id) state.textContent = "已完成，可编辑";
  } catch (error) {
    if (error.name === "AbortError") {
      if (selectedRecordId === record.id) state.textContent = "生成已停止，内容未完成";
    } else {
      if (selectedRecordId === record.id) {
        state.textContent = "";
        section.classList.toggle("hidden", !output);
      }
      if (output) {
        if (selectedRecordId === record.id) state.textContent = "连接中断，内容未完成";
      }
      if (selectedRecordId === record.id) el("#speech-status").textContent = error.message || "DeepSeek 请求失败，请重试";
    }
  } finally {
    if (outputFrame) cancelAnimationFrame(outputFrame);
    flushOutput();
    if (generationController === controller) {
      generationController = null;
      generationRecordId = null;
      updateTranscriptSelection();
      el("#communication-cancel-ai").classList.add("hidden");
    }
    renderRecordList();
  }
}

function cancelGeneration() {
  if (!generationController) return;
  generationController.abort();
}

function persistAiEdit(outputName, value) {
  const state = el(outputName === "summary" ? "#summary-state" : "#reply-state");
  state.textContent = value.trim() ? "已编辑，点击存入语音框后保存" : "";
}

async function saveOutputToBlock(outputName) {
  const record = selectedRecord();
  const saveButton = el(outputName === "summary" ? "#summary-save-to-block" : "#reply-save-to-block");
  const sourceBlockId = saveButton.dataset.blockId;
  const block = record && ensureRecordBlocks(record).find((item) => item.id === sourceBlockId);
  const value = el(outputName === "summary" ? "#summary-text" : "#reply-text").value.trim();
  if (!block || !value) return;
  block[outputName] = value;
  record.updatedAt = new Date().toISOString();
  const target = [...el("#communication-transcript").querySelectorAll(".transcript-block-note-editor")]
    .find((item) => item.dataset.blockId === block.id && item.dataset.blockField === outputName);
  if (target) target.value = value;
  saveButton.disabled = true;
  try {
    await saveRecord(record);
    saveButton.textContent = "已存入本框";
    el(outputName === "summary" ? "#summary-state" : "#reply-state").textContent = "已保存在此语音框";
    sortRecords();
    renderRecordList();
  } catch (error) {
    showStorageError(error);
    saveButton.textContent = "保存失败，请重试";
  } finally {
    saveButton.disabled = false;
  }
}

function copyReply() {
  const text = el("#reply-text").value.trim();
  if (!text) return;
  navigator.clipboard?.writeText(text).then(() => {
    el("#reply-state").textContent = "已复制";
  }).catch(() => {
    el("#reply-state").textContent = "复制失败，请手动选择文字";
  });
}

export async function initializeCommunication(options = {}) {
  apiBase = options.apiBase || "";
  el("#communication-new").addEventListener("click", () => createCommunication().catch(showStorageError));
  el("#communication-start-empty").addEventListener("click", () => startListening().catch(showStorageError));
  el("#communication-add-block").addEventListener("click", () => {
    const record = selectedRecord();
    if (record) createTranscriptBlock(record);
  });
  el("#communication-start").addEventListener("click", () => startListening().catch(showStorageError));
  el("#communication-resume").addEventListener("click", () => startListening().catch(showStorageError));
  el("#communication-pause").addEventListener("click", pauseListening);
  el("#communication-stop").addEventListener("click", finishListening);
  el("#communication-delete").addEventListener("click", async () => {
    const record = selectedRecord();
    if (!record || !window.confirm(`删除“${record.title}”？此操作无法撤销。`)) return;
    if (generationRecordId === record.id) cancelGeneration();
    if (shouldListen || listening) finishListening();
    try {
      await deleteRecordById(record.id);
      records = records.filter((item) => item.id !== record.id);
      selectedRecordId = records[0]?.id || null;
      renderSelectedRecord();
      setSpeechControls("idle");
    } catch (error) {
      showStorageError(error);
    }
  });
  el("#communication-clear-all").addEventListener("click", async () => {
    if (!records.length || !window.confirm("清空当前账号的全部沟通记录？此操作无法撤销。")) return;
    if (generationController) cancelGeneration();
    if (shouldListen || listening) finishListening();
    try {
      await clearRecords();
      records = [];
      selectedRecordId = null;
      renderSelectedRecord();
      setSpeechControls("idle");
    } catch (error) {
      showStorageError(error);
    }
  });
  el("#record-search").addEventListener("input", renderRecordList);
  el("#communication-record-list").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-record-id]");
    if (!button || button.dataset.recordId === selectedRecordId) return;
    if (shouldListen || listening) finishListening();
    if (generationController) cancelGeneration();
    selectedRecordId = button.dataset.recordId;
    selectedTranscriptText = "";
    selectedTextRecordId = selectedRecordId;
    selectedTextBlockId = null;
    renderSelectedRecord();
    setSpeechControls("idle");
  });
  el("#communication-transcript").addEventListener("input", updateTranscriptBlock);
  el("#communication-transcript").addEventListener("click", activateTranscriptBlock);
  el("#communication-transcript").addEventListener("focusin", activateTranscriptBlock);
  el("#communication-transcript").addEventListener("blur", () => persistCurrentRecord().catch(showStorageError));
  document.addEventListener("selectionchange", updateTranscriptSelection);
  el("#transcript-map").addEventListener("click", (event) => {
    const marker = event.target.closest("[data-block-id]");
    if (!marker) return;
    const block = [...el("#communication-transcript").querySelectorAll(".transcript-block")]
      .find((item) => item.dataset.blockId === marker.dataset.blockId);
    if (!block) return;
    activeBlockId = block.dataset.blockId;
    renderBlockOutputs(selectedRecord(), activeBlockId);
    block.classList.add("transcript-block-current", "active");
    el("#communication-transcript").querySelectorAll(".transcript-block").forEach((item) => {
      item.classList.toggle("active", item.dataset.blockId === activeBlockId);
    });
    block.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
    renderTranscriptNavigation(selectedRecord());
    setTimeout(() => block.classList.remove("transcript-block-current"), 1400);
    if (shouldListen || listening) setSpeechControls("listening");
  });
  el("#communication-title").addEventListener("input", (event) => {
    const record = selectedRecord();
    if (!record) return;
    record.title = event.target.value;
    queuePersist();
  });
  el("#communication-title").addEventListener("blur", () => persistCurrentRecord().catch(showStorageError));
  el("#communication-summarize").addEventListener("click", () => generateWithDeepSeek("summarize"));
  el("#communication-reply").addEventListener("click", () => generateWithDeepSeek("reply"));
  for (const button of [el("#communication-summarize"), el("#communication-reply")]) {
    button.addEventListener("pointerdown", () => { preservingSelectionForAction = true; });
    button.addEventListener("focus", () => { preservingSelectionForAction = true; });
    button.addEventListener("click", () => {
      setTimeout(() => { preservingSelectionForAction = false; }, 0);
    });
  }
  el("#communication-cancel-ai").addEventListener("click", cancelGeneration);
  el("#summary-text").addEventListener("input", (event) => persistAiEdit("summary", event.target.value));
  el("#reply-text").addEventListener("input", (event) => persistAiEdit("reply", event.target.value));
  el("#summary-save-to-block").addEventListener("click", () => saveOutputToBlock("summary"));
  el("#reply-save-to-block").addEventListener("click", () => saveOutputToBlock("reply"));
  el("#reply-copy").addEventListener("click", copyReply);
  el("#history-assistant-open").addEventListener("click", () => {
    el("#communication-record-layout").classList.add("is-history-assistant");
    el("#history-assistant-panel").classList.remove("hidden");
    loadHistoryChats().catch((error) => { el("#history-assistant-status").textContent = error.message; });
    el("#history-assistant-question").focus();
  });
  el("#history-assistant-close").addEventListener("click", () => {
    el("#communication-record-layout").classList.remove("is-history-assistant");
    el("#history-assistant-panel").classList.add("hidden");
  });
  el("#history-assistant-form").addEventListener("submit", askHistory);
  el("#history-assistant-chat-list").addEventListener("click", (event) => {
    const item = event.target.closest("[data-history-chat]");
    if (!item) return;
    const chat = historyChats.find((entry) => entry.id === item.dataset.historyChat);
    if (chat) renderHistoryAnswer(chat.answer, chat.sources || []);
  });
  el("#history-assistant-sources").addEventListener("click", (event) => {
    const button = event.target.closest("[data-conversation-id]");
    if (button) jumpToHistorySource(button.dataset.conversationId, button.dataset.blockId);
  });
  document.querySelectorAll("[data-quick-phrase]").forEach((button) => button.addEventListener("click", () => {
    el("#direct-reply-text").value = button.dataset.quickPhrase;
    el("#direct-reply-text").focus();
  }));
  el("#direct-reply-display").addEventListener("click", () => {
    const value = el("#direct-reply-text").value.trim();
    if (!value) return;
    el("#response-display-text").textContent = value;
    el("#response-display").classList.remove("hidden");
  });
  el("#response-display-close").addEventListener("click", () => el("#response-display").classList.add("hidden"));
  try {
    await refreshRecords();
    setSpeechControls("idle");
  } catch (error) {
    el("#record-count").textContent = "云端记录暂不可用";
    el("#speech-status").textContent = error.message;
  }
}

let historyChats = [];

async function loadHistoryChats() {
  const result = await authenticatedRequest("/api/history/chats");
  historyChats = result.chats || [];
  const list = el("#history-assistant-chat-list");
  list.replaceChildren(...historyChats.map((chat) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.historyChat = chat.id;
    button.textContent = chat.query;
    item.append(button);
    return item;
  }));
}

function renderHistoryAnswer(answer, sources) {
  const output = el("#history-assistant-answer");
  output.textContent = answer || "";
  output.classList.toggle("hidden", !answer);
  const list = el("#history-assistant-sources");
  list.replaceChildren(...sources.map((source) => {
    const item = document.createElement("li");
    item.className = "history-source";
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.conversationId = source.conversationId;
    button.dataset.blockId = source.blockId;
    button.textContent = `${source.conversationTitle} · ${source.blockTitle}`;
    const text = document.createElement("p");
    text.textContent = source.originalText || "";
    const time = document.createElement("time");
    time.textContent = source.createdAt ? fullDateFormatter.format(new Date(source.createdAt)) : "";
    item.append(button, text, time);
    return item;
  }));
}

async function askHistory(event) {
  event.preventDefault();
  const question = el("#history-assistant-question").value.trim();
  if (!question) return;
  const submit = el("#history-assistant-submit");
  submit.disabled = true;
  el("#history-assistant-status").textContent = "正在检索当前账号的记录…";
  renderHistoryAnswer("", []);
  let answer = "";
  try {
    await streamAssistant({
      url: `${apiBase}/api/history/ask`,
      payload: { question },
      onMeta: (message) => { el("#history-assistant-status").textContent = message; },
      onSources: (sources) => {
        currentHistorySources = sources;
        renderHistoryAnswer("", sources);
      },
      onToken: (text) => {
        answer += text;
        renderHistoryAnswer(answer, currentHistorySources);
      },
    });
    el("#history-assistant-status").textContent = "回答完成";
    await loadHistoryChats();
  } catch (error) {
    el("#history-assistant-status").textContent = error.message;
  } finally {
    submit.disabled = false;
  }
}

let currentHistorySources = [];

async function jumpToHistorySource(conversationId, blockId) {
  if (shouldListen || listening) finishListening();
  if (generationController) cancelGeneration();
  let record = records.find((item) => item.id === conversationId);
  if (!record) {
    await refreshRecords();
    record = records.find((item) => item.id === conversationId);
  }
  if (!record) {
    el("#history-assistant-status").textContent = "来源记录已不存在。";
    return;
  }
  selectedRecordId = record.id;
  activeBlockId = blockId;
  el("#communication-record-layout").classList.remove("is-history-assistant");
  el("#history-assistant-panel").classList.add("hidden");
  renderSelectedRecord();
  setSpeechControls("idle");
  requestAnimationFrame(() => {
    const block = [...el("#communication-transcript").querySelectorAll(".transcript-block")]
      .find((item) => item.dataset.blockId === blockId);
    block?.scrollIntoView({ behavior: "smooth", block: "center" });
    block?.classList.add("transcript-block-current");
    if (block) setTimeout(() => block.classList.remove("transcript-block-current"), 1400);
  });
}
