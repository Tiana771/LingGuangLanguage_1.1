import { FRAME_COUNT, frameFeatures, taskHandednessToTrainingLabel, validFrameCount } from "./camera_features.mjs";
import { PredictionSmoother } from "./prediction_smoother.mjs";
import { RecognitionSession } from "./recognition_session.mjs";
import { initializeCommunication, stopCommunicationListening } from "./communication_assistant.mjs";
import { initializeSpeechOutput } from "./speech_output.mjs";
import { initializeAuth } from "./auth_client.mjs";
import { initializeLearning } from "./learning_client.mjs";

const modes = {
  recognize: { title: "手语识别", description: "在画面中比划，查看系统识别到的手语词汇。" },
  video: { title: "查手语视频", description: "输入词语查找对应的手语示范；尚未收录的表达会明确提示。" },
  knowledge: { title: "问手语知识", description: "检索国家通用手语词典的已审核内容，展示原文、书中图片和出处。" },
  communication: { title: "沟通助手", description: "实时查看口语转写，保存沟通记录，并按需整理重点或生成回复建议。" },
  learning: { title: "手语学习", description: "按自己的节奏学习已审核词条，并用复习反馈安排下一次复习。" },
  profile: { title: "个人中心", description: "管理个人资料、偏好设置和账号安全。" },
};

const navItems = [...document.querySelectorAll(".nav-item")];
const modeButtons = [...document.querySelectorAll(".mode-button")];
const recognizePanel = document.querySelector("#recognize-panel");
const lookupPanel = document.querySelector("#lookup-panel");
const communicationPanel = document.querySelector("#communication-panel");
const profilePanel = document.querySelector("#profile-panel");
const learningPanel = document.querySelector("#learning-panel");
const termList = document.querySelector("#term-list");
const input = document.querySelector("#query-input");
const feedback = document.querySelector("#query-feedback");
const result = document.querySelector("#lookup-result");
const API_BASE = window.SIGN_API_BASE || ( ["8000", "8080"].includes(location.port)
  ? `${location.protocol}//${location.hostname}:5001`
  : location.origin);
const KNOWLEDGE_HISTORY_KEY = "linguang.knowledge-search.v1";
const KNOWLEDGE_HISTORY_LIMIT = 20;
const KNOWLEDGE_HISTORY_TTL_MS = 24 * 60 * 60 * 1000;
let knowledgeSources = [];
let knowledgeListsLoaded = false;
let knowledgeLists = [];
let appToastTimer = null;
let currentMode = "recognize";
let cameraStream;
let recognition;
let entries = [];
let handLandmarker;
let handConnections = [];
let inferenceCanvas;
let inferenceContext;
let overlayContext;
let animationFrame = 0;
let lastFrameAt = 0;
let lastPredictionAt = 0;
let predictionInFlight = false;
let cameraSession = 0;
let predictionGeneration = 0;
let frameBuffer = [];
let stableLabel = null;
let noHandWindowCleared = false;
let activePredictionController = null;

function showAppToast(message) {
  const toast = document.querySelector("#app-toast");
  if (!toast || !message) return;
  clearTimeout(appToastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");
  appToastTimer = setTimeout(() => toast.classList.remove("is-visible"), 2800);
}

window.linguangToast = showAppToast;
let lastRecognitionRenderAt = 0;
const predictionSmoother = new PredictionSmoother({ windowSize: 3, majority: 2, lockMs: 1500 });
const recognitionSession = new RecognitionSession({ noHandResetMs: 500, maxHistory: 30 });
const CAMERA_FRAME_INTERVAL_MS = 40;
const PREDICTION_INTERVAL_MS = 500;
let MIN_VALID_HAND_FRAMES = null;

function setMode(mode) {
  if (mode !== "recognize" && cameraStream) stopCamera();
  if (mode !== "communication") stopCommunicationListening();
  currentMode = mode;
  document.querySelector(".app-shell").classList.toggle("recognize-mode", mode === "recognize");
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  modeButtons.forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  const isRecognize = mode === "recognize";
  const isLookup = mode === "video" || mode === "knowledge";
  recognizePanel.classList.toggle("hidden", !isRecognize);
  lookupPanel.classList.toggle("hidden", !isLookup);
  document.querySelector("#book-knowledge-panel").classList.toggle("hidden", mode !== "knowledge");
  document.querySelector("#lookup-mode-switch").classList.toggle("hidden", mode === "knowledge");
  document.querySelector(".lookup-main").classList.toggle("hidden", mode === "knowledge");
  communicationPanel.classList.toggle("hidden", mode !== "communication");
  profilePanel.classList.toggle("hidden", mode !== "profile");
  learningPanel.classList.toggle("hidden", mode !== "learning");
  document.querySelector("#page-title").textContent = modes[mode].title;
  document.querySelector("#page-description").textContent = modes[mode].description;
  document.querySelector("#breadcrumb-current").textContent = modes[mode].title;
  document.querySelector("#page-eyebrow-detail").textContent = mode === "communication" ? "实时字幕与记录" : mode === "profile" ? "账号与偏好" : mode === "learning" ? "个人学习计划" : "限定词表";
  document.querySelector("#result-label").textContent = mode === "video" ? "词条匹配" : "资料检索";
  input.placeholder = mode === "video" ? "输入一个词语，例如：谢谢" : "例如：谢谢适合什么场合用？";
  document.querySelector("#query-hint").textContent = mode === "video"
    ? "语音会先转成可编辑文字，再由你确认查询。"
    : "检索仅覆盖演示词条与别名；无审核依据时会明确告知未收录。";
  feedback.textContent = "";
  if (isLookup && result.querySelector(".lookup-empty")) input.focus();
  if (mode === "knowledge") loadKnowledgeStatus();
  if (mode === "learning") window.linguangLearning?.open();
}

navItems.forEach((item) => item.addEventListener("click", () => {
  if (!window.linguangAuth?.requireLogin(item.dataset.mode)) return;
  setMode(item.dataset.mode);
}));
modeButtons.forEach((item) => item.addEventListener("click", () => {
  if (!window.linguangAuth?.requireLogin(item.dataset.mode)) return;
  setMode(item.dataset.mode);
}));

const fontSizeSelect = document.querySelector("#app-font-size");
const FONT_SIZE_STORAGE_KEY = "linguang-app-font-size";
function applyAppFontSize(value) {
  const fontSize = Number(value);
  if (!Number.isInteger(fontSize) || fontSize < 12 || fontSize > 24 || fontSize % 2 !== 0) return false;
  document.body.style.setProperty("--app-font-scale", String(fontSize / 14));
  document.querySelector("#font-size-preview").style.fontSize = `${fontSize}px`;
  fontSizeSelect.value = String(fontSize);
  return true;
}

let storedFontSize = "14";
try {
  const savedPreference = localStorage.getItem(FONT_SIZE_STORAGE_KEY);
  storedFontSize = ({ small: "12", medium: "14", large: "16" })[savedPreference] || savedPreference || "14";
} catch {
  // Use the default size when browser storage is unavailable.
}
if (!applyAppFontSize(storedFontSize)) storedFontSize = "14";
applyAppFontSize(storedFontSize);
try {
  localStorage.setItem(FONT_SIZE_STORAGE_KEY, storedFontSize);
} catch {
  // Keep the selected size for this page session.
}
fontSizeSelect.addEventListener("change", () => {
  applyAppFontSize(fontSizeSelect.value);
  try {
    localStorage.setItem(FONT_SIZE_STORAGE_KEY, fontSizeSelect.value);
  } catch {
    // Keep the selected size for this page session.
  }
});
termList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-term]");
  if (!button) return;
  input.value = button.dataset.term;
  runLookup(button.dataset.term);
});

async function loadTerms() {
  try {
    const response = await fetch(`${API_BASE}/api/terms`);
    if (!response.ok) throw new Error("词条服务暂不可用");
    const payload = await response.json();
    entries = payload.terms;
    termList.replaceChildren(...entries.map((entry) => {
      const button = document.createElement("button");
      button.className = "term-chip";
      button.type = "button";
      button.dataset.term = entry.term;
      button.textContent = entry.term;
      return button;
    }));
  } catch (error) {
    feedback.textContent = `${error.message}，请确认 Flask 服务已启动；文字输入仍可保留。`;
  }
}

async function checkModel() {
  const title = document.querySelector("#model-status-title");
  const status = document.querySelector(".model-status");
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const payload = await response.json();
    if (!response.ok || !payload.model_loaded) throw new Error(payload.error || "模型尚未加载");
    if (!payload.camera_features_ready) throw new Error("本地 MediaPipe 运行库或 hand_landmarker.task 未就绪");
    MIN_VALID_HAND_FRAMES = Number(payload.min_selected_frames);
    if (!Number.isInteger(MIN_VALID_HAND_FRAMES) || MIN_VALID_HAND_FRAMES < 1) throw new Error("服务端有效帧门槛配置无效");
    title.textContent = "识别功能已就绪";
    status.classList.add("is-ready");
    status.classList.remove("is-error");
  } catch (error) {
    title.textContent = "识别功能暂不可用";
    status.classList.add("is-error");
    status.classList.remove("is-ready");
  }
}

function setRecognition(note, pipelineState = "") {
  const now = performance.now();
  const transient = /收集动作窗口|未检测到手|动作不清晰|有效手部帧不足/.test(note);
  if (transient && now - lastRecognitionRenderAt < 600) return;
  lastRecognitionRenderAt = now;
  document.querySelector("#recognition-note").textContent = note;
  if (pipelineState) setPipelineState("pipeline-result", pipelineState);
}

function setPipelineState(id, state) {
  const status = document.querySelector(`#${id}`);
  status.textContent = state;
  const row = status.closest(".pipeline-row");
  row.dataset.state = /运行中|完成|已识别/.test(state) ? "done"
    : /加载中|缓冲中/.test(state) ? "active" : "pending";
}

function setRecognitionResult(label) {
  const resultElement = document.querySelector("#recognition-result");
  const panel = document.querySelector(".result-empty");
  resultElement.textContent = label;
  panel.classList.toggle("is-recognized", Boolean(label) && label !== "暂未识别，请重试");
}

function renderRecognitionHistory() {
  const list = document.querySelector("#recognition-history-list");
  const items = recognitionSession.history;
  list.replaceChildren();
  if (items.length === 0) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = "本轮还没有识别结果";
    list.append(empty);
  } else {
    items.forEach(({ label }, index) => {
      const item = document.createElement("li");
      item.className = "history-item";
      const number = document.createElement("span");
      number.className = "history-index";
      number.textContent = String(index + 1).padStart(2, "0");
      const word = document.createElement("strong");
      word.textContent = label;
      item.append(number, word);
      list.append(item);
    });
    list.scrollTop = list.scrollHeight;
  }
  document.querySelector("#history-undo").disabled = items.length === 0;
  document.querySelector("#history-clear").disabled = items.length === 0;
}

async function createHandLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(`${API_BASE}/vendor/mediapipe/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${API_BASE}/vendor/mediapipe/wasm`);
  handConnections = HandLandmarker.HAND_CONNECTIONS;
  return HandLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: `${API_BASE}/assets/hand_landmarker.task`, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

function prepareInferenceCanvas(video) {
  inferenceCanvas = document.createElement("canvas");
  inferenceCanvas.width = video.videoWidth;
  inferenceCanvas.height = video.videoHeight;
  inferenceContext = inferenceCanvas.getContext("2d", { alpha: false });
  overlayContext = document.querySelector("#hand-overlay").getContext("2d");
  const overlay = document.querySelector("#hand-overlay");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function drawMirroredInput(video) {
  const { width, height } = inferenceCanvas;
  inferenceContext.save();
  inferenceContext.translate(width, 0);
  inferenceContext.scale(-1, 1);
  inferenceContext.drawImage(video, 0, 0, width, height);
  inferenceContext.restore();
}

function drawHands(result) {
  const overlay = document.querySelector("#hand-overlay");
  const { width, height } = overlay;
  overlayContext.clearRect(0, 0, width, height);
  overlayContext.strokeStyle = "#59d6a7";
  overlayContext.fillStyle = "#f4fff9";
  overlayContext.lineWidth = Math.max(2, width / 500);
  for (const landmarks of result.landmarks || []) {
    for (const connection of handConnections) {
      const from = landmarks[connection.start];
      const to = landmarks[connection.end];
      overlayContext.beginPath();
      overlayContext.moveTo(from.x * width, from.y * height);
      overlayContext.lineTo(to.x * width, to.y * height);
      overlayContext.stroke();
    }
    for (const point of landmarks) {
      overlayContext.beginPath();
      overlayContext.arc(point.x * width, point.y * height, Math.max(2, width / 320), 0, Math.PI * 2);
      overlayContext.fill();
    }
  }
}

function votePrediction(payload) {
  const outcome = predictionSmoother.push(payload);
  if (outcome.label) {
    stableLabel = outcome.label;
    setRecognitionResult(outcome.label);
    if (outcome.changed && recognitionSession.recordAccepted({ accepted: true, label: outcome.label }, Date.now())) renderRecognitionHistory();
    const note = outcome.changed
      ? `已确认 · 置信度 ${(payload.confidence * 100).toFixed(1)}%`
      : outcome.locked ? "识别结果已保留" : "稳定识别";
    setRecognition(note, outcome.locked ? "结果锁定 1.5 秒" : "稳定");
  } else if (payload.accepted) {
    setRecognition("动作已检测，正在确认结果", "确认中");
  } else {
    const lowFrames = Number(payload.valid_frame_count) < MIN_VALID_HAND_FRAMES;
    setRecognition(lowFrames ? "请将当前动作需要的手完整放入画面" : "动作不清晰，没看清请重试", lowFrames ? "质量不足" : "拒识");
  }
}

async function requestWindowPrediction(sequence, session, generation) {
  const controller = new AbortController();
  activePredictionController = controller;
  predictionInFlight = true;
  const requestStartedAt = performance.now();
  let timedOut = false;
  const timeoutId = window.setTimeout(() => { timedOut = true; controller.abort(); }, 3000);
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || payload.message || "推理服务暂不可用");
    console.info("[sign-predict]", {
      accepted: payload.accepted,
      label: payload.label,
      candidate_label: payload.candidate_label,
      confidence: payload.confidence,
      valid_frame_count: payload.valid_frame_count,
      message: payload.message,
    });
    if (session === cameraSession && generation === predictionGeneration && performance.now() - requestStartedAt <= 1500) {
      votePrediction(payload);
    }
  } catch (error) {
    if (session === cameraSession && generation === predictionGeneration && (error.name !== "AbortError" || timedOut)) {
      predictionSmoother.reset();
      setRecognition(timedOut ? "预测请求超时，请检查本地服务" : `识别服务暂不可用：${error.message}`, "连接错误");
    }
  } finally {
    window.clearTimeout(timeoutId);
    if (activePredictionController === controller) {
      activePredictionController = null;
      predictionInFlight = false;
    }
  }
}

function processCameraFrame(now) {
  animationFrame = requestAnimationFrame(processCameraFrame);
  if (!handLandmarker || !cameraStream || now - lastFrameAt < CAMERA_FRAME_INTERVAL_MS) return;
  const video = document.querySelector("#camera-video");
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return;
  lastFrameAt = now;
  const session = cameraSession;
  try {
    drawMirroredInput(video);
    const result = handLandmarker.detectForVideo(inferenceCanvas, now);
    drawHands(result);
    const detections = (result.landmarks || []).map((landmarks, index) => ({
      landmarks,
      handedness: taskHandednessToTrainingLabel(result.handedness?.[index]?.[0]?.categoryName),
      score: result.handedness?.[index]?.[0]?.score,
    }));
    const frame = frameFeatures(detections);

    if (frame.handCount === 0) {
      predictionGeneration += 1;
      activePredictionController?.abort();
      const segmentEnded = recognitionSession.observeNoHands(now);
      if (segmentEnded) {
        frameBuffer = [];
        stableLabel = null;
        predictionSmoother.reset();
        noHandWindowCleared = true;
      } else if (!noHandWindowCleared) {
        frameBuffer.push(frame.features);
        if (frameBuffer.length > FRAME_COUNT) frameBuffer.shift();
      }
      setRecognition(segmentEnded ? "手已放下，结果已保留；可以开始下一个动作" : "未检测到手，已保留最近结果", "等待下一动作");
      document.querySelector("#camera-state").textContent = segmentEnded ? "等待下一动作" : "未检测到手";
      return;
    }
    recognitionSession.observeHands();
    noHandWindowCleared = false;
    frameBuffer.push(frame.features);
    if (frameBuffer.length > FRAME_COUNT) frameBuffer.shift();
    const validFrames = validFrameCount(frameBuffer);
    document.querySelector("#camera-state").textContent = `已检测到${frame.handCount}只手`;
    setPipelineState("pipeline-hands", "运行中");
    if (frameBuffer.length < FRAME_COUNT) {
      setRecognition(`正在收集动作窗口 ${frameBuffer.length}/${FRAME_COUNT} 帧`, "缓冲中");
      setPipelineState("pipeline-inference", "缓冲中");
      return;
    }
    if (validFrames < MIN_VALID_HAND_FRAMES) {
      predictionGeneration += 1;
      activePredictionController?.abort();
      if (stableLabel && predictionSmoother.isLocked()) {
        setRecognition("识别结果已保留", "结果锁定 1.5 秒");
      } else {
        setRecognition("请将当前动作需要的手完整放入画面", "质量不足");
      }
      return;
    }
    setPipelineState("pipeline-inference", "运行中");
    if (!predictionInFlight && now - lastPredictionAt >= PREDICTION_INTERVAL_MS) {
      lastPredictionAt = now;
      requestWindowPrediction(frameBuffer.slice(), session, predictionGeneration);
    }
  } catch (error) {
    setRecognition(`手部特征提取失败：${error.message}`, "特征错误");
  }
}

function showEmpty(title, copy) {
  const empty = document.createElement("div");
  empty.className = "lookup-empty";
  const icon = document.createElement("span");
  icon.className = "empty-search";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "⌕";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const description = document.createElement("p");
  description.textContent = copy;
  empty.append(icon, heading, description);
  result.replaceChildren(empty);
}

function renderTerm(entry) {
  const card = document.createElement("article");
  card.className = "entry-card";
  const head = document.createElement("div");
  head.className = "entry-card-head";
  const term = document.createElement("strong");
  term.textContent = entry.term;
  const badge = document.createElement("span");
  badge.className = "review-badge";
  badge.textContent = "词条匹配";
  head.append(term, badge);

  const body = document.createElement("div");
  body.className = "entry-body";
  if (entry.explanation) {
    const explanation = document.createElement("p");
    explanation.className = "entry-explanation";
    explanation.textContent = entry.explanation;
    body.append(explanation);
  }
  if (entry.sign_description) {
    const description = document.createElement("p");
    description.className = "entry-sign-description";
    description.textContent = entry.sign_description;
    body.append(description);
  }
  const label = document.createElement("label");
  label.textContent = currentMode === "video" ? "示范视频" : "词条状态";
  body.append(label);

  if (entry.status === "ready" && typeof entry.video_url === "string" && entry.video_url) {
    const video = document.createElement("video");
    video.className = "sign-video";
    video.controls = true;
    video.preload = "metadata";
    video.src = new URL(entry.video_url, API_BASE).href;
    video.addEventListener("error", () => {
      const message = document.createElement("p");
      message.className = "video-error";
      message.textContent = "视频加载失败，请检查素材文件后重试。";
      const retry = document.createElement("button");
      retry.className = "text-button";
      retry.type = "button";
      retry.textContent = "重试加载";
      retry.addEventListener("click", () => video.load());
      body.replaceChildren(label, message, retry);
    });
    body.append(video);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "video-placeholder";
    placeholder.textContent = "示范视频待补充";
    body.append(placeholder);
  }
  card.append(head, body);
  result.replaceChildren(card);
}

function renderCandidates(candidates, message) {
  const wrapper = document.createElement("div");
  wrapper.className = "lookup-empty candidate-results";
  const heading = document.createElement("strong");
  heading.textContent = message;
  wrapper.append(heading);
  candidates.forEach((entry) => {
    const button = document.createElement("button");
    button.className = "term-chip";
    button.type = "button";
    button.textContent = `查看“${entry.term}”`;
    button.addEventListener("click", () => renderTerm(entry));
    wrapper.append(button);
  });
  result.replaceChildren(wrapper);
}

async function runLookup(query) {
  feedback.textContent = "";
  showEmpty("正在查询", "正在连接本地词条服务…");
  try {
    const response = await fetch(`${API_BASE}/api/terms/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "词条服务暂不可用");
    if (payload.match_status === "matched") {
      renderTerm(payload.term_data);
    } else if (["ambiguous", "confirmation_required"].includes(payload.status)) {
      renderCandidates(payload.candidates, payload.message);
    } else {
      showEmpty("当前暂未收录", "没有匹配到当前 10 个词条，请尝试单个词语或已配置别名。");
    }
  } catch (error) {
    showEmpty("查询暂不可用", error.message);
    feedback.textContent = "请确认 Flask 服务已启动；你输入的文字仍保留在输入框中。";
  }
}

async function loadKnowledgeStatus() {
  const badge = document.querySelector("#knowledge-mode-badge");
  const status = document.querySelector("#book-import-status");
  try {
    const response = await fetch(`${API_BASE}/api/knowledge/status`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "状态读取失败");
    if (data.knowledgeApiVersion !== 2) throw new Error("当前连到的是旧版服务。请关闭原来的服务终端，再运行 start_sample_system.py。");
    badge.textContent = data.retrieval === "exact_title" ? "词条匹配" : data.retrieval === "atlas_vector" ? "语义检索" : "资料查找";
    const datasetLabel = data.collection === "SignKnowledgeChunks_v2" ? "新版词条资料" : "旧版资料";
    status.textContent = `当前使用${datasetLabel}，已审核 ${data.approvedCount} 条，待审核 ${data.pendingCount} 条。${data.externalProcessingAllowed ? "已允许书籍内容发送到配置的模型服务。" : "当前直接显示原书内容，不调用模型。"}`;
  } catch (error) {
    badge.textContent = "暂不可用";
    status.textContent = error.message;
  }
}

function renderBookSources(sources) {
  const list = document.querySelector("#book-sources");
  list.replaceChildren();
  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "book-source";
    const body = document.createElement("div");
    body.className = "book-source-body";
    const content = document.createElement("div");
    content.className = "book-source-content";
    const selection = document.createElement("label");
    selection.className = "book-source-select";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = source.id || "";
    checkbox.dataset.knowledgeEntryId = source.id || "";
    checkbox.disabled = !source.id;
    checkbox.setAttribute("aria-label", `选择词条：${source.title || "词条"}`);
    checkbox.addEventListener("change", updateKnowledgeSelection);
    selection.append(checkbox, document.createTextNode("选择"));
    const heading = document.createElement("strong");
    heading.textContent = source.title || "词条片段";
    const tags = document.createElement("div");
    tags.className = "book-source-tags";
    for (const label of [
      source.book ? `第 ${source.book} 册` : "手语词典",
      source.page ? `PDF 第 ${source.page} 页` : "EPUB 章节",
      source.sourceFile,
    ].filter(Boolean)) {
      const tag = document.createElement("span");
      tag.className = "book-source-tag";
      tag.textContent = label;
      tags.append(tag);
    }
    const text = document.createElement("p");
    text.textContent = source.text;
    content.append(selection, heading, tags, text);
    if (source.recommendationReason) {
      const reason = document.createElement("p");
      reason.className = "book-source-reason";
      reason.textContent = `推荐原因：${source.recommendationReason}`;
      content.append(reason);
    }
    const images = document.createElement("div");
    images.className = "book-source-images";
    for (const imageUrl of source.images || []) {
      const image = document.createElement("img");
      image.src = new URL(imageUrl, API_BASE).href;
      image.alt = `${source.title}，国家通用手语词典原图`;
      image.loading = "lazy";
      image.className = "book-source-image";
      images.append(image);
    }
    body.append(content);
    if (images.childElementCount) body.append(images);
    item.append(body);
    list.append(item);
  }
  const toolbar = document.querySelector("#knowledge-selection-toolbar");
  toolbar.classList.toggle("hidden", sources.length === 0);
  document.querySelector("#knowledge-select-all").checked = false;
  document.querySelector("#knowledge-save-status").textContent = "";
  knowledgeSources = sources;
  updateKnowledgeSelection();
}

function updateKnowledgeSelection() {
  const boxes = [...document.querySelectorAll("[data-knowledge-entry-id]")];
  const selected = boxes.filter((box) => box.checked && box.value).length;
  document.querySelector("#knowledge-selected-count").textContent = `已选 ${selected} 个词`;
  document.querySelector("#knowledge-select-all").checked = boxes.length > 0 && selected === boxes.filter((box) => box.value).length;
  document.querySelector("#knowledge-add-favorites").disabled = selected === 0;
  document.querySelector("#knowledge-add-today-plan").disabled = selected === 0;
  document.querySelector("#knowledge-add-list").disabled = selected === 0 || !document.querySelector("#knowledge-list-select").value;
  document.querySelector("#knowledge-create-topic-list").disabled = selected === 0 || !document.querySelector("#knowledge-new-list-title").value.trim();
}

function selectedKnowledgeIds() {
  return [...document.querySelectorAll("[data-knowledge-entry-id]:checked")]
    .map((box) => box.value).filter(Boolean);
}

async function loadKnowledgeLists() {
  const auth = window.linguangAuth;
  if (!auth?.requireLogin("knowledge")) return false;
  const select = document.querySelector("#knowledge-list-select");
  if (!knowledgeListsLoaded) {
    const data = await window.linguangRequest("/api/learning/lists");
    knowledgeLists = data.lists || [];
    knowledgeListsLoaded = true;
  }
  const selectedValue = select.value;
  select.replaceChildren(new Option("选择已有词单…", ""));
  knowledgeLists.forEach((item) => select.append(new Option(`${item.title}（${item.entries.length}）`, item.id)));
  select.value = knowledgeLists.some((item) => item.id === selectedValue) ? selectedValue : "";
  updateKnowledgeSelection();
  return true;
}

async function saveSelectedKnowledge(action, listId = "", listTitle = "") {
  if (!selectedKnowledgeIds().length) return;
  if (!window.linguangAuth?.requireLogin("knowledge")) return;
  const button = action === "favorite" ? document.querySelector("#knowledge-add-favorites")
    : action === "plan" ? document.querySelector("#knowledge-add-today-plan")
    : listTitle ? document.querySelector("#knowledge-create-topic-list") : document.querySelector("#knowledge-add-list");
  const status = document.querySelector("#knowledge-save-status");
  const selected = selectedKnowledgeIds();
  button.disabled = true;
  status.textContent = "正在保存…";
  try {
    const result = await window.linguangRequest("/api/learning/batch-save", {
      method: "POST",
      body: JSON.stringify({ action, entryIds: selected, listId, listTitle }),
    });
    status.textContent = action === "favorite"
      ? `已收藏 ${result.added} 个${result.alreadyPresent ? `，${result.alreadyPresent} 个此前已收藏` : ""}${result.skipped ? `，跳过 ${result.skipped} 个不可用词条` : ""}。`
      : action === "plan"
        ? `已加入今日学习计划 ${result.added} 个${result.alreadyPresent ? `，${result.alreadyPresent} 个原已在计划中` : ""}${result.skipped ? `，跳过 ${result.skipped} 个不可用词条` : ""}。前往“手语学习”并刷新计划即可开始。`
      : `已将 ${result.added} 个词加入“${result.listTitle}”${result.alreadyPresent ? `，${result.alreadyPresent} 个已在词单中` : ""}。`;
    const toastMessage = action === "favorite"
      ? `已加入收藏 ${result.added} 个词${result.alreadyPresent ? `，${result.alreadyPresent} 个已收藏` : ""}`
      : action === "plan"
        ? `已加入今日学习计划 ${result.added} 个词${result.alreadyPresent ? `，${result.alreadyPresent} 个已在计划中` : ""}`
        : `已加入词单“${result.listTitle}” ${result.added} 个词${result.alreadyPresent ? `，${result.alreadyPresent} 个已存在` : ""}`;
    showAppToast(toastMessage);
    if (action === "list" && listTitle) {
      knowledgeListsLoaded = false;
      document.querySelector("#knowledge-new-list-title").value = "";
      await loadKnowledgeLists();
      document.querySelector("#knowledge-list-select").value = result.listId;
    }
    selected.forEach((id) => {
      const source = knowledgeSources.find((item) => item.id === id);
      if (source && action === "favorite") source.favorite = true;
    });
  } catch (error) {
    status.textContent = error.message;
  } finally {
    updateKnowledgeSelection();
  }
}

document.querySelector("#knowledge-select-all").addEventListener("change", (event) => {
  document.querySelectorAll("[data-knowledge-entry-id]").forEach((box) => { box.checked = event.currentTarget.checked; });
  updateKnowledgeSelection();
});
document.querySelector("#knowledge-add-favorites").addEventListener("click", () => saveSelectedKnowledge("favorite"));
document.querySelector("#knowledge-add-today-plan").addEventListener("click", () => saveSelectedKnowledge("plan"));
document.querySelector("#knowledge-list-select").addEventListener("focus", () => {
  loadKnowledgeLists().catch((error) => { document.querySelector("#knowledge-save-status").textContent = error.message; });
});
document.querySelector("#knowledge-list-select").addEventListener("change", updateKnowledgeSelection);
document.querySelector("#knowledge-add-list").addEventListener("click", () => {
  saveSelectedKnowledge("list", document.querySelector("#knowledge-list-select").value);
});
document.querySelector("#knowledge-new-list-title").addEventListener("input", updateKnowledgeSelection);
document.querySelector("#knowledge-create-topic-list").addEventListener("click", () => {
  saveSelectedKnowledge("list", "", document.querySelector("#knowledge-new-list-title").value.trim());
});

function readKnowledgeHistory() {
  try {
    const entries = JSON.parse(localStorage.getItem(KNOWLEDGE_HISTORY_KEY) || "[]");
    if (!Array.isArray(entries)) return [];
    const now = Date.now();
    const valid = entries.filter((entry) => entry && typeof entry.query === "string"
      && Number.isFinite(entry.savedAt) && entry.savedAt <= now
      && Number.isFinite(new Date(entry.savedAt).getTime())
      && now - entry.savedAt < KNOWLEDGE_HISTORY_TTL_MS
      && entry.data && Array.isArray(entry.data.sources));
    if (valid.length !== entries.length) localStorage.setItem(KNOWLEDGE_HISTORY_KEY, JSON.stringify(valid));
    return valid.slice(0, KNOWLEDGE_HISTORY_LIMIT);
  } catch {
    return [];
  }
}

function renderKnowledgeHistory() {
  const list = document.querySelector("#knowledge-search-history-list");
  const empty = document.querySelector("#knowledge-search-history-empty");
  const clear = document.querySelector("#knowledge-history-clear");
  if (!list || !empty || !clear) return;
  const entries = readKnowledgeHistory();
  list.replaceChildren();
  empty.hidden = entries.length > 0;
  clear.disabled = entries.length === 0;
  entries.forEach((entry) => {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "knowledge-search-history-item";
    button.setAttribute("aria-label", `打开搜索记录：${entry.query}`);
    const query = document.createElement("span");
    query.className = "knowledge-search-history-query";
    query.textContent = entry.query;
    const time = document.createElement("time");
    time.dateTime = new Date(entry.savedAt).toISOString();
    time.textContent = new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    }).format(entry.savedAt);
    button.append(query, time);
    button.addEventListener("click", () => {
      document.querySelector("#book-question").value = entry.query;
      showKnowledgeResult(entry.data, { cachedAt: entry.savedAt });
    });
    row.append(button);
    list.append(row);
  });
}

function saveKnowledgeHistory(query, data) {
  const savedAt = Date.now();
  const cachedData = {
    retrieval: data.retrieval,
    message: data.message,
    answerMode: data.answerMode,
    answer: data.answer,
    intent: data.intent,
    topic: data.topic,
    suggestedListTitle: data.suggestedListTitle,
    recommendationMode: data.recommendationMode,
    sources: (data.sources || []).map((source) => ({
      id: source.id,
      title: source.title,
      book: source.book,
      page: source.page,
      sourceFile: source.sourceFile,
      text: source.text,
      images: source.images || [],
      recommendationReason: source.recommendationReason || "",
    })),
  };
  const key = query.toLocaleLowerCase().replace(/\s+/g, "");
  const entries = readKnowledgeHistory().filter((entry) =>
    entry.query.toLocaleLowerCase().replace(/\s+/g, "") !== key);
  entries.unshift({ query, savedAt, data: cachedData });
  try {
    localStorage.setItem(KNOWLEDGE_HISTORY_KEY, JSON.stringify(entries.slice(0, KNOWLEDGE_HISTORY_LIMIT)));
  } catch {
    return;
  }
  renderKnowledgeHistory();
}

function showKnowledgeResult(data, { cachedAt = null } = {}) {
  const status = document.querySelector("#book-question-status");
  const answer = document.querySelector("#book-answer");
  const retrievalLabel = data.retrieval === "scenario" ? "场景候选筛选"
    : data.retrieval === "exact_title" ? "词条精确匹配"
    : data.retrieval === "similar" ? "相似词推荐"
    : data.retrieval === "atlas_vector" ? "语义检索" : "词条检索";
  status.textContent = cachedAt
    ? `${data.message} 检索方式：${retrievalLabel}；已打开本地缓存结果。`
    : `${data.message} 检索方式：${retrievalLabel}；${data.answerMode || ""}`;
  const topicInput = document.querySelector("#knowledge-new-list-title");
  if (data.intent === "scenario" && data.suggestedListTitle && !topicInput.value.trim()) {
    topicInput.value = data.suggestedListTitle;
  }
  if (data.answer) {
    answer.textContent = data.answer;
    answer.classList.remove("hidden");
  } else {
    answer.textContent = "";
    answer.classList.add("hidden");
  }
  renderBookSources(data.sources || []);
}

document.querySelector("#knowledge-history-clear").addEventListener("click", () => {
  try {
    localStorage.removeItem(KNOWLEDGE_HISTORY_KEY);
  } catch {
    // Storage may be unavailable in restricted browser contexts.
  }
  renderKnowledgeHistory();
});

document.querySelector("#book-question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const field = document.querySelector("#book-question");
  const status = document.querySelector("#book-question-status");
  const answer = document.querySelector("#book-answer");
  const submit = event.currentTarget.querySelector("button[type=submit]");
  if (!field.value.trim()) {
    status.textContent = "先输入你想问的内容。";
    field.focus();
    return;
  }
  submit.disabled = true;
  status.textContent = "正在检索已审核的书籍内容…";
  answer.classList.add("hidden");
  answer.textContent = "";
  document.querySelector("#book-sources").replaceChildren();
  try {
    const response = await fetch(`${API_BASE}/api/knowledge/search`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: field.value.trim() }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "检索失败");
    if (data.knowledgeApiVersion !== 2) throw new Error("当前连到的是旧版服务，结果未显示。请关闭原来的服务终端，再运行 start_sample_system.py。");
    showKnowledgeResult(data);
    saveKnowledgeHistory(field.value.trim(), data);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

document.querySelector("#query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!input.value.trim()) {
    feedback.textContent = "请先输入词语或问题。";
    input.focus();
    return;
  }
  runLookup(input.value);
});

document.querySelector("#camera-start").addEventListener("click", async () => {
  const title = document.querySelector("#camera-title");
  const copy = document.querySelector("#camera-copy");
  const startButton = document.querySelector("#camera-start");
  startButton.disabled = true;
  const session = cameraSession + 1;
  recognitionSession.resetRound();
  renderRecognitionHistory();
  setRecognitionResult("暂未识别，请重试");
  stableLabel = null;
  frameBuffer = [];
  noHandWindowCleared = false;
  predictionSmoother.reset();
  setRecognition("正在启动识别组件", "初始化中");
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("unsupported");
    const healthResponse = await fetch(`${API_BASE}/api/health`);
    const health = await healthResponse.json();
    if (!healthResponse.ok || !health.model_loaded) throw new Error("识别模型或 API 未就绪");
    if (!health.camera_features_ready) throw new Error("本地 MediaPipe 运行库或手部模型文件未就绪");
    MIN_VALID_HAND_FRAMES = Number(health.min_selected_frames);
    if (!Number.isInteger(MIN_VALID_HAND_FRAMES) || MIN_VALID_HAND_FRAMES < 1) throw new Error("服务端有效帧门槛配置无效");
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    cameraSession = session;
    const video = document.querySelector("#camera-video");
    video.srcObject = cameraStream;
    await video.play();
    if (!video.videoWidth) await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));
    frameBuffer = [];
    predictionSmoother.reset();
    stableLabel = null;
    lastFrameAt = 0;
    lastPredictionAt = 0;
    document.querySelector("#camera-frame").classList.add("is-live");
    document.querySelector("#camera-stop").disabled = false;
    document.querySelector("#camera-state").textContent = "正在加载本地识别模型";
    setRecognition("正在加载本地手部识别模型", "初始化中");
    setPipelineState("pipeline-hands", "加载中");
    const landmarker = await createHandLandmarker();
    if (session !== cameraSession || !cameraStream) {
      landmarker.close();
      return;
    }
    handLandmarker = landmarker;
    prepareInferenceCanvas(video);
    document.querySelector("#camera-state").textContent = "正在检测双手";
    title.textContent = "摄像头识别已启动";
    setPipelineState("pipeline-hands", "运行中");
    animationFrame = requestAnimationFrame(processCameraFrame);
    cameraStream.getVideoTracks()[0]?.addEventListener("ended", () => {
      stopCamera();
      title.textContent = "摄像头连接中断";
      document.querySelector("#camera-state").textContent = "设备不可用";
    }, { once: true });
  } catch (error) {
    if (session === cameraSession && cameraStream) stopCamera();
    title.textContent = error.name === "NotAllowedError" ? "摄像头权限未获允许"
      : error.name === "NotFoundError" ? "未找到可用摄像头"
        : error.message === "unsupported" ? "当前环境不支持摄像头"
          : "识别模块启动失败";
    document.querySelector("#camera-state").textContent = error.name === "NotAllowedError" ? "权限未允许" : "启动失败";
    setRecognition(error.name === "NotAllowedError" ? "请允许浏览器使用摄像头" : `启动失败：${error.message}`, "启动失败");
  } finally {
    startButton.disabled = false;
  }
});

document.querySelector("#camera-stop").addEventListener("click", stopCamera);
document.querySelector("#history-undo").addEventListener("click", () => {
  recognitionSession.undo();
  renderRecognitionHistory();
});
document.querySelector("#history-clear").addEventListener("click", () => {
  recognitionSession.clear();
  renderRecognitionHistory();
});
function stopCamera() {
  cameraSession += 1;
  predictionGeneration += 1;
  activePredictionController?.abort();
  activePredictionController = null;
  cancelAnimationFrame(animationFrame);
  handLandmarker?.close();
  handLandmarker = undefined;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  document.querySelector("#camera-video").srcObject = null;
  document.querySelector("#hand-overlay").getContext("2d")?.clearRect(0, 0, document.querySelector("#hand-overlay").width, document.querySelector("#hand-overlay").height);
  document.querySelector("#camera-frame").classList.remove("is-live");
  document.querySelector("#camera-state").textContent = "预览未启动";
  document.querySelector("#camera-stop").disabled = true;
  document.querySelector("#camera-start").disabled = false;
  document.querySelector("#camera-title").textContent = "摄像头尚未开启";
  document.querySelector("#camera-copy").textContent = "允许摄像头访问后，在本地提取双手关键点并识别。";
  frameBuffer = [];
  noHandWindowCleared = false;
  predictionSmoother.reset();
  stableLabel = null;
  recognitionSession.resetRound();
  renderRecognitionHistory();
  predictionInFlight = false;
  setRecognitionResult("暂未识别，请重试");
  setRecognition("摄像头已关闭；再次开启将开始新一轮", "已关闭");
  setPipelineState("pipeline-hands", "等待开启");
  setPipelineState("pipeline-inference", "等待开启");
}

document.querySelector("#mic-button").addEventListener("click", () => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    feedback.textContent = "此浏览器不支持语音输入，请直接使用文字查询。";
    return;
  }
  recognition?.abort();
  recognition = new SpeechRecognition();
  recognition.lang = "zh-CN";
  recognition.interimResults = false;
  recognition.onresult = (event) => {
    input.value = event.results[0][0].transcript;
    feedback.textContent = "语音已转为文字，请检查后点击查询。";
  };
  recognition.onerror = (event) => {
    feedback.textContent = event.error === "not-allowed" ? "麦克风权限未获允许，仍可使用文字查询。" : "语音输入暂不可用，请改用文字查询。";
  };
  try {
    recognition.start();
    feedback.textContent = "正在听取语音…";
  } catch {
    feedback.textContent = "语音输入暂不可用，请直接使用文字查询。";
  }
});

window.addEventListener("pagehide", stopCamera);
window.addEventListener("pagehide", stopCommunicationListening);
loadTerms();
renderKnowledgeHistory();
checkModel();
initializeSpeechOutput();
let communicationInitialized = false;
let learningInitialized = false;
window.linguangNavigate = setMode;
initializeAuth({
  apiBase: API_BASE,
  requireLoginAtStart: true,
  onAuthenticated: (_user, newlyAuthenticated, destination) => {
    if (!communicationInitialized) {
      communicationInitialized = true;
      initializeCommunication({ apiBase: API_BASE });
    }
    if (!learningInitialized) {
      learningInitialized = true;
      initializeLearning({ apiBase: API_BASE });
    }
    if (newlyAuthenticated) setMode(destination || "communication");
  },
});
