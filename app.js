import { FRAME_COUNT, frameFeatures, validFrameCount } from "./camera_features.mjs";

const modes = {
  recognize: { title: "手语识别", description: "限定词表仅覆盖十个训练词；摄像头特征在本地提取。" },
  video: { title: "查手语视频", description: "输入词语查找对应的手语示范；尚未收录的表达会明确提示。" },
  knowledge: { title: "问手语知识", description: "基于本地候选词条查询收录状态；未经审核的内容不会作为知识回答。" },
};

const navItems = [...document.querySelectorAll(".nav-item")];
const modeButtons = [...document.querySelectorAll(".mode-button")];
const recognizePanel = document.querySelector("#recognize-panel");
const lookupPanel = document.querySelector("#lookup-panel");
const termList = document.querySelector("#term-list");
const input = document.querySelector("#query-input");
const feedback = document.querySelector("#query-feedback");
const result = document.querySelector("#lookup-result");
const API_BASE = window.SIGN_API_BASE || ( ["8000", "8080"].includes(location.port)
  ? `${location.protocol}//${location.hostname}:5001`
  : location.origin);
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
let predictionVotes = [];
let stableLabel = null;
const CAMERA_FRAME_INTERVAL_MS = 40;
const PREDICTION_INTERVAL_MS = 450;
const MIN_VALID_HAND_FRAMES = 13;
const VOTE_WINDOW = 5;

function setMode(mode) {
  if (mode !== "recognize" && cameraStream) stopCamera();
  currentMode = mode;
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  modeButtons.forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  const isRecognize = mode === "recognize";
  recognizePanel.classList.toggle("hidden", !isRecognize);
  lookupPanel.classList.toggle("hidden", isRecognize);
  document.querySelector("#page-title").textContent = modes[mode].title;
  document.querySelector("#page-description").textContent = modes[mode].description;
  document.querySelector("#breadcrumb-current").textContent = modes[mode].title;
  document.querySelector("#result-label").textContent = mode === "video" ? "词条匹配" : "资料检索";
  document.querySelector("#aside-title").textContent = mode === "video" ? "词条状态" : "资料范围";
  document.querySelector("#aside-copy").textContent = mode === "video"
    ? "词条由 Flask 本地接口提供；视频素材尚未补充时会显示等待状态。"
    : "当前通过本地词条接口检索；没有资料时明确提示未收录，不调用大模型。";
  input.placeholder = mode === "video" ? "输入一个词语，例如：谢谢" : "例如：谢谢适合什么场合用？";
  document.querySelector("#query-hint").textContent = mode === "video"
    ? "语音会先转成可编辑文字，再由你确认查询。"
    : "检索仅覆盖演示词条与别名；无审核依据时会明确告知未收录。";
  feedback.textContent = "";
  if (!isRecognize && result.querySelector(".lookup-empty")) input.focus();
}

navItems.forEach((item) => item.addEventListener("click", () => setMode(item.dataset.mode)));
modeButtons.forEach((item) => item.addEventListener("click", () => setMode(item.dataset.mode)));
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
    document.querySelector("#term-count").textContent = String(entries.length);
  } catch (error) {
    feedback.textContent = `${error.message}，请确认 Flask 服务已启动；文字输入仍可保留。`;
  }
}

async function checkModel() {
  const title = document.querySelector("#model-status-title");
  const copy = document.querySelector("#model-status-copy");
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const payload = await response.json();
    if (!response.ok || !payload.model_loaded) throw new Error(payload.error || "模型尚未加载");
    if (!payload.camera_features_ready) throw new Error("本地 MediaPipe 运行库或 hand_landmarker.task 未就绪");
    title.textContent = "模型与本地识别组件已就绪";
    copy.textContent = "摄像头关键点在浏览器处理，特征序列仅发送到本机 Flask 接口。";
  } catch (error) {
    title.textContent = "模型或 API 尚未就绪";
    copy.textContent = `${error.message}；请确认本机 Flask 服务已启动并加载模型。`;
  }
}

function setRecognition(title, note, pipelineState = "") {
  document.querySelector("#recognition-result").textContent = title;
  document.querySelector("#recognition-note").textContent = note;
  if (pipelineState) document.querySelector("#pipeline-result").textContent = pipelineState;
}

async function createHandLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(`${API_BASE}/vendor/mediapipe/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${API_BASE}/vendor/mediapipe/wasm`);
  handConnections = HandLandmarker.HAND_CONNECTIONS;
  return HandLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: `${API_BASE}/assets/hand_landmarker.task`, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,
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
  predictionVotes.push(payload.accepted ? payload.label : null);
  predictionVotes = predictionVotes.slice(-VOTE_WINDOW);
  const counts = new Map();
  predictionVotes.filter(Boolean).forEach((label) => counts.set(label, (counts.get(label) || 0) + 1));
  const winner = [...counts].find(([, count]) => count >= 3)?.[0];
  if (winner) {
    stableLabel = winner;
    setRecognition(winner, `连续窗口稳定识别 · 置信度 ${(payload.confidence * 100).toFixed(1)}%`, "稳定");
    return;
  }
  const rejectedVotes = predictionVotes.filter((label) => label === null).length;
  if (stableLabel && rejectedVotes < 3) {
    setRecognition(stableLabel, "正在确认后续窗口", "确认中");
  } else if (payload.accepted) {
    setRecognition("正在确认动作", "等待连续窗口给出一致结果", "确认中");
  } else {
    stableLabel = null;
    setRecognition("暂未识别，请重试", "置信度不足", "拒识");
  }
}

async function requestWindowPrediction(sequence, session, generation) {
  predictionInFlight = true;
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || payload.message || "推理服务暂不可用");
    if (session === cameraSession && generation === predictionGeneration) votePrediction(payload);
  } catch (error) {
    if (session === cameraSession && generation === predictionGeneration) {
      stableLabel = null;
      predictionVotes = [];
      setRecognition("识别服务不可用", error.message, "连接错误");
    }
  } finally {
    predictionInFlight = false;
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
      handedness: result.handedness?.[index]?.[0]?.categoryName,
      score: result.handedness?.[index]?.[0]?.score,
    }));
    const frame = frameFeatures(detections);
    frameBuffer.push(frame.features);
    if (frameBuffer.length > FRAME_COUNT) frameBuffer.shift();
    const validFrames = validFrameCount(frameBuffer);

    if (frame.handCount === 0) {
      predictionGeneration += 1;
      predictionVotes = [];
      stableLabel = null;
      setRecognition("暂未识别，请重试", "未检测到手", "无手");
      document.querySelector("#camera-state").textContent = "未检测到手";
      return;
    }
    document.querySelector("#camera-state").textContent = `已检测到${frame.handCount}只手`;
    document.querySelector("#pipeline-hands").textContent = "运行中";
    if (frameBuffer.length < FRAME_COUNT) {
      setRecognition("正在收集动作", `时序窗口 ${frameBuffer.length}/${FRAME_COUNT} 帧`, "缓冲中");
      document.querySelector("#pipeline-inference").textContent = "缓冲中";
      return;
    }
    if (validFrames < MIN_VALID_HAND_FRAMES) {
      setRecognition("暂未识别，请重试", `有效手部帧不足（${validFrames}/${MIN_VALID_HAND_FRAMES}）`, "质量不足");
      return;
    }
    document.querySelector("#pipeline-inference").textContent = "運行中";
    if (!predictionInFlight && now - lastPredictionAt >= PREDICTION_INTERVAL_MS) {
      lastPredictionAt = now;
      requestWindowPrediction(frameBuffer.slice(), session, predictionGeneration);
    }
  } catch (error) {
    setRecognition("手部特征提取失败", error.message, "特征错误");
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
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("unsupported");
    const healthResponse = await fetch(`${API_BASE}/api/health`);
    const health = await healthResponse.json();
    if (!healthResponse.ok || !health.model_loaded) throw new Error("识别模型或 API 未就绪");
    if (!health.camera_features_ready) throw new Error("本地 MediaPipe 运行库或手部模型文件未就绪");
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    cameraSession = session;
    const video = document.querySelector("#camera-video");
    video.srcObject = cameraStream;
    await video.play();
    if (!video.videoWidth) await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));
    frameBuffer = [];
    predictionVotes = [];
    stableLabel = null;
    lastFrameAt = 0;
    lastPredictionAt = 0;
    document.querySelector("#camera-frame").classList.add("is-live");
    document.querySelector("#camera-stop").disabled = false;
    document.querySelector("#camera-state").textContent = "正在加载本地识别模型";
    setRecognition("正在启动手部识别", "正在加载本地 Hand Landmarker 模型", "初始化中");
    document.querySelector("#pipeline-hands").textContent = "加载中";
    const landmarker = await createHandLandmarker();
    if (session !== cameraSession || !cameraStream) {
      landmarker.close();
      return;
    }
    handLandmarker = landmarker;
    prepareInferenceCanvas(video);
    document.querySelector("#camera-state").textContent = "正在检测双手";
    title.textContent = "摄像头识别已启动";
    copy.textContent = "画面在浏览器本地处理；特征序列发送到本机 API。";
    document.querySelector("#pipeline-hands").textContent = "运行中";
    animationFrame = requestAnimationFrame(processCameraFrame);
    cameraStream.getVideoTracks()[0]?.addEventListener("ended", () => {
      stopCamera();
      title.textContent = "摄像头连接中断";
      copy.textContent = "设备已断开，请重新连接后再试。";
      document.querySelector("#camera-state").textContent = "设备不可用";
    }, { once: true });
  } catch (error) {
    if (session === cameraSession && cameraStream) stopCamera();
    title.textContent = error.name === "NotAllowedError" ? "摄像头权限未获允许"
      : error.name === "NotFoundError" ? "未找到可用摄像头"
        : error.message === "unsupported" ? "当前环境不支持摄像头"
          : "识别模块启动失败";
    copy.textContent = error.message === "unsupported"
      ? "当前环境不支持摄像头访问；可改用 localhost 或 HTTPS。"
      : error.message;
    document.querySelector("#camera-state").textContent = error.name === "NotAllowedError" ? "权限未允许" : "启动失败";
  } finally {
    startButton.disabled = false;
  }
});

document.querySelector("#camera-stop").addEventListener("click", stopCamera);
function stopCamera() {
  cameraSession += 1;
  predictionGeneration += 1;
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
  predictionVotes = [];
  stableLabel = null;
  predictionInFlight = false;
  setRecognition("暂未识别，请重试", "摄像头已关闭", "已关闭");
  document.querySelector("#pipeline-hands").textContent = "等待开启";
  document.querySelector("#pipeline-inference").textContent = "等待开启";
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
loadTerms();
checkModel();
