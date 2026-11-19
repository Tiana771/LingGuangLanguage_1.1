// AI 助手抽屉：右下角悬浮球 + 右侧面板。
// 只负责界面与流式解析，检索与工具调用都在服务端 /api/agent/probe 完成。

const TOOL_LABELS = {
  lookup_term: "查词典",
  search_scene: "按场景检索",
  get_sign_video: "调取示范视频",
};

const SEP = String.fromCharCode(10, 10);

export function initializeAgentPanel({ apiBase = "" } = {}) {
  const ball = document.querySelector("#agent-ball");
  const drawer = document.querySelector("#agent-drawer");
  if (!ball || !drawer) return;

  const closeButton = drawer.querySelector("#agent-close");
  const form = drawer.querySelector("#agent-form");
  const input = drawer.querySelector("#agent-question");
  const sendButton = drawer.querySelector("#agent-send");
  const stepsBox = drawer.querySelector("#agent-steps");
  const cardsBox = drawer.querySelector("#agent-cards");
  const answerBox = drawer.querySelector("#agent-answer");
  const statusBox = drawer.querySelector("#agent-status");

  let running = false;
  let turns = [];

  function open() {
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    ball.setAttribute("aria-expanded", "true");
    window.setTimeout(() => input.focus(), 120);
  }

  function close() {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    ball.setAttribute("aria-expanded", "false");
    ball.focus();
  }

  function toggle() {
    drawer.classList.contains("open") ? close() : open();
  }

  function addStep(text) {
    const row = document.createElement("div");
    row.className = "agent-step running";
    const dot = document.createElement("span");
    dot.className = "agent-step-dot";
    const label = document.createElement("button");
    label.type = "button";
    label.className = "agent-step-label";
    label.textContent = text;
    const detail = document.createElement("pre");
    detail.className = "agent-step-detail";
    label.addEventListener("click", () => row.classList.toggle("open"));
    row.append(dot, label, detail);
    stepsBox.append(row);
    return row;
  }

  function renderCards(cards) {
    cardsBox.replaceChildren();
    (cards || []).forEach((card) => {
      const item = document.createElement("article");
      item.className = "agent-card";
      if (card.video) {
        const video = document.createElement("video");
        video.src = card.video;
        video.controls = true;
        video.preload = "metadata";
        video.className = "agent-card-video";
        item.append(video);
      } else if (Array.isArray(card.images) && card.images.length) {
        const image = document.createElement("img");
        image.src = card.images[0];
        image.loading = "lazy";
        image.alt = card.title || "词典插图";
        image.className = "agent-card-image";
        item.append(image);
      }
      const body = document.createElement("div");
      body.className = "agent-card-body";
      const meta = document.createElement("div");
      meta.className = "agent-card-meta";
      meta.textContent =
        [card.book ? `第 ${card.book} 册` : "", card.page ? `第 ${card.page} 页` : ""]
          .filter(Boolean)
          .join(" · ") || "已审核词典";
      const name = document.createElement("div");
      name.className = "agent-card-name";
      name.textContent = card.title || "";
      body.append(meta, name);
      if (card.text) {
        const desc = document.createElement("p");
        desc.className = "agent-card-desc";
        desc.textContent = card.text;
        body.append(desc);
      }
      item.append(body);
      cardsBox.append(item);
    });
  }

  function reset() {
    stepsBox.replaceChildren();
    cardsBox.replaceChildren();
    answerBox.textContent = "";
    statusBox.textContent = "正在处理…";
  }

  function handleBlock(block, state) {
    let event = "";
    let data = "";
    block.split(String.fromCharCode(10)).forEach((row) => {
      if (row.startsWith("event:")) event = row.slice(6).trim();
      else if (row.startsWith("data:")) data += row.slice(5).trim();
    });
    if (!event) return;
    let payload = {};
    try {
      payload = JSON.parse(data);
    } catch {
      return;
    }

    if (event === "tool_start") {
      state.running = addStep(`正在${TOOL_LABELS[payload.name] || payload.name}…`);
      state.running.querySelector(".agent-step-detail").textContent = payload.raw_arguments || "";
    } else if (event === "tool_result") {
      const row = state.running || addStep("");
      state.running = null;
      row.classList.remove("running");
      row.classList.add("done");
      const label = TOOL_LABELS[payload.name] || payload.name;
      const titles = Array.isArray(payload.titles) && payload.titles.length ? ` · ${payload.titles.join("、")}` : "";
      row.querySelector(".agent-step-label").textContent = payload.ok
        ? `${label}：找到 ${payload.count || 0} 条${titles}`
        : `${label}：未成功${payload.error ? `（${payload.error}）` : ""}`;
      row.querySelector(".agent-step-detail").textContent = `${payload.raw_arguments || ""}耗时 ${payload.took_ms}ms · 命中 ${payload.count || 0} 条`;
    } else if (event === "token") {
      state.text += payload.text || "";
      answerBox.textContent = state.text;
    } else if (event === "cards") {
      renderCards(payload.cards);
    } else if (event === "error") {
      statusBox.textContent = `出错：${payload.message || ""}`;
      statusBox.classList.add("error");
    } else if (event === "done") {
      const seconds = (payload.elapsed_ms / 1000).toFixed(1);
      statusBox.textContent = `用时 ${seconds} 秒 · ${payload.tools} 次工具调用`;
      statusBox.classList.remove("error");
    }
  }

  async function ask(question) {
    if (running || !question) return;
    running = true;
    sendButton.disabled = true;
    reset();
    const state = { text: "", running: null };
    try {
      const response = await fetch(`${apiBase}/api/agent/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: turns.slice(-8) }),
      });
      if (response.status === 503) {
        statusBox.textContent = "AI 能力暂未开启。";
        return;
      }
      if (!response.ok || !response.body) {
        statusBox.textContent = `请求失败（HTTP ${response.status}）。`;
        statusBox.classList.add("error");
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        pending += decoder.decode(chunk.value, { stream: true });
        let index = pending.indexOf(SEP);
        while (index >= 0) {
          handleBlock(pending.slice(0, index), state);
          pending = pending.slice(index + SEP.length);
          index = pending.indexOf(SEP);
        }
      }
      if (state.text) {
        turns.push({ role: "user", content: question });
        turns.push({ role: "assistant", content: state.text });
        if (turns.length > 8) turns = turns.slice(-8);
      }
    } catch (error) {
      statusBox.textContent = "连接中断，请重试。";
      statusBox.classList.add("error");
    } finally {
      running = false;
      sendButton.disabled = false;
    }
  }

  ball.addEventListener("click", toggle);
  closeButton.addEventListener("click", close);
  drawer.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    ask(question);
  });
  drawer.querySelectorAll("[data-agent-chip]").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.agentChip;
      input.focus();
    });
  });

  window.linguangAgent = {
    show() {
      ball.classList.remove("hidden");
    },
    hide() {
      ball.classList.add("hidden");
      if (drawer.classList.contains("open")) close();
    },
  };
}
