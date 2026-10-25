const VOICE_STORAGE_KEY = "linguang-speech-voice-uri";
const PREVIEW_TEXT = "你好，这是一段语音试听。";

export function initializeSpeechOutput() {
  const synth = window.speechSynthesis;
  const voiceSelect = document.querySelector("#speech-voice-select");
  const voiceStatus = document.querySelector("#speech-voice-status");
  const previewButton = document.querySelector("#speech-voice-preview");
  const speakButton = document.querySelector("#reply-speak");
  const pauseButton = document.querySelector("#reply-speech-pause");
  const resumeButton = document.querySelector("#reply-speech-resume");
  const stopButton = document.querySelector("#reply-speech-stop");
  const outputStatus = document.querySelector("#reply-speech-status");

  document.querySelector("#communication-transcript").addEventListener("click", (event) => {
    const button = event.target.closest(".transcript-speak");
    if (!button) return;
    const editor = button.closest(".transcript-block")?.querySelector(".transcript-block-editor");
    speak(editor?.innerText || "", document.querySelector("#transcript-speech-status"), setPlaybackState);
  });

  if (!synth || !window.SpeechSynthesisUtterance) {
    voiceSelect.replaceChildren(new Option("浏览器不支持语音朗读", ""));
    voiceSelect.disabled = true;
    voiceStatus.textContent = "当前浏览器不支持语音朗读。";
    speakButton.disabled = true;
    previewButton.disabled = true;
    return;
  }

  let voices = [];
  let activeUtterance = null;
  let activeStatusTarget = outputStatus;

  function selectedVoice() {
    return voices.find((voice) => voice.voiceURI === voiceSelect.value) || null;
  }

  function renderVoices() {
    let storedVoiceUri = "";
    try {
      storedVoiceUri = localStorage.getItem(VOICE_STORAGE_KEY) || "";
    } catch {
      // The voice list remains usable when browser storage is unavailable.
    }
    voices = synth.getVoices().slice().sort((left, right) => {
      const leftChinese = left.lang.toLowerCase().startsWith("zh") ? 0 : 1;
      const rightChinese = right.lang.toLowerCase().startsWith("zh") ? 0 : 1;
      return leftChinese - rightChinese || left.lang.localeCompare(right.lang) || left.name.localeCompare(right.name);
    });
    voiceSelect.replaceChildren();
    if (!voices.length) {
      voiceSelect.append(new Option("暂无可用音色", ""));
      voiceSelect.disabled = true;
      previewButton.disabled = true;
      speakButton.disabled = true;
      voiceStatus.textContent = "Edge 暂未提供可用音色。请检查 Windows 语音设置后重试。";
      return;
    }

    voices.forEach((voice) => {
      voiceSelect.append(new Option(`${voice.name} (${voice.lang || "未知语言"})`, voice.voiceURI));
    });
    const preferredVoice = voices.find((voice) => voice.voiceURI === storedVoiceUri)
      || voices.find((voice) => voice.lang.toLowerCase().startsWith("zh"))
      || voices[0];
    voiceSelect.value = preferredVoice.voiceURI;
    voiceSelect.disabled = false;
    previewButton.disabled = false;
    speakButton.disabled = false;
    voiceStatus.textContent = `${voices.length} 个可用音色；音色由当前浏览器和设备提供。`;
  }

  function setPlaybackState(state) {
    const active = state === "speaking" || state === "paused";
    pauseButton.disabled = state !== "speaking";
    resumeButton.disabled = state !== "paused";
    stopButton.disabled = !active;
    if (state !== "speaking" && state !== "paused") activeUtterance = null;
  }

  function speak(text, statusTarget, onStateChange = () => {}) {
    const content = text.trim();
    if (!synth || !window.SpeechSynthesisUtterance) {
      statusTarget.textContent = "当前浏览器不支持语音朗读。";
      return;
    }
    if (!content) {
      statusTarget.textContent = "没有可朗读的文字。";
      return;
    }
    const voice = selectedVoice();
    if (!voice) {
      statusTarget.textContent = "当前没有可用音色。";
      return;
    }
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.voice = voice;
    utterance.lang = voice.lang || "zh-CN";
    utterance.rate = 1;
    activeUtterance = utterance;
    activeStatusTarget = statusTarget;
    onStateChange("speaking");
    statusTarget.textContent = "正在朗读…";
    utterance.onend = () => {
      if (activeUtterance !== utterance) return;
      statusTarget.textContent = "朗读完成。";
      onStateChange("idle");
    };
    utterance.onerror = (event) => {
      if (activeUtterance !== utterance) return;
      statusTarget.textContent = event.error === "canceled" || event.error === "interrupted"
        ? "朗读已停止。"
        : "朗读失败，请检查设备音量和语音服务。";
      onStateChange("idle");
    };
    synth.speak(utterance);
  }

  voiceSelect.addEventListener("change", () => {
    try {
      localStorage.setItem(VOICE_STORAGE_KEY, voiceSelect.value);
    } catch {
      voiceStatus.textContent = "所选音色仅在本次打开期间生效。";
    }
  });
  previewButton.addEventListener("click", () => {
    synth.cancel();
    setPlaybackState("idle");
    speak(PREVIEW_TEXT, voiceStatus);
  });
  speakButton.addEventListener("click", () => {
    speak(document.querySelector("#reply-text").value, outputStatus, setPlaybackState);
  });
  document.querySelector("#direct-reply-speak").addEventListener("click", () => {
    speak(document.querySelector("#direct-reply-text").value, document.querySelector("#direct-reply-status"), setPlaybackState);
  });
  pauseButton.addEventListener("click", () => {
    if (!synth.speaking || synth.paused) return;
    synth.pause();
    activeStatusTarget.textContent = "已暂停。";
    setPlaybackState("paused");
  });
  resumeButton.addEventListener("click", () => {
    if (!synth.paused) return;
    synth.resume();
    activeStatusTarget.textContent = "正在朗读…";
    setPlaybackState("speaking");
  });
  stopButton.addEventListener("click", () => {
    synth.cancel();
    activeStatusTarget.textContent = "朗读已停止。";
    setPlaybackState("idle");
  });
  window.addEventListener("pagehide", () => synth.cancel());
  synth.addEventListener?.("voiceschanged", renderVoices);
  renderVoices();
  if (!voices.length) setTimeout(renderVoices, 500);
}
