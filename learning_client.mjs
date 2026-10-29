let apiBase = "";
let activeView = "today";
let initialized = false;
let todayQueue = [];
let cardIndex = 0;
let statsDays = 7;
let detailReturnScrollY = 0;
let detailReturnLabel = "返回学习";

const $ = (selector) => document.querySelector(selector);
const status = () => $("#learning-status");

async function request(path, options = {}) {
  return window.linguangRequest(path, options);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function sourceLabel(entry) {
  const book = entry.book ? `第 ${entry.book} 册` : "词典";
  const page = entry.page ? `第 ${entry.page} 页` : entry.anchor || "";
  return [book, page].filter(Boolean).join(" · ");
}

function appendImages(container, entry) {
  const images = (entry.images || []).slice(0, 4);
  if (!images.length) return;
  const gallery = element("div", "learning-image-gallery");
  for (const source of images) {
    const image = element("img", "learning-entry-image");
    image.src = source;
    image.alt = `${entry.title}手语词典原图`;
    image.loading = "lazy";
    image.onerror = () => image.remove();
    gallery.append(image);
  }
  container.append(gallery);
}

function dateGroupLabel(value) {
  if (!value) return "日期未记录";
  if (value === "逾期") return "逾期";
  const date = new Date(value.length === 10 ? `${value}T12:00:00` : value);
  if (Number.isNaN(date.getTime())) return "日期未记录";
  const localDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const today = new Date();
  const todayDate = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  if (localDate.getTime() === todayDate.getTime()) return "今天";
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
}

function renderGroupedEntries(container, entries, dateOf, options = {}) {
  const groups = new Map();
  entries.forEach((entry) => {
    const key = dateGroupLabel(dateOf(entry));
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  });
  groups.forEach((items, label) => {
    const group = element("details", "learning-date-group");
    group.append(element("summary", "learning-date-heading", `${label}（${items.length}）`));
    const list = element("div", "learning-date-entries");
    items.forEach((entry) => list.append(renderWordRow(entry, options)));
    group.append(list);
    container.append(group);
  });
}

function renderWordRow(entry, options = {}) {
  const row = element("div", "learning-history-word");
  const line = element("div", "learning-history-word-line");
  const word = element("button", "learning-history-word-button", entry.title);
  word.type = "button";
  word.setAttribute("aria-label", `展开${entry.title}的简要说明`);
  word.setAttribute("aria-expanded", "false");
  const detail = element("button", "learning-word-detail-button", "›");
  detail.type = "button";
  detail.title = "打开大字详情";
  detail.setAttribute("aria-label", `打开${entry.title}的大字详情`);
  detail.addEventListener("click", () => openEntryDetail(entry, options));
  line.append(word, detail);
  if (options.favorite || options.listId) {
    const remove = element("button", "learning-word-remove", options.favorite ? "取消收藏" : "从词单移除");
    remove.type = "button";
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      try {
        if (options.favorite) {
          await request(`/api/learning/entries/${encodeURIComponent(entry.id)}/favorite`, {
            method: "POST", body: JSON.stringify({ favorite: false }),
          });
          window.linguangToast?.(`已取消收藏“${entry.title}”`);
          await loadFavorites();
        } else {
          await updateListEntry(options.listId, entry.id, false, false);
          window.linguangToast?.(`已从词单移除“${entry.title}”`);
          await loadLists();
        }
      } catch (error) {
        status().textContent = error.message;
        window.linguangToast?.(error.message);
        remove.disabled = false;
      }
    });
    line.append(remove);
  }
  const preview = element("div", "learning-history-word-preview hidden");
  word.addEventListener("click", async () => {
    const expanded = word.getAttribute("aria-expanded") === "true";
    word.setAttribute("aria-expanded", String(!expanded));
    preview.classList.toggle("hidden", expanded);
    if (expanded || preview.dataset.loaded === "true" || preview.dataset.loading === "true") return;
    preview.dataset.loading = "true";
    preview.replaceChildren(element("span", "learning-word-preview-loading", "正在读取词语说明…"));
    try {
      const data = entry.text && Array.isArray(entry.images) ? { entry }
        : await request(`/api/learning/entries/${encodeURIComponent(entry.id)}`);
      const fullEntry = data.entry || entry;
      preview.replaceChildren(
        element("span", "learning-word-preview-source", sourceLabel(fullEntry)),
        element("p", "learning-word-preview-description", fullEntry.text || "暂无文字说明。"),
      );
      preview.dataset.loaded = "true";
    } catch (error) {
      preview.replaceChildren(element("span", "learning-word-preview-loading", error.message));
      delete preview.dataset.loading;
      return;
    }
    delete preview.dataset.loading;
  });
  row.append(line, preview);
  return row;
}

async function openEntryDetail(entry, { favorite = false, listId = "" } = {}) {
  const page = $("#learning-entry-page");
  detailReturnScrollY = window.scrollY;
  detailReturnLabel = favorite ? "返回我的收藏"
    : listId ? "返回我的词单"
      : activeView === "library" ? "返回已学词条"
        : activeView === "stats" ? "返回学习统计" : "返回今日学习";
  $("#learning-detail-back").replaceChildren(element("span", "", "←"), document.createTextNode(` ${detailReturnLabel}`));
  $("#learning-panel").classList.add("hidden");
  page.classList.remove("hidden");
  page.scrollTop = 0;
  const panel = $("#learning-entry-detail");
  panel.replaceChildren(element("div", "learning-loading", "正在读取词条详情…"));
  try {
    const data = entry.text && Array.isArray(entry.images) ? { entry } : await request(`/api/learning/entries/${encodeURIComponent(entry.id)}`);
    const fullEntry = data.entry || entry;
    panel.replaceChildren();
    const back = element("button", "learning-detail-back-inline", `← ${detailReturnLabel}`);
    back.type = "button";
    back.addEventListener("click", returnFromEntryDetail);
    panel.append(back);
    const title = element("h2", "learning-detail-title", fullEntry.title || entry.title);
    title.id = "learning-detail-title";
    panel.append(title);
    panel.append(element("p", "learning-entry-source", sourceLabel(fullEntry)));
    panel.append(element("p", "learning-detail-description", fullEntry.text || "暂无文字说明。"));
    appendImages(panel, fullEntry);
    const actions = element("div", "learning-detail-actions");
    if (favorite || listId) {
      const remove = element("button", "button button-secondary", favorite ? "取消收藏" : "从词单移除");
      remove.type = "button";
      remove.addEventListener("click", async () => {
        remove.disabled = true;
        try {
          if (favorite) {
            await request(`/api/learning/entries/${encodeURIComponent(entry.id)}/favorite`, {
              method: "POST", body: JSON.stringify({ favorite: false }),
            });
          } else {
            await updateListEntry(listId, entry.id, false, false);
          }
          returnFromEntryDetail();
          status().textContent = favorite ? "已从收藏中移除。" : "已从词单移除。";
          window.linguangToast?.(favorite ? `已取消收藏“${entry.title}”` : `已从词单移除“${entry.title}”`);
          if (favorite) await loadFavorites();
          else await loadLists();
        } catch (error) {
          status().textContent = error.message;
          remove.disabled = false;
        }
      });
      actions.append(remove);
    }
    const close = element("button", "button button-primary", "返回列表");
    close.type = "button";
    close.addEventListener("click", returnFromEntryDetail);
    actions.append(close);
    panel.append(actions);
  } catch (error) {
    panel.replaceChildren(element("div", "learning-empty", error.message));
  }
}

function returnFromEntryDetail() {
  const page = $("#learning-entry-page");
  page.classList.add("hidden");
  $("#learning-panel").classList.remove("hidden");
  window.scrollTo(0, detailReturnScrollY);
}

function setView(view) {
  activeView = view;
  document.querySelectorAll(".learning-tab").forEach((tab) => {
    const selected = tab.dataset.learningView === view;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll(".learning-view").forEach((panel) => panel.classList.toggle("hidden", panel.id !== `learning-${view}-view`));
  status().textContent = "";
  if (view === "today") loadToday();
  if (view === "library") loadLibrary();
  if (view === "favorites") loadFavorites();
  if (view === "lists") loadLists();
  if (view === "stats") loadStats(statsDays);
}

async function loadToday() {
  const area = $("#learning-card-area");
  area.replaceChildren(element("div", "learning-loading", "正在读取今日学习计划…"));
  $("#learning-queue-empty").classList.add("hidden");
  try {
    const data = await request("/api/learning/today");
    const counts = data.counts || {};
    const initialCards = [...(data.due || []), ...(data.new || [])];
    const repeats = [...(data.todayRepeat || [])];
    todayQueue = [];
    while (initialCards.length) {
      todayQueue.push(...initialCards.splice(0, 2));
      if (repeats.length) todayQueue.push(repeats.shift());
    }
    todayQueue.push(...repeats);
    cardIndex = 0;
    $("#learning-today-date").textContent = counts.today || "今天";
    $("#learning-streak").textContent = counts.streak || 0;
    $("#learning-completed").textContent = counts.completed || 0;
    $("#learning-daily-target").value = String(data.settings?.dailyNewTarget || 5);
    $("#learning-queue-summary").textContent = `待复习 ${counts.due || 0} 个 · 新词 ${counts.new || 0} 个`;
    renderLearningCard();
  } catch (error) {
    area.replaceChildren(element("div", "learning-empty", error.message));
  }
}

function renderLearningCard() {
  const area = $("#learning-card-area");
  area.replaceChildren();
  const entry = todayQueue[cardIndex];
  if (!entry) {
    $("#learning-queue-empty").classList.remove("hidden");
    return;
  }
  $("#learning-queue-empty").classList.add("hidden");
  const card = element("article", "learning-study-card");
  const top = element("div", "learning-card-top");
  top.append(element("span", `learning-kind ${entry.isNew ? "is-new" : "is-review"}`, entry.isNew ? "新词" : "复习"));
  top.append(element("span", "learning-progress", `${cardIndex + 1} / ${todayQueue.length}`));
  card.append(top);
  card.append(element("h3", "learning-word", entry.title));
  card.append(element("p", "learning-card-source", sourceLabel(entry)));
  if (entry.todayRepeat) {
    const left = Math.max(0, Number(entry.masteryRequired || 0) - Number(entry.masterySuccesses || 0));
    card.append(element("p", "learning-mastery-progress", `这个词还需要连续答对 ${left} 次`));
  }
  const reveal = element("button", "button button-primary learning-reveal", "查看动作说明");
  reveal.type = "button";
  card.append(reveal);
  const details = element("div", "learning-answer hidden");
  details.append(element("p", "learning-original-text", entry.text || "词条暂时没有文字说明。"));
  appendImages(details, entry);
  details.append(element("p", "learning-feedback-prompt", "回想动作后，选择你这次记得的程度："));
  const ratings = element("div", "learning-rating-buttons");
  [["forgot", "忘记了"], ["fuzzy", "有点模糊"], ["remembered", "记得"]].forEach(([rating, label]) => {
    const button = element("button", `learning-rating rating-${rating}`, label);
    button.type = "button";
    button.addEventListener("click", () => submitReview(entry, rating, button));
    ratings.append(button);
  });
  details.append(ratings);
  reveal.addEventListener("click", () => {
    details.classList.remove("hidden");
    reveal.classList.add("hidden");
    details.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  card.append(details);
  area.append(card);
}

async function submitReview(entry, rating, button) {
  const buttons = [...button.parentElement.querySelectorAll("button")];
  buttons.forEach((item) => { item.disabled = true; });
  try {
    const eventId = globalThis.crypto?.randomUUID?.() || `${Date.now()}_${Math.random().toString(36).slice(2, 12)}`;
    const result = await request("/api/learning/review", { method: "POST", body: JSON.stringify({ entryId: entry.id, rating, eventId }) });
    if (result.dayComplete) $("#learning-completed").textContent = Number($("#learning-completed").textContent || 0) + 1;
    if (result.repeatToday) {
      entry.todayRepeat = true;
      entry.masterySuccesses = result.masterySuccesses;
      entry.masteryRequired = result.masteryRequired;
      const nextOtherCard = todayQueue.findIndex((item, index) => index > cardIndex && item.id !== entry.id);
      const insertAt = Math.min(todayQueue.length, Math.max(cardIndex + 2, nextOtherCard < 0 ? todayQueue.length : nextOtherCard + 2));
      todayQueue.splice(insertAt, 0, entry);
      $("#learning-queue-summary").textContent = `待复习 ${Math.max(0, todayQueue.length - cardIndex - 1)} 个（含稍后重练）`;
    }
    cardIndex += 1;
    renderLearningCard();
  } catch (error) {
    status().textContent = error.message;
    buttons.forEach((item) => { item.disabled = false; });
  }
}

async function loadLibrary() {
  const list = $("#learning-entry-list");
  list.replaceChildren(element("div", "learning-loading", "正在读取已学词条…"));
  try {
    list.replaceChildren();
    const data = await request("/api/learning/stats/words?category=learned&days=30");
    if (!data.groups?.length) list.append(element("div", "learning-empty", "还没有已学词条。完成一次今日学习后，词条会出现在这里。"));
    data.groups?.forEach((group) => {
      const section = element("details", "learning-learned-date");
      section.append(element("summary", "learning-date-heading", `${dateGroupLabel(group.date)}（${group.entries.length}）`));
      const words = element("div", "learning-date-entries");
      group.entries.forEach((word) => words.append(renderWordRow(word)));
      section.append(words);
      list.append(section);
    });
  } catch (error) {
    list.replaceChildren(element("div", "learning-empty", error.message));
  }
}

async function loadLists() {
  $("#learning-custom-lists").replaceChildren();
  try {
    const listsCache = await request("/api/learning/lists");
    listsCache.lists?.forEach((item) => {
      const section = element("details", "learning-list-group learning-custom-list");
      const summary = element("summary", "learning-custom-list-summary", `${item.title}（${item.entries.length}）`);
      const heading = element("div", "learning-list-heading");
      const rename = element("button", "text-button", "重命名");
      rename.type = "button";
      rename.addEventListener("click", async () => {
        const title = prompt("输入新的词单名称", item.title)?.trim();
        if (!title || title === item.title) return;
        try { await request(`/api/learning/lists/${encodeURIComponent(item.id)}`, { method: "PATCH", body: JSON.stringify({ title }) }); await loadLists(); window.linguangToast?.(`词单已重命名为“${title}”`); }
        catch (error) { status().textContent = error.message; }
      });
      const remove = element("button", "text-button", "删除词单");
      remove.type = "button";
      remove.addEventListener("click", async () => {
        if (!confirm(`删除词单“${item.title}”？词条收藏和学习记录不会删除。`)) return;
        try { await request(`/api/learning/lists/${encodeURIComponent(item.id)}`, { method: "DELETE" }); await loadLists(); window.linguangToast?.(`已删除词单“${item.title}”`); }
        catch (error) { status().textContent = error.message; }
      });
      heading.append(rename, remove);
      section.append(summary, heading);
      if (!item.entries.length) section.append(element("p", "learning-empty-inline", "这个词单还没有词条。可在手语词典中添加。"));
      item.entries.forEach((entry) => section.append(renderWordRow(entry, { listId: item.id })));
      $("#learning-custom-lists").append(section);
    });
    if (!listsCache.lists?.length) {
      $("#learning-custom-lists").append(element("div", "learning-empty", "还没有主题词单。可以新建词单，再从手语词典添加词语。"));
    }
  } catch (error) { $("#learning-custom-lists").replaceChildren(element("div", "learning-empty", error.message)); }
}

async function loadFavorites() {
  const favorites = $("#learning-favorites");
  favorites.replaceChildren(element("div", "learning-loading", "正在读取收藏…"));
  try {
    const listsCache = await request("/api/learning/lists");
    favorites.replaceChildren();
    if (!listsCache.favorites?.length) favorites.append(element("div", "learning-empty", "还没有收藏。可以在手语词典中收藏想学的词。"));
    else renderGroupedEntries(favorites, listsCache.favorites, (entry) => entry.favoritedAt, { favorite: true });
  } catch (error) {
    favorites.replaceChildren(element("div", "learning-empty", error.message));
  }
}

async function updateListEntry(listId, entryId, add, refresh = true) {
  await request(`/api/learning/lists/${encodeURIComponent(listId)}/entries/${encodeURIComponent(entryId)}`, { method: add ? "POST" : "DELETE", ...(add ? { body: "{}" } : {}) });
  if (refresh) await loadLists();
}

async function loadStats(days) {
  statsDays = days;
  const chart = $("#learning-chart");
  chart.replaceChildren(element("div", "learning-loading", "正在读取统计…"));
  try {
    const data = await request(`/api/learning/stats?days=${days}`);
    const totals = data.totals || {};
    const stats = [["今日已学", totals.todayWords || 0, "today"], ["练习次数", totals.practiceCount || 0, "all"], ["记得", totals.remembered || 0, "remembered"], ["有点模糊", totals.fuzzy || 0, "fuzzy"], ["忘记了", totals.forgot || 0, "forgot"], ["待复习", totals.due || 0, "due"]];
    const totalArea = $("#learning-stat-totals");
    totalArea.replaceChildren(...stats.map(([label, value, category]) => {
      const block = element("button", "learning-stat", "");
      block.type = "button";
      block.setAttribute("aria-expanded", "false");
      block.append(element("span", "learning-label", label), element("strong", "", String(value)));
      block.addEventListener("click", () => loadStatWords(category, label, block));
      return block;
    }));
    chart.replaceChildren();
    chart.style.gridTemplateColumns = `repeat(${Math.max(1, data.chart?.length || 1)}, minmax(12px, 1fr))`;
    const max = Math.max(1, ...(data.chart || []).map((day) => day.total));
    (data.chart || []).forEach((day) => {
      const item = element("div", "learning-chart-day");
      const value = element("span", "learning-chart-value", String(day.total));
      const bar = element("div", "learning-chart-bar");
      bar.style.setProperty("--bar-height", `${Math.max(day.total ? 5 : 2, (day.total / max) * 100)}%`);
      bar.setAttribute("aria-hidden", "true");
      item.title = `${day.date}：${day.total} 次`;
      item.append(value, bar, element("span", "learning-chart-date", day.date.slice(5)));
      chart.append(item);
    });
    const hard = $("#learning-hard-words");
    hard.replaceChildren();
    if (!data.hardWords?.length) hard.append(element("p", "learning-empty-inline", "今天没有忘记的词，也没有连续多天答错的词。"));
    data.hardWords?.forEach((word) => {
      const button = element("button", "learning-hard-word", `${word.title} · ${word.reason}`);
      button.type = "button";
      button.addEventListener("click", () => openEntryDetail(word));
      hard.append(button);
    });
  } catch (error) { chart.replaceChildren(element("div", "learning-empty", error.message)); }
}

async function loadStatWords(category, title, selectedCard) {
  const details = $("#learning-stat-details");
  const content = $("#learning-stat-detail-content");
  const isOpen = !details.classList.contains("hidden") && details.dataset.category === category;
  document.querySelectorAll(".learning-stat").forEach((card) => card.setAttribute("aria-expanded", "false"));
  if (isOpen) { details.classList.add("hidden"); return; }
  details.classList.remove("hidden");
  details.dataset.category = category;
  selectedCard.setAttribute("aria-expanded", "true");
  $("#learning-stat-detail-close").classList.remove("hidden");
  content.replaceChildren(element("div", "learning-loading", `正在读取“${title}”词汇…`));
  try {
    const data = await request(`/api/learning/stats/words?category=${encodeURIComponent(category)}&days=${statsDays}`);
    content.replaceChildren(element("h3", "learning-stat-detail-title", `${title}对应的词汇`));
    if (!data.groups?.length) content.append(element("p", "learning-empty-inline", "这个统计周期内没有对应词汇。"));
    data.groups?.forEach((group) => {
      const dateGroup = element("details", "learning-date-group");
      const groupTitle = category === "today"
        ? (group.type === "new" ? "今日新词" : "今日复习")
        : dateGroupLabel(group.date);
      dateGroup.append(element("summary", "learning-date-heading", `${groupTitle}（${group.entries.length}）`));
      const words = element("div", "learning-date-entries");
      group.entries.forEach((word) => words.append(renderWordRow(word)));
      dateGroup.append(words);
      content.append(dateGroup);
    });
  } catch (error) { content.replaceChildren(element("div", "learning-empty", error.message)); }
}

export function initializeLearning(options = {}) {
  if (initialized) return;
  initialized = true;
  apiBase = options.apiBase || "";
  document.querySelectorAll(".learning-tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.learningView)));
  $("#learning-daily-target").addEventListener("change", async (event) => {
    try {
      await request("/api/learning/settings", { method: "PUT", body: JSON.stringify({ dailyNewTarget: Number(event.currentTarget.value) }) });
      await loadToday();
    } catch (error) { status().textContent = error.message; }
  });
  $("#learning-refresh").addEventListener("click", loadToday);
  $("#learning-detail-back").addEventListener("click", returnFromEntryDetail);
  $("#learning-stat-detail-close").addEventListener("click", () => {
    $("#learning-stat-details").classList.add("hidden");
    document.querySelectorAll(".learning-stat").forEach((card) => card.setAttribute("aria-expanded", "false"));
  });
  $("#learning-create-list").addEventListener("click", async () => {
    const title = prompt("给词单起个名字，例如：日常问候")?.trim();
    if (!title) return;
    try { await request("/api/learning/lists", { method: "POST", body: JSON.stringify({ title }) }); await loadLists(); window.linguangToast?.(`已新建词单“${title}”`); }
    catch (error) { status().textContent = error.message; }
  });
  document.querySelectorAll("[data-stats-days]").forEach((button) => button.addEventListener("click", () => {
    statsDays = Number(button.dataset.statsDays);
    document.querySelectorAll("[data-stats-days]").forEach((item) => { const selected = item === button; item.classList.toggle("active", selected); item.setAttribute("aria-pressed", String(selected)); });
    loadStats(statsDays);
  }));
  window.linguangLearning = { open: () => setView(activeView) };
}
