"use strict";

const modelDefaults = {
  openai: "gpt-4o-mini",
  gemini: "gemini-1.5-flash",
  deepseek: "deepseek-chat",
  ollama: "llama3.1",
  gigachat: "GigaChat",
};
const embeddingDefaults = [
  "sentence-transformers/all-MiniLM-L6-v2",
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "sentence-transformers/all-mpnet-base-v2",
];
const paramDefaults = {
  chunkSize: 1200,
  overlap: 200,
  topK: 4,
  minScore: 0.2,
};

const state = {
  mode: "cloud",
  provider: "openai",
  model: modelDefaults.openai,
  cloudReady: false,
  localReady: false,
  webReady: false,
  cloudConfirmed: false,
  defaults: {
    chunk_size: paramDefaults.chunkSize,
    overlap: paramDefaults.overlap,
    top_k: paramDefaults.topK,
    min_score: paramDefaults.minScore,
  },
  indexMeta: {
    cloud: {},
    local: {},
    web: {},
  },
  paramTouched: {
    chunkSize: false,
    overlap: false,
    topK: false,
    minScore: false,
  },
};

const elements = {
  modeButtons: Array.from(document.querySelectorAll("[data-mode]")),
  modePanels: Array.from(document.querySelectorAll("[data-panel]")),
  statusCloud: document.getElementById("status-cloud"),
  statusLocal: document.getElementById("status-local"),
  statusWeb: document.getElementById("status-web"),
  paramChunkSize: document.getElementById("param-chunk-size"),
  paramOverlap: document.getElementById("param-overlap"),
  paramTopK: document.getElementById("param-top-k"),
  paramMinScore: document.getElementById("param-min-score"),
  buildCloud: document.getElementById("build-cloud"),
  buildLocal: document.getElementById("build-local"),
  buildWeb: document.getElementById("build-web"),
  cloudHint: document.getElementById("cloud-hint"),
  localFiles: document.getElementById("local-files"),
  webUrl: document.getElementById("web-url"),
  webCrawl: document.getElementById("web-crawl"),
  webAllowlist: document.getElementById("web-allowlist"),
  webMaxPages: document.getElementById("web-max-pages"),
  provider: document.getElementById("provider"),
  model: document.getElementById("model"),
  embeddingModel: document.getElementById("embedding-model"),
  chatStatus: document.getElementById("chat-status"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  toast: document.getElementById("toast"),
  refreshHealth: document.getElementById("refresh-health"),
  progressCloud: document.getElementById("progress-cloud"),
  progressLocal: document.getElementById("progress-local"),
  progressWeb: document.getElementById("progress-web"),
  progressCloudLabel: document.getElementById("progress-cloud-label"),
  progressLocalLabel: document.getElementById("progress-local-label"),
  progressWebLabel: document.getElementById("progress-web-label"),
  progressCloudMessage: document.getElementById("progress-cloud-message"),
  progressLocalMessage: document.getElementById("progress-local-message"),
  progressWebMessage: document.getElementById("progress-web-message"),
  progressCloudMeta: document.getElementById("progress-cloud-label").parentElement,
  progressLocalMeta: document.getElementById("progress-local-label").parentElement,
  progressWebMeta: document.getElementById("progress-web-label").parentElement,
};

const jobPollers = {
  cloud: null,
  local: null,
  web: null,
};

const progressMap = {
  cloud: {
    bar: elements.progressCloud,
    label: elements.progressCloudLabel,
    message: elements.progressCloudMessage,
    meta: elements.progressCloudMeta,
  },
  local: {
    bar: elements.progressLocal,
    label: elements.progressLocalLabel,
    message: elements.progressLocalMessage,
    meta: elements.progressLocalMeta,
  },
  web: {
    bar: elements.progressWeb,
    label: elements.progressWebLabel,
    message: elements.progressWebMessage,
    meta: elements.progressWebMeta,
  },
};

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatInline(text) {
  let escaped = escapeHtml(text);
  const codePlaceholders = [];
  escaped = escaped.replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${codePlaceholders.length}@@`;
    codePlaceholders.push(`<code>${code}</code>`);
    return token;
  });
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  codePlaceholders.forEach((value, idx) => {
    escaped = escaped.replace(`@@CODE${idx}@@`, value);
  });
  return escaped;
}

function renderMarkdown(text) {
  if (!text) {
    return "";
  }
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let inCode = false;
  let codeLines = [];
  let listType = null;
  let listItemOpen = false;

  const closeListItem = () => {
    if (listItemOpen) {
      html += "</li>";
      listItemOpen = false;
    }
  };

  const flushList = () => {
    if (listType) {
      closeListItem();
      html += `</${listType}>`;
      listType = null;
    }
  };

  const flushCode = () => {
    if (inCode) {
      html += `<pre><code>${codeLines.join("\n")}</code></pre>`;
      codeLines = [];
      inCode = false;
    }
  };

  const openList = (type) => {
    if (listType && listType !== type) {
      flushList();
    }
    if (!listType) {
      listType = type;
      html += `<${type}>`;
    }
  };

  const openListItem = (content) => {
    closeListItem();
    html += `<li>${content}`;
    listItemOpen = true;
  };

  lines.forEach((line) => {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        flushCode();
      } else {
        if (!listItemOpen) {
          flushList();
        }
        inCode = true;
        codeLines = [];
      }
      return;
    }

    if (inCode) {
      codeLines.push(escapeHtml(line));
      return;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      if (listItemOpen) {
        html += '<div class="paragraph-gap"></div>';
      } else {
        flushList();
        html += '<div class="paragraph-gap"></div>';
      }
      return;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.+)/);
    if (ulMatch) {
      openList("ul");
      openListItem(formatInline(ulMatch[1]));
      return;
    }

    const olMatch = trimmed.match(/^\d+\.\s+(.+)/);
    if (olMatch) {
      openList("ol");
      openListItem(formatInline(olMatch[1]));
      return;
    }

    if (listItemOpen) {
      html += `<p>${formatInline(line)}</p>`;
      return;
    }

    flushList();
    html += `<p>${formatInline(line)}</p>`;
  });

  flushCode();
  flushList();
  return html;
}

function setToast(message) {
  if (!message) {
    return;
  }
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.setTimeout(() => elements.toast.classList.remove("show"), 2400);
}

function updateProgress(mode, progress, message, status) {
  const payload = progressMap[mode];
  if (!payload) {
    return;
  }
  const value = typeof progress === "number" ? Math.round(progress) : 0;
  payload.bar.style.width = `${value}%`;
  payload.label.textContent = `${value}%`;
  payload.message.textContent = message || "";
  payload.meta.classList.remove("is-done", "is-error");
  if (status === "done") {
    payload.meta.classList.add("is-done");
  }
  if (status === "error") {
    payload.meta.classList.add("is-error");
  }
}

function setStatusPill(pill, ready) {
  pill.classList.remove("is-ready", "is-warning", "is-error", "neutral");
  if (ready === null) {
    pill.textContent = "неизвестно";
    pill.classList.add("neutral");
    return;
  }
  if (ready) {
    pill.textContent = "готово";
    pill.classList.add("is-ready");
  } else {
    pill.textContent = "не готово";
    pill.classList.add("is-warning");
  }
}

function updateStatus() {
  setStatusPill(elements.statusCloud, state.cloudReady);
  setStatusPill(elements.statusLocal, state.localReady);
  setStatusPill(elements.statusWeb, state.webReady);
  updateChatState();
}

function setMode(mode) {
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  elements.modePanels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === mode);
  });
  applyParamsFromMode();
  updateChatState();
}

function isModeReady() {
  if (state.mode === "cloud") {
    return state.cloudReady && state.cloudConfirmed;
  }
  if (state.mode === "local") {
    return state.localReady;
  }
  return state.webReady;
}

function updateChatState() {
  const ready = isModeReady();
  elements.chatInput.disabled = !ready;
  elements.sendBtn.disabled = !ready;
  if (state.mode === "cloud" && state.cloudReady && !state.cloudConfirmed) {
    elements.chatStatus.textContent = "создайте индекс";
    elements.chatStatus.className = "status-pill is-warning";
    elements.cloudHint.textContent =
      "Для документации нужно пересобрать индекс.";
  } else {
    elements.chatStatus.textContent = ready ? "готово" : "индекс не готов";
    elements.chatStatus.className = `status-pill ${ready ? "is-ready" : "neutral"}`;
    elements.cloudHint.textContent =
      "Сначала постройте эмбеддинги для этого режима.";
  }
}

function toInt(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function toFloat(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function markParamTouched(key) {
  if (state.paramTouched && key in state.paramTouched) {
    state.paramTouched[key] = true;
  }
}

function setParamValue(element, value, key) {
  if (!element || (state.paramTouched && state.paramTouched[key])) {
    return;
  }
  if (value === null || value === undefined || Number.isNaN(value)) {
    return;
  }
  element.value = value;
}

function applyParamsFromMode() {
  const defaults = state.defaults || {};
  const meta = state.indexMeta?.[state.mode] || {};
  const chunkSize = toInt(meta.chunk_size);
  const overlap = toInt(meta.overlap);
  setParamValue(
    elements.paramChunkSize,
    chunkSize ?? defaults.chunk_size ?? paramDefaults.chunkSize,
    "chunkSize",
  );
  setParamValue(
    elements.paramOverlap,
    overlap ?? defaults.overlap ?? paramDefaults.overlap,
    "overlap",
  );
  setParamValue(
    elements.paramTopK,
    defaults.top_k ?? paramDefaults.topK,
    "topK",
  );
  setParamValue(
    elements.paramMinScore,
    defaults.min_score ?? paramDefaults.minScore,
    "minScore",
  );
}

function getParamInt(element) {
  return toInt(element ? element.value : null);
}

function getParamFloat(element) {
  return toFloat(element ? element.value : null);
}

function getEmbeddingModel() {
  const value = elements.embeddingModel ? elements.embeddingModel.value.trim() : "";
  return value;
}

function setEmbeddingOptions(models, selected) {
  if (!elements.embeddingModel) {
    return;
  }
  elements.embeddingModel.innerHTML = "";
  const unique = Array.from(new Set(models.filter(Boolean)));
  unique.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    elements.embeddingModel.appendChild(option);
  });
  if (selected && unique.includes(selected)) {
    elements.embeddingModel.value = selected;
  } else if (unique.length > 0) {
    elements.embeddingModel.value = unique[0];
  }
}

async function fetchEmbeddingModels() {
  try {
    const response = await fetch("/embedding-models");
    if (!response.ok) {
      setEmbeddingOptions(embeddingDefaults, embeddingDefaults[0]);
      return;
    }
    const payload = await response.json();
    const models = Array.isArray(payload.models) ? payload.models : embeddingDefaults;
    const selected = payload.default || models[0] || embeddingDefaults[0];
    setEmbeddingOptions(models, selected);
  } catch (error) {
    setEmbeddingOptions(embeddingDefaults, embeddingDefaults[0]);
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    state.cloudReady = Boolean(payload.index_ready);
    state.localReady = Boolean(payload.local_index_ready);
    state.webReady = Boolean(payload.web_index_ready);
    state.defaults = payload.defaults || state.defaults;
    state.indexMeta = payload.indexes || state.indexMeta;
    applyParamsFromMode();
    updateStatus();
  } catch (error) {
    setToast("Не удалось получить статус.");
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = JSON.parse(text);
  } catch (error) {
    payload = null;
  }
  return { response, text, payload };
}

function stopPolling(mode) {
  if (jobPollers[mode]) {
    window.clearInterval(jobPollers[mode]);
    jobPollers[mode] = null;
  }
}

async function pollJob(jobId, mode) {
  stopPolling(mode);
  jobPollers[mode] = window.setInterval(async () => {
    try {
      const response = await fetch(`/index-jobs/${jobId}`);
      if (!response.ok) {
        return;
      }
      const job = await response.json();
      updateProgress(mode, job.progress, job.message, job.status);
      if (job.status === "done") {
        stopPolling(mode);
        if (mode === "cloud") {
          state.cloudReady = true;
          state.cloudConfirmed = true;
        } else if (mode === "local") {
          state.localReady = true;
        } else {
          state.webReady = true;
        }
        await fetchHealth();
        setToast("Индекс обновлен.");
      }
      if (job.status === "error") {
        stopPolling(mode);
        setToast(job.error || "Ошибка при построении индекса.");
      }
    } catch (error) {
      stopPolling(mode);
      setToast("Не удалось получить прогресс.");
    }
  }, 1200);
}

async function buildCloudIndex() {
  setToast("Строим индекс Cloud...");
  state.cloudConfirmed = false;
  updateProgress("cloud", 2, "запуск", "running");
  const payload = { async_build: true };
  const embeddingModel = getEmbeddingModel();
  if (embeddingModel) {
    payload.embedding_model = embeddingModel;
  }
  const chunkSize = getParamInt(elements.paramChunkSize);
  const overlap = getParamInt(elements.paramOverlap);
  if (chunkSize !== null) {
    payload.chunk_size = chunkSize;
  }
  if (overlap !== null) {
    payload.overlap = overlap;
  }
  const { response, text, payload: data } = await postJson("/cloud-index", payload);
  if (!response.ok) {
    setToast(text || "Не удалось построить индекс Cloud.");
    return;
  }
  if (data && data.job_id) {
    pollJob(data.job_id, "cloud");
    return;
  }
  state.cloudReady = true;
  state.cloudConfirmed = true;
  await fetchHealth();
  setToast("Индекс Cloud обновлен.");
}

async function buildLocalIndex() {
  const files = elements.localFiles.files;
  if (!files || files.length === 0) {
    setToast("Загрузите хотя бы один файл.");
    return;
  }
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    formData.append("files", file, file.name);
  });
  setToast("Строим локальный индекс...");
  updateProgress("local", 2, "загрузка файлов", "running");
  const embeddingModel = getEmbeddingModel();
  const params = new URLSearchParams();
  params.set("async_build", "true");
  if (embeddingModel) {
    params.set("embedding_model", embeddingModel);
  }
  const chunkSize = getParamInt(elements.paramChunkSize);
  const overlap = getParamInt(elements.paramOverlap);
  if (chunkSize !== null) {
    params.set("chunk_size", String(chunkSize));
  }
  if (overlap !== null) {
    params.set("overlap", String(overlap));
  }
  const response = await fetch(`/local-index?${params.toString()}`, {
    method: "POST",
    body: formData,
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = JSON.parse(text);
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    setToast(text || "Не удалось построить локальный индекс.");
    return;
  }
  if (payload && payload.job_id) {
    pollJob(payload.job_id, "local");
    return;
  }
  state.localReady = true;
  await fetchHealth();
  setToast("Локальный индекс обновлен.");
}

async function buildWebIndex() {
  const url = elements.webUrl.value.trim();
  if (!url) {
    setToast("Введите ссылку на сайт.");
    return;
  }
  const payload = {
    url,
    crawl: elements.webCrawl.checked,
    max_pages: Number(elements.webMaxPages.value || 30),
  };
  const embeddingModel = getEmbeddingModel();
  if (embeddingModel) {
    payload.embedding_model = embeddingModel;
  }
  const chunkSize = getParamInt(elements.paramChunkSize);
  const overlap = getParamInt(elements.paramOverlap);
  if (chunkSize !== null) {
    payload.chunk_size = chunkSize;
  }
  if (overlap !== null) {
    payload.overlap = overlap;
  }
  const allowlist = elements.webAllowlist.value.trim();
  if (allowlist) {
    payload.allowlist = allowlist;
  }
  setToast("Строим индекс сайта...");
  updateProgress("web", 2, "запуск", "running");
  const { response, text, payload: data } = await postJson("/web-index", {
    ...payload,
    async_build: true,
  });
  if (!response.ok) {
    setToast(text || "Не удалось построить индекс сайта.");
    return;
  }
  if (data && data.job_id) {
    pollJob(data.job_id, "web");
    return;
  }
  state.webReady = true;
  await fetchHealth();
  setToast("Индекс сайта обновлен.");
}

function stripSources(text) {
  const marker = "\nSources:";
  const idx = text.indexOf(marker);
  if (idx >= 0) {
    return text.slice(0, idx).trim();
  }
  const ruMarker = "\nИсточники:";
  const ruIdx = text.indexOf(ruMarker);
  if (ruIdx >= 0) {
    return text.slice(0, ruIdx).trim();
  }
  return text.trim();
}

function appendMessage(role, content) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "Вы" : "Репетитор";
  const body = document.createElement("div");
  body.className = "content";
  body.textContent = content;
  wrapper.appendChild(meta);
  wrapper.appendChild(body);
  elements.chatLog.appendChild(wrapper);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  return wrapper;
}

function renderAssistantContent(wrapper, payload) {
  const body = wrapper.querySelector(".content");
  const answerText = stripSources(payload.answer || "");
  body.innerHTML = renderMarkdown(answerText || "Ответ не получен.");

  const selfCheck = payload.self_check_question;
  if (selfCheck) {
    const selfBlock = document.createElement("div");
    selfBlock.className = "self-check";
    selfBlock.textContent = `Самопроверка: ${selfCheck}`;
    wrapper.appendChild(selfBlock);
  }

  const quizzes = Array.isArray(payload.quizzes)
    ? payload.quizzes
    : payload.quiz
      ? [payload.quiz]
      : [];
  if (quizzes.length > 0) {
    quizzes.forEach((quiz, idx) => {
      if (!quiz || !quiz.question || !Array.isArray(quiz.options)) {
        return;
      }
      const quizBlock = document.createElement("div");
      quizBlock.className = "quiz";
      const title = document.createElement("h4");
      title.textContent = quizzes.length > 1 ? `Тест ${idx + 1}` : "Проверочный тест";
      const question = document.createElement("p");
      question.textContent = quiz.question;
      const options = document.createElement("div");
      options.className = "quiz-options";
      const feedback = document.createElement("div");
      feedback.className = "quiz-feedback";
      const correctIndex = Number(quiz.correct_index);

      quiz.options.forEach((option, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = option;
        button.addEventListener("click", () => {
          if (quizBlock.dataset.answered === "1") {
            return;
          }
          quizBlock.dataset.answered = "1";
          if (index === correctIndex) {
            button.classList.add("is-correct");
            feedback.textContent = "Верно!";
          } else {
            button.classList.add("is-wrong");
            const correct = quiz.options[correctIndex] || "ответ";
            feedback.textContent = `Неверно. Правильный ответ: ${correct}.`;
          }
        });
        options.appendChild(button);
      });

      quizBlock.appendChild(title);
      quizBlock.appendChild(question);
      quizBlock.appendChild(options);
      quizBlock.appendChild(feedback);
      wrapper.appendChild(quizBlock);
    });
  }

  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  if (citations.length > 0) {
    const sources = document.createElement("div");
    sources.className = "sources";
    const title = document.createElement("h4");
    title.textContent = "Источники";
    const list = document.createElement("ul");
    citations.forEach((item) => {
      const li = document.createElement("li");
      const label = item.title || item.url || "источник";
      if (item.url && item.url.startsWith("http")) {
        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = label;
        li.appendChild(link);
      } else {
        li.textContent = label;
      }
      list.appendChild(li);
    });
    sources.appendChild(title);
    sources.appendChild(list);
    wrapper.appendChild(sources);
  }
}

async function sendQuestion(text) {
  const message = text.trim();
  if (!message) {
    return;
  }
  if (!isModeReady()) {
    setToast("Сначала постройте индекс.");
    return;
  }
  appendMessage("user", message);
  const assistantMessage = appendMessage("assistant", "Думаю...");
  elements.chatInput.value = "";
  const body = {
    question: message,
    mode: state.mode,
    provider: elements.provider.value,
    model: elements.model.value,
  };
  const topK = getParamInt(elements.paramTopK);
  const minScore = getParamFloat(elements.paramMinScore);
  if (topK !== null) {
    body.top_k = topK;
  }
  if (minScore !== null) {
    body.min_score = minScore;
  }
  const { response, text: rawText, payload } = await postJson("/query", body);
  if (!response.ok) {
    assistantMessage.querySelector(".content").textContent =
      rawText || "Не удалось получить ответ.";
    return;
  }
  if (payload) {
    renderAssistantContent(assistantMessage, payload);
    return;
  }
  try {
    const parsed = JSON.parse(rawText || "{}");
    renderAssistantContent(assistantMessage, parsed);
  } catch (error) {
    assistantMessage.querySelector(".content").textContent = "Ошибка разбора ответа.";
  }
}

function bindEvents() {
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  elements.provider.addEventListener("change", (event) => {
    const nextProvider = event.target.value;
    elements.model.value = modelDefaults[nextProvider] || "";
  });

  if (elements.paramChunkSize) {
    elements.paramChunkSize.addEventListener("input", () => markParamTouched("chunkSize"));
  }
  if (elements.paramOverlap) {
    elements.paramOverlap.addEventListener("input", () => markParamTouched("overlap"));
  }
  if (elements.paramTopK) {
    elements.paramTopK.addEventListener("input", () => markParamTouched("topK"));
  }
  if (elements.paramMinScore) {
    elements.paramMinScore.addEventListener("input", () => markParamTouched("minScore"));
  }

  elements.buildCloud.addEventListener("click", buildCloudIndex);
  elements.buildLocal.addEventListener("click", buildLocalIndex);
  elements.buildWeb.addEventListener("click", buildWebIndex);
  elements.refreshHealth.addEventListener("click", fetchHealth);

  elements.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendQuestion(elements.chatInput.value);
  });

  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuestion(elements.chatInput.value);
    }
  });
}

function init() {
  elements.provider.value = state.provider;
  elements.model.value = state.model;
  fetchEmbeddingModels();
  setMode(state.mode);
  bindEvents();
  fetchHealth();
  updateStatus();
}

init();
