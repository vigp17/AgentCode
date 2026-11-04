// webview script — no ts-check (acquireVsCodeApi is injected at runtime)
const vscode = acquireVsCodeApi();

const messagesEl = document.getElementById("messages");
const inputEl = /** @type {HTMLTextAreaElement} */ (document.getElementById("input"));
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const modelLabel = document.getElementById("model-label");
const modelDropdown = document.getElementById("model-dropdown");

let currentAssistantEl = null;
let currentToolsEl = null;
let busy = false;

// ── Send ────────────────────────────────────────────────────────────────────

function send() {
  const content = inputEl.value.trim();
  if (!content || busy) return;

  appendUserMessage(content);
  inputEl.value = "";
  inputEl.style.height = "auto";
  setBusy(true);
  vscode.postMessage({ type: "send", content });
}

sendBtn?.addEventListener("click", send);
inputEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
inputEl?.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
});
clearBtn?.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  currentAssistantEl = null;
  currentToolsEl = null;
  setBusy(false);
  vscode.postMessage({ type: "clear" });
});

// ── Model picker ─────────────────────────────────────────────────────────────
// The list is pushed by the backend ("models" message): every model your API
// keys can use, discovered live from Anthropic/OpenAI/Google, plus AgentCode's
// fine-tunes once they're configured. FALLBACK_MODELS only covers the moment
// before the first result arrives (or when discovery is turned off).

const FALLBACK_MODELS = [
  { id: "claude-sonnet-5", provider: "anthropic", label: "Sonnet 5" },
  { id: "claude-opus-5", provider: "anthropic", label: "Opus 5" },
  { id: "claude-fable-5", provider: "anthropic", label: "Fable 5" },
  { id: "claude-haiku-4-5", provider: "anthropic", label: "Haiku 4.5" },
  { id: "gpt-5.6-sol", provider: "openai", label: "GPT-5.6 Sol" },
  { id: "gpt-5.6-terra", provider: "openai", label: "GPT-5.6 Terra" },
  { id: "gpt-5.6-luna", provider: "openai", label: "GPT-5.6 Luna" },
  { id: "gemini/gemini-3.1-pro-preview", provider: "gemini", label: "Gemini 3.1 Pro" },
  { id: "gemini/gemini-3.6-flash", provider: "gemini", label: "Gemini 3.6 Flash" },
  { id: "gemini/gemini-3.5-flash-lite", provider: "gemini", label: "Gemini 3.5 Flash-Lite" },
];

const PROVIDER_ORDER = ["agentcode", "anthropic", "openai", "gemini"];
const PROVIDER_LABELS = {
  agentcode: "AgentCode (fine-tuned)",
  anthropic: "Claude",
  openai: "GPT",
  gemini: "Gemini",
};
const PER_GROUP = 8; // newest N per provider; the filter box searches all

const savedState = vscode.getState() || {};
let models = savedState.models && savedState.models.length ? savedState.models : FALLBACK_MODELS;
let currentModel = savedState.currentModel || "claude-sonnet-5";
let modelFilter = "";
const expandedGroups = new Set();

function setModels(list) {
  // A provider discovery returned nothing for (no key, offline) keeps its
  // fallback entries so the picker never loses a whole group.
  const found = new Set(list.map((m) => m.provider));
  models = [...list, ...FALLBACK_MODELS.filter((m) => !found.has(m.provider))];
  vscode.setState({ ...(vscode.getState() || {}), models: list, currentModel });
  renderModelDropdown();
}

function renderModelDropdown() {
  if (!modelDropdown) return;
  const q = modelFilter.toLowerCase();
  modelDropdown.innerHTML = "";

  const filter = document.createElement("input");
  filter.id = "model-filter";
  filter.placeholder = "Filter models…";
  filter.value = modelFilter;
  filter.addEventListener("click", (e) => e.stopPropagation());
  filter.addEventListener("input", () => {
    modelFilter = filter.value;
    renderModelDropdown();
    const again = document.getElementById("model-filter");
    again?.focus();
    again?.setSelectionRange(again.value.length, again.value.length);
  });
  modelDropdown.appendChild(filter);

  const providers = [...new Set([...PROVIDER_ORDER, ...models.map((m) => m.provider)])];
  for (const provider of providers) {
    const rows = models.filter(
      (m) => m.provider === provider &&
        (!q || m.id.toLowerCase().includes(q) || (m.label || "").toLowerCase().includes(q))
    );
    if (!rows.length) continue;
    const group = document.createElement("div");
    group.className = "model-group";
    const label = document.createElement("div");
    label.className = "model-group-label";
    label.textContent = PROVIDER_LABELS[provider] || provider;
    group.appendChild(label);

    const showAll = q || expandedGroups.has(provider);
    const visible = showAll ? rows : rows.slice(0, PER_GROUP);
    // Keep the active model visible even if it's older than the newest N.
    const active = rows.find((m) => m.id === currentModel);
    if (active && !visible.includes(active)) visible.push(active);

    for (const m of visible) {
      const opt = document.createElement("div");
      opt.className = "model-option" + (m.id === currentModel ? " active" : "");
      opt.dataset.model = m.id;
      opt.textContent = m.label || m.id;
      if (m.label && m.label !== m.id) {
        const sub = document.createElement("span");
        sub.className = "model-id";
        sub.textContent = m.id;
        opt.appendChild(sub);
      }
      opt.addEventListener("click", () => selectModel(m.id));
      group.appendChild(opt);
    }
    if (!showAll && rows.length > visible.length) {
      const more = document.createElement("div");
      more.className = "model-more";
      more.textContent = `Show ${rows.length - visible.length} more…`;
      more.addEventListener("click", (e) => {
        e.stopPropagation();
        expandedGroups.add(provider);
        renderModelDropdown();
      });
      group.appendChild(more);
    }
    modelDropdown.appendChild(group);
  }

  const refresh = document.createElement("div");
  refresh.className = "model-refresh";
  refresh.textContent = "↻ Check for new models";
  refresh.addEventListener("click", (e) => {
    e.stopPropagation();
    refresh.textContent = "Checking…";
    vscode.postMessage({ type: "refresh_models" });
  });
  modelDropdown.appendChild(refresh);
}

function setCurrentModel(model) {
  currentModel = model;
  if (modelLabel) modelLabel.textContent = "▾ " + model;
  vscode.setState({ ...(vscode.getState() || {}), currentModel: model });
  renderModelDropdown();
}

function selectModel(model) {
  modelDropdown?.classList.remove("open");
  setCurrentModel(model);
  vscode.postMessage({ type: "set_model", model });
}

modelLabel?.addEventListener("click", (e) => {
  e.stopPropagation();
  modelDropdown?.classList.toggle("open");
});

modelDropdown?.addEventListener("click", (e) => e.stopPropagation());

document.addEventListener("click", () => {
  modelDropdown?.classList.remove("open");
});

renderModelDropdown();
if (modelLabel) modelLabel.textContent = "▾ " + currentModel;

// ── Message rendering ────────────────────────────────────────────────────────

function appendUserMessage(content) {
  currentAssistantEl = null;
  currentToolsEl = null;
  const div = document.createElement("div");
  div.className = "message user";
  div.innerHTML = `<div class="message-role">You</div>
    <div class="message-content">${escHtml(content)}</div>`;
  messagesEl?.appendChild(div);
  scrollToBottom();
}

function ensureAssistantMessage() {
  if (!currentAssistantEl) {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.innerHTML = `<div class="message-role">AgentCode</div>
      <div class="message-content"></div>`;
    messagesEl?.appendChild(div);
    currentAssistantEl = div.querySelector(".message-content");
    currentToolsEl = null;
  }
  return currentAssistantEl;
}

function appendToolCall(name, args) {
  if (!currentToolsEl) {
    currentToolsEl = document.createElement("div");
    currentToolsEl.className = "tool-calls";
    messagesEl?.appendChild(currentToolsEl);
  }
  const el = document.createElement("div");
  el.className = "tool-call";
  el.dataset.tool = name;
  const argsStr = formatArgs(args);
  el.innerHTML = `<div class="spinner"></div><span><b>${escHtml(name)}</b>${argsStr ? " · " + argsStr : ""}</span>`;
  currentToolsEl.appendChild(el);
  scrollToBottom();
  return el;
}

function markToolDone(name) {
  const el = currentToolsEl?.querySelector(`[data-tool="${name}"]:not(.done)`);
  if (el) el.classList.add("done");
}

function appendError(message) {
  const div = document.createElement("div");
  div.className = "error-message";
  div.textContent = "⚠ " + message;
  messagesEl?.appendChild(div);
  scrollToBottom();
}

function appendCostBadge(cost) {
  if (!cost) return;
  const div = document.createElement("div");
  div.className = "cost-badge";
  div.textContent = `$${cost.toFixed(4)}`;
  messagesEl?.appendChild(div);
}

// ── Extension → webview messages ────────────────────────────────────────────

window.addEventListener("message", (event) => {
  const msg = event.data;

  switch (msg.type) {
    case "ready":
      setCurrentModel(msg.model);
      break;

    case "models":
      if (Array.isArray(msg.models) && msg.models.length) setModels(msg.models);
      else renderModelDropdown();
      break;

    case "text":
      ensureAssistantMessage().textContent += msg.delta;
      scrollToBottom();
      break;

    case "tool_call":
      appendToolCall(msg.name, msg.args);
      break;

    case "tool_result":
      markToolDone(msg.name);
      break;

    case "done":
      currentAssistantEl = null;
      currentToolsEl = null;
      appendCostBadge(msg.cost);
      setBusy(false);
      break;

    case "error":
      appendError(msg.message);
      setBusy(false);
      break;

    case "cleared":
      messagesEl.innerHTML = "";
      currentAssistantEl = null;
      currentToolsEl = null;
      setBusy(false);
      break;
  }
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function setBusy(state) {
  busy = state;
  if (sendBtn) sendBtn.disabled = state;
  if (inputEl) inputEl.disabled = state;
}

function scrollToBottom() {
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatArgs(args) {
  const interesting = ["path", "command", "pattern", "message"];
  for (const key of interesting) {
    if (args[key]) return escHtml(String(args[key]).slice(0, 60));
  }
  return "";
}
