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

modelLabel?.addEventListener("click", (e) => {
  e.stopPropagation();
  modelDropdown?.classList.toggle("open");
});

document.addEventListener("click", () => {
  modelDropdown?.classList.remove("open");
});

document.querySelectorAll(".model-option").forEach((el) => {
  el.addEventListener("click", () => {
    const model = el.getAttribute("data-model");
    if (!model) return;
    document.querySelectorAll(".model-option").forEach((o) => o.classList.remove("active"));
    el.classList.add("active");
    if (modelLabel) modelLabel.textContent = "▾ " + model;
    modelDropdown?.classList.remove("open");
    vscode.postMessage({ type: "set_model", model });
  });
});

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
      if (modelLabel) modelLabel.textContent = "▾ " + msg.model;
      document.querySelectorAll(".model-option").forEach((el) => {
        el.classList.toggle("active", el.getAttribute("data-model") === msg.model);
      });
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
