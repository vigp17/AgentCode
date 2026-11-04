// webview script — no ts-check (acquireVsCodeApi is injected at runtime)
const vscode = acquireVsCodeApi();

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const emptyState = $("empty-state");
const inputEl = /** @type {HTMLTextAreaElement} */ ($("input"));
const sendBtn = $("send-btn");
const clearBtn = $("clear-btn");
const modelButton = $("model-button");
const modelNameEl = $("model-name");
const modelDot = $("model-dot");
const modelDropdown = $("model-dropdown");
const effortEl = $("effort");
const hintEl = $("composer-hint");
const slashMenu = $("slash-menu");

let currentAssistantEl = null;
let currentToolsEl = null;
let busy = false;

// ── Models ──────────────────────────────────────────────────────────────────
// The backend pushes the list ("models" message): the AgentCode lineup plus
// every frontier model your API keys can use, discovered live. FALLBACK_MODELS
// only covers the moment before the first result arrives.

const ECOSYSTEM = [
  { id: "axon-1", provider: "agentcode", label: "axon-1", kind: "chat", supports_effort: true,
    available: false, description: "Unified code model: everyday coding, reads and surgical edits" },
  { id: "tandav-1", provider: "agentcode", label: "tandav-1", kind: "chat", available: false,
    description: "Apex architect: hardest multi-file work, up to 50 agent steps" },
  { id: "veda-1", provider: "agentcode", label: "veda-1", kind: "chat", available: false,
    description: "Deep research: docs, web context, cited specs" },
  { id: "chitra-1", provider: "agentcode", label: "chitra-1", kind: "image", available: false,
    description: "Visual assets: icons, diagrams, images into your workspace" },
];

const FALLBACK_MODELS = [
  ...ECOSYSTEM,
  { id: "claude-sonnet-5", provider: "anthropic", label: "Sonnet 5" },
  { id: "claude-opus-5", provider: "anthropic", label: "Opus 5" },
  { id: "claude-haiku-4-5", provider: "anthropic", label: "Haiku 4.5" },
  { id: "gpt-5.6-sol", provider: "openai", label: "GPT-5.6 Sol" },
  { id: "gpt-5.6-terra", provider: "openai", label: "GPT-5.6 Terra" },
  { id: "gpt-5.6-luna", provider: "openai", label: "GPT-5.6 Luna" },
  { id: "gemini/gemini-3.1-pro-preview", provider: "gemini", label: "Gemini 3.1 Pro" },
  { id: "gemini/gemini-3.6-flash", provider: "gemini", label: "Gemini 3.6 Flash" },
];

const PROVIDER_ORDER = ["agentcode", "anthropic", "openai", "gemini"];
const PROVIDER_LABELS = {
  agentcode: "AgentCode",
  anthropic: "Claude",
  openai: "GPT",
  gemini: "Gemini",
};
const PER_GROUP = 6; // newest N per frontier provider; the filter box searches all
const EFFORTS = ["low", "medium", "high"];
const EFFORT_HINTS = {
  low: "Low effort · fast answers",
  medium: "Medium effort · standard coding",
  high: "High effort · deep multi-file fixes",
};

const saved = vscode.getState() || {};
let models = mergeWithFallback(saved.models || []);
let currentModel = saved.currentModel || "claude-sonnet-5";
let effort = EFFORTS.includes(saved.effort) ? saved.effort : "medium";
let modelFilter = "";
const expandedGroups = new Set();
// A switch requested mid-turn is sent when the turn ends: while a tool
// approval is pending the backend only reads permission responses.
let pendingSwitch = null;

function persist() {
  vscode.setState({ ...(vscode.getState() || {}), currentModel, effort });
}

function mergeWithFallback(list) {
  const found = new Set(list.map((m) => m.provider));
  const ids = new Set(list.map((m) => m.id));
  return [
    ...list,
    // The ecosystem is always listed, even before the backend reports it.
    ...ECOSYSTEM.filter((m) => !ids.has(m.id)),
    ...FALLBACK_MODELS.filter((m) => m.provider !== "agentcode" && !found.has(m.provider)),
  ];
}

function findModel(id) {
  return models.find((m) => m.id === id);
}

function setModels(list) {
  models = mergeWithFallback(list);
  vscode.setState({ ...(vscode.getState() || {}), models: list });
  renderModelDropdown();
  renderComposerState();
}

function renderModelDropdown() {
  if (!modelDropdown) return;
  const q = modelFilter.toLowerCase();
  modelDropdown.innerHTML = "";

  const filter = document.createElement("input");
  filter.id = "model-filter";
  filter.placeholder = "Search models…";
  filter.value = modelFilter;
  filter.addEventListener("input", () => {
    modelFilter = filter.value;
    renderModelDropdown();
    const again = $("model-filter");
    again?.focus();
    again?.setSelectionRange(again.value.length, again.value.length);
  });
  filter.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDropdown();
    if (e.key === "Enter") {
      const first = modelDropdown.querySelector(".model-option");
      if (first) selectModel(first.dataset.model);
    }
  });
  modelDropdown.appendChild(filter);

  const providers = [...new Set([...PROVIDER_ORDER, ...models.map((m) => m.provider)])];
  for (const provider of providers) {
    const rows = models.filter(
      (m) => m.provider === provider &&
        (!q || m.id.toLowerCase().includes(q) || (m.label || "").toLowerCase().includes(q) ||
          (m.description || "").toLowerCase().includes(q))
    );
    if (!rows.length) continue;

    const group = document.createElement("div");
    group.className = "model-group";
    const label = document.createElement("div");
    label.className = "model-group-label";
    label.textContent = PROVIDER_LABELS[provider] || provider;
    group.appendChild(label);

    const showAll = q || provider === "agentcode" || expandedGroups.has(provider);
    const visible = showAll ? rows : rows.slice(0, PER_GROUP);
    const active = rows.find((m) => m.id === currentModel);
    if (active && !visible.includes(active)) visible.push(active);

    for (const m of visible) group.appendChild(modelOption(m));

    if (!showAll && rows.length > visible.length) {
      const more = document.createElement("div");
      more.className = "model-more";
      more.textContent = `Show ${rows.length - visible.length} more…`;
      more.addEventListener("click", () => {
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
  refresh.addEventListener("click", () => {
    refresh.textContent = "Checking…";
    vscode.postMessage({ type: "refresh_models" });
  });
  modelDropdown.appendChild(refresh);
}

function modelOption(m) {
  const opt = document.createElement("div");
  opt.className = "model-option";
  if (m.id === currentModel) opt.classList.add("active");
  if (m.available === false) opt.classList.add("unavailable");
  opt.dataset.model = m.id;
  opt.setAttribute("role", "option");

  const title = document.createElement("span");
  title.className = "model-title";
  title.textContent = m.label || m.id;
  if (m.kind === "image") title.appendChild(badge("image"));
  if (m.supports_effort) title.appendChild(badge("effort"));
  if (m.available === false) title.appendChild(badge("not connected", true));
  opt.appendChild(title);

  const check = document.createElement("span");
  check.className = "check";
  check.textContent = "✓";
  opt.appendChild(check);

  const subText = m.description || (m.label && m.label !== m.id ? m.id : "");
  if (subText) {
    const sub = document.createElement("span");
    sub.className = "model-sub";
    sub.textContent = subText;
    opt.appendChild(sub);
  }
  opt.addEventListener("click", () => selectModel(m.id));
  return opt;
}

function badge(text, off = false) {
  const b = document.createElement("span");
  b.className = "badge" + (off ? " off" : "");
  b.textContent = text;
  return b;
}

function openDropdown() {
  modelFilter = "";
  renderModelDropdown();
  modelDropdown?.classList.add("open");
  modelButton?.setAttribute("aria-expanded", "true");
  $("model-filter")?.focus();
}

function closeDropdown() {
  modelDropdown?.classList.remove("open");
  modelButton?.setAttribute("aria-expanded", "false");
}

modelButton?.addEventListener("click", (e) => {
  e.stopPropagation();
  modelDropdown?.classList.contains("open") ? closeDropdown() : openDropdown();
});
modelDropdown?.addEventListener("click", (e) => e.stopPropagation());
document.addEventListener("click", () => {
  closeDropdown();
  hideSlashMenu();
});

// ── Model / effort switching (→ server.py over stdio) ───────────────────────

function selectModel(id) {
  closeDropdown();
  if (!id) return;
  applyModelLocally(id);
  sendOrQueue({ type: "set_model", model: id });
  inputEl?.focus();
}

function selectEffort(level) {
  if (!EFFORTS.includes(level)) return;
  effort = level;
  persist();
  renderComposerState();
  sendOrQueue({ type: "set_effort", effort: level });
}

function sendOrQueue(msg) {
  if (busy) {
    pendingSwitch = { ...(pendingSwitch || {}), [msg.type]: msg };
    notice("Applies after this reply finishes");
    return;
  }
  vscode.postMessage(msg);
}

function flushPendingSwitch() {
  if (!pendingSwitch) return;
  for (const msg of Object.values(pendingSwitch)) vscode.postMessage(msg);
  pendingSwitch = null;
}

function applyModelLocally(id) {
  currentModel = id;
  persist();
  renderComposerState();
  renderModelDropdown();
}

function renderComposerState() {
  const m = findModel(currentModel);
  if (modelNameEl) modelNameEl.textContent = currentModel;
  modelDot?.classList.toggle("off", m?.available === false);
  if (modelButton) {
    modelButton.title = m?.available === false
      ? `${currentModel} isn't connected yet`
      : `Model: ${currentModel}`;
  }

  // Reasoning-effort pills: only for models that support effort (axon-1).
  const showEffort = Boolean(m?.supports_effort);
  if (effortEl) effortEl.hidden = !showEffort;
  effortEl?.querySelectorAll(".effort-pill").forEach((pill) => {
    pill.setAttribute("aria-checked", String(pill.dataset.effort === effort));
    pill.tabIndex = pill.dataset.effort === effort ? 0 : -1;
  });

  let hint = "";
  if (m?.available === false) hint = "Not connected · see finetune/README.md";
  else if (m?.kind === "image") hint = "Image model · /image coming soon";
  else if (showEffort) hint = EFFORT_HINTS[effort];
  if (hintEl) hintEl.textContent = hint;

  if (inputEl) {
    inputEl.placeholder = m?.kind === "image"
      ? "Describe an image to generate…"
      : `Message ${currentModel}…  (/ for commands)`;
  }
}

effortEl?.addEventListener("click", (e) => {
  const pill = e.target.closest(".effort-pill");
  if (pill) selectEffort(pill.dataset.effort);
});
effortEl?.addEventListener("keydown", (e) => {
  // Radio-group arrow keys.
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  const i = EFFORTS.indexOf(effort) + (e.key === "ArrowRight" ? 1 : -1);
  selectEffort(EFFORTS[(i + EFFORTS.length) % EFFORTS.length]);
  effortEl.querySelector(`[data-effort="${effort}"]`)?.focus();
});

// ── Slash commands & autocomplete ───────────────────────────────────────────

const COMMANDS = [
  { name: "/model", arg: "<name>", desc: "Switch model" },
  { name: "/effort", arg: "<low|medium|high>", desc: "Set reasoning effort (axon-1)" },
  { name: "/clear", arg: "", desc: "Start a new chat" },
];

let slashItems = [];
let slashIndex = 0;

/** Suggestions for the current input, or [] when it isn't a slash command. */
function slashSuggestions(text) {
  if (!text.startsWith("/") || text.includes("\n")) return [];

  const modelArg = text.match(/^\/model\s+(\S*)$/);
  if (modelArg) {
    const q = modelArg[1].toLowerCase();
    return models
      .filter((m) => !q || m.id.toLowerCase().includes(q) || (m.label || "").toLowerCase().includes(q))
      .sort((a, b) => rank(a, q) - rank(b, q))
      .slice(0, 8)
      .map((m) => ({
        main: m.id,
        sub: m.available === false ? "not connected" : (m.description || m.label || ""),
        insert: `/model ${m.id}`,
        run: () => selectModel(m.id),
      }));
  }

  const effortArg = text.match(/^\/effort\s+(\S*)$/);
  if (effortArg) {
    return EFFORTS.filter((lvl) => lvl.startsWith(effortArg[1].toLowerCase())).map((lvl) => ({
      main: lvl,
      sub: EFFORT_HINTS[lvl],
      insert: `/effort ${lvl}`,
      run: () => selectEffort(lvl),
    }));
  }

  if (/^\/\S*$/.test(text)) {
    return COMMANDS.filter((c) => c.name.startsWith(text)).map((c) => ({
      main: `${c.name} ${c.arg}`.trim(),
      sub: c.desc,
      insert: c.arg ? `${c.name} ` : c.name,
      // Commands with an argument complete to "/model " and keep the menu
      // open for the argument; /clear runs straight away.
      run: c.arg ? null : () => runCommand(c.name),
    }));
  }
  return [];
}

function rank(m, q) {
  // Prefix matches first, then the AgentCode lineup, then everything else.
  const prefix = q && m.id.toLowerCase().startsWith(q) ? 0 : 1;
  const eco = m.provider === "agentcode" ? 0 : 1;
  return prefix * 2 + eco;
}

function updateSlashMenu() {
  slashItems = slashSuggestions(inputEl.value);
  if (!slashItems.length) return hideSlashMenu();
  slashIndex = Math.min(slashIndex, slashItems.length - 1);
  slashMenu.innerHTML = "";
  slashItems.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "slash-item" + (i === slashIndex ? " selected" : "");
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(i === slashIndex));
    const main = document.createElement("span");
    main.className = "slash-main";
    main.textContent = item.main;
    const sub = document.createElement("span");
    sub.className = "slash-sub";
    sub.textContent = item.sub || "";
    row.append(main, sub);
    row.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep focus in the textarea
      acceptSlash(i);
    });
    slashMenu.appendChild(row);
  });
  slashMenu.hidden = false;
  slashMenu.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
}

function hideSlashMenu() {
  if (slashMenu) slashMenu.hidden = true;
  slashItems = [];
  slashIndex = 0;
}

function acceptSlash(i) {
  const item = slashItems[i];
  if (!item) return;
  if (item.run) {
    clearInput();
    hideSlashMenu();
    item.run();
  } else {
    inputEl.value = item.insert;
    slashIndex = 0;
    updateSlashMenu();
  }
}

/** Run a fully typed slash command. Returns true if the text was one. */
function runCommand(text) {
  const [cmd, ...rest] = text.trim().split(/\s+/);
  const arg = rest.join(" ");
  if (cmd === "/clear") {
    clearChat();
    return true;
  }
  if (cmd === "/model") {
    if (!arg) notice(`Current model: ${currentModel}`);
    else {
      selectModel(arg);
      notice(`Switched to ${arg}`);
    }
    return true;
  }
  if (cmd === "/effort") {
    if (EFFORTS.includes(arg)) {
      selectEffort(arg);
      notice(`Effort: ${arg}`);
    } else notice("Usage: /effort low | medium | high");
    return true;
  }
  return false;
}

// ── Send ────────────────────────────────────────────────────────────────────

function send() {
  const content = inputEl.value.trim();
  if (!content || busy) return;
  hideSlashMenu();
  if (content.startsWith("/") && runCommand(content)) {
    clearInput();
    return;
  }
  appendUserMessage(content);
  clearInput();
  setBusy(true);
  vscode.postMessage({ type: "send", content });
}

function clearInput() {
  inputEl.value = "";
  autosize();
}

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
}

sendBtn?.addEventListener("click", send);

inputEl?.addEventListener("keydown", (e) => {
  const menuOpen = slashMenu && !slashMenu.hidden && slashItems.length;
  if (menuOpen) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const d = e.key === "ArrowDown" ? 1 : -1;
      slashIndex = (slashIndex + d + slashItems.length) % slashItems.length;
      updateSlashMenu();
      return;
    }
    if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
      e.preventDefault();
      acceptSlash(slashIndex);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideSlashMenu();
      return;
    }
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

inputEl?.addEventListener("input", () => {
  autosize();
  slashIndex = 0;
  updateSlashMenu();
});

document.querySelectorAll(".suggestion").forEach((el) => {
  el.addEventListener("click", () => {
    inputEl.value = el.dataset.prompt || "";
    send();
  });
});

function clearChat() {
  messagesEl.querySelectorAll(":scope > :not(#empty-state)").forEach((el) => el.remove());
  if (emptyState) emptyState.hidden = false;
  currentAssistantEl = null;
  currentToolsEl = null;
  setBusy(false);
  vscode.postMessage({ type: "clear" });
}

clearBtn?.addEventListener("click", clearChat);

// ── Message rendering ───────────────────────────────────────────────────────

function hideEmptyState() {
  if (emptyState) emptyState.hidden = true;
}

function appendUserMessage(content) {
  hideEmptyState();
  currentAssistantEl = null;
  currentToolsEl = null;
  const div = document.createElement("div");
  div.className = "message user";
  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  div.appendChild(body);
  messagesEl?.appendChild(div);
  scrollToBottom();
}

function ensureAssistantMessage() {
  if (!currentAssistantEl) {
    hideEmptyState();
    const div = document.createElement("div");
    div.className = "message assistant";
    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = currentModel;
    const body = document.createElement("div");
    body.className = "message-content";
    div.append(role, body);
    messagesEl?.appendChild(div);
    currentAssistantEl = body;
    currentToolsEl = null;
  }
  return currentAssistantEl;
}

function appendToolCall(name, args) {
  hideEmptyState();
  if (!currentToolsEl) {
    currentToolsEl = document.createElement("div");
    currentToolsEl.className = "tool-calls";
    messagesEl?.appendChild(currentToolsEl);
  }
  const el = document.createElement("div");
  el.className = "tool-call";
  el.dataset.tool = name;
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  const text = document.createElement("span");
  const b = document.createElement("b");
  b.textContent = name;
  text.appendChild(b);
  const argsStr = formatArgs(args);
  if (argsStr) text.append(" · " + argsStr);
  el.append(spinner, text);
  currentToolsEl.appendChild(el);
  scrollToBottom();
  return el;
}

function markToolDone(name) {
  const el = currentToolsEl?.querySelector(`[data-tool="${CSS.escape(name)}"]:not(.done)`);
  if (el) el.classList.add("done");
}

function appendError(message) {
  hideEmptyState();
  const div = document.createElement("div");
  div.className = "error-message";
  div.textContent = "⚠ " + message;
  messagesEl?.appendChild(div);
  scrollToBottom();
}

function notice(text) {
  hideEmptyState();
  const div = document.createElement("div");
  div.className = "notice";
  div.textContent = text;
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
      applyModelLocally(msg.model);
      if (msg.effort && msg.effort !== effort) {
        // The webview remembers the user's effort across backend restarts.
        vscode.postMessage({ type: "set_effort", effort });
      }
      break;

    case "model_changed":
      currentModel = msg.model;
      if (EFFORTS.includes(msg.effort)) effort = msg.effort;
      persist();
      renderComposerState();
      renderModelDropdown();
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
      appendToolCall(msg.name, msg.args || {});
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
      currentAssistantEl = null;
      currentToolsEl = null;
      setBusy(false);
      break;
  }
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function setBusy(state) {
  busy = state;
  if (sendBtn) sendBtn.disabled = state;
  if (inputEl) inputEl.disabled = state;
  if (!state) {
    flushPendingSwitch();
    inputEl?.focus();
  }
}

function scrollToBottom() {
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatArgs(args) {
  const interesting = ["path", "command", "pattern", "message"];
  for (const key of interesting) {
    if (args[key]) return String(args[key]).slice(0, 60);
  }
  return "";
}

// ── Init ────────────────────────────────────────────────────────────────────

renderModelDropdown();
renderComposerState();
