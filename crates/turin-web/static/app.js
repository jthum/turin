const state = {
  status: null,
  apps: [],
  selectedAppId: null,
  selectedScreenId: null,
  listCache: new Map(),
  loadingLists: new Set(),
  notices: [],
  refreshing: false,
};

const els = {
  appList: document.querySelector("#app-list"),
  appTitle: document.querySelector("#app-title"),
  appAbout: document.querySelector("#app-about"),
  connectionStatus: document.querySelector("#connection-status"),
  runtimeTarget: document.querySelector("#runtime-target"),
  refreshButton: document.querySelector("#refresh-button"),
  screenNav: document.querySelector("#screen-nav"),
  screen: document.querySelector("#screen"),
  notices: document.querySelector("#notice-stack"),
  emptyTemplate: document.querySelector("#empty-state-template"),
};

els.refreshButton.addEventListener("click", () => refresh({ reason: "manual" }));

bootstrap();

async function bootstrap() {
  await refresh({ reason: "initial" });
  connectEvents();
}

async function refresh({ reason } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  els.connectionStatus.textContent = reason === "initial" ? "Connecting..." : "Refreshing...";
  try {
    const status = await getJson("/api/status");
    state.status = status;
    state.apps = Object.values(status.ui?.apps ?? {});
    selectDefaults();
    await loadVisibleLists();
    render();
    els.connectionStatus.textContent = "Live";
  } catch (error) {
    pushNotice("error", "Refresh failed", error.message);
    els.connectionStatus.textContent = "Disconnected";
    render();
  } finally {
    state.refreshing = false;
  }
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.addEventListener("open", () => {
    els.connectionStatus.textContent = "Live";
  });
  source.addEventListener("runtime.snapshot", () => {
    invalidateLists();
    refresh({ reason: "event" });
  });
  source.addEventListener("ui.intent", () => {
    invalidateLists();
    refresh({ reason: "ui" });
  });
  source.addEventListener("harness.action_ran", () => {
    invalidateLists();
    refresh({ reason: "action" });
  });
  source.addEventListener("web.error", event => {
    pushNotice("error", "Event stream error", event.data || "Unknown event stream error");
    render();
  });
  source.addEventListener("error", () => {
    els.connectionStatus.textContent = "Reconnecting...";
  });
}

function selectDefaults() {
  if (!state.apps.length) {
    state.selectedAppId = null;
    state.selectedScreenId = null;
    return;
  }
  if (!state.apps.some(app => app.id === state.selectedAppId)) {
    state.selectedAppId = state.apps[0].id;
  }
  const app = selectedApp();
  const screenIds = Object.keys(app?.screens ?? {});
  if (!screenIds.length) {
    state.selectedScreenId = null;
    return;
  }
  if (!screenIds.includes(state.selectedScreenId)) {
    state.selectedScreenId = app.opens_with || screenIds[0];
  }
}

function render() {
  renderChrome();
  renderNotices();
  renderApps();
  renderScreens();
  renderScreen();
}

function renderChrome() {
  const app = selectedApp();
  const health = state.status?.snapshot?.health;
  els.runtimeTarget.textContent = health
    ? `${state.status.web.connection_kind} · ${state.status.web.connection_target}`
    : "Runtime";
  els.appTitle.textContent = app?.definition?.title || "Turin Web";
  els.appAbout.textContent =
    app?.definition?.about || "Semantic harness UI rendered in the browser.";
}

function renderApps() {
  clear(els.appList);
  if (!state.apps.length) {
    const node = els.emptyTemplate.content.cloneNode(true);
    els.appList.append(node);
    return;
  }
  for (const app of state.apps) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "nav-button";
    button.setAttribute("aria-current", app.id === state.selectedAppId ? "true" : "false");
    button.textContent = app.definition?.title || app.id;
    button.addEventListener("click", () => {
      state.selectedAppId = app.id;
      state.selectedScreenId = app.opens_with || Object.keys(app.screens ?? {})[0] || null;
      loadVisibleLists().then(render);
    });
    els.appList.append(button);
  }
}

function renderScreens() {
  clear(els.screenNav);
  const app = selectedApp();
  const screens = Object.values(app?.screens ?? {});
  if (!screens.length) {
    els.screenNav.innerHTML = `<p class="muted">No screens declared.</p>`;
    return;
  }

  for (const menu of app.menus ?? []) {
    const title = document.createElement("p");
    title.className = "eyebrow";
    title.textContent = menu.title;
    els.screenNav.append(title);
    for (const item of menu.items ?? []) renderMenuItem(item, 0);
  }

  if (!(app.menus ?? []).length) {
    for (const screen of screens) renderScreenButton(screen.id, screen.title, 0);
  }
}

function renderMenuItem(item, depth) {
  renderScreenButton(item.opens, item.label, depth, item.badge);
  for (const child of item.items ?? []) renderMenuItem(child, depth + 1);
}

function renderScreenButton(screenId, label, depth, badge) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "nav-button";
  button.style.marginLeft = `${depth * 0.75}rem`;
  button.setAttribute("aria-current", screenId === state.selectedScreenId ? "true" : "false");
  button.textContent = badge ? `${label} · ${badge}` : label;
  button.addEventListener("click", () => {
    state.selectedScreenId = screenId;
    loadVisibleLists().then(render);
  });
  els.screenNav.append(button);
}

function renderScreen() {
  clear(els.screen);
  const app = selectedApp();
  const screen = selectedScreen();
  if (!app || !screen) {
    const node = els.emptyTemplate.content.cloneNode(true);
    els.screen.append(node);
    return;
  }

  const heading = document.createElement("div");
  heading.className = "screen-heading";
  heading.innerHTML = `<h2>${escapeHtml(screen.title)}</h2><span class="muted code">${escapeHtml(screen.id)}</span>`;
  els.screen.append(heading);

  const stack = document.createElement("div");
  stack.className = "node-stack";
  for (const node of screen.nodes ?? []) {
    stack.append(renderNode(node, app));
  }
  if (!screen.nodes?.length) {
    stack.innerHTML = `<div class="empty-state"><span>Empty screen</span><p>This screen has no declared nodes.</p></div>`;
  }
  els.screen.append(stack);
}

function renderNode(node, app) {
  switch (node.kind) {
    case "section":
      return renderSection(node, app);
    case "text":
      return renderPanel(node.text, "text-node");
    case "action":
      return renderAction(node, app);
    case "list":
      return renderList(node);
    case "activity":
    case "detail":
    case "report":
    case "chart":
      return renderPlaceholder(node);
    case "form":
      return renderForm(node, app);
    default:
      return renderPanel(`Unsupported node kind: ${node.kind || "unknown"}`, "muted");
  }
}

function renderSection(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const stack = document.createElement("div");
  stack.className = "node-stack";
  for (const child of node.nodes ?? []) stack.append(renderNode(child, app));
  panel.append(stack);
  return panel;
}

function renderAction(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  const row = document.createElement("div");
  row.className = "action-row";
  const button = document.createElement("button");
  button.type = "button";
  button.className = node.confirm ? "danger-button" : "primary-button";
  button.textContent = node.label;
  button.addEventListener("click", () => runAction(node, app));
  row.append(button);
  panel.append(row);
  return panel;
}

function renderList(node) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const key = listKey(node);
  const cached = state.listCache.get(key);
  if (state.loadingLists.has(key)) {
    panel.append(renderText("Loading list...", "muted"));
    return panel;
  }
  if (!cached) {
    panel.append(renderText("List not loaded yet.", "muted"));
    return panel;
  }
  if (cached.error) {
    panel.append(renderText(cached.error, "muted"));
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    panel.append(renderText("No matching items.", "muted"));
    return panel;
  }

  const fields = node.fields?.length ? node.fields : ["title", "status", "kind", "priority"];
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${fields.map(field => `<th>${escapeHtml(field)}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const item of items) {
    const row = document.createElement("tr");
    row.innerHTML = fields
      .map(field => `<td>${escapeHtml(fieldValue(item, field))}</td>`)
      .join("");
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  panel.append(wrap);
  return panel;
}

function renderForm(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const form = document.createElement("form");
  form.className = "form-grid";
  for (const field of node.fields ?? []) {
    form.append(renderField(field));
  }
  const button = document.createElement("button");
  button.type = "submit";
  button.className = "primary-button";
  button.textContent = "Run";
  form.append(button);
  form.addEventListener("submit", event => {
    event.preventDefault();
    const params = { ...(node.params && typeof node.params === "object" ? node.params : {}) };
    const data = new FormData(form);
    for (const field of node.fields ?? []) {
      params[field.name] = coerceFieldValue(data.get(field.name), field.kind);
    }
    runAction({ action: node.action, label: node.title, params, confirm: false }, app);
  });
  panel.append(form);
  return panel;
}

function renderField(field) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";
  wrapper.innerHTML = `<span class="field-label">${escapeHtml(field.label)}</span>`;
  const kind = field.kind || "text";
  let input;
  if (kind === "textarea") {
    input = document.createElement("textarea");
  } else if (kind === "boolean") {
    input = document.createElement("select");
    input.innerHTML = `<option value="false">False</option><option value="true">True</option>`;
  } else if (field.options?.length) {
    input = document.createElement("select");
    for (const option of field.options) {
      const value = scalarLabel(option);
      input.append(new Option(value, value));
    }
  } else {
    input = document.createElement("input");
    input.type = kind === "number" || kind === "integer" ? "number" : "text";
  }
  input.name = field.name;
  if (field.required) input.required = true;
  if (field.default !== undefined && field.default !== null) input.value = scalarLabel(field.default);
  wrapper.append(input);
  return wrapper;
}

function renderPlaceholder(node) {
  const detail = node.source ? `Source: ${node.source}` : "No source";
  return renderPanel(`${node.title || node.kind}: ${detail}`, "muted");
}

function renderPanel(text, className) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.append(renderText(text, className));
  return panel;
}

function renderText(text, className) {
  const paragraph = document.createElement("p");
  paragraph.className = className;
  paragraph.textContent = text;
  return paragraph;
}

async function runAction(node, app) {
  if (node.confirm && !window.confirm(`Run ${node.label}?`)) return;
  try {
    const result = await postJson("/api/actions/run", {
      action: node.action,
      harness_id: app?.source?.harness_id || null,
      params: node.params ?? null,
    });
    pushNotice("success", "Action completed", `${result.result.action} finished.`);
    invalidateLists();
    await refresh({ reason: "action" });
  } catch (error) {
    pushNotice("error", "Action failed", error.message);
    render();
  }
}

async function loadVisibleLists() {
  const screen = selectedScreen();
  const nodes = flattenNodes(screen?.nodes ?? []);
  const lists = nodes.filter(node => node.kind === "list");
  await Promise.all(lists.map(loadList));
}

async function loadList(node) {
  const key = listKey(node);
  if (state.listCache.has(key) || state.loadingLists.has(key)) return;
  state.loadingLists.add(key);
  try {
    const payload = {
      source: node.source,
      where: node.where || {},
      limit: node.limit,
    };
    const response = await postJson("/api/ui/list", payload);
    state.listCache.set(key, response);
  } catch (error) {
    state.listCache.set(key, { error: error.message });
  } finally {
    state.loadingLists.delete(key);
  }
}

function invalidateLists() {
  state.listCache.clear();
}

function selectedApp() {
  return state.apps.find(app => app.id === state.selectedAppId) || null;
}

function selectedScreen() {
  const app = selectedApp();
  return app?.screens?.[state.selectedScreenId] || null;
}

function flattenNodes(nodes) {
  const out = [];
  for (const node of nodes) {
    out.push(node);
    if (node.kind === "section") out.push(...flattenNodes(node.nodes ?? []));
  }
  return out;
}

function listKey(node) {
  return JSON.stringify({
    source: node.source,
    where: node.where || {},
    limit: node.limit || null,
  });
}

function fieldValue(item, field) {
  if (item[field] !== undefined && item[field] !== null) return scalarLabel(item[field]);
  if (item.metadata?.[field] !== undefined && item.metadata[field] !== null) {
    return scalarLabel(item.metadata[field]);
  }
  return "";
}

function coerceFieldValue(value, kind) {
  if (kind === "number" || kind === "integer") return Number(value);
  if (kind === "boolean") return value === "true";
  return value ?? "";
}

function scalarLabel(value) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function renderNotices() {
  clear(els.notices);
  for (const notice of state.notices.slice(-4)) {
    const node = document.createElement("article");
    node.className = "notice";
    node.dataset.level = notice.level;
    node.innerHTML = `<strong>${escapeHtml(notice.title)}</strong><p>${escapeHtml(notice.body)}</p>`;
    els.notices.append(node);
  }
}

function pushNotice(level, title, body) {
  state.notices.push({ level, title, body });
  if (state.notices.length > 8) state.notices.shift();
}

async function getJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  return decodeJsonResponse(response);
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return decodeJsonResponse(response);
}

async function decodeJsonResponse(response) {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error?.message || `${response.status} ${response.statusText}`);
  }
  return body;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
