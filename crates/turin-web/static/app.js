const state = {
  status: null,
  apps: [],
  selectedAppId: null,
  selectedScreenId: null,
  listCache: new Map(),
  selectedListItems: new Map(),
  loadingLists: new Set(),
  formDrafts: new Map(),
  runningActions: new Set(),
  notices: [],
  latestActionResult: null,
  appliedStatusUiRequests: false,
  refreshing: false,
};

const ACTIVITY_LIMIT = 12;
const DETAIL_LIMIT = 25;
const REPORT_LIMIT = 100;
const CHART_LIMIT = 100;

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
    applyStatusUiRequestsOnce();
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
  source.addEventListener("ui.intent", event => {
    applyUiIntentEvent(event);
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

function applyStatusUiRequestsOnce() {
  if (state.appliedStatusUiRequests) return;
  state.appliedStatusUiRequests = true;
  const ui = state.status?.ui;
  for (const open of ui?.opens ?? []) applyUiIntentPayload({ ...open, type: "open" });
  for (const show of ui?.shows ?? []) applyUiIntentPayload({ ...show, type: "show" });
  for (const focus of ui?.focuses ?? []) applyUiIntentPayload({ ...focus, type: "focus" });
  for (const notice of ui?.notices ?? []) {
    applyUiIntentPayload({ ...notice, type: "notify" });
  }
}

function applyUiIntentEvent(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (error) {
    pushNotice("error", "UI event parse failed", error.message);
    return false;
  }
  return applyUiIntentPayload(payload);
}

function applyUiIntentPayload(payload) {
  switch (payload?.type) {
    case "open":
      return applyUiOpen(payload.app_id, payload.target, "open");
    case "show":
      return applyUiShow(payload.app_id, payload.target);
    case "focus":
      return applyUiFocus(payload.app_id, payload.target);
    case "notify":
      pushNotice(normalizeNoticeLevel(payload.level), payload.title || "UI notice", payload.body || "");
      return true;
    default:
      return false;
  }
}

function applyUiOpen(appId, target, label) {
  const app = selectAppById(appId);
  if (!app) return false;
  const screenId = screenIdForTarget(app, target);
  if (!screenId) {
    pushNotice("error", `UI ${label} failed`, `Target '${target}' is not a screen in '${appId}'.`);
    return false;
  }
  state.selectedScreenId = screenId;
  return true;
}

function applyUiShow(appId, target) {
  const app = selectAppById(appId);
  if (!app) return false;
  if (screenIdForTarget(app, target)) return applyUiOpen(appId, target, "show");
  if (app.panes?.[target]) {
    pushNotice("info", "Pane requested", `turin-web noted ui.show pane '${target}'.`);
    return true;
  }
  pushNotice("error", "UI show failed", `Target '${target}' is not a screen or pane in '${appId}'.`);
  return false;
}

function applyUiFocus(appId, target) {
  const app = selectAppById(appId);
  if (!app) return false;
  const screenId = focusScreenIdForTarget(app, target);
  if (!screenId) {
    pushNotice("error", "UI focus failed", `Target '${target}' was not found in '${appId}'.`);
    return false;
  }
  state.selectedScreenId = screenId;
  return true;
}

function selectAppById(appId) {
  const app = state.apps.find(candidate => candidate.id === appId);
  if (!app) {
    pushNotice("error", "UI request ignored", `App '${appId}' is not available.`);
    return null;
  }
  state.selectedAppId = app.id;
  return app;
}

function screenIdForTarget(app, target) {
  if (app?.screens?.[target]) return target;
  return Object.values(app?.screens ?? []).find(screen => screen.title === target)?.id || null;
}

function focusScreenIdForTarget(app, target) {
  const screenId = screenIdForTarget(app, target);
  if (screenId) return screenId;
  for (const screen of Object.values(app?.screens ?? {})) {
    if ((screen.nodes ?? []).some(node => nodeContainsTarget(node, target))) return screen.id;
  }
  return null;
}

function nodeContainsTarget(node, target) {
  if (node?.id === target) return true;
  switch (node?.kind) {
    case "section":
      return (node.nodes ?? []).some(child => nodeContainsTarget(child, target));
    case "action":
      return node.action === target || node.label === target;
    case "form":
      return node.action === target || node.title === target;
    default:
      return false;
  }
}

function normalizeNoticeLevel(level) {
  if (level === "success" || level === "warning" || level === "error") return level;
  return "info";
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
    for (const item of menu.items ?? []) renderMenuItem(app, item, 0);
  }

  if (!(app.menus ?? []).length) {
    for (const screen of screens) {
      renderScreenButton(app, screen.id, screen.title, 0, screen.presentation);
    }
  }
}

function renderMenuItem(app, item, depth) {
  renderScreenButton(app, item.opens, item.label, depth, item.badge);
  for (const child of item.items ?? []) renderMenuItem(app, child, depth + 1);
}

function renderScreenButton(app, screenId, label, depth, fallbackBadge) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "nav-button";
  button.style.marginLeft = `${depth * 0.75}rem`;
  button.setAttribute("aria-current", screenId === state.selectedScreenId ? "true" : "false");
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  button.append(labelNode);
  const badge = badgeForTarget(app, screenId, fallbackBadge);
  if (badge) {
    const badgeNode = document.createElement("span");
    badgeNode.className = "nav-badge";
    badgeNode.dataset.level = badge.level;
    badgeNode.textContent = badge.text;
    button.append(badgeNode);
  }
  button.addEventListener("click", () => {
    state.selectedScreenId = screenId;
    loadVisibleLists().then(render);
  });
  els.screenNav.append(button);
}

function badgeForTarget(app, target, fallback) {
  const dynamic = app?.badges?.[target];
  const text = badgeText(dynamic, fallback);
  if (!text) return null;
  return {
    text,
    level: dynamic?.level || "neutral",
  };
}

function badgeText(badge, fallback) {
  const label = badge?.label || fallback || null;
  const hasCount = badge?.count !== undefined && badge?.count !== null;
  if (label && hasCount) return `${label} ${badge.count}`;
  if (label) return label;
  if (hasCount) return String(badge.count);
  return null;
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
  if (state.latestActionResult) stack.append(renderActionResult(state.latestActionResult));
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
      return renderList(node, app);
    case "activity":
      return renderActivity(node);
    case "detail":
      return renderDetail(node, app);
    case "report":
      return renderReport(node);
    case "chart":
      return renderChart(node);
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
  const actionKey = actionRunKey(node.action);
  button.type = "button";
  button.className = node.confirm ? "danger-button" : "primary-button";
  button.disabled = state.runningActions.has(actionKey);
  button.textContent = state.runningActions.has(actionKey) ? "Running..." : node.label;
  button.addEventListener("click", () => runAction(node, app));
  row.append(button);
  panel.append(row);
  return panel;
}

function renderList(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const request = dataRequestForNode(node);
  if (!request) {
    panel.append(renderText("List source is missing.", "muted"));
    return panel;
  }
  const key = listKey(request);
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
  const selectedItem = selectedListItem(key, items);
  if (selectedItem) state.selectedListItems.set(key, itemKey(selectedItem));
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${fields.map(field => `<th>${escapeHtml(field)}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const item of items) {
    const row = document.createElement("tr");
    const selected = selectedItem && itemKey(item) === itemKey(selectedItem);
    row.className = "list-row";
    row.tabIndex = 0;
    row.setAttribute("aria-selected", selected ? "true" : "false");
    row.innerHTML = fields
      .map(field => `<td>${escapeHtml(fieldValue(item, field))}</td>`)
      .join("");
    const select = () => {
      state.selectedListItems.set(key, itemKey(item));
      render();
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      select();
    });
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  panel.append(wrap);
  if (selectedItem) {
    const detail = document.createElement("div");
    detail.className = "list-selection";
    detail.innerHTML = `<h4>Selected item</h4>`;
    detail.append(renderWorkItemDetail(selectedItem, app));
    panel.append(detail);
  }
  return panel;
}

function renderActivity(node) {
  const request = dataRequestForNode(node);
  if (!request) {
    return renderPanel(
      `${node.title}: no browser activity adapter exists for ${node.source}.`,
      "muted",
    );
  }
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    panel.append(renderText("Loading activity data...", "muted"));
    return panel;
  }
  if (!cached) {
    panel.append(renderText("Activity data not loaded yet.", "muted"));
    return panel;
  }
  if (cached.error) {
    panel.append(renderText(cached.error, "muted"));
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    panel.append(renderText("No worklist activity yet.", "muted"));
    return panel;
  }

  const list = document.createElement("div");
  list.className = "activity-list";
  for (const item of items.slice(0, ACTIVITY_LIMIT)) {
    const row = document.createElement("article");
    row.className = "activity-item";
    row.innerHTML = `
      <strong>${escapeHtml(item.title)}</strong>
      <span>${escapeHtml(item.status)} · ${escapeHtml(item.kind)} · priority ${escapeHtml(item.priority)}</span>
    `;
    list.append(row);
  }
  panel.append(list);
  return panel;
}

function renderDetail(node, app) {
  const request = dataRequestForNode(node);
  if (!request) {
    return renderPanel(
      `${node.title}: no browser detail adapter exists for ${node.source}.`,
      "muted",
    );
  }
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    panel.append(renderText("Loading detail data...", "muted"));
    return panel;
  }
  if (!cached) {
    panel.append(renderText("Detail data not loaded yet.", "muted"));
    return panel;
  }
  if (cached.error) {
    panel.append(renderText(cached.error, "muted"));
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    panel.append(renderText("No worklist items available for detail.", "muted"));
    return panel;
  }
  const item = selectDetailItem(node, items);
  if (!item) {
    panel.append(renderText(`Work item '${node.item_id}' was not found.`, "muted"));
    return panel;
  }
  panel.append(renderWorkItemDetail(item, app));
  return panel;
}

function renderWorkItemDetail(item, app) {
  const wrapper = document.createElement("div");
  wrapper.className = "detail-grid work-item-detail";
  const metadata = item.metadata && typeof item.metadata === "object" ? item.metadata : {};
  const fields = [
    ["Title", item.title],
    ["Status", item.status],
    ["Kind", item.kind],
    ["Priority", item.priority],
    ["Worklist", item.worklist_id],
    ["Release", metadata.release],
    ["Lane", metadata.lane],
    ["Updated", item.updated_at],
  ];
  wrapper.innerHTML = fields
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(
      ([label, value]) => `
        <div>
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(scalarLabel(value))}</strong>
        </div>
      `,
    )
    .join("");
  if (item.action?.name) {
    const actions = document.createElement("div");
    actions.className = "action-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "danger-button";
    button.textContent = `Run ${item.action.name}`;
    button.addEventListener("click", () => {
      runAction(
        {
          action: item.action.name,
          label: `Work item: ${item.title}`,
          params: item.action.params ?? null,
          confirm: true,
        },
        app,
      );
    });
    actions.append(button);
    wrapper.append(actions);
  }
  return wrapper;
}

function renderReport(node) {
  const request = dataRequestForNode(node);
  if (!request) return renderPlaceholder(node);
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  if (node.prompt) panel.append(renderText(node.prompt, "muted"));
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    panel.append(renderText("Loading report data...", "muted"));
    return panel;
  }
  if (!cached) {
    panel.append(renderText("Report data not loaded yet.", "muted"));
    return panel;
  }
  if (cached.error) {
    panel.append(renderText(cached.error, "muted"));
    return panel;
  }
  const items = cached.list?.items ?? [];
  panel.append(renderMetricGrid(reportMetrics(items)));
  return panel;
}

function renderChart(node) {
  const request = dataRequestForNode(node);
  if (!request) return renderPlaceholder(node);
  const panel = document.createElement("section");
  panel.className = "panel";
  const label = node.render_as ? `${node.intent || "breakdown"} · ${node.render_as}` : node.intent || "breakdown";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3><p class="muted">${escapeHtml(label)}</p>`;
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    panel.append(renderText("Loading chart data...", "muted"));
    return panel;
  }
  if (!cached) {
    panel.append(renderText("Chart data not loaded yet.", "muted"));
    return panel;
  }
  if (cached.error) {
    panel.append(renderText(cached.error, "muted"));
    return panel;
  }
  const items = cached.list?.items ?? [];
  panel.append(renderBars(groupCounts(items, chartGroupField(node))));
  return panel;
}

function renderMetricGrid(metrics) {
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  for (const metric of metrics) {
    const tile = document.createElement("article");
    tile.className = "metric-tile";
    tile.innerHTML = `<span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong>`;
    grid.append(tile);
  }
  return grid;
}

function renderBars(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const wrapper = document.createElement("div");
  wrapper.className = "chart-bars";
  if (!entries.length) {
    wrapper.append(renderText("No chart data yet.", "muted"));
    return wrapper;
  }
  const max = Math.max(...entries.map(([, count]) => count), 1);
  for (const [label, count] of entries) {
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <div><i style="width: ${(count / max) * 100}%"></i></div>
      <strong>${escapeHtml(count)}</strong>
    `;
    wrapper.append(row);
  }
  return wrapper;
}

function renderForm(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h3>${escapeHtml(node.title)}</h3>`;
  const form = document.createElement("form");
  const formKey = formDraftKey(node);
  form.className = "form-grid";
  for (const field of node.fields ?? []) {
    form.append(renderField(field, formKey));
  }
  const button = document.createElement("button");
  const actionKey = actionRunKey(node.action);
  button.type = "submit";
  button.className = "primary-button";
  button.disabled = state.runningActions.has(actionKey);
  button.textContent = state.runningActions.has(actionKey) ? "Running..." : "Run";
  form.append(button);
  form.addEventListener("submit", event => {
    event.preventDefault();
    const params = collectFormParams(form, node);
    if (!params) return;
    runAction({ action: node.action, label: node.title, params, confirm: false }, app);
  });
  panel.append(form);
  return panel;
}

function renderField(field, formKey) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";
  const kind = normalizeFieldKind(field.kind);
  const required = field.required === true;
  wrapper.innerHTML = `<span class="field-label">${escapeHtml(field.label)}${required ? " *" : ""}</span>`;
  let input;
  if (kind === "textarea") {
    input = document.createElement("textarea");
  } else if (kind === "boolean") {
    input = document.createElement("select");
    input.append(new Option("False", "false"));
    input.append(new Option("True", "true"));
  } else if (field.options?.length) {
    input = document.createElement("select");
    for (const option of field.options) {
      const value = scalarLabel(option);
      input.append(new Option(value, encodeFieldValue(option)));
    }
  } else {
    input = document.createElement("input");
    input.type = kind === "number" || kind === "integer" ? "number" : "text";
    if (kind === "integer") input.step = "1";
  }
  input.name = field.name;
  input.dataset.kind = kind;
  if (required) input.required = true;
  setInputValue(input, draftValueForField(formKey, field), kind);
  input.addEventListener("input", () => {
    rememberFormDraft(formKey, field.name, coerceFieldValue(input.value, kind, field));
  });
  input.addEventListener("change", () => {
    rememberFormDraft(formKey, field.name, coerceFieldValue(input.value, kind, field));
  });
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

function renderActionResult(result) {
  const panel = document.createElement("section");
  panel.className = "panel action-result";
  panel.dataset.level = result.level;
  panel.innerHTML = `
    <p class="eyebrow">${escapeHtml(result.level)}</p>
    <h3>${escapeHtml(result.title)}</h3>
    <p class="muted">${escapeHtml(result.body)}</p>
  `;
  if (result.detail !== undefined && result.detail !== null) {
    const pre = document.createElement("pre");
    pre.className = "json-preview";
    pre.textContent = jsonPreview(result.detail, 2400);
    panel.append(pre);
  }
  return panel;
}

async function runAction(node, app) {
  if (node.confirm && !window.confirm(`Run ${node.label}?`)) return;
  const actionKey = actionRunKey(node.action);
  if (state.runningActions.has(actionKey)) return;
  state.runningActions.add(actionKey);
  state.latestActionResult = {
    level: "info",
    title: "Action running",
    body: `Running ${node.label || node.action}.`,
    detail: null,
  };
  pushNotice("info", "Action started", `Running ${node.label || node.action}.`);
  render();
  try {
    const result = await postJson("/api/actions/run", {
      action: node.action,
      harness_id: app?.source?.harness_id || null,
      params: node.params ?? null,
    });
    state.latestActionResult = {
      level: "success",
      title: "Action completed",
      body: `${result.result.action} finished.`,
      detail: result.result.result,
    };
    pushNotice("success", "Action completed", `${result.result.action} finished.`);
    invalidateLists();
    await refresh({ reason: "action" });
  } catch (error) {
    state.latestActionResult = {
      level: "error",
      title: "Action failed",
      body: error.message,
      detail: null,
    };
    pushNotice("error", "Action failed", error.message);
    render();
  } finally {
    state.runningActions.delete(actionKey);
    render();
  }
}

async function loadVisibleLists() {
  const screen = selectedScreen();
  const nodes = flattenNodes(screen?.nodes ?? []);
  const requests = nodes.map(dataRequestForNode).filter(Boolean);
  await Promise.all(requests.map(loadDataRequest));
}

async function loadDataRequest(request) {
  const key = listKey(request);
  if (state.listCache.has(key) || state.loadingLists.has(key)) return;
  state.loadingLists.add(key);
  try {
    const payload = {
      source: request.source,
      where: request.where || {},
      limit: request.limit,
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

function dataRequestForNode(node) {
  if (!node?.source) return null;
  if (node.kind === "list") {
    return {
      source: node.source,
      where: node.where || {},
      limit: node.limit || null,
    };
  }
  if (node.kind === "activity" && node.source.startsWith("worklists.")) {
    return {
      source: node.source,
      where: {},
      limit: ACTIVITY_LIMIT,
    };
  }
  if (node.kind === "detail" && node.source.startsWith("worklists.")) {
    return {
      source: node.source,
      where: {},
      limit: DETAIL_LIMIT,
    };
  }
  if (node.kind === "report" && node.source.startsWith("worklists.")) {
    return {
      source: node.source,
      where: {},
      limit: REPORT_LIMIT,
    };
  }
  if (node.kind === "chart" && node.source.startsWith("worklists.")) {
    return {
      source: node.source,
      where: {},
      limit: CHART_LIMIT,
    };
  }
  return null;
}

function listKey(request) {
  return JSON.stringify({
    source: request.source,
    where: request.where || {},
    limit: request.limit || null,
  });
}

function selectDetailItem(node, items) {
  if (node.item_id) {
    return (
      items.find(item => item.public_id === node.item_id || String(item.id) === node.item_id) ||
      null
    );
  }
  return items.find(item => item.status === "pending") || items[0] || null;
}

function selectedListItem(key, items) {
  const selectedId = state.selectedListItems.get(key);
  if (selectedId) {
    const selected = items.find(item => itemKey(item) === selectedId);
    if (selected) return selected;
  }
  return items[0] || null;
}

function itemKey(item) {
  return item.public_id || String(item.id);
}

function fieldValue(item, field) {
  if (item[field] !== undefined && item[field] !== null) return scalarLabel(item[field]);
  if (item.metadata?.[field] !== undefined && item.metadata[field] !== null) {
    return scalarLabel(item.metadata[field]);
  }
  return "";
}

function reportMetrics(items) {
  const counts = groupCounts(items, "status");
  return [
    { label: "Total", value: items.length },
    { label: "Pending", value: counts.pending || 0 },
    { label: "Completed", value: counts.completed || 0 },
    { label: "Failed", value: counts.failed || 0 },
  ];
}

function chartGroupField(node) {
  if (node.intent === "kind_breakdown") return "kind";
  if (node.intent === "priority_breakdown") return "priority";
  return "status";
}

function groupCounts(items, field) {
  return items.reduce((counts, item) => {
    const label = fieldValue(item, field) || "unknown";
    counts[label] = (counts[label] || 0) + 1;
    return counts;
  }, {});
}

function collectFormParams(form, node) {
  if (!form.reportValidity()) return null;
  const params = { ...(node.params && typeof node.params === "object" ? node.params : {}) };
  const data = new FormData(form);
  try {
    for (const field of node.fields ?? []) {
      params[field.name] = coerceFieldValue(data.get(field.name), normalizeFieldKind(field.kind), field);
    }
    return params;
  } catch (error) {
    pushNotice("error", "Invalid form value", error.message);
    render();
    return null;
  }
}

function coerceFieldValue(value, kind, field = {}) {
  if (field.options?.length) return decodeFieldValue(value);
  if (value === null || value === "") return null;
  if (kind === "integer") {
    const parsed = Number(value);
    if (!Number.isInteger(parsed)) throw new Error(`${field.label || field.name} must be an integer.`);
    return parsed;
  }
  if (kind === "number") {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) throw new Error(`${field.label || field.name} must be a number.`);
    return parsed;
  }
  if (kind === "boolean") return value === "true";
  return value ?? "";
}

function normalizeFieldKind(kind) {
  const normalized = (kind || "text").toLowerCase();
  if (normalized === "bool" || normalized === "checkbox" || normalized === "switch") return "boolean";
  if (normalized === "int") return "integer";
  if (normalized === "multiline") return "textarea";
  return normalized;
}

function formDraftKey(node) {
  return node.id || node.action || node.title;
}

function fieldDraftKey(formKey, fieldName) {
  return `${formKey}:${fieldName}`;
}

function draftValueForField(formKey, field) {
  const key = fieldDraftKey(formKey, field.name);
  if (state.formDrafts.has(key)) return state.formDrafts.get(key);
  if (field.default !== undefined) return field.default;
  if (normalizeFieldKind(field.kind) === "boolean") return false;
  return "";
}

function rememberFormDraft(formKey, fieldName, value) {
  state.formDrafts.set(fieldDraftKey(formKey, fieldName), value);
}

function setInputValue(input, value, kind) {
  if (kind === "boolean") {
    input.value = value === true || value === "true" ? "true" : "false";
    return;
  }
  if (input.tagName === "SELECT" && input.options.length) {
    const encoded = encodeFieldValue(value);
    input.value = Array.from(input.options).some(option => option.value === encoded)
      ? encoded
      : input.options[0].value;
    return;
  }
  input.value = value === undefined || value === null ? "" : scalarLabel(value);
}

function encodeFieldValue(value) {
  return JSON.stringify(value);
}

function decodeFieldValue(value) {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function actionRunKey(action) {
  return action || "unknown";
}

function scalarLabel(value) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function jsonPreview(value, maxLength) {
  const rendered = JSON.stringify(value, null, 2);
  if (rendered.length <= maxLength) return rendered;
  return `${rendered.slice(0, maxLength)}\n... truncated`;
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
