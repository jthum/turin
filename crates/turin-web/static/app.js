const state = {
  status: null,
  apps: [],
  selectedAppId: null,
  selectedScreenId: null,
  listCache: new Map(),
  selectedListItems: new Map(),
  localBadges: new Map(),
  loadingLists: new Set(),
  formDrafts: new Map(),
  runningActions: new Set(),
  notices: [],
  latestActionResult: null,
  activePaneId: null,
  pendingAction: null,
  appliedStatusUiRequests: false,
  refreshing: false,
  connection: {
    http: "connecting",
    events: "pending",
    eventErrors: 0,
  },
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
};

els.refreshButton.addEventListener("click", () => refresh({ reason: "manual" }));
document.addEventListener("keydown", event => {
  const dialog = activeOverlayDialog();
  if (event.key === "Tab" && dialog) {
    trapDialogTab(event, dialog);
    return;
  }
  if (event.key === "Escape" && state.pendingAction) {
    clearPendingAction();
    return;
  }
  if (event.key === "Escape" && state.activePaneId) {
    closePane();
  }
});

bootstrap();

async function bootstrap() {
  await refresh({ reason: "initial" });
  connectEvents();
}

async function refresh({ reason } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  setHttpStatus(reason === "initial" ? "connecting" : "refreshing");
  try {
    const status = await getJson("/api/status");
    state.status = status;
    state.apps = Object.values(status.ui?.apps ?? {});
    applyLocalBadgesToApps();
    selectDefaults();
    applyStatusUiRequestsOnce();
    await loadVisibleLists();
    render();
    setHttpStatus("live");
  } catch (error) {
    pushNotice("error", "Refresh failed", error.message);
    setHttpStatus("disconnected");
    render();
  } finally {
    state.refreshing = false;
  }
}

function connectEvents() {
  setEventStatus("connecting");
  const source = new EventSource("/api/events");
  source.addEventListener("open", () => {
    setEventStatus("live");
  });
  source.addEventListener("runtime.snapshot", () => {
    setEventStatus("live");
    invalidateLists();
    refresh({ reason: "event" });
  });
  source.addEventListener("ui.intent", event => {
    setEventStatus("live");
    const applied = applyUiIntentEvent(event, { reloadRefresh: false });
    if (!applied) invalidateLists();
    refresh({ reason: "ui" });
  });
  source.addEventListener("harness.action_ran", () => {
    setEventStatus("live");
    invalidateLists();
    refresh({ reason: "action" });
  });
  source.addEventListener("web.error", event => {
    state.connection.eventErrors += 1;
    setEventStatus("stream-error");
    pushNotice("error", "Event stream error", event.data || "Unknown event stream error");
    render();
  });
  source.addEventListener("error", () => {
    state.connection.eventErrors += 1;
    setEventStatus(source.readyState === EventSource.CLOSED ? "closed" : "reconnecting");
  });
}

function setHttpStatus(status) {
  state.connection.http = status;
  renderConnectionStatus();
}

function setEventStatus(status) {
  state.connection.events = status;
  renderConnectionStatus();
}

function renderConnectionStatus() {
  const http = state.connection.http;
  const events = state.connection.events;
  els.connectionStatus.textContent = `${httpStatusLabel(http)} · ${eventStatusLabel(events)}`;
  els.connectionStatus.dataset.state = connectionStatusLevel(http, events);
  els.connectionStatus.title = connectionStatusTitle(http, events);
}

function httpStatusLabel(status) {
  switch (status) {
    case "connecting":
      return "Runtime connecting";
    case "refreshing":
      return "Runtime refreshing";
    case "live":
      return "Runtime live";
    case "disconnected":
      return "Runtime disconnected";
    default:
      return "Runtime unknown";
  }
}

function eventStatusLabel(status) {
  switch (status) {
    case "pending":
      return "events pending";
    case "connecting":
      return "events connecting";
    case "live":
      return "events live";
    case "reconnecting":
      return "events reconnecting";
    case "stream-error":
      return "events errored";
    case "closed":
      return "events closed";
    default:
      return "events unknown";
  }
}

function connectionStatusLevel(http, events) {
  if (http === "disconnected" || events === "stream-error" || events === "closed") {
    return "error";
  }
  if (http === "refreshing" || events === "reconnecting") return "warning";
  if (http === "connecting" || events === "pending" || events === "connecting") return "info";
  return "success";
}

function connectionStatusTitle(http, events) {
  const pieces = [`HTTP refresh: ${http}`, `Event stream: ${events}`];
  if (state.connection.eventErrors > 0) {
    pieces.push(`Event stream errors observed: ${state.connection.eventErrors}`);
  }
  return pieces.join("; ");
}

function selectDefaults() {
  if (!state.apps.length) {
    state.selectedAppId = null;
    state.selectedScreenId = null;
    state.pendingAction = null;
    return;
  }
  if (!state.apps.some(app => app.id === state.selectedAppId)) {
    state.selectedAppId = state.apps[0].id;
    state.pendingAction = null;
  }
  const app = selectedApp();
  const screenIds = Object.keys(app?.screens ?? {});
  if (state.activePaneId && !app?.panes?.[state.activePaneId]) {
    state.activePaneId = null;
  }
  if (!screenIds.length) {
    state.selectedScreenId = null;
    state.pendingAction = null;
    return;
  }
  if (!screenIds.includes(state.selectedScreenId)) {
    state.selectedScreenId = defaultScreenIdForApp(app);
    state.pendingAction = null;
  }
}

function applyStatusUiRequestsOnce() {
  if (state.appliedStatusUiRequests) return;
  state.appliedStatusUiRequests = true;
  const ui = state.status?.ui;
  for (const open of ui?.opens ?? []) applyUiIntentPayload({ ...open, type: "open" });
  for (const show of ui?.shows ?? []) applyUiIntentPayload({ ...show, type: "show" });
  for (const focus of ui?.focuses ?? []) applyUiIntentPayload({ ...focus, type: "focus" });
  for (const refresh of ui?.refreshes ?? []) {
    applyUiIntentPayload({ ...refresh, type: "refresh" }, { reloadRefresh: false });
  }
  for (const notice of ui?.notices ?? []) {
    applyUiIntentPayload({ ...notice, type: "notify" });
  }
}

function applyUiIntentEvent(event, options = {}) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (error) {
    pushNotice("error", "UI event parse failed", error.message);
    return false;
  }
  return applyUiIntentPayload(payload, options);
}

function applyUiIntentPayload(payload, options = {}) {
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
    case "badge":
      return applyUiBadge(payload.app_id, payload.target, payload);
    case "refresh":
      return applyUiRefresh(payload.binding, options);
    default:
      return false;
  }
}

function applyUiIntentMessages(messages, options = {}) {
  let applied = false;
  for (const message of messages ?? []) {
    applied = applyUiIntentPayload(message, options) || applied;
  }
  return applied;
}

function applyUiRefresh(binding, { reloadRefresh = true } = {}) {
  if (!binding) return false;
  invalidateListBinding(binding);
  if (reloadRefresh) loadVisibleLists().then(render);
  return true;
}

function applyUiBadge(appId, target, payload) {
  const app = appById(appId);
  if (!app || !target) return false;
  const badge = {
    app_id: appId,
    target,
    count: payload.count ?? null,
    label: payload.label ?? null,
    level: normalizeBadgeLevel(payload.level),
    data: payload.data ?? {},
  };
  state.localBadges.set(localBadgeKey(appId, target), badge);
  app.badges = app.badges || {};
  app.badges[target] = badge;
  return true;
}

function applyLocalBadgesToApps() {
  for (const badge of state.localBadges.values()) {
    const app = appById(badge.app_id);
    if (!app) continue;
    app.badges = app.badges || {};
    app.badges[badge.target] = badge;
  }
}

function localBadgeKey(appId, target) {
  return `${appId}\n${target}`;
}

function normalizeBadgeLevel(level) {
  if (level === "success" || level === "warning" || level === "error" || level === "info") {
    return level;
  }
  return null;
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
  state.activePaneId = null;
  state.pendingAction = null;
  return true;
}

function applyUiShow(appId, target) {
  const app = selectAppById(appId);
  if (!app) return false;
  if (screenIdForTarget(app, target)) return applyUiOpen(appId, target, "show");
  if (app.panes?.[target]) {
    state.activePaneId = target;
    state.pendingAction = null;
    loadVisibleLists().then(render);
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
  const app = appById(appId);
  if (!app) {
    pushNotice("error", "UI request ignored", `App '${appId}' is not available.`);
    return null;
  }
  state.selectedAppId = app.id;
  return app;
}

function appById(appId) {
  return state.apps.find(candidate => candidate.id === appId) || null;
}

function screenIdForTarget(app, target) {
  if (app?.screens?.[target]) return target;
  return Object.values(app?.screens ?? []).find(screen => screen.title === target)?.id || null;
}

function defaultScreenIdForApp(app) {
  return screenIdForTarget(app, app?.opens_with) || Object.keys(app?.screens ?? {})[0] || null;
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
  renderPane();
  renderActionConfirmation();
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = "nav-button";
    button.setAttribute("aria-current", "true");
    const label = document.createElement("span");
    label.textContent = "Default Console";
    const badge = document.createElement("span");
    badge.className = "nav-badge";
    badge.dataset.level = "info";
    badge.textContent = "runtime";
    button.append(label, badge);
    els.appList.append(button);
    els.appList.append(
      emptyState(
        "No custom harness UI",
        "The default operator console is active. Declare ui.app(...) only when a harness needs workflow-specific screens."
      )
    );
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
      state.selectedScreenId = defaultScreenIdForApp(app);
      state.activePaneId = null;
      state.pendingAction = null;
      loadVisibleLists().then(render);
    });
    els.appList.append(button);
  }
}

function renderScreens() {
  clear(els.screenNav);
  const app = selectedApp();
  if (!app) {
    const title = document.createElement("p");
    title.className = "eyebrow";
    title.textContent = "Default";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "nav-button";
    button.setAttribute("aria-current", "true");
    button.textContent = "Runtime Overview";
    els.screenNav.append(title, button);
    return;
  }
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
    state.activePaneId = null;
    state.pendingAction = null;
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
  if (!app) {
    els.screen.append(renderDefaultConsole());
    return;
  }
  if (!screen) {
    els.screen.append(
      emptyState(
        "No screens declared",
        "This harness UI app exists, but it has no app:home(...) or app:screen(...) surfaces yet."
      )
    );
    return;
  }

  const heading = document.createElement("div");
  heading.className = "screen-heading";
  heading.innerHTML = `<h2>${escapeHtml(screen.title)}</h2><span class="muted code">${escapeHtml(screen.id)}</span>`;
  els.screen.append(heading);

  const stack = document.createElement("div");
  stack.className = "node-stack";
  const latestActionResult = latestActionResultForApp(app);
  if (latestActionResult) stack.append(renderActionResult(latestActionResult));
  for (const node of screen.nodes ?? []) {
    stack.append(renderNode(node, app));
  }
  if (!screen.nodes?.length) {
    stack.innerHTML = `<div class="empty-state"><span>Empty screen</span><p>This screen has no declared nodes.</p></div>`;
  }
  els.screen.append(stack);
}

function renderDefaultConsole() {
  const snapshot = state.status?.snapshot;
  const health = snapshot?.health;
  const daemon = snapshot?.status;
  const web = state.status?.web;
  const ui = state.status?.ui;
  const uiRequestCount =
    (ui?.opens?.length ?? 0) +
    (ui?.shows?.length ?? 0) +
    (ui?.focuses?.length ?? 0) +
    (ui?.refreshes?.length ?? 0);

  const wrapper = document.createElement("div");
  wrapper.className = "default-console";

  const heading = document.createElement("div");
  heading.className = "screen-heading";
  const title = document.createElement("div");
  title.innerHTML = `
    <p class="eyebrow">Default Operator Console</p>
    <h2>Turin is ready</h2>
  `;
  const target = document.createElement("span");
  target.className = "muted code";
  target.textContent = web ? `${web.connection_kind} target` : "runtime";
  heading.append(title, target);
  wrapper.append(heading);

  const lead = document.createElement("p");
  lead.className = "lede";
  lead.textContent =
    "You can inspect runtime health, sessions, tasks, events, and APIs without any harness-defined UI. Custom harness screens appear here only when a harness declares ui.app(...).";
  wrapper.append(lead);

  const grid = document.createElement("div");
  grid.className = "default-grid";
  grid.append(
    renderMetricPanel("Connection", [
      ["Status", health?.ready ? "ready" : "not ready"],
      ["Version", health?.version || web?.version || "unknown"],
      ["Transport", health?.transport || web?.connection_kind || "unknown"],
      ["Target", web?.connection_target || "unknown"],
    ]),
    renderMetricPanel("Registry", [
      ["Agents", daemon?.registry?.agents?.length ?? health?.agent_count ?? 0],
      ["Harnesses", daemon?.harnesses?.length ?? health?.harness_count ?? 0],
      ["Channels", daemon?.registry?.channels?.length ?? health?.channel_count ?? 0],
      ["Issues", daemon?.registry?.issues?.length ?? health?.issue_count ?? 0],
    ]),
    renderMetricPanel("Work", [
      ["Live sessions", snapshot?.live_sessions?.length ?? 0],
      ["Stored sessions", snapshot?.sessions?.length ?? 0],
      ["Tracked tasks", snapshot?.tasks?.length ?? 0],
      ["Active tasks", health?.active_task_count ?? 0],
    ]),
    renderMetricPanel("UI Signals", [
      ["Harness apps", state.apps.length],
      ["Notices", ui?.notices?.length ?? 0],
      ["Requests", uiRequestCount],
      ["Local notices", state.notices.length],
    ])
  );
  wrapper.append(grid);

  const guidance = document.createElement("section");
  guidance.className = "panel";
  guidance.innerHTML = "<h3>When to add harness UI</h3>";
  guidance.append(
    renderState(
      "Simple stays simple",
      "Keep using the default console until a workflow needs its own screens, lists, forms, reports, panes, badges, or action buttons.",
      "info"
    )
  );
  wrapper.append(guidance);

  const latestActionResult = latestActionResultForApp(null);
  if (latestActionResult) {
    wrapper.append(renderActionResult(latestActionResult));
  }

  return wrapper;
}

function renderMetricPanel(title, rows) {
  const panel = document.createElement("section");
  panel.className = "panel";
  const heading = document.createElement("h3");
  heading.textContent = title;
  panel.append(heading);
  const grid = document.createElement("div");
  grid.className = "stat-grid";
  for (const [label, value] of rows) {
    const card = document.createElement("div");
    card.className = "stat-card";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const valueNode = document.createElement("strong");
    valueNode.textContent = String(value ?? "unknown");
    card.append(labelNode, valueNode);
    grid.append(card);
  }
  panel.append(grid);
  return panel;
}

function renderPane() {
  const preferredAction = focusedDialogAction(".pane-sheet");
  document.querySelectorAll(".pane-overlay").forEach(node => node.remove());
  const app = selectedApp();
  const pane = selectedPane();
  if (!app || !pane) return;

  const overlay = document.createElement("div");
  overlay.className = "pane-overlay";
  overlay.addEventListener("click", event => {
    if (event.target === overlay) closePane();
  });

  const sheet = document.createElement("aside");
  sheet.className = "pane-sheet";
  sheet.setAttribute("role", "dialog");
  sheet.setAttribute("aria-modal", "true");
  sheet.setAttribute("aria-label", pane.title || pane.id);

  const header = document.createElement("div");
  header.className = "pane-header";
  const title = document.createElement("div");
  title.innerHTML = `<p class="eyebrow">Pane</p><h2>${escapeHtml(pane.title || pane.id)}</h2>`;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "ghost-button";
  close.dataset.autofocus = "true";
  close.dataset.dialogAction = "close";
  close.textContent = "Close";
  close.addEventListener("click", closePane);
  header.append(title, close);
  sheet.append(header);

  const stack = document.createElement("div");
  stack.className = "node-stack";
  const latestActionResult = latestActionResultForApp(app);
  if (latestActionResult) stack.append(renderActionResult(latestActionResult));
  for (const node of pane.nodes ?? []) {
    stack.append(renderNode(node, app));
  }
  if (!pane.nodes?.length) {
    stack.innerHTML = `<div class="empty-state"><span>Empty pane</span><p>This pane has no declared nodes.</p></div>`;
  }
  sheet.append(stack);
  overlay.append(sheet);
  document.body.append(overlay);
  focusDialogAction(sheet, preferredAction);
}

function closePane() {
  state.activePaneId = null;
  render();
}

function renderActionConfirmation() {
  const preferredAction = focusedDialogAction(".confirm-dialog");
  document.querySelectorAll(".confirm-overlay").forEach(node => node.remove());
  const pending = state.pendingAction;
  if (!pending) return;

  const overlay = document.createElement("div");
  overlay.className = "confirm-overlay";
  overlay.addEventListener("click", event => {
    if (event.target === overlay) clearPendingAction();
  });

  const dialog = document.createElement("section");
  dialog.className = "confirm-dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  dialog.setAttribute("aria-label", `Confirm ${pending.label}`);
  dialog.innerHTML = `
    <p class="eyebrow">Confirm action</p>
    <h2>${escapeHtml(pending.label)}</h2>
    <p class="muted">Run <span class="code">${escapeHtml(pending.action)}</span>?</p>
  `;

  if (pending.params !== undefined && pending.params !== null) {
    const pre = document.createElement("pre");
    pre.className = "json-preview";
    pre.textContent = jsonPreview(pending.params, 900);
    dialog.append(pre);
  }

  const row = document.createElement("div");
  row.className = "action-row confirm-actions";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "ghost-button";
  cancel.dataset.autofocus = "true";
  cancel.dataset.dialogAction = "cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", clearPendingAction);
  const run = document.createElement("button");
  run.type = "button";
  run.className = "danger-button";
  run.dataset.dialogAction = "run";
  run.textContent = "Confirm and run";
  run.disabled = state.runningActions.has(actionRunKey(pending.action));
  run.addEventListener("click", () => {
    const action = state.pendingAction;
    if (!action) return;
    state.pendingAction = null;
    render();
    runAction(action.node, action.app, { confirmed: true });
  });
  row.append(cancel, run);
  dialog.append(row);
  overlay.append(dialog);
  document.body.append(overlay);
  focusDialogAction(dialog, preferredAction);
}

function focusedDialogAction(selector) {
  const active = document.activeElement;
  if (!active?.closest?.(selector)) return null;
  return active.dataset.dialogAction || null;
}

function focusDialogAction(dialog, preferredAction = null) {
  const target =
    dialogActionButton(dialog, preferredAction) ||
    dialog.querySelector("[data-autofocus]:not(:disabled)") ||
    dialog.querySelector("button:not(:disabled)");
  target?.focus();
}

function dialogActionButton(dialog, action) {
  if (!action) return null;
  return Array.from(dialog.querySelectorAll("[data-dialog-action]")).find(
    button => button.dataset.dialogAction === action && !button.disabled
  ) || null;
}

function activeOverlayDialog() {
  return document.querySelector(".confirm-dialog") || document.querySelector(".pane-sheet");
}

function trapDialogTab(event, dialog) {
  const controls = focusableDialogControls(dialog);
  if (controls.length === 0) {
    event.preventDefault();
    return;
  }
  const first = controls[0];
  const last = controls[controls.length - 1];
  const active = document.activeElement;
  if (!dialog.contains(active)) {
    event.preventDefault();
    first.focus();
    return;
  }
  if (event.shiftKey && active === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
}

function focusableDialogControls(dialog) {
  return Array.from(dialog.querySelectorAll(
    "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]:not([tabindex='-1'])"
  )).filter(element => element.tabIndex >= 0);
}

function requestActionConfirmation(node, app) {
  state.pendingAction = {
    node,
    app,
    label: node.label || node.title || node.action,
    action: node.action,
    params: node.params ?? null,
  };
  render();
}

function clearPendingAction() {
  state.pendingAction = null;
  render();
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
      return renderActivity(node, app);
    case "detail":
      return renderDetail(node, app);
    case "report":
      return renderReport(node, app);
    case "chart":
      return renderChart(node, app);
    case "form":
      return renderForm(node, app);
    default:
      return renderPanel(`Unsupported node kind: ${node.kind || "unknown"}`, "muted");
  }
}

function renderSection(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
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
  appendNodeBadge(row, node, app);
  panel.append(row);
  return panel;
}

function renderList(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
  if (!node.source) {
    appendState(panel, "warning", "List source missing", unsupportedSourceMessage("list", node.source));
    return panel;
  }
  if (!isWorklistSource(node.source)) {
    appendState(panel, "warning", "Unsupported list source", unsupportedSourceMessage("list", node.source));
    return panel;
  }
  appendListMetadata(panel, node);
  const request = dataRequestForNode(node);
  if (!request) return panel;
  const key = listKey(request);
  const cached = state.listCache.get(key);
  if (state.loadingLists.has(key)) {
    appendState(panel, "loading", "Loading list", "Fetching current worklist rows.");
    return panel;
  }
  if (!cached) {
    appendState(panel, "info", "List not loaded yet", dataNotLoadedMessage("list"));
    return panel;
  }
  if (cached.error) {
    appendCachedDataError(panel, "List failed to load", cached, request);
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    appendState(panel, "empty", "No matching items", emptyListMessage(node));
    return panel;
  }

  const fields = node.fields?.length ? node.fields : ["title", "status", "kind", "priority"];
  const selectedItem = selectedListItem(key, items);
  if (selectedItem) state.selectedListItems.set(key, itemKey(selectedItem));
  const selectedIndex = selectedListItemIndex(selectedItem, items);
  appendListSummary(panel, items.length, selectedIndex);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const headers = [...fields.map(field => fieldHeaderLabel(field, node)), "Action"];
  table.innerHTML = `<thead><tr>${headers.map(header => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const item of items) {
    const row = document.createElement("tr");
    const selected = selectedItem && itemKey(item) === itemKey(selectedItem);
    row.className = "list-row";
    row.tabIndex = 0;
    row.dataset.listKey = key;
    row.dataset.itemKey = itemKey(item);
    row.setAttribute("aria-selected", selected ? "true" : "false");
    const cells = [
      ...fields.map(field => fieldValue(item, field)),
      workItemActionMarker(item),
    ];
    row.innerHTML = cells.map(value => `<td>${escapeHtml(value)}</td>`).join("");
    const select = () => {
      selectListItem(key, item);
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
        return;
      }
      const itemIndex = items.findIndex(candidate => itemKey(candidate) === itemKey(item));
      if (event.key === "ArrowDown") {
        event.preventDefault();
        selectListItemAt(key, items, itemIndex + 1, { focus: true });
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        selectListItemAt(key, items, itemIndex - 1, { focus: true });
      } else if (event.key === "Home") {
        event.preventDefault();
        selectListItemAt(key, items, 0, { focus: true });
      } else if (event.key === "End") {
        event.preventDefault();
        selectListItemAt(key, items, items.length - 1, { focus: true });
      }
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

function workItemActionMarker(item) {
  return item.action?.name ? "action" : "-";
}

function renderActivity(node, app) {
  const request = dataRequestForNode(node);
  if (!request) {
    return renderStatePanel("Unsupported activity source", unsupportedSourceMessage("activity", node.source), "warning");
  }
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    appendState(panel, "loading", "Loading activity", "Fetching recent worklist activity.");
    return panel;
  }
  if (!cached) {
    appendState(panel, "info", "Activity not loaded yet", dataNotLoadedMessage("activity"));
    return panel;
  }
  if (cached.error) {
    appendCachedDataError(panel, "Activity failed to load", cached, request);
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    appendState(panel, "empty", "No worklist activity yet", "Activity will appear after work items are created or updated.");
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
    return renderStatePanel("Unsupported detail source", unsupportedSourceMessage("detail", node.source), "warning");
  }
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    appendState(panel, "loading", "Loading detail", "Fetching worklist detail data.");
    return panel;
  }
  if (!cached) {
    appendState(panel, "info", "Detail not loaded yet", dataNotLoadedMessage("detail"));
    return panel;
  }
  if (cached.error) {
    appendCachedDataError(panel, "Detail failed to load", cached, request);
    return panel;
  }
  const items = cached.list?.items ?? [];
  if (!items.length) {
    appendState(panel, "empty", "No worklist items available", "Detail surfaces need at least one loaded work item.");
    return panel;
  }
  const item = selectDetailItem(node, items);
  if (!item) {
    appendState(panel, "empty", "Work item not found", `Work item '${node.item_id}' was not found in the loaded detail data.`);
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
    ["ID", item.public_id || String(item.id)],
    ["Title", item.title],
    ["Status", item.status],
    ["Kind", item.kind],
    ["Priority", item.priority],
    ["Worklist", item.worklist_id],
    ["Created", item.created_at],
    ["Updated", item.updated_at],
    ["Parent", item.parent_id],
    ["Paused", item.paused ? "yes" : null],
    ["Pause reason", item.pause_reason],
    ["Claimed by", item.claim_agent_id],
    ["Claimed at", item.claimed_at],
    ["Completed", item.completed_at],
    ["Release", metadata.release],
    ["Lane", metadata.lane],
    ["Failure", item.failure_reason],
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
    button.textContent = "Queue for confirmation";
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
    const hint = document.createElement("p");
    hint.className = "muted";
    hint.textContent = `Action ${item.action.name} requires confirmation before running.`;
    actions.append(hint, button);
    wrapper.append(actions);
  }
  return wrapper;
}

function renderReport(node, app) {
  const request = dataRequestForNode(node);
  if (!request) return renderPlaceholder(node, app);
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
  if (node.prompt) panel.append(renderText(node.prompt, "muted"));
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    appendState(panel, "loading", "Loading report", "Fetching worklist rows for this report.");
    return panel;
  }
  if (!cached) {
    appendState(panel, "info", "Report not loaded yet", dataNotLoadedMessage("report"));
    return panel;
  }
  if (cached.error) {
    appendCachedDataError(panel, "Report failed to load", cached, request);
    return panel;
  }
  const items = cached.list?.items ?? [];
  panel.append(renderMetricGrid(reportMetrics(items)));
  if (!items.length) {
    appendState(panel, "empty", "No report data yet", "This report will populate when the backing worklist has rows.");
    return panel;
  }
  const next = highestPriorityPendingItem(items);
  if (next) panel.append(renderReportHighlight(next, app));
  return panel;
}

function renderChart(node, app) {
  const request = dataRequestForNode(node);
  if (!request) return renderPlaceholder(node, app);
  const panel = document.createElement("section");
  panel.className = "panel";
  const groupField = chartGroupField(node);
  const label = node.render_as ? `${node.intent || "breakdown"} · ${node.render_as}` : node.intent || "breakdown";
  appendPanelHeading(panel, node.title, node, app);
  panel.append(renderText(`${label} · grouped by ${chartGroupLabel(node)}`, "muted"));
  const cached = state.listCache.get(listKey(request));
  if (state.loadingLists.has(listKey(request))) {
    appendState(panel, "loading", "Loading chart", "Fetching worklist rows for this chart.");
    return panel;
  }
  if (!cached) {
    appendState(panel, "info", "Chart not loaded yet", dataNotLoadedMessage("chart"));
    return panel;
  }
  if (cached.error) {
    appendCachedDataError(panel, "Chart failed to load", cached, request);
    return panel;
  }
  const items = cached.list?.items ?? [];
  panel.append(renderBars(groupCounts(items, groupField)));
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
    wrapper.append(renderState("No chart data yet", "This chart will populate when the backing worklist has rows.", "empty"));
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

function renderReportHighlight(item, app) {
  const wrapper = document.createElement("div");
  wrapper.className = "list-selection";
  const heading = document.createElement("h4");
  heading.textContent = "Next highest-priority pending item";
  wrapper.append(heading, renderWorkItemDetail(item, app));
  return wrapper;
}

function renderForm(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title, node, app);
  const form = document.createElement("form");
  const formKey = formDraftKey(node);
  form.className = "form-grid";
  for (const field of node.fields ?? []) {
    form.append(renderField(field, node, formKey));
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

function renderField(field, node, formKey) {
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
    input.type = inputTypeForFieldKind(kind);
    if (kind === "number") input.step = "any";
    if (kind === "integer") input.step = "1";
  }
  input.name = field.name;
  input.dataset.kind = kind;
  if (required) input.required = true;
  setInputValue(input, draftValueForField(formKey, field, node), kind);
  input.addEventListener("input", () => {
    rememberFormDraft(formKey, field.name, draftValueFromInput(input, kind, field));
  });
  input.addEventListener("change", () => {
    rememberFormDraft(formKey, field.name, draftValueFromInput(input, kind, field));
  });
  wrapper.append(input);
  return wrapper;
}

function renderPlaceholder(node, app) {
  const panel = document.createElement("section");
  panel.className = "panel";
  appendPanelHeading(panel, node.title || node.kind || "Unsupported", node, app);
  appendState(
    panel,
    "warning",
    `Unsupported ${node.kind || "surface"} source`,
    unsupportedSourceMessage(node.kind || "surface", node.source),
  );
  return panel;
}

function unsupportedSourceMessage(kind, source) {
  const surface = String(kind || "").trim() || "surface";
  const normalizedSource = String(source || "").trim();
  if (!normalizedSource) return `This ${surface} is declared and visible, but no source was provided. Add a worklists.<name> source or a deliberate adapter for this client.`;
  return `This ${surface} is declared and visible, but source '${normalizedSource}' cannot load in the browser yet. Only named worklists.<name> sources load today; model this data as a worklist or add a deliberate adapter for this client.`;
}

function dataNotLoadedMessage(kind) {
  const surface = String(kind || "").trim() || "surface";
  return `This ${surface} is visible, but its backing data has not loaded yet. It will appear after the client requests and receives the current data.`;
}

function renderPanel(text, className) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.append(renderText(text, className));
  return panel;
}

function renderStatePanel(title, body, level = "info") {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.append(renderState(title, body, level));
  return panel;
}

function appendState(panel, level, title, body) {
  panel.append(renderState(title, body, level));
}

function appendCachedDataError(panel, title, cached, request) {
  const details = cached.envelope?.details || {};
  const context = [cached.envelope?.code, details.source]
    .filter(value => typeof value === "string" && value.trim())
    .join(" · ");
  const body = context ? `${cached.error} (${context})` : cached.error;
  const node = renderState(title, body, "error");
  if (request) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "ghost-button";
    retry.textContent = "Retry data load";
    retry.addEventListener("click", () => {
      retryDataRequest(request).catch(error => {
        pushNotice("error", "Retry failed", error.message);
      });
    });
    node.append(retry);
  }
  panel.append(node);
}

function renderState(title, body, level = "info") {
  const node = document.createElement("div");
  node.className = "surface-state";
  node.dataset.level = level;
  const heading = document.createElement("strong");
  heading.textContent = title;
  const message = document.createElement("p");
  message.textContent = body || "";
  node.append(heading, message);
  return node;
}

function renderText(text, className) {
  const paragraph = document.createElement("p");
  paragraph.className = className;
  paragraph.textContent = text;
  return paragraph;
}

function appendPanelHeading(panel, title, node, app) {
  const heading = document.createElement("div");
  heading.className = "node-heading";
  const h3 = document.createElement("h3");
  h3.textContent = title;
  heading.append(h3);
  appendNodeBadge(heading, node, app);
  panel.append(heading);
}

function appendNodeBadge(container, node, app) {
  const badge = nodeBadge(app, node);
  if (!badge) return;
  const pill = document.createElement("span");
  pill.className = "node-badge";
  pill.dataset.level = badge.level;
  pill.textContent = badge.text;
  container.append(pill);
}

function nodeBadge(app, node) {
  if (!node?.id) return null;
  const badge = app?.badges?.[node.id];
  const text = badgeText(badge, null);
  if (!text) return null;
  return {
    text,
    level: badge?.level || "neutral",
  };
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
  const meta = actionResultMeta(result);
  if (meta.length) {
    const row = document.createElement("p");
    row.className = "action-result-meta";
    row.textContent = meta.join(" · ");
    panel.append(row);
  }
  if (result.detail !== undefined && result.detail !== null) {
    const pre = document.createElement("pre");
    pre.className = "json-preview";
    pre.textContent = jsonPreview(result.detail, 2400);
    panel.append(pre);
  }
  return panel;
}

function latestActionResultForApp(app) {
  const result = state.latestActionResult;
  if (!result) return null;
  const appId = app?.id || null;
  return (result.appId || null) === appId ? result : null;
}

function actionResultMeta(result) {
  return [
    result.action ? `Action ${result.action}` : null,
    result.agentId ? `Agent ${result.agentId}` : null,
    result.harnessId ? `Harness ${result.harnessId}` : null,
  ].filter(Boolean);
}

function actionCompletedBody(result) {
  return result.result === undefined || result.result === null
    ? `${result.action} completed without a result payload.`
    : `${result.action} finished.`;
}

async function runAction(node, app, options = {}) {
  if (node.confirm && !options.confirmed) {
    requestActionConfirmation(node, app);
    return;
  }
  const actionKey = actionRunKey(node.action);
  if (state.runningActions.has(actionKey)) return;
  state.runningActions.add(actionKey);
  state.latestActionResult = {
    appId: app?.id || null,
    action: node.action,
    agentId: app?.source?.agent_id || null,
    harnessId: app?.source?.harness_id || null,
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
      appId: app?.id || null,
      action: result.result.action,
      agentId: result.result.agent_id || app?.source?.agent_id || null,
      harnessId: result.result.harness_id || app?.source?.harness_id || null,
      level: "success",
      title: "Action completed",
      body: actionCompletedBody(result.result),
      detail: result.result.result,
    };
    pushNotice("success", "Action completed", actionCompletedBody(result.result));
    applyUiIntentMessages(result.result.ui_intents, { reloadRefresh: false });
    invalidateLists();
    await refresh({ reason: "action" });
  } catch (error) {
    state.latestActionResult = {
      appId: app?.id || null,
      action: node.action,
      agentId: app?.source?.agent_id || null,
      harnessId: app?.source?.harness_id || null,
      level: "error",
      title: "Action failed",
      body: error.message,
      detail: error.envelope || null,
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
  const pane = selectedPane();
  const nodes = [
    ...flattenNodes(screen?.nodes ?? []),
    ...flattenNodes(pane?.nodes ?? []),
  ];
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
    state.listCache.set(key, {
      error: error.message,
      envelope: error.envelope || null,
      status: error.status || null,
    });
  } finally {
    state.loadingLists.delete(key);
  }
}

async function retryDataRequest(request) {
  state.listCache.delete(listKey(request));
  const loading = loadDataRequest(request);
  render();
  await loading;
  render();
}

function invalidateLists() {
  state.listCache.clear();
}

function invalidateListBinding(binding) {
  for (const key of Array.from(state.listCache.keys())) {
    const request = parseListKey(key);
    if (request?.source === binding) state.listCache.delete(key);
  }
}

function parseListKey(key) {
  try {
    return JSON.parse(key);
  } catch {
    return null;
  }
}

function selectedApp() {
  return state.apps.find(app => app.id === state.selectedAppId) || null;
}

function selectedScreen() {
  const app = selectedApp();
  return app?.screens?.[state.selectedScreenId] || null;
}

function selectedPane() {
  const app = selectedApp();
  return app?.panes?.[state.activePaneId] || null;
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
    if (!isWorklistSource(node.source)) return null;
    return {
      source: node.source,
      where: node.where || {},
      limit: node.limit || null,
    };
  }
  if (node.kind === "activity" && isWorklistSource(node.source)) {
    return {
      source: node.source,
      where: {},
      limit: ACTIVITY_LIMIT,
    };
  }
  if (node.kind === "detail" && isWorklistSource(node.source)) {
    return {
      source: node.source,
      where: {},
      limit: DETAIL_LIMIT,
    };
  }
  if (node.kind === "report" && isWorklistSource(node.source)) {
    return {
      source: node.source,
      where: {},
      limit: REPORT_LIMIT,
    };
  }
  if (node.kind === "chart" && isWorklistSource(node.source)) {
    return {
      source: node.source,
      where: {},
      limit: CHART_LIMIT,
    };
  }
  return null;
}

function isWorklistSource(source) {
  return (
    typeof source === "string" &&
    source.startsWith("worklists.") &&
    source.slice("worklists.".length).trim().length > 0
  );
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

function selectedListItemIndex(selectedItem, items) {
  if (!selectedItem) return -1;
  return items.findIndex(item => itemKey(item) === itemKey(selectedItem));
}

function appendListSummary(panel, itemCount, selectedIndex) {
  const summary = document.createElement("p");
  summary.className = "list-summary";
  const selected =
    selectedIndex >= 0 && selectedIndex < itemCount ? ` · selected ${selectedIndex + 1}` : "";
  summary.textContent = `Rows 1-${itemCount} of ${itemCount}${selected}`;
  panel.append(summary);
}

function appendListMetadata(panel, node) {
  const parts = listMetadataParts(node);
  if (!parts.length) return;
  const meta = document.createElement("p");
  meta.className = "list-summary";
  meta.textContent = parts.join(" · ");
  panel.append(meta);
}

function listMetadataParts(node) {
  const parts = [];
  const whereFields = filterFields(node.where);
  if (whereFields.length) parts.push(`Where ${whereFields.join(",")}`);
  const sortFields = sortFieldsForDisplay(node.sort || []);
  if (sortFields.length) parts.push(`Sort ${sortFields.join(",")}`);
  if (node.limit) parts.push(`Limit ${node.limit}`);
  return parts;
}

function filterFields(where) {
  if (!where || typeof where !== "object" || Array.isArray(where)) return [];
  return Object.keys(where).sort();
}

function sortFieldsForDisplay(sort) {
  return sort.map(sortEntryField).filter(Boolean);
}

function emptyListMessage(node) {
  const whereCount =
    node.where && typeof node.where === "object" ? Object.keys(node.where).length : 0;
  return whereCount
    ? `This worklist query returned no rows after applying ${whereCount} declared filter(s).`
    : "This worklist query returned no rows.";
}

function selectListItem(key, item, options = {}) {
  state.selectedListItems.set(key, itemKey(item));
  render();
  if (options.focus) queueMicrotask(() => focusSelectedListRow(key));
}

function selectListItemAt(key, items, index, options = {}) {
  if (!items.length) return;
  const boundedIndex = Math.max(0, Math.min(index, items.length - 1));
  selectListItem(key, items[boundedIndex], options);
}

function focusSelectedListRow(key) {
  const selectedId = state.selectedListItems.get(key);
  if (!selectedId) return;
  const row = Array.from(document.querySelectorAll(".list-row")).find(
    candidate => candidate.dataset.listKey === key && candidate.dataset.itemKey === selectedId,
  );
  row?.focus();
}

function itemKey(item) {
  return item.public_id || String(item.id);
}

function fieldValue(item, field) {
  if (field === "id" || field === "public_id") return itemKey(item);
  if (field === "internal_id") return String(item.id);
  if (item[field] !== undefined && item[field] !== null) return scalarLabel(item[field]);
  if (item.metadata?.[field] !== undefined && item.metadata[field] !== null) {
    return scalarLabel(item.metadata[field]);
  }
  return "";
}

function fieldLabel(field) {
  return String(field || "")
    .split(/[_.]/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function fieldHeaderLabel(field, node) {
  const label = fieldLabel(field);
  const index = sortFieldIndex(field, node?.sort || []);
  if (index < 0) return label;
  const direction = sortFieldDirection(field, node?.sort || []);
  return `${label} [sort ${index + 1}${direction ? ` ${direction}` : ""}]`;
}

function sortFieldIndex(field, sort) {
  return sort.findIndex(entry => sortEntryField(entry) === field);
}

function sortFieldDirection(field, sort) {
  const entry = sort.find(candidate => sortEntryField(candidate) === field);
  return sortEntryDirection(entry);
}

function sortEntryField(entry) {
  return String(entry || "")
    .trim()
    .replace(/^[+-]+/, "")
    .trim()
    .split(/\s+/)[0]
    .split(":")[0];
}

function sortEntryDirection(entry) {
  const raw = String(entry || "").trim();
  const prefixDirection = raw.startsWith("-") ? "desc" : raw.startsWith("+") ? "asc" : "";
  const normalized = raw.replace(/^[+-]+/, "").trim();
  const [fieldToken, directionToken] = normalized.split(/\s+/);
  const colonDirection = directionLabel((fieldToken || "").split(":")[1]);
  return colonDirection || directionLabel(directionToken) || prefixDirection;
}

function directionLabel(value) {
  const normalized = String(value || "").trim().replace(/[,;]+$/, "").toLowerCase();
  if (normalized === "asc" || normalized === "ascending") return "asc";
  if (normalized === "desc" || normalized === "descending") return "desc";
  return "";
}

function reportMetrics(items) {
  const counts = groupCounts(items, "status");
  const metrics = [
    { label: "Total", value: items.length },
    { label: "Pending", value: counts.pending || 0 },
    { label: "Claimed", value: counts.claimed || 0 },
    { label: "Done", value: counts.done || 0 },
    { label: "Failed", value: counts.failed || 0 },
  ];
  const known =
    (counts.pending || 0) +
    (counts.claimed || 0) +
    (counts.done || 0) +
    (counts.failed || 0);
  const other = items.length - known;
  if (other > 0) metrics.push({ label: "Other", value: other });
  return metrics;
}

function highestPriorityPendingItem(items) {
  return items
    .filter(item => item.status === "pending")
    .sort((left, right) => (right.priority || 0) - (left.priority || 0))[0] || null;
}

function chartGroupField(node) {
  if (node.intent === "kind_breakdown") return "kind";
  if (node.intent === "priority_breakdown") return "priority";
  return "status";
}

function chartGroupLabel(node) {
  return fieldLabel(chartGroupField(node));
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
      const rawValue = data.get(field.name);
      if (!field.required && (rawValue === null || rawValue === "")) continue;
      params[field.name] = coerceFieldValue(rawValue, normalizeFieldKind(field.kind), field);
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
  if (normalized === "float" || normalized === "decimal") return "number";
  if (normalized === "multiline" || normalized === "markdown") return "textarea";
  if (normalized === "secret" || normalized === "passphrase") return "password";
  return normalized;
}

function inputTypeForFieldKind(kind) {
  if (kind === "number" || kind === "integer") return "number";
  if (kind === "password") return "password";
  return "text";
}

function formDraftKey(node) {
  return node.id || node.action || node.title;
}

function fieldDraftKey(formKey, fieldName) {
  return `${formKey}:${fieldName}`;
}

function draftValueForField(formKey, field, node) {
  const key = fieldDraftKey(formKey, field.name);
  if (state.formDrafts.has(key)) return state.formDrafts.get(key);
  if (field.default !== undefined) return field.default;
  if (node?.params && typeof node.params === "object" && field.name in node.params) {
    return node.params[field.name];
  }
  if (normalizeFieldKind(field.kind) === "boolean") return false;
  return "";
}

function rememberFormDraft(formKey, fieldName, value) {
  state.formDrafts.set(fieldDraftKey(formKey, fieldName), value);
}

function draftValueFromInput(input, kind, field) {
  if (input.tagName === "SELECT" && field.options?.length) {
    return decodeFieldValue(input.value);
  }
  if (kind === "boolean") {
    return input.value === "true";
  }
  return input.value;
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
    const error = new Error(errorMessageFromEnvelope(body, response));
    error.envelope = body?.error || null;
    error.status = response.status;
    throw error;
  }
  return body;
}

function errorMessageFromEnvelope(body, response) {
  const message = body?.error?.message || `${response.status} ${response.statusText}`;
  const guidance = body?.error?.details?.guidance;
  if (typeof guidance === "string" && guidance.trim()) {
    return `${message} ${guidance}`;
  }
  return message;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function emptyState(title, body) {
  const node = document.createElement("div");
  node.className = "empty-state";
  const heading = document.createElement("span");
  heading.textContent = title;
  const message = document.createElement("p");
  message.textContent = body;
  node.append(heading, message);
  return node;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
