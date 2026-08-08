local WORK_LIST = "development-desk-work"
local REVIEW_LIST = "development-desk-reviews"
local BRIEF_LIST = "development-desk-briefs"

local WORK_BINDING = "worklists." .. WORK_LIST
local REVIEW_BINDING = "worklists." .. REVIEW_LIST
local BRIEF_BINDING = "worklists." .. BRIEF_LIST

local app = ui.app("Turin Development Desk", {
  id = "turin-development-desk",
  about = "Plan, build, review, and ship Turin from one durable agent workspace.",
  icon = "sparkles",
})

local function work()
  return worklist(WORK_LIST)
end

local function reviews()
  return worklist(REVIEW_LIST)
end

local function briefs()
  return worklist(BRIEF_LIST)
end

local function text_param(params, name, fallback)
  local value = params and params[name]
  if value == nil or tostring(value) == "" then
    return fallback
  end
  return tostring(value)
end

local function number_param(params, name, fallback)
  return tonumber(params and params[name]) or fallback
end

local function notify_refresh(title, body, level, ...)
  app:notice(title, {
    body = body,
    level = level or "info",
  })
  for _, binding in ipairs({ ... }) do
    app:refresh(binding)
  end
end

local function add_work_item(list, item)
  return list:add({
    title = item.title,
    kind = item.kind or "task",
    prompt = item.prompt or item.title,
    priority = item.priority or 0,
    metadata = {
      area = item.area or "Core",
      effort = item.effort or "Medium",
      stage = item.stage or "Backlog",
      source = item.source or "development-desk",
    },
  })
end

action.define("desk.seed", function(_ctx, _params)
  local list = work()
  if not list:empty() then
    notify_refresh(
      "Development desk already has work",
      "Starter work was not added because the backlog is not empty.",
      "info",
      WORK_BINDING
    )
    return { status = "unchanged", count = 0 }
  end

  local starter_work = {
    {
      title = "Dogfood a complete coding task through Turin",
      prompt = "Choose a contained improvement, use Turin to investigate and implement it, then record where the workflow helped or got in the way.",
      area = "Product",
      effort = "Medium",
      stage = "Ready",
      priority = 100,
    },
    {
      title = "Review the Development Desk visual hierarchy",
      prompt = "Review the desktop workspace at normal and compact widths in light and dark modes. Capture concrete hierarchy, density, contrast, and interaction issues.",
      area = "Desktop UI",
      effort = "Medium",
      stage = "Ready",
      priority = 90,
    },
    {
      title = "Run Turin's full pre-push validation",
      prompt = "Run scripts/prepush_ci.sh all, investigate failures, and keep behavior changes separate from cleanup.",
      area = "Quality",
      effort = "Small",
      stage = "Ready",
      priority = 80,
    },
    {
      title = "Explain one Turin workflow to a non-developer",
      prompt = "Choose one practical Turin scenario and explain the outcome, setup, and limits without leading with runtime implementation details.",
      area = "Docs",
      effort = "Small",
      stage = "Backlog",
      priority = 70,
    },
  }

  for _, item in ipairs(starter_work) do
    add_work_item(list, item)
  end

  app:badge("work", { count = #starter_work, level = "info" })
  notify_refresh(
    "Development desk is ready",
    "Added four practical starter tasks. Replace or complete them as you begin dogfooding.",
    "success",
    WORK_BINDING
  )
  return { status = "seeded", count = #starter_work }
end)

action.define("desk.capture", function(_ctx, params)
  local title = text_param(params, "title", nil)
  if title == nil then
    error("A task title is required")
  end

  local item = add_work_item(work(), {
    title = title,
    prompt = text_param(params, "description", title),
    area = text_param(params, "area", "Core"),
    effort = text_param(params, "effort", "Medium"),
    stage = "Backlog",
    priority = number_param(params, "priority", 50),
    source = "operator",
  })

  notify_refresh(
    "Task captured",
    title .. " is now in the development backlog.",
    "success",
    WORK_BINDING
  )
  app:open("work")
  app:focus("development-work")
  return { status = "captured", item_id = item.id, title = title }
end)

action.define("desk.complete_next", function(_ctx, params)
  local area = text_param(params, "area", nil)
  local options = nil
  if area ~= nil and area ~= "Any" then
    options = { where = { area = area } }
  end
  local item = work():next(options)
  if item == nil then
    notify_refresh(
      "No work is waiting",
      "Capture a task or change the selected area before trying again.",
      "warning",
      WORK_BINDING
    )
    return { status = "empty" }
  end

  item:done({
    outcome = text_param(params, "outcome", "Completed from the Development Desk"),
    completed_at = time.now_utc(),
  })
  notify_refresh("Task completed", item.title, "success", WORK_BINDING)
  return { status = "completed", item_id = item.id, title = item.title }
end)

action.define("desk.queue_review", function(_ctx, params)
  local item = work():next()
  if item == nil then
    notify_refresh(
      "No task is ready for review",
      "Capture work before creating a review request.",
      "warning",
      WORK_BINDING,
      REVIEW_BINDING
    )
    return { status = "empty" }
  end

  local review = reviews():add({
    title = "Review: " .. item.title,
    kind = "review",
    prompt = table.concat({
      "Review this Turin development task for correctness, regressions, and missing validation.",
      "",
      "Task: " .. item.title,
      "",
      item.prompt or "No additional task context was supplied.",
      "",
      "Operator note: " .. text_param(params, "note", "No additional note."),
    }, "\n"),
    priority = item.priority,
    metadata = {
      area = item.metadata and item.metadata.area or "Core",
      effort = item.metadata and item.metadata.effort or "Medium",
      stage = "Review",
      source_item = item.id,
    },
  })

  item:done({ review_item = review.id, outcome = "Sent to review" })
  app:badge("reviews", { count = 1, level = "warning" })
  notify_refresh(
    "Review queued",
    item.title .. " moved into the review queue.",
    "success",
    WORK_BINDING,
    REVIEW_BINDING
  )
  app:open("reviews")
  return { status = "queued", item_id = item.id, review_id = review.id }
end)

action.define("desk.approve_next_review", function(_ctx, _params)
  local item = reviews():next()
  if item == nil then
    notify_refresh(
      "Review queue is clear",
      "There are no pending review items.",
      "info",
      REVIEW_BINDING
    )
    return { status = "empty" }
  end

  item:done({ decision = "approved", decided_at = time.now_utc() })
  notify_refresh("Review approved", item.title, "success", REVIEW_BINDING)
  return { status = "approved", item_id = item.id, title = item.title }
end)

action.define("desk.create_plan", function(_ctx, params)
  local task = text_param(params, "task", nil)
  if task == nil then
    error("A planning task is required")
  end

  local plan = runtime.agent("planner"):ask(table.concat({
    "Prepare a pragmatic implementation plan for this Turin task.",
    "Identify likely files, sequencing, risks, and focused validation.",
    "Do not pad the plan with generic process steps.",
    "",
    task,
  }, "\n"))
  if plan == nil or plan == "" then
    error("The planner returned an empty plan")
  end

  local item = briefs():add({
    title = text_param(params, "title", "Implementation plan"),
    kind = "plan",
    prompt = plan,
    priority = 80,
    metadata = {
      area = text_param(params, "area", "Core"),
      effort = "Generated",
      stage = "Plan",
      generated_at = time.now_utc(),
    },
  })
  fs.write(".turin/runtime/development-desk/latest-plan.md", plan)

  notify_refresh(
    "Plan prepared",
    "The planner's brief is available in Plans & briefs.",
    "success",
    BRIEF_BINDING
  )
  app:open("briefs")
  return { status = "planned", item_id = item.id, title = item.title }
end)

action.define("desk.generate_review", function(_ctx, params)
  local change = text_param(params, "change", nil)
  if change == nil then
    error("A change summary is required")
  end

  local review = runtime.agent("reviewer"):ask(table.concat({
    "Review this Turin change as a strict senior engineer.",
    "Lead with concrete bugs or risks, then identify missing tests and residual uncertainty.",
    "",
    change,
  }, "\n"))
  if review == nil or review == "" then
    error("The reviewer returned an empty review")
  end

  local item = reviews():add({
    title = text_param(params, "title", "Generated change review"),
    kind = "review",
    prompt = review,
    priority = 90,
    metadata = {
      area = text_param(params, "area", "Core"),
      effort = "Generated",
      stage = "Review",
      generated_at = time.now_utc(),
    },
  })
  fs.write(".turin/runtime/development-desk/latest-review.md", review)

  app:badge("reviews", { count = 1, level = "warning" })
  notify_refresh(
    "Review prepared",
    "The generated review is waiting in the review queue.",
    "success",
    REVIEW_BINDING
  )
  app:open("reviews")
  return { status = "reviewed", item_id = item.id, title = item.title }
end)

action.define("desk.status", function(_ctx, _params)
  return {
    work = work():progress(),
    reviews = reviews():progress(),
    briefs = briefs():progress(),
  }
end)

app:home("Development overview", function(screen)
  screen:text("A working surface for shaping Turin: keep the conversation close, make durable work visible, and move deliberately from idea to review.")

  screen:section("Start here", function(section)
    section:action("Set up starter work", "desk.seed", {
      id = "seed-development-desk",
    })
    section:action("Capture a task", "desk.open_capture", {
      id = "open-task-capture",
    })
    section:action("Open project brief", "desk.show_project_brief", {
      id = "show-project-brief",
    })
  end)

  screen:worklist("Current work", {
    id = "current-work",
    from = WORK_LIST,
    where = { status = "pending" },
    fields = { "title", "area", "stage", "effort", "priority", "status" },
    intent = "tasks",
    as = "table",
    limit = 8,
  })

  screen:activity("Recent desk activity", {
    id = "desk-activity",
    from = WORK_BINDING,
  })
end)

action.define("desk.open_capture", function(_ctx, _params)
  app:open("capture")
  app:focus("capture-task")
  return { status = "opened", target = "capture-task" }
end)

action.define("desk.show_project_brief", function(_ctx, _params)
  app:show("project-brief", { presentation = "sheet" })
  return { status = "shown", target = "project-brief" }
end)

app:screen("work", "Work", function(screen)
  screen:text("The durable development backlog. Select a row for context, or use the controls to move the highest-priority pending task forward.")
  screen:section("Work controls", function(section)
    section:action("Capture task", "desk.open_capture")
    section:action("Send next to review", "desk.queue_review", { confirm = true })
    section:action("Complete next", "desk.complete_next", { confirm = true })
  end)
  screen:worklist("Development work", {
    id = "development-work",
    from = WORK_LIST,
    fields = { "title", "area", "stage", "effort", "priority", "status" },
    intent = "tasks",
    as = "table",
    limit = 40,
  })
end)

app:screen("reviews", "Reviews", function(screen)
  screen:text("Human and agent review stays explicit. Generated reviews are drafts for inspection, not automatic approval.")
  screen:section("Review controls", function(section)
    section:action("Generate review", "desk.open_review")
    section:action("Approve next review", "desk.approve_next_review", { confirm = true })
  end)
  screen:worklist("Review queue", {
    id = "review-queue",
    from = REVIEW_LIST,
    fields = { "title", "area", "stage", "priority", "status" },
    intent = "approval",
    as = "table",
    limit = 30,
  })
end)

action.define("desk.open_review", function(_ctx, _params)
  app:open("new-review")
  app:focus("generate-review")
  return { status = "opened", target = "generate-review" }
end)

app:screen("briefs", "Plans & briefs", function(screen)
  screen:text("Generated plans and durable working briefs remain inspectable after the conversation moves on.")
  screen:section("Planning controls", function(section)
    section:action("Create implementation plan", "desk.open_plan")
  end)
  screen:worklist("Planning briefs", {
    id = "planning-briefs",
    from = BRIEF_LIST,
    fields = { "title", "area", "stage", "created_at", "status" },
    intent = "documents",
    as = "table",
    limit = 30,
  })
end)

action.define("desk.open_plan", function(_ctx, _params)
  app:open("new-plan")
  app:focus("create-plan")
  return { status = "opened", target = "create-plan" }
end)

app:screen("capture", "Capture task", function(screen)
  screen:text("Capture enough context to make the task useful later. Keep implementation discussion in the Assistant surface.")
  screen:form("Add to development work", {
    id = "capture-task",
    action = "desk.capture",
    fields = {
      { name = "title", label = "Task", type = "text", required = true },
      { name = "description", label = "Outcome and context", type = "textarea" },
      { name = "area", label = "Area", type = "select", default = "Core", options = { "Core", "Runtime", "Desktop UI", "TUI", "Web", "Quality", "Docs", "Product" } },
      { name = "effort", label = "Effort", type = "select", default = "Medium", options = { "Small", "Medium", "Large" } },
      { name = "priority", label = "Priority", type = "integer", default = 50 },
    },
  })
end)

app:screen("new-plan", "Create plan", function(screen)
  screen:text("Delegate a bounded planning pass. The result becomes a durable brief and remains available to every Turin client.")
  screen:form("Prepare implementation plan", {
    id = "create-plan",
    action = "desk.create_plan",
    fields = {
      { name = "title", label = "Brief title", type = "text", default = "Implementation plan" },
      { name = "task", label = "Task and constraints", type = "textarea", required = true },
      { name = "area", label = "Area", type = "select", default = "Core", options = { "Core", "Runtime", "Desktop UI", "TUI", "Web", "Quality", "Docs", "Product" } },
    },
  })
end)

app:screen("new-review", "Generate review", function(screen)
  screen:text("Ask the reviewer specialist for a risk-first assessment, then inspect and approve the result from the Reviews screen.")
  screen:form("Review a change", {
    id = "generate-review",
    action = "desk.generate_review",
    fields = {
      { name = "title", label = "Review title", type = "text", default = "Generated change review" },
      { name = "change", label = "Change summary, diff, or review context", type = "textarea", required = true },
      { name = "area", label = "Area", type = "select", default = "Core", options = { "Core", "Runtime", "Desktop UI", "TUI", "Web", "Quality", "Docs", "Product" } },
    },
  })
end)

app:screen("insights", "Insights", function(screen)
  screen:text("A compact operational picture derived from the same durable work shown elsewhere in the desk.")
  screen:section("Delivery snapshot", function(section)
    section:report("Work summary", {
      id = "work-summary",
      from = WORK_BINDING,
      prompt = "Summarize current Turin development work, risks, and next priorities.",
    })
    section:chart("Work by status", {
      id = "work-status",
      from = WORK_BINDING,
      intent = "status_breakdown",
      as = "bar",
    })
  end)
end)

app:pane("project-brief", "Project brief", function(pane)
  pane:text("Turin is a lean, event-driven agentic runtime with composable Lua harnesses. This desk is the dogfooding surface: use Assistant for active development, Work for durable commitments, Reviews for explicit decisions, and Plans & briefs for reusable generated context.")
  pane:detail("Current delivery snapshot", {
    id = "project-delivery-snapshot",
    from = WORK_BINDING,
  })
end, {
  presentation = "sheet",
})

app:menu("Workspace", function(menu)
  menu:item("Overview", "home", { icon = "layout-dashboard" })
  menu:item("Work", "work", { icon = "list-checks", badge = "work" }, function(submenu)
    submenu:item("All work", "work")
    submenu:item("Capture task", "capture")
  end)
  menu:item("Reviews", "reviews", { icon = "shield-check", badge = "reviews" }, function(submenu)
    submenu:item("Review queue", "reviews")
    submenu:item("Generate review", "new-review")
  end)
  menu:item("Plans & briefs", "briefs", { icon = "file-text" }, function(submenu)
    submenu:item("Saved briefs", "briefs")
    submenu:item("Create plan", "new-plan")
  end)
  menu:item("Insights", "insights", { icon = "chart-bar" })
end)

function on_turn_prepare(ctx)
  local project_guidance = try(fs.read, "AGENTS.md")
  local work_progress = work():progress()
  local review_progress = reviews():progress()

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "You are operating inside Turin Development Desk.",
    "Treat the repository as a real product: inspect before editing, preserve behavior deliberately, and validate focused changes.",
    "Development work: " .. tostring(work_progress.done) .. "/" .. tostring(work_progress.total) .. " complete.",
    "Reviews: " .. tostring(review_progress.done) .. "/" .. tostring(review_progress.total) .. " complete.",
    project_guidance and ("\nRepository guidance:\n" .. project_guidance) or "",
  }, "\n")

  return ALLOW
end
