local RELEASE_WORKLIST = "release"
local RELEASE_BINDING = "worklists." .. RELEASE_WORKLIST

local app = ui.app("Release Operator", {
  id = "release-operator",
  about = "Coordinate release approvals, intake, readiness, and notes from one focused operator console.",
  icon = "rocket",
})

local function release_list()
  return worklist(RELEASE_WORKLIST)
end

local function selected_release(params)
  if params and params.release then
    return tostring(params.release)
  end
  return "2026.06"
end

local function selected_release_mode(params)
  if params and params.release_mode then
    return tostring(params.release_mode)
  end
  return "standard"
end

local function notify_refresh(title, body, level)
  app:notice(title, {
    body = body,
    level = level or "info",
  })
  app:refresh(RELEASE_BINDING)
end

action.define("release.seed_demo_work", function(_ctx, params)
  local release = selected_release(params)
  local release_mode = selected_release_mode(params)
  local count = tonumber(params and params.count or 3) or 3
  local risk_threshold = tonumber(params and params.risk_threshold or 0.75) or 0.75
  local list = release_list()

  for i = 1, count do
    list:add({
      title = "Approve " .. release .. " checkpoint " .. tostring(i),
      kind = "approval",
      action = "release.approve_next",
      params = {
        release = release,
      },
      priority = count - i + 1,
      metadata = {
        lane = i % 2 == 0 and "qa" or "ops",
        release = release,
        release_mode = release_mode,
        risk_threshold = risk_threshold,
      },
    })
  end

  app:badge("approvals", {
    count = count,
    level = "info",
  })
  notify_refresh(
    "Seeded release work",
    "Created " .. tostring(count) .. " approval items for " .. release .. ".",
    "success"
  )

  return {
    status = "seeded",
    release = release,
    release_mode = release_mode,
    count = count,
    risk_threshold = risk_threshold,
  }
end)

action.define("release.approve_next", function(_ctx, params)
  local release = selected_release(params)
  local item = release_list():next({
    where = {
      kind = "approval",
      release = release,
    },
  })

  if item == nil then
    notify_refresh(
      "No approvals waiting",
      "There are no pending approvals for " .. release .. ".",
      "warning"
    )
    return {
      status = "empty",
      release = release,
    }
  end

  item:done({
    decision = "approved",
    decided_by = "release-operator",
  })

  notify_refresh("Approved next item", item.title, "success")

  return {
    status = "approved",
    release = release,
    item_id = item.id,
    title = item.title,
  }
end)

action.define("release.reject_next", function(_ctx, params)
  local release = selected_release(params)
  local item = release_list():next({
    where = {
      kind = "approval",
      release = release,
    },
  })

  if item == nil then
    notify_refresh(
      "No approvals waiting",
      "There are no pending approvals for " .. release .. ".",
      "warning"
    )
    return {
      status = "empty",
      release = release,
    }
  end

  item:fail("Rejected from the Release Operator UI")
  notify_refresh("Rejected next item", item.title, "warning")

  return {
    status = "rejected",
    release = release,
    item_id = item.id,
    title = item.title,
  }
end)

action.define("release.show_notes", function(_ctx, _params)
  app:show("release-notes", {
    presentation = "sheet",
  })
  return {
    status = "shown",
    target = "release-notes",
  }
end)

action.define("release.open_intake", function(_ctx, _params)
  app:open("intake")
  app:focus("seed-demo-form")
  return {
    status = "opened",
    target = "intake",
    focus = "seed-demo-form",
  }
end)

action.define("release.fail_diagnostic", function(_ctx, params)
  local reason = params and params.reason or "Release Operator diagnostic failure"
  error(reason)
end)

app:home("Release Desk", function(screen)
  screen:text("Track release approvals, seed demo work, and review readiness without leaving the operator console.")

  screen:section("Release Commands", function(section)
    section:action("Seed Approval Work", "release.seed_demo_work", {
      id = "seed-demo-work",
      params = {
        release = "2026.06",
        release_mode = "standard",
        count = 4,
      },
    })
    section:action("Approve Next", "release.approve_next", {
      id = "approve-next",
      confirm = true,
      params = {
        release = "2026.06",
      },
    })
    section:action("Reject Next", "release.reject_next", {
      id = "reject-next",
      confirm = true,
      params = {
        release = "2026.06",
      },
    })
    section:action("Open Release Notes", "release.show_notes", {
      id = "show-release-notes",
    })
    section:action("Open Intake Form", "release.open_intake", {
      id = "open-intake-form",
    })
  end)

  screen:worklist("Recent Release Work", {
    id = "recent-release-work",
    from = RELEASE_WORKLIST,
    limit = 8,
    fields = { "title", "status", "kind", "lane", "release", "priority" },
    intent = "tasks",
    as = "table",
  })

  screen:activity("Release Activity", {
    id = "release-activity",
    from = RELEASE_BINDING,
  })
end)

app:screen("approvals", "Approvals", function(screen)
  screen:worklist("Pending Approvals", {
    id = "pending-approvals",
    from = RELEASE_WORKLIST,
    where = {
      kind = "approval",
      status = "pending",
    },
    fields = { "title", "lane", "release", "priority", "status" },
    intent = "approval",
    as = "table",
  })
end)

app:screen("intake", "Intake", function(screen)
  screen:form("Create Demo Approval Batch", {
    id = "seed-demo-form",
    action = "release.seed_demo_work",
    params = {
      release = "2026.06",
      release_mode = "standard",
      count = 1,
      risk_threshold = 0.75,
    },
    fields = {
      { name = "release", label = "Release", type = "text", default = "2026.06" },
      { name = "release_mode", label = "Release Mode", type = "select", options = { "standard", "hotfix", "rollback" } },
      { name = "count", label = "Count", type = "number" },
      { name = "risk_threshold", label = "Risk Threshold", type = "decimal" },
    },
  })
end)

app:screen("overview", "Overview", function(screen)
  screen:section("Summary Surfaces", function(section)
    section:detail("Release Snapshot", {
      id = "release-snapshot",
      from = RELEASE_BINDING,
    })
    section:report("Release Readiness Report", {
      id = "release-readiness",
      from = RELEASE_BINDING,
      prompt = "Summarize current release approval readiness.",
    })
    section:chart("Approval Flow", {
      id = "approval-flow",
      from = RELEASE_BINDING,
      intent = "status_breakdown",
      as = "bar",
    })
  end)
end)

app:badge("release-readiness", {
  label = "live",
  level = "info",
})

app:pane("release-notes", "Release Notes", function(pane)
  pane:text("A lightweight pane can hold contextual workflow surfaces without changing the active screen.")
  pane:detail("Current Release Snapshot", {
    id = "pane-release-snapshot",
    from = RELEASE_BINDING,
  })
end, {
  presentation = "sheet",
})

app:menu("Main", function(menu)
  menu:item("Dashboard", "home", { icon = "layout-dashboard" })
  menu:item("Work", "approvals", { icon = "list-checks", badge = "approvals" }, function(sub)
    sub:item("Approvals", "approvals")
    sub:item("Intake", "intake")
  end)
  menu:item("Overview", "overview", { icon = "chart-bar" })
end)
